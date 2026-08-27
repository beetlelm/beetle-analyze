"""blimp_eval_oneshot.py — single-model BLiMP runner for the new-org sweep.

Why a separate runner: ``legacy/eval_model.py`` infers applicable benchmarks
through ``MODEL_GROUPS``, which only knows the {humanscale, fineweb} datasets
in the slug regex. The new orgs (``Beetle-FineWeb-100M``, ``Beetle-FineWeb-2B``,
``Beetle-FineWeb2-24B``, ``Beetle-nld-eng-Variants``) do not match that regex,
so the auto-registration logic silently skips them or assigns the wrong
benchmark set. This module bypasses ``MODEL_GROUPS`` entirely: it parses the
slug locally, decides which BLiMP suites are applicable from the (l1, l2)
pair, and calls the same ``run_minimal_pairs`` / ``preload_all_datasets``
helpers that ``legacy/eval_model.py`` uses.

Writes results into the canonical CSVs that ``analyze.blimp_scale_curriculum``
already reads from:

    results/evals/blimp/blimp_results.csv
    results/blimp/{benchmark}_results.csv  (legacy mirror, optional)

Resume is honoured via the same ``utils.already_done`` cache.
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "legacy"))

# legacy/eval_model.py installs a basicConfig with format "[rank%(rank)s] …".
# We are not running under its CLI, so log records will not carry a `rank`
# attribute and emission will explode. Inject a default rank=0 onto every
# record before legacy's basicConfig fires.
_old_factory = logging.getLogRecordFactory()
def _rank_factory(*a, **kw):
    rec = _old_factory(*a, **kw)
    if not hasattr(rec, "rank"):
        rec.rank = 0
    return rec
logging.setLogRecordFactory(_rank_factory)

# These imports come from legacy/eval_model.py.
from eval_model import (  # type: ignore[import-not-found]
    BENCHMARKS,
    preload_all_datasets,
    run_minimal_pairs,
)
from utils import already_done, append_result  # type: ignore[import-not-found]

from analyze.blimp_scale_curriculum import parse_slug

log = logging.getLogger("blimp_oneshot")

# Global, lazy-loaded dataset cache so the runner can iterate models in a
# single Python process and not reload BLiMP data each invocation.
_DATASET_CACHE: dict[str, dict[str, list]] = {}


def _ensure_datasets(benchmarks: list[str]) -> dict[str, dict[str, list]]:
    needed = [b for b in benchmarks if b not in _DATASET_CACHE]
    if needed:
        os.environ["BLIMP_TASKS"] = ",".join(needed)
        loaded = preload_all_datasets(log)
        _DATASET_CACHE.update(loaded)
    return _DATASET_CACHE


def applicable_benchmarks(repo: str) -> list[tuple[str, str, str]]:
    """Return [(bm_name, lang_code, lang_name), ...] given just the repo slug.

    Uses parse_slug to recover (l1, l2). For mono models, only the model's
    single language counts. For bilinguals both l1 and l2 are tested if the
    BLiMP suite has a config for that ISO.
    """
    slug = parse_slug(repo)
    if slug is None:
        return []
    isos = {x for x in (slug.l1, slug.l2) if x}
    out: list[tuple[str, str, str]] = []
    for bm_name, cfg in BENCHMARKS.items():
        for lang_code, lang_name in cfg["langs"].items():
            if lang_code in isos:
                out.append((bm_name, lang_code, lang_name))
    # Deduplicate while preserving order.
    seen: set[tuple[str, str]] = set()
    uniq: list[tuple[str, str, str]] = []
    for row in out:
        key = (row[0], row[1])
        if key not in seen:
            seen.add(key)
            uniq.append(row)
    return uniq


def _csv_for(bm_name: str, results_dir: Path) -> Path:
    """Where each benchmark's per-row CSV lives. Mirror legacy/eval_model.py."""
    return results_dir / f"{bm_name}_results.csv"


def _write_evals_blimp_row(repo: str, bm_name: str, lang_name: str, iso: str,
                           accuracy: float, n_correct: int, n_total: int,
                           output_root: Path) -> None:
    """Append a row to results/evals/blimp/blimp_results.csv (final-state)."""
    out = output_root / "evals" / "blimp" / "blimp_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    cols = ["benchmark", "model", "model_family", "eval_language",
            "iso_code", "accuracy", "n_correct", "n_total"]
    org = repo.split("/", 1)[0].lower().replace("-", "_")
    row = [bm_name, repo, org, lang_name, iso,
           f"{accuracy:.6f}", str(n_correct), str(n_total)]
    with open(out, "a", encoding="utf-8") as fh:
        if new:
            fh.write(",".join(cols) + "\n")
        fh.write(",".join(row) + "\n")


def evaluate_model(repo: str, *, output_root: Path,
                   batch_size: int = 64, hf_token: str | None = None,
                   benchmarks: list[str] | None = None) -> dict:
    """Run all applicable BLiMP suites on a single model's main checkpoint.

    Returns a summary dict {bm_name: accuracy} plus error info.
    """
    tasks = applicable_benchmarks(repo)
    if benchmarks:
        tasks = [t for t in tasks if t[0] in set(benchmarks)]
    if not tasks:
        log.warning("[%s] no applicable benchmarks (slug parse failed?)", repo)
        return {"repo": repo, "scores": {}, "skipped": True}

    legacy_results_dir = output_root / "blimp"
    legacy_results_dir.mkdir(parents=True, exist_ok=True)

    needed: list[tuple[str, str, str, Path]] = []
    for bm_name, lang_code, lang_name in tasks:
        csv_path = _csv_for(bm_name, legacy_results_dir)
        if already_done(str(csv_path), repo, "main", lang_code, bm_name):
            log.info("[%s] %s/%s already done — skip", repo, bm_name, lang_code)
            continue
        needed.append((bm_name, lang_code, lang_name, csv_path))

    if not needed:
        return {"repo": repo, "scores": {}, "skipped": "all-done"}

    _ensure_datasets([t[0] for t in needed])

    log.info("[%s] loading model …", repo)
    t0 = time.monotonic()
    try:
        model = (AutoModelForCausalLM
                 .from_pretrained(repo, revision="main",
                                  trust_remote_code=True,
                                  torch_dtype=torch.float16,
                                  low_cpu_mem_usage=True,
                                  token=hf_token)
                 .to("cuda:0").eval())
        tok = AutoTokenizer.from_pretrained(repo, revision="main",
                                            trust_remote_code=True,
                                            token=hf_token)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    except Exception as e:
        log.error("[%s] model load failed: %s", repo, e)
        return {"repo": repo, "scores": {}, "error": str(e)[:300]}

    scores: dict[str, float] = {}
    for bm_name, lang_code, lang_name, csv_path in needed:
        cfg = BENCHMARKS[bm_name]
        try:
            data = _DATASET_CACHE[bm_name][lang_code]
        except KeyError:
            log.warning("[%s] no data for %s/%s — skip", repo, bm_name, lang_code)
            continue
        try:
            acc, n_correct, n_total = run_minimal_pairs(
                model, tok, data, torch.device("cuda:0"), batch_size,
            )
            log.info("[%s] %-10s %-8s acc=%.4f (%d/%d)",
                     repo, bm_name, lang_code, acc, n_correct, n_total)
            append_result(str(csv_path), {
                "benchmark": bm_name,
                "model": repo,
                "lang_pair": "",
                "bilingual_type": "",
                "checkpoint": "main",
                "eval_language": lang_name,
                "lang_code": lang_code,
                "accuracy": round(acc, 6),
                "n_correct": n_correct,
                "n_total": n_total,
            })
            _write_evals_blimp_row(repo, bm_name, lang_name, lang_code,
                                   acc, n_correct, n_total, output_root)
            scores[f"{bm_name}/{lang_code}"] = acc
        except Exception as e:
            log.error("[%s] %s/%s eval failed: %s", repo, bm_name, lang_code, e)

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("[%s] done in %.1fs", repo, time.monotonic() - t0)
    return {"repo": repo, "scores": scores}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repos", nargs="+", required=True,
                   help="One or more HF repo slugs, e.g. "
                        "Beetle-FineWeb-100M/beetle-bilingual-...-nld-eng")
    p.add_argument("--output-root", type=Path, default=ROOT / "results")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--benchmarks", nargs="*", default=None,
                   help="Restrict to a subset of "
                        "{blimp_eng, blimp_nl, zhoblimp, multiblimp}")
    p.add_argument("--walltime-seconds", type=int, default=0,
                   help="Stop scheduling new models once this many seconds "
                        "have elapsed since process start (0 = no cap).")
    p.add_argument("--regen-figures", action="store_true",
                   help="After each model, re-run analyze.blimp_scale_curriculum.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")
    t_start = time.monotonic()
    deadline = t_start + args.walltime_seconds if args.walltime_seconds else None
    summary: list[dict] = []
    for repo in args.repos:
        if deadline and time.monotonic() > deadline:
            log.warning("walltime reached — stopping (remaining %d models skipped)",
                        len(args.repos) - len(summary))
            break
        try:
            res = evaluate_model(repo, output_root=args.output_root,
                                 batch_size=args.batch_size,
                                 benchmarks=args.benchmarks)
            summary.append(res)
        except Exception as e:
            log.exception("[%s] uncaught error: %s", repo, e)
            summary.append({"repo": repo, "error": str(e)[:300]})
        if args.regen_figures:
            try:
                from analyze.blimp_scale_curriculum import render_all
                render_all()
                log.info("figures regenerated")
            except Exception as e:
                log.exception("figure regen failed: %s", e)

    elapsed = time.monotonic() - t_start
    log.info("oneshot complete: %d models in %.1f min",
             len(summary), elapsed / 60.0)


if __name__ == "__main__":
    main()
