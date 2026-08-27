#!/usr/bin/env python3
"""
evals/eval_blimp.py — Minimal-pair grammaticality evaluation.

Evaluates models on four BLiMP-style benchmarks:
  blimp_eng   — English  (nyu-mll/blimp, ~67 k pairs)
  blimp_nl    — Dutch    (juletxara/blimp-nl)
  zhoblimp    — Chinese  (Junrui1202/zhoblimp)
  multiblimp  — 5 langs  (jumelet/multiblimp: nld/deu/fra/fas/bul)

Completed (repo, benchmark, lang) triples are recorded in the eval ledger
(analyze/eval_ledger.py) and skipped on re-runs without re-scoring.

Usage
-----
  # All Beetle orgs + all baselines, all benchmarks, 8-GPU fan-out
  CUDA_VISIBLE_DEVICES=0 python evals/eval_blimp.py \\
      --models beetle-humanscale beetle-fineweb baselines \\
      --rank 0 --world-size 8 --output results/evals/blimp

  # Quick test: BLOOM, English BLiMP only
  python evals/eval_blimp.py --models bloom --benchmarks blimp_eng \\
      --output /tmp/test-evals
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import get_dataset_config_names, load_dataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "analyze"))

from evals.baseline_models import resolve_models
from evals.model_loader import (
    DEVICE, load_model, unload_model, setup_logging, log_gpu_mem,
    resolve_vocab_size,
)

import logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ledger (best-effort — never crashes the eval loop)
# ---------------------------------------------------------------------------

try:
    from analyze.eval_ledger import is_done as _ledger_is_done
    from analyze.eval_ledger import record as _ledger_record
except Exception:
    def _ledger_is_done(*_a, **_kw) -> bool: return False   # type: ignore[misc]
    def _ledger_record(*_a, **_kw) -> None: pass             # type: ignore[misc]

# ---------------------------------------------------------------------------
# Benchmark definitions (mirrors legacy/eval_model.py BENCHMARKS)
# ---------------------------------------------------------------------------

BENCHMARKS: dict[str, dict] = {
    "blimp_eng": dict(
        hf_id="nyu-mll/blimp",
        config_mode="all_configs",
        good_col="sentence_good",
        bad_col="sentence_bad",
        eval_language="English",
        iso_code="eng",
        pkl_name="blimp_eng.pkl",
    ),
    "blimp_nl": dict(
        hf_id="juletxara/blimp-nl",
        config_mode="all_configs",
        good_col="sentence_good",
        bad_col="sentence_bad",
        eval_language="Dutch",
        iso_code="nld",
        pkl_name="blimp_nl.pkl",
    ),
    "zhoblimp": dict(
        hf_id="Junrui1202/zhoblimp",
        config_mode="all_configs",
        good_col="sentence_good",
        bad_col="sentence_bad",
        eval_language="Chinese",
        iso_code="zho",
        pkl_name="zhoblimp.pkl",
    ),
    "multiblimp": dict(
        hf_id="jumelet/multiblimp",
        config_mode="per_lang",
        good_col="sen",
        bad_col="wrong_sen",
        langs={
            "nld": ("Dutch",      "nld"),
            "deu": ("German",     "deu"),
            "fra": ("French",     "fra"),
            "fas": ("Persian",    "fas"),
            "bul": ("Bulgarian",  "bul"),
            "ukr": ("Ukrainian",  "ukr"),
            "eng": ("English",    "eng"),
            "jpn": ("Japanese",   "jpn"),
            "zho": ("Chinese",    "zho"),
        },
    ),
}

# ---------------------------------------------------------------------------
# Dataset loading (with pkl cache + exponential-backoff retry)
# ---------------------------------------------------------------------------

def _load_all_configs(
    hf_id: str, good_col: str, bad_col: str, pkl_name: str, pkl_dir: str,
) -> List[Tuple[str, str]]:
    path = os.path.join(pkl_dir, pkl_name) if pkl_dir else ""
    if path and os.path.exists(path):
        with open(path, "rb") as fh:
            pairs = pickle.load(fh)
        logger.info(f"  pkl cache hit: {len(pairs):,} pairs ({pkl_name})")
        return pairs

    logger.info(f"  Downloading {hf_id} (no pkl at {path or 'PKL_DIR not set'}) …")
    configs = get_dataset_config_names(hf_id)
    pairs: List[Tuple[str, str]] = []
    for i, cfg in enumerate(configs):
        if i > 0:
            time.sleep(0.1)
        for attempt in range(4):
            try:
                try:
                    ds = load_dataset(hf_id, cfg, split="train")
                except Exception:
                    ds_dict = load_dataset(hf_id, cfg)
                    ds = ds_dict[list(ds_dict.keys())[0]]
                pairs.extend(zip(ds[good_col], ds[bad_col]))
                del ds
                break
            except Exception as exc:
                if "429" in str(exc) and attempt < 3:
                    wait = 30 * (2 ** attempt)
                    logger.warning(f"    429 on '{cfg}' — waiting {wait}s …")
                    time.sleep(wait)
                else:
                    logger.warning(f"    Skipping config '{cfg}': {exc}")
                    break
        if (i + 1) % 25 == 0:
            logger.info(f"  … {i+1}/{len(configs)} configs loaded")

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(pairs, fh)
        logger.info(f"  Cached {len(pairs):,} pairs → {pkl_name}")

    gc.collect()
    return pairs


def _load_per_lang(hf_id: str, lang: str, good_col: str, bad_col: str) -> List[Tuple[str, str]]:
    ds = load_dataset(hf_id, lang, split="train")
    pairs = list(zip(ds[good_col], ds[bad_col]))
    del ds
    return pairs


def load_benchmark_data(
    name: str, pkl_dir: str, lang_filter: Optional[set[str]] = None,
) -> dict[str, tuple[list, str]]:
    """Return {eval_language: (pairs, iso_code)} for benchmark *name*.

    If *lang_filter* is given, only ISO codes in that set are loaded.  For
    non-multiblimp benchmarks, the single iso_code is checked against the
    filter (and the benchmark is skipped if not present).
    """
    cfg = BENCHMARKS[name]

    if name == "multiblimp":
        result: dict[str, tuple[list, str]] = {}
        for iso, (lang_label, iso_code) in cfg["langs"].items():
            if lang_filter is not None and iso_code not in lang_filter:
                continue
            try:
                pairs = _load_per_lang(cfg["hf_id"], iso, cfg["good_col"], cfg["bad_col"])
                result[lang_label] = (pairs, iso_code)
                logger.info(f"  multiblimp/{iso}: {len(pairs):,} pairs")
            except Exception as exc:
                logger.warning(f"  multiblimp/{iso}: SKIP — {exc}")
        return result

    if lang_filter is not None and cfg["iso_code"] not in lang_filter:
        return {}

    pairs = _load_all_configs(
        cfg["hf_id"], cfg["good_col"], cfg["bad_col"], cfg["pkl_name"], pkl_dir,
    )
    return {cfg["eval_language"]: (pairs, cfg["iso_code"])}


# ---------------------------------------------------------------------------
# Scoring (batched, with autocast for A100)
# ---------------------------------------------------------------------------

def _is_oom(exc: BaseException) -> bool:
    """Recognise CUDA OOMs across torch versions."""
    if torch.cuda.is_available():
        oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


_HEARTBEAT_EVERY_BATCHES = 25


@torch.no_grad()
def score_sentences(
    model, tokenizer, sentences: List[str], device: str, batch_size: int,
) -> torch.Tensor:
    """Return per-sentence sum log-probability (higher = more likely).

    Adapts batch_size downward on CUDA OOM and retries the same chunk.
    Logs a heartbeat every _HEARTBEAT_EVERY_BATCHES so callers can tell
    a running scoring loop apart from a hung process.
    """
    vocab_size = resolve_vocab_size(model)
    all_scores: list[torch.Tensor] = []
    n_total = len(sentences)
    cur_bs = batch_size
    batches_done = 0
    i = 0
    while i < n_total:
        batch = sentences[i: i + cur_bs]
        try:
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=512, return_tensors="pt")
            input_ids     = enc["input_ids"].to(device).clamp(0, vocab_size - 1)
            attention_mask = enc["attention_mask"].to(device)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                outputs = model(input_ids=input_ids)
            log_probs = F.log_softmax(outputs.logits.float(), dim=-1)
            del outputs

            lp_shifted   = log_probs[:, :-1, :]
            ids_shifted  = input_ids[:, 1:]
            mask_shifted = attention_mask[:, 1:].float()
            token_lp     = lp_shifted.gather(-1, ids_shifted.unsqueeze(-1)).squeeze(-1)
            sent_lp      = (token_lp * mask_shifted).sum(dim=-1)
            all_scores.append(sent_lp.cpu())
        except BaseException as exc:
            if _is_oom(exc) and cur_bs > 1:
                new_bs = max(1, cur_bs // 2)
                logger.warning(
                    f"    CUDA OOM at batch_size={cur_bs}; halving to {new_bs} and retrying"
                )
                cur_bs = new_bs
                # Drop refs and reclaim before retry
                for _name in ("outputs", "log_probs", "lp_shifted", "ids_shifted",
                              "mask_shifted", "token_lp", "sent_lp",
                              "input_ids", "attention_mask", "enc"):
                    if _name in locals():
                        del locals()[_name]
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            # Non-recoverable: log a real traceback before letting it bubble up
            # so the outer ledger handler in evaluate() can record + continue.
            logger.error(
                f"    score_sentences failed at i={i}, batch_size={cur_bs}",
                exc_info=True,
            )
            raise

        i += cur_bs
        batches_done += 1
        if batches_done % _HEARTBEAT_EVERY_BATCHES == 0:
            # cuda.synchronize() forces async device errors to surface here
            # (as Python exceptions) rather than in a later, unrelated call.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info(f"    progress: {min(i, n_total):,}/{n_total:,} sentences scored "
                        f"(batch_size={cur_bs})")

    return torch.cat(all_scores)


def run_minimal_pairs(
    model, tokenizer, pairs: List[Tuple[str, str]], device: str, batch_size: int,
) -> Tuple[float, int, int]:
    if not pairs:
        return 0.0, 0, 0
    s_good = score_sentences(model, tokenizer, [p[0] for p in pairs], device, batch_size)
    s_bad  = score_sentences(model, tokenizer, [p[1] for p in pairs], device, batch_size)
    n_correct = int((s_good > s_bad).sum().item())
    return n_correct / len(pairs), n_correct, len(pairs)


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def _append_row(output_file: Path, row: dict) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        output_file, mode="a", header=not output_file.exists(), index=False,
    )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    output_dir  = Path(args.output)
    output_file = output_dir / "blimp_results.csv"
    pkl_dir = args.pkl_dir or os.environ.get(
        "PKL_DIR", str(output_dir / ".pkl_cache")
    )

    setup_logging("blimp", output_dir, args.rank)

    all_model_pairs = resolve_models(args.models, output_dir=output_dir.parent)
    my_models = [p for i, p in enumerate(all_model_pairs) if i % args.world_size == args.rank]
    logger.info(f"[rank {args.rank}/{args.world_size}] {len(my_models)} models assigned")

    bench_names = args.benchmarks or list(BENCHMARKS.keys())

    lang_filter: Optional[set[str]] = set(args.langs) if args.langs else None
    if lang_filter:
        logger.info(f"Language filter active: {sorted(lang_filter)}")

    logger.info("Pre-loading benchmark data …")
    bench_data: dict[str, dict[str, tuple[list, str]]] = {}
    for name in bench_names:
        if name not in BENCHMARKS:
            logger.warning(f"Unknown benchmark '{name}', skipping")
            continue
        bench_data[name] = load_benchmark_data(name, pkl_dir, lang_filter)
        for lang, (pairs, iso) in bench_data[name].items():
            logger.info(f"  {name}/{lang} ({iso}): {len(pairs):,} pairs ready")

    for repo, family in my_models:
        # Collect tasks not yet recorded in ledger
        tasks = [
            (bname, lang, iso)
            for bname, lang_data in bench_data.items()
            for lang, (pairs, iso) in lang_data.items()
            if not _ledger_is_done("blimp", repo, args.revision, bname, iso,
                                   output_root=str(output_dir))
        ]
        if not tasks:
            logger.info(f"SKIP {repo}@{args.revision} — all benchmarks already in ledger")
            continue

        result = load_model(repo, DEVICE, revision=args.revision)
        if result is None:
            continue
        model, tokenizer = result

        try:
            for bname, lang, iso in tasks:
                pairs, _ = bench_data[bname][lang]
                logger.info(f"  {repo}@{args.revision} | {bname} | {lang}: {len(pairs):,} pairs …")
                t0 = time.time()
                try:
                    acc, n_correct, n_total = run_minimal_pairs(
                        model, tokenizer, pairs, DEVICE, args.batch_size,
                    )
                    elapsed = time.time() - t0
                    pairs_per_sec = n_total / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"    accuracy={acc:.4f}  ({n_correct}/{n_total})  "
                        f"{pairs_per_sec:.0f} pairs/s  [{elapsed:.1f}s]"
                    )
                    _append_row(output_file, {
                        "benchmark":      bname,
                        "model":          repo,
                        "branch":         args.revision,
                        "model_family":   family,
                        "eval_language":  lang,
                        "iso_code":       iso,
                        "accuracy":       round(acc, 6),
                        "n_correct":      n_correct,
                        "n_total":        n_total,
                    })
                    _ledger_record("blimp", repo, args.revision, bname, iso, "ok",
                                   rank=args.rank, output_root=str(output_dir))
                except Exception as exc:
                    logger.error(f"    ERROR {repo}@{args.revision}|{bname}|{lang}: {exc}")
                    _ledger_record("blimp", repo, args.revision, bname, iso, "error",
                                   rank=args.rank, output_root=str(output_dir))
        finally:
            # Unload regardless of how the inner loop terminated; otherwise
            # an interpreter-level error would leak the model to the next
            # iteration's load and double host-RAM peak.
            unload_model(model, tokenizer)
            del model, tokenizer

    logger.info(f"Done. Results at {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BLiMP / BLiMP-NL / ZhoBLIMP / MultiBLiMP evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--models", nargs="+", required=True, metavar="SPEC",
                   help="beetle-humanscale | beetle-fineweb | baselines | "
                        "eurollm | xglm | bloom | <org/repo>")
    p.add_argument("--benchmarks", nargs="+", choices=list(BENCHMARKS.keys()),
                   default=None, help="Benchmarks to run (default: all)")
    p.add_argument("--langs", nargs="+", default=None, metavar="ISO",
                   help="Restrict evaluation to these ISO-639-3 codes "
                        "(applies to multiblimp sub-languages AND to the "
                        "single-language benchmarks blimp_eng/blimp_nl/"
                        "zhoblimp — skipped if their iso_code is not listed).")
    p.add_argument("--output", default="results/evals/blimp")
    p.add_argument("--pkl-dir", default=None,
                   help="Pickled dataset cache dir (default: <output>/.pkl_cache)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Scoring batch size (default: 64, safe for A100 80 GB up to ~9B)")
    p.add_argument("--revision", default="main",
                   help="HuggingFace branch / revision to evaluate (default: main). "
                        "Recorded as the 'branch' column in the output CSV and "
                        "as the checkpoint slot in the eval ledger.")
    p.add_argument("--rank",       type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
