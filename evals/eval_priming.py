#!/usr/bin/env python3
"""
evals/eval_priming.py — Replicate Arnett (2025) loss curves and crosslingual
structural priming on Beetle bilingual eng-X / X-eng models, at every
checkpoint.

Two stages, run together by default:

  Stage A — §3.1 Loss Patterns
    Mean surprisal on a held-out evaluation set for each language the model
    was trained on (FLORES-200 devtest, matching evals/eval_flores_ppl.py
    conventions). One row per (repo, revision, language).
    Output: results/evals/priming/loss_curves.csv

  Stage B — §4 Structural Priming Effects
    For each Arnett stimulus file whose language pair matches the model
    (Schoonbaert/Bernolet → eng↔nld, Hartsuiker → eng↔spa, Kotzochampou →
    eng↔ell, Fleischer → eng↔pol), score every (Prime, Target) pair as the
    conditional surprisal of TargetSentence given PrimeSentence and emit per-
    stimulus rows in the column schema Arnett's
    final_structural_priming_analysis.Rmd preprocess loop expects, so the
    same downstream R analysis drops in unchanged.
    Output: results/evals/priming/<dataset>__<repo_safe>__<revision>.tsv

The script enumerates Beetle bilingual models from BEETLE_ORGS via the
existing baseline_models.resolve_models() helper, restricts to repos whose
(L1, L2) includes English, and walks every step-N branch plus 'main'
(matching evals/eval_evp_trajectory.list_checkpoints).

BGPT models are NOT re-evaluated here — per project decision, we use Arnett's
published numbers from arnett-2025-priming-raw/ for the BGPT side of the
comparison. See evals/compare_priming_to_bgpt.py for the join.

Usage
-----
  # All Beetle-HumanScale + Beetle-FineWeb bilingual models, every checkpoint
  python evals/eval_priming.py --models beetle-humanscale beetle-fineweb

  # 8-GPU fan-out, rank 3 only
  CUDA_VISIBLE_DEVICES=3 python evals/eval_priming.py \\
      --models beetle-humanscale --rank 3 --world-size 8

  # Final-checkpoint smoke test
  python evals/eval_priming.py --models beetle-humanscale --revisions main

  # Subsample to N evenly-spaced step-N branches plus main
  python evals/eval_priming.py --models beetle-humanscale --max-checkpoints 12

  # Skip stage A or stage B independently
  python evals/eval_priming.py --models beetle-humanscale --skip-loss
  python evals/eval_priming.py --models beetle-humanscale --skip-priming
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Repo-root imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "analyze"))

from evals.baseline_models import resolve_models, model_family_label
from evals.model_loader import DEVICE, log_gpu_mem, setup_logging
from evals.priming_stimuli import (
    ARNETT_STIMULI,
    LANG_TITLE_TO_ISO3,
    load_stimuli,
    stimuli_for_model_pair,
)

from analyze.model_naming import parse_repo, RepoMeta
from analyze.ppl_utils import load_flores_sentences, FLORES_LANG_MAP

# Optional ledger
try:
    from analyze.eval_ledger import is_done as _ledger_is_done
    from analyze.eval_ledger import record as _ledger_record
except Exception:
    def _ledger_is_done(*_a, **_kw) -> bool: return False  # type: ignore[misc]
    def _ledger_record(*_a, **_kw) -> None: pass            # type: ignore[misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT = _REPO_ROOT / "results" / "evals" / "priming"
LOSS_CSV_NAME = "loss_curves.csv"
PRIMING_SUMMARY_NAME = "priming_all.tsv"

# Arnett's loss eval is mean per-token surprisal on held-out text. We use
# FLORES devtest (1012 sentences/lang) to mirror evals/eval_flores_ppl.py.
LOSS_MAX_SENTENCES = 1012


# ---------------------------------------------------------------------------
# VRAM preflight — protect the GPU from cooperative OOMs across ranks
# ---------------------------------------------------------------------------

def _apply_cuda_preflight() -> None:
    """Set per-process CUDA defaults that reduce the chance of an OOM
    *cascade* taking down peer ranks on the same node.

    Called once at process start. Idempotent.

    What this does:
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (already set by
        scripts/_sweep_safety.sh in normal launches; we re-apply here so
        single-rank `python evals/eval_priming.py` invocations get it too).
      - Optional per-process VRAM cap via PRIMING_CUDA_MEM_FRACTION
        (e.g. ``0.9`` to leave a 10% headroom buffer); useful when running
        multiple ranks per physical GPU via MIG slices.
      - Reset peak memory counters so log_gpu_mem reports per-process peaks.
    """
    if not torch.cuda.is_available():
        return
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    frac_str = os.environ.get("PRIMING_CUDA_MEM_FRACTION", "")
    if frac_str:
        try:
            frac = float(frac_str)
            if 0.0 < frac <= 1.0:
                torch.cuda.set_per_process_memory_fraction(frac)
                logger.info(f"CUDA preflight: set per-process memory fraction to {frac}")
        except Exception as exc:
            logger.warning(f"CUDA preflight: could not apply mem fraction {frac_str!r}: {exc}")
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Repo / revision filtering
# ---------------------------------------------------------------------------

def _repo_safe(repo: str) -> str:
    return repo.replace("/", "_")


def _is_bilingual_eng_pair(meta: RepoMeta) -> bool:
    """True iff the model is bilingual and English is one of its languages."""
    if meta.condition == "mono":
        return False
    if meta.condition == "unknown":
        return False
    return meta.l1 == "eng" or meta.l2 == "eng"


def _filter_to_eng_bilingual(model_pairs: list[tuple[str, str]]) -> list[tuple[str, str, RepoMeta]]:
    """Keep only Beetle bilingual configs containing English."""
    out: list[tuple[str, str, RepoMeta]] = []
    for repo, family in model_pairs:
        meta = parse_repo(repo)
        if _is_bilingual_eng_pair(meta):
            out.append((repo, family, meta))
        else:
            logger.debug(f"  skip non-eng-bilingual: {repo} (cond={meta.condition}, l1={meta.l1}, l2={meta.l2})")
    return out


def list_checkpoints(repo: str, max_n: int = 0) -> list[str]:
    """Enumerate step-N branches plus 'main', subsampled to max_n if set.

    Same convention as evals/eval_evp_trajectory.list_checkpoints — kept here
    instead of imported to avoid pulling that module's heavy dependencies.
    """
    from huggingface_hub import list_repo_refs
    try:
        refs = list_repo_refs(repo)
    except Exception as e:
        logger.warning(f"list_repo_refs({repo}) failed: {e}")
        return ["main"]
    revs: list[tuple[int, str]] = []
    for b in refs.branches:
        if b.name.startswith("step-"):
            try:
                step = int(b.name.split("-", 1)[1])
                revs.append((step, b.name))
            except ValueError:
                continue
    revs.sort()
    if max_n and len(revs) > max_n:
        idx = np.linspace(0, len(revs) - 1, max_n).round().astype(int).tolist()
        revs = [revs[i] for i in idx]
    out = [r[1] for r in revs]
    out.append("main")
    return out


# ---------------------------------------------------------------------------
# Surprisal scoring helpers (built on minicons IncrementalLMScorer, matching
# the legacy/modern API fallback already used by evals/eval_meco.py)
# ---------------------------------------------------------------------------

def _conditional_surprisal(scorer_obj, prime: str, target: str) -> float:
    """Return -sum(log P(target tokens | prime context)).

    Uses the same legacy/modern minicons API fallback as eval_meco.py.
    Returns NaN on internal scorer error so a single bad row never aborts
    the whole eval.
    """
    # Legacy path (minicons < 0.3.x).
    try:
        return float(
            scorer_obj.conditional_score(
                prime, target,
                bos_token=False,
                reduction=lambda x: -x.sum(0).item(),
            )[0]
        )
    except TypeError:
        pass
    except Exception:
        return float("nan")

    # Modern path: per-token log-probs, sum and negate.
    try:
        out = scorer_obj.conditional_score(prime, target, bos_token=False)
        first = out[0] if out else None
        if first is None:
            return float("nan")
        if hasattr(first, "__iter__") and not isinstance(first, (str, bytes)):
            return -float(sum(first))
        return -float(first)
    except Exception:
        return float("nan")


def _sentence_surprisal(scorer_obj, sentence: str) -> float:
    """Mean per-token surprisal of an unconditional sentence.

    Uses minicons sequence_score(reduction=mean), with a fallback to
    summing per-token log-probs from token_score().
    """
    try:
        s = scorer_obj.sequence_score(sentence, reduction=lambda x: -x.mean(0).item())
        return float(s[0]) if hasattr(s, "__iter__") else float(s)
    except Exception:
        pass
    try:
        # token_score returns list of [(token, log_prob), ...] per sequence.
        toks = scorer_obj.token_score([sentence])[0]
        log_probs = [lp for _, lp in toks if lp == lp]  # drop NaN
        if not log_probs:
            return float("nan")
        return -float(sum(log_probs) / len(log_probs))
    except Exception:
        return float("nan")


def _cleanup() -> None:
    """Release model + activation memory between revisions.

    Called both on the success path and inside OOM/error handlers so that
    one bad checkpoint cannot bleed VRAM into the next iteration. The
    `reset_peak_memory_stats` call ensures `log_gpu_mem` reads accurate
    peaks for the next model rather than carrying yesterday's high water
    mark forward.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def _is_oom(exc: BaseException) -> bool:
    """True iff *exc* is a CUDA OOM (works across torch versions)."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):  # type: ignore[attr-defined]
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


# ---------------------------------------------------------------------------
# Stage A — §3.1 mean surprisal on held-out FLORES sentences
# ---------------------------------------------------------------------------

def score_loss_curve(
    scorer_obj,
    languages: list[str],
    max_sentences: int = LOSS_MAX_SENTENCES,
) -> pd.DataFrame:
    """Mean per-token surprisal over FLORES devtest for each language.

    Returns one row per language with columns:
        language, n_sentences, mean_surprisal
    """
    rows: list[dict] = []
    for iso in languages:
        if iso not in FLORES_LANG_MAP:
            logger.warning(f"  No FLORES tag for {iso!r}; skipping.")
            rows.append({"language": iso, "n_sentences": 0, "mean_surprisal": float("nan")})
            continue
        sents = load_flores_sentences(iso)[:max_sentences]
        if not sents:
            rows.append({"language": iso, "n_sentences": 0, "mean_surprisal": float("nan")})
            continue
        per_sentence = [
            _sentence_surprisal(scorer_obj, s) for s in tqdm(sents, desc=f"flores:{iso}", leave=False)
        ]
        per_sentence = [v for v in per_sentence if v == v]   # drop NaN
        mean = float(sum(per_sentence) / len(per_sentence)) if per_sentence else float("nan")
        rows.append({"language": iso, "n_sentences": len(per_sentence), "mean_surprisal": mean})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage B — §4 structural priming
# ---------------------------------------------------------------------------

def score_priming(
    scorer_obj,
    stimuli_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score every (Prime, Target) row.

    Returns a copy of *stimuli_df* with an added ``Surprisal`` column equal
    to ``-sum(log P(TargetSentence | PrimeSentence))``.

    The column schema after this call is exactly what
    final_structural_priming_analysis.Rmd's preprocess loop reads in: it
    pivots wider on TargetType to compute the relative-probability metric.
    """
    res = stimuli_df.copy()
    surprisals: list[float] = []
    for prime, target in tqdm(
        zip(res["PrimeSentence"].astype(str), res["TargetSentence"].astype(str)),
        total=len(res), desc="priming", leave=False,
    ):
        surprisals.append(_conditional_surprisal(scorer_obj, prime, target))
    res["Surprisal"] = surprisals
    # Rename FullSentence → FullText to match Arnett's expected join key.
    if "FullText" not in res.columns:
        res = res.rename(columns={"FullSentence": "FullText"})
    return res


# ---------------------------------------------------------------------------
# Per-(repo, revision) orchestration
# ---------------------------------------------------------------------------

def evaluate_one_revision(
    repo: str,
    revision: str,
    meta: RepoMeta,
    output_dir: Path,
    skip_loss: bool,
    skip_priming: bool,
) -> Optional[dict]:
    """Run both stages for a single (repo, revision). Returns a dict
    summarising what was produced or None if the model failed to load."""
    from minicons import scorer as _scorer

    # ----- Fast-skip if Stage B has no compatible stimuli AND Stage A is off
    # so we never load a multi-GB model just to write zero rows. Trajectory
    # tiers regularly hit this path (deu/zho/eus/etc bilinguals).
    if skip_loss and not skip_priming:
        if not stimuli_for_model_pair(meta.l1 or "", meta.l2 or ""):
            logger.info(
                f"SKIP {repo}@{revision}: no Arnett stimuli for "
                f"{meta.l1}-{meta.l2} and Stage A skipped — nothing to do."
            )
            return {"repo": repo, "revision": revision, "loss_rows": 0,
                    "priming_files": [], "skipped": "no_stage_b_coverage"}

    logger.info(f"Loading {repo} @ {revision} …")
    ilm = None
    summary: dict = {"repo": repo, "revision": revision, "loss_rows": 0, "priming_files": []}
    try:
        try:
            ilm = _scorer.IncrementalLMScorer(
                repo, DEVICE,
                tokenizer=repo,
                revision=revision,
                trust_remote_code=True,
            )
        except Exception as exc:
            if _is_oom(exc):
                # OOM at construction: free everything we can and bail. The
                # caller's ledger will mark this revision as 'error' so it
                # is retried later (possibly on a smaller-tier rank).
                logger.error(
                    f"OOM loading {repo}@{revision}: {str(exc).splitlines()[0]}"
                )
                _cleanup()
                return None
            logger.warning(f"SKIP {repo}@{revision}: {str(exc).splitlines()[0]}")
            return None
        log_gpu_mem(logger, prefix=repo.split("/")[-1])

        # ----- Stage A: §3.1 loss curves -----
        if not skip_loss:
            languages = [c for c in (meta.l1, meta.l2) if c]
            try:
                loss_df = score_loss_curve(ilm, languages)
            except Exception as exc:
                if _is_oom(exc):
                    logger.error(
                        f"OOM during Stage A on {repo}@{revision}; aborting "
                        f"this revision: {str(exc).splitlines()[0]}"
                    )
                    return None
                raise
            loss_df.insert(0, "repo", repo)
            loss_df.insert(1, "revision", revision)
            loss_df.insert(2, "family", meta.family)
            loss_df.insert(3, "condition", meta.condition)
            loss_df.insert(4, "variant", meta.variant or "")
            loss_df.insert(5, "l1", meta.l1 or "")
            loss_df.insert(6, "l2", meta.l2 or "")
            loss_df.insert(7, "model_family", model_family_label(repo))
            loss_csv = output_dir / LOSS_CSV_NAME
            # Append (header iff file is new). atomic-ish: append per-row chunks.
            write_header = not loss_csv.exists()
            loss_df.to_csv(loss_csv, mode="a", header=write_header, index=False)
            summary["loss_rows"] = len(loss_df)
            logger.info(f"  Stage A: wrote {len(loss_df)} loss rows → {loss_csv}")

        # ----- Stage B: §4 priming -----
        if not skip_priming:
            compatible = stimuli_for_model_pair(meta.l1 or "", meta.l2 or "")
            if not compatible:
                logger.info(f"  Stage B: no Arnett stimuli for {meta.l1}-{meta.l2}, skipping priming.")
            for stim in compatible:
                stim_df = load_stimuli(stim)
                try:
                    scored = score_priming(ilm, stim_df)
                except Exception as exc:
                    if _is_oom(exc):
                        # One stimulus file's worth of rows lost; free cache and
                        # try the next file rather than abort the whole revision.
                        logger.error(
                            f"OOM during Stage B on {repo}@{revision} "
                            f"dataset={stim.dataset}: {str(exc).splitlines()[0]}"
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise
                scored["model"] = repo
                scored["revision"] = revision
                scored["checkpoint"] = _revision_to_step(revision)
                scored["family"] = meta.family
                scored["condition"] = meta.condition
                scored["l1"] = meta.l1 or ""
                scored["l2"] = meta.l2 or ""
                out_path = output_dir / f"{stim.dataset}__{_repo_safe(repo)}__{revision}.tsv"
                scored.to_csv(out_path, sep="\t", index=False)
                summary["priming_files"].append(str(out_path))
                logger.info(f"  Stage B: wrote {len(scored)} rows → {out_path}")
        return summary
    finally:
        # Always release the model — even on success, exception, or OOM —
        # so we never keep two sets of weights resident across revisions.
        if ilm is not None:
            try:
                # minicons IncrementalLMScorer keeps the HF model on the
                # `.model` attribute (legacy + modern API).
                if hasattr(ilm, "model"):
                    del ilm.model
            except Exception:
                pass
        del ilm
        _cleanup()


def _revision_to_step(revision: str) -> Optional[int]:
    """Parse 'step-N' → N, 'main' → None, plain int → int."""
    if revision in (None, "", "main"):
        return None
    if revision.startswith("step-"):
        try:
            return int(revision.split("-", 1)[1])
        except ValueError:
            return None
    try:
        return int(revision)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("priming", output_dir, args.rank)

    # GPU safety: apply allocator config and (optional) per-process memory
    # fraction *before* we touch any model.
    _apply_cuda_preflight()

    # 1. Resolve and filter to bilingual eng-X / X-eng models.
    all_pairs = resolve_models(args.models, output_dir=output_dir.parent)
    bilingual = _filter_to_eng_bilingual(all_pairs)
    logger.info(
        f"Resolved {len(all_pairs)} models from {args.models}; "
        f"{len(bilingual)} are bilingual eng-X / X-eng."
    )

    # 2. Shard by rank.
    my_models = [t for i, t in enumerate(bilingual) if i % args.world_size == args.rank]
    logger.info(f"[rank {args.rank}/{args.world_size}] {len(my_models)} models")

    # 3. For each model, enumerate revisions.
    for repo, family, meta in my_models:
        if args.revisions is not None:
            revisions = list(args.revisions)
        else:
            revisions = list_checkpoints(repo, max_n=args.max_checkpoints)
        logger.info(f"[{repo}] {len(revisions)} revisions: "
                    f"{revisions[:3]}{'...' if len(revisions) > 3 else ''}")

        for rev in revisions:
            ledger_key = ("priming", repo, rev, "priming", None)
            if _ledger_is_done(*ledger_key, output_root=str(output_dir)):
                logger.info(f"  SKIP {repo}@{rev} — ledger says done")
                continue
            summary = evaluate_one_revision(
                repo, rev, meta, output_dir,
                skip_loss=args.skip_loss,
                skip_priming=args.skip_priming,
            )
            status = "ok" if summary is not None else "error"
            _ledger_record("priming", repo, rev, "priming", None, status,
                           rank=args.rank, output_root=str(output_dir))

    # 4. Rank 0 assembles the priming summary TSV.
    if args.rank == 0 and not args.skip_priming:
        _write_priming_summary(output_dir)

    logger.info("Done.")


def _write_priming_summary(output_dir: Path) -> None:
    """Concatenate every per-(dataset, repo, revision) priming TSV into a
    single long-form summary TSV the comparison script can consume."""
    parts: list[pd.DataFrame] = []
    for tsv in sorted(output_dir.glob("*__*__*.tsv")):
        if tsv.name == PRIMING_SUMMARY_NAME:
            continue
        try:
            parts.append(pd.read_csv(tsv, sep="\t"))
        except Exception as exc:
            logger.warning(f"  could not read {tsv}: {exc}")
    if not parts:
        return
    combined = pd.concat(parts, ignore_index=True)
    out = output_dir / PRIMING_SUMMARY_NAME
    combined.to_csv(out, sep="\t", index=False)
    logger.info(f"Combined priming surprisals → {out} ({len(combined)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replicate Arnett (2025) loss + structural priming on Beetle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--models", nargs="+", required=True,
        metavar="SPEC",
        help="Model specs (see baseline_models.resolve_models). "
             "Typical: 'beetle-humanscale beetle-fineweb'.",
    )
    p.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--revisions", nargs="+", default=None,
        help="Specific HF revisions to evaluate (e.g. step-10000 main). "
             "If omitted, every step-N branch + main is enumerated.",
    )
    p.add_argument(
        "--max-checkpoints", type=int, default=0,
        help="Subsample step-N branches to N evenly-spaced ones (0 = all).",
    )
    p.add_argument("--skip-loss", action="store_true", help="Skip Stage A (§3.1).")
    p.add_argument("--skip-priming", action="store_true", help="Skip Stage B (§4).")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
