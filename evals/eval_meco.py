#!/usr/bin/env python3
"""
evals/eval_meco.py — MECO L2 eye-tracking surprisal evaluation.

Scores every word in the MECO stimuli TSV under each model and writes
per-model surprisal CSVs. Surprisal values can then be correlated with
human reading times to measure cognitive fit.

Stimuli format (data/meco_l2_stims.tsv):
  itemid  wordnum  word  FullText  FullTextMarked  [rt columns...]
The target word is delimited by asterisks in FullTextMarked:
  "The cat *sat* on the mat"

Two model sources are supported:
  1. Beetle orgs — Beetle-HumanScale and Beetle-FineWeb
  2. Baselines   — EuroLLM, XGLM, BLOOM at multiple sizes

Completed repos are recorded in the eval ledger and skipped on re-runs.

Usage
-----
  # All Beetle-HumanScale models + all baselines
  python evals/eval_meco.py --models beetle-humanscale baselines

  # BLOOM only, custom stimuli, 8-GPU fan-out
  CUDA_VISIBLE_DEVICES=3 python evals/eval_meco.py \\
      --models bloom \\
      --stimuli data/meco_l2_stims.tsv \\
      --rank 3 --world-size 8
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm

tqdm.pandas()

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "analyze"))

from evals.baseline_models import resolve_models, model_family_label
from evals.model_loader import DEVICE, log_gpu_mem, setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STIMULI = _REPO_ROOT / "data" / "meco_l2_stims.tsv"

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
# Surprisal scoring (minicons, with legacy/modern API fallback)
# ---------------------------------------------------------------------------

def _get_surprisal(ilm_model, marked_text: str) -> float:
    """Return -sum(log P(target | context)) for a marked sentence.

    The target is delimited by asterisks: "The cat *sat* on the mat".
    Handles both legacy (< 0.3.x) and modern minicons APIs.
    """
    parts = marked_text.split("*")
    if len(parts) < 3:
        return float("nan")

    context = parts[0].strip()
    target = parts[1].strip()

    # Legacy path
    try:
        return ilm_model.conditional_score(
            context, target,
            bos_token=False,
            reduction=lambda x: -x.sum(0).item(),
        )[0]
    except TypeError:
        pass
    except Exception:
        return float("nan")

    # Modern path (>= 0.3.x): per-token log-probs, sum and negate
    try:
        out = ilm_model.conditional_score(context, target, bos_token=False)
        first = out[0] if out else None
        if first is None:
            return float("nan")
        if hasattr(first, "__iter__") and not isinstance(first, (str, bytes)):
            return -float(sum(first))
        return -float(first)
    except Exception:
        return float("nan")


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Per-model scoring
# ---------------------------------------------------------------------------

def _model_slug(repo: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", repo)


def score_model(
    repo: str,
    stimuli: pd.DataFrame,
    device: str,
    revision: str = "main",
) -> Optional[pd.DataFrame]:
    """Score all stimulus rows under *repo*@*revision*. Returns None on load failure."""
    from minicons import scorer

    logger.info(f"Loading {repo}@{revision} …")
    try:
        ilm_model = scorer.IncrementalLMScorer(
            repo, device,
            tokenizer=repo,
            revision=revision,
            trust_remote_code=True,
        )
    except Exception as exc:
        logger.warning(f"SKIP {repo}@{revision}: {str(exc).split(chr(10))[0]}")
        return None

    log_gpu_mem(logger, prefix=repo.split("/")[-1])

    result = stimuli.copy()
    # progress_apply is broken on pandas>=2.x with the installed tqdm
    # (AttributeError: 'Series' object has no attribute '_is_builtin_func').
    # Fall back to plain Series.apply — the per-item compute is heavy enough
    # that the lost progress bar is fine.
    result["surprisal"] = result["FullTextMarked"].apply(
        lambda t: _get_surprisal(ilm_model, t)
    )
    result["model"] = repo
    result["branch"] = revision
    result["model_family"] = model_family_label(repo)
    result["revision"] = revision

    del ilm_model
    _cleanup()
    return result


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def _output_path(output_dir: Path, repo: str, revision: str = "main") -> Path:
    suffix = "" if revision == "main" else f"__{_model_slug(revision)}"
    return output_dir / f"surprisals_{_model_slug(repo)}{suffix}.csv"


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    stimuli_path = Path(args.stimuli)
    if not stimuli_path.exists():
        logger.error(f"Stimuli file not found: {stimuli_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging("meco", output_dir, args.rank)

    stimuli = pd.read_csv(stimuli_path, sep="\t")
    logger.info(f"Loaded {len(stimuli)} stimulus rows from {stimuli_path}")
    if "FullTextMarked" not in stimuli.columns:
        logger.error("Stimuli TSV must have a 'FullTextMarked' column.")
        sys.exit(1)

    # Resolve model list and shard by rank
    all_model_pairs = resolve_models(args.models, output_dir=output_dir.parent)
    my_models = [p for i, p in enumerate(all_model_pairs) if i % args.world_size == args.rank]
    logger.info(f"[rank {args.rank}/{args.world_size}] {len(my_models)} models to evaluate")

    for repo, _family in my_models:
        if _ledger_is_done("meco", repo, args.revision, "meco", None,
                           output_root=str(output_dir)):
            logger.info(f"SKIP {repo}@{args.revision} — already in ledger")
            continue

        df = score_model(repo, stimuli, DEVICE, revision=args.revision)
        if df is None:
            _ledger_record("meco", repo, args.revision, "meco", None, "error",
                           rank=args.rank, output_root=str(output_dir))
            continue

        out_path = _output_path(output_dir, repo, args.revision)
        df.to_csv(out_path, index=False)
        logger.info(f"  Wrote {len(df)} rows → {out_path}")
        _ledger_record("meco", repo, args.revision, "meco", None, "ok",
                       rank=args.rank, output_root=str(output_dir))

    # Rank 0: assemble combined summary CSV
    if args.rank == 0:
        _write_summary(output_dir, all_model_pairs)

    logger.info("Done.")


def _write_summary(output_dir: Path, all_model_pairs: list) -> None:
    """Concatenate all per-(model, revision) CSVs into a single summary file."""
    parts: list[pd.DataFrame] = []
    for repo, _family in all_model_pairs:
        slug = _model_slug(repo)
        for path in output_dir.glob(f"surprisals_{slug}.csv"):
            try:
                parts.append(pd.read_csv(path))
            except Exception:
                pass
        for path in output_dir.glob(f"surprisals_{slug}__*.csv"):
            try:
                parts.append(pd.read_csv(path))
            except Exception:
                pass
    if not parts:
        return
    combined = pd.concat(parts, ignore_index=True)
    summary_path = output_dir / "meco_all_surprisals.csv"
    combined.to_csv(summary_path, index=False)
    logger.info(f"Combined surprisals → {summary_path} ({len(combined)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MECO L2 eye-tracking surprisal evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--models", nargs="+", required=True,
        metavar="SPEC",
        help=(
            "Model specs: beetle-humanscale | beetle-fineweb | baselines | "
            "eurollm | xglm | bloom | <org/repo>"
        ),
    )
    p.add_argument(
        "--stimuli", default=str(DEFAULT_STIMULI),
        help=f"Path to MECO stimuli TSV (default: {DEFAULT_STIMULI})",
    )
    p.add_argument(
        "--output", default="results/evals/meco",
        help="Output directory (default: results/evals/meco)",
    )
    p.add_argument(
        "--revision", default="main",
        help="HuggingFace branch / revision to evaluate (default: main). "
             "Recorded as the 'branch' column and namespaces the output CSV.",
    )
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
