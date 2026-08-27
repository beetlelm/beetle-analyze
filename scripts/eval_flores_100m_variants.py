"""
eval_flores_100m_variants.py — re-run FLORES PPL on the 100M nld-eng B3
manipulation cohort with proper per-(repo, revision) cache namespacing.

The first round produced byte-identical NLLs across `pre-pretrain` ≡ `safe`
≡ `baseline` because the bespoke driver that wrote
`results/100m-variants/flores/ppl_summary.csv` keyed its cache only by step,
not by repo. This driver bypasses that path entirely and uses
`analyze.ppl_eval.run_all_languages`, which writes
`ppl_<iso>_<repo-slug>_<revision>.csv` — slug-namespaced, so two variants
can never alias.

Usage
-----
GPU host:
    python scripts/eval_flores_100m_variants.py --overwrite

Multi-GPU (sharded across ranks via _rank_barrier):
    WORLD_SIZE=8 RANK=0 python scripts/eval_flores_100m_variants.py &
    WORLD_SIZE=8 RANK=1 python scripts/eval_flores_100m_variants.py &
    ...

CSVs land in:
    results/ppl/ppl_<iso>_<slug>_<revision>.csv     (per-sentence)
    results/ppl/ppl_summary.csv                      (rank-0, all variants)

The grid plot (`analyze/flores_grid_100m.py`) reads from there directly.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "analyze"))

from analyze.ppl_eval import run_all_languages  # noqa: E402

ORG = "Beetle-FineWeb-100M"
STEM = "beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-nld-eng"

VARIANTS = [
    "baseline",       # convention: bare stem, no suffix
    "safe",
    "ewc",
    "lamol",
    "maml",
    "pre-pretrain",
    "alibi",
    "layer-growth",
]


def _repo_for(variant: str) -> str:
    return f"{ORG}/{STEM}" if variant == "baseline" else f"{ORG}/{STEM}-{variant}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=VARIANTS,
                   help="subset of variants to eval (default: all 8)")
    p.add_argument("--overwrite", action="store_true",
                   help="re-score even if per-(repo, revision) CSV exists")
    p.add_argument("--final-only", action="store_true",
                   help="score only the final step-9096 branch (skip trajectory)")
    p.add_argument("--langs", nargs="+", default=["eng", "nld"])
    args = p.parse_args()

    repos = [_repo_for(v) for v in args.variants]
    print(f"[flores-100m] cohort ({len(repos)}):")
    for r in repos:
        print(f"  · {r}")

    # Sharding env vars are read by run_all_languages internally if set,
    # but we pass explicit rank/world to be sure.
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))

    summary = run_all_languages(
        lang_codes=args.langs,
        overwrite=args.overwrite,
        rank=rank,
        world_size=world,
        output_root=str(REPO_ROOT / "results"),
        repos=repos,
        all_checkpoints=not args.final_only,
    )
    if rank == 0 and not summary.empty:
        # Filter the summary down to our cohort and persist a side-car CSV
        # under results/100m-variants/ so the grid plotter has a stable path.
        cohort_slugs = {r.replace("/", "__") for r in repos}
        sub = summary[summary["repo"].apply(
            lambda x: x.replace("/", "__") in cohort_slugs)]
        out = REPO_ROOT / "results" / "100m-variants" / "flores" / "ppl_summary_v2.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out, index=False)
        print(f"[flores-100m] cohort summary → {out}  ({len(sub)} rows)")


if __name__ == "__main__":
    main()
