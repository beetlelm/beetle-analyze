#!/usr/bin/env python3
"""
scripts/priming_interim_effects.py — fast Beetle-only effects computer for
in-flight tiers. Replicates the Beetle half of compare_priming_to_bgpt.py
without the slow priming_combined.tsv concat, then merges in pre-computed
BGPT effects from a reference tier so the headline panel still has BGPT
reference bars.

Usage:
    python scripts/priming_interim_effects.py \
        --beetle-dir results/evals/priming/tier2_headline_trajectory \
        --bgpt-effects-from results/evals/priming/tier1_final/comparison/priming_effects_per_model.csv

Writes:
    {beetle_dir}/comparison/priming_effects_per_model.csv
    {beetle_dir}/comparison/summary.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evals.compare_priming_to_bgpt import DATASET_PIVOT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s")
logger = logging.getLogger("interim-effects")


def beetle_effects(beetle_dir: Path) -> pd.DataFrame:
    p = beetle_dir / "priming_all.tsv"
    if not p.exists():
        logger.error("no priming_all.tsv at %s", p); sys.exit(2)

    long = pd.read_csv(p, sep="\t", low_memory=False)
    if "checkpoint" in long.columns and long["checkpoint"].isna().all():
        long["checkpoint"] = long["revision"].astype(str) if "revision" in long.columns else "main"
    elif "checkpoint" in long.columns and "revision" in long.columns:
        long["checkpoint"] = long["checkpoint"].fillna(long["revision"])

    out_rows: list[dict] = []
    for ds, sub in long.groupby("dataset"):
        if ds not in DATASET_PIVOT:
            continue
        a, b, prob_col = DATASET_PIVOT[ds]
        idx_cols = [c for c in (
            "dataset", "model", "revision", "checkpoint",
            "Set", "ExtraCondition",
            "PrimeLanguage", "PrimeType", "TargetLanguage",
        ) if c in sub.columns and not sub[c].isna().all()]
        wide = sub.pivot_table(
            index=idx_cols, columns="TargetType", values="Surprisal",
            aggfunc="first", dropna=False,
        ).reset_index()
        wide.columns.name = None
        if a in wide.columns and b in wide.columns:
            denom = np.exp(-wide[a]) + np.exp(-wide[b])
            wide[prob_col] = np.exp(-wide[a]) / denom.replace(0, np.nan)
        else:
            continue

        # Vectorized collapse: a Python-level groupby loop on ~40k groups
        # was the bottleneck (5+ min). Use a single agg over (group_cols ×
        # is_match) then unstack — pure pandas, ~seconds.
        target_prefix = a.replace("_Target", "")
        group_cols = [c for c in (
            "model", "dataset", "checkpoint",
            "PrimeLanguage", "TargetLanguage",
        ) if c in wide.columns]
        wide = wide.dropna(subset=[prob_col])
        if wide.empty:
            continue
        wide["_is_match"] = wide["PrimeType"].astype(str).str.startswith(target_prefix)
        agg = (wide.groupby(group_cols + ["_is_match"], dropna=False)[prob_col]
                   .agg(["mean", "count"])
                   .unstack("_is_match"))
        # agg.columns is a MultiIndex like [('mean', False), ('mean', True), ...]
        if (True not in agg["mean"].columns) or (False not in agg["mean"].columns):
            continue
        keep = agg["mean"][True].notna() & agg["mean"][False].notna()
        agg = agg.loc[keep]
        if agg.empty:
            continue
        block = pd.DataFrame({
            "mean_match":    agg["mean"][True].astype(float),
            "mean_mismatch": agg["mean"][False].astype(float),
            "n_match":       agg["count"][True].fillna(0).astype(int),
            "n_mismatch":    agg["count"][False].fillna(0).astype(int),
        })
        block["priming_effect"] = block["mean_match"] - block["mean_mismatch"]
        block["model_source"]   = "beetle"
        block = block.reset_index()
        out_rows.extend(block.to_dict("records"))
    return pd.DataFrame(out_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beetle-dir", required=True, type=Path)
    ap.add_argument("--bgpt-effects-from", type=Path,
                    help="Path to a pre-computed priming_effects_per_model.csv "
                         "(BGPT rows are extracted and concatenated to the "
                         "Beetle effects).")
    args = ap.parse_args()

    beetle_dir = args.beetle_dir.resolve()
    out_dir = beetle_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("computing Beetle effects from %s", beetle_dir / "priming_all.tsv")
    beetle = beetle_effects(beetle_dir)
    logger.info("%d Beetle effect rows", len(beetle))

    parts = [beetle]
    if args.bgpt_effects_from and args.bgpt_effects_from.exists():
        ref = pd.read_csv(args.bgpt_effects_from)
        bgpt = ref[ref.model_source == "bgpt"].copy()
        logger.info("merging %d BGPT rows from %s", len(bgpt), args.bgpt_effects_from)
        parts.append(bgpt)
    else:
        logger.info("no bgpt-effects-from supplied; output will be Beetle-only")

    combined = pd.concat(parts, ignore_index=True, sort=False)
    out_csv = out_dir / "priming_effects_per_model.csv"
    combined.to_csv(out_csv, index=False)
    logger.info("wrote %d rows -> %s", len(combined), out_csv)

    pivot = (combined
             .groupby(["model_source", "dataset"])["priming_effect"].mean()
             .unstack("model_source"))
    md = ["# Beetle vs. BGPT priming comparison (interim)",
          "",
          f"- Beetle priming dir: `{beetle_dir}`",
          f"- Source CSV: {out_csv}",
          "",
          "## Per-model priming effects",
          "",
          pivot.to_markdown(floatfmt="+.4f")]
    (out_dir / "summary.md").write_text("\n".join(md))
    logger.info("wrote %s", out_dir / "summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
