#!/usr/bin/env python3
"""
analyze/rcmcq_rescore.py — Aggregate score-variant RC-MCQ summaries.

Walks ``results/evals_paper/03_rcmcq_variants/<model>[__rev=...]/summary.csv``
and concatenates them into:

  results/evals_paper/03_rcmcq_variants/_aggregated_wide.csv
      one row per (model, revision, prompt, cefr) with columns
      acc_mean, acc_sum, acc_pmi_mean, acc_pmi_sum, n_questions, n_unacceptable.

  results/evals_paper/03_rcmcq_variants/_aggregated_long.csv
      one row per (model, revision, prompt, cefr, rule).

  results/evals_paper/03_rcmcq_variants/_per_model_means.csv
      one row per (model, revision); columns are mean accuracy (across all
      prompt × cefr cells) for each scoring rule, plus the delta of each rule
      over ``acc_mean`` (the eval's default scoring).

Optional: pass --merge-rcmcq to also pull in the original
``results/evals_paper/03_rcmcq/<model>/summary.csv`` baseline as a row with
rule=baseline_mean. Useful when comparing across models that have variant
results vs models that only have the original eval.

Usage:
  python -m analyze.rcmcq_rescore
  python -m analyze.rcmcq_rescore --root results/evals_paper/03_rcmcq_variants \
      --merge-rcmcq results/evals_paper/03_rcmcq
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

REV_RE = re.compile(r"__rev=(?P<rev>[^/]+)$")
RULE_COLUMNS = ["acc_mean", "acc_sum", "acc_pmi_mean", "acc_pmi_sum"]


def _split_rev(dir_name: str) -> tuple[str, str]:
    m = REV_RE.search(dir_name)
    if m:
        return dir_name[: m.start()], m.group("rev")
    return dir_name, "main"


def _collect_variants(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    if not root.exists():
        return pd.DataFrame()
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_"):
            continue
        sf = model_dir / "summary.csv"
        if not sf.exists():
            continue
        df = pd.read_csv(sf)
        if df.empty:
            continue
        repo, rev = _split_rev(model_dir.name)
        df["model_dir"] = model_dir.name
        df["repo_safe"] = repo
        df["revision"] = rev
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _collect_baseline(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    if not root.exists():
        return pd.DataFrame()
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_"):
            continue
        sf = model_dir / "summary.csv"
        if not sf.exists():
            continue
        df = pd.read_csv(sf)
        df = df[df["cefr"].isin(["A2", "B1", "B2", "C1", "C2"])].copy()
        if df.empty:
            continue
        repo, rev = _split_rev(model_dir.name)
        df["model_dir"] = model_dir.name
        df["repo_safe"] = repo
        df["revision"] = rev
        df = df.rename(columns={"accuracy": "acc_mean"})
        for c in ("acc_sum", "acc_pmi_mean", "acc_pmi_sum"):
            df[c] = float("nan")
        rows.append(df[["model", "model_dir", "repo_safe", "revision",
                        "prompt", "cefr", "n_questions", "n_unacceptable",
                        "acc_mean", "acc_sum", "acc_pmi_mean", "acc_pmi_sum"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path,
                    default=Path("results/evals_paper/03_rcmcq_variants"))
    ap.add_argument("--merge-rcmcq", type=Path, default=None,
                    help="If set, also load the original 03_rcmcq summaries "
                         "and merge them under acc_mean (other rules NaN).")
    args = ap.parse_args()

    wide = _collect_variants(args.root)
    if args.merge_rcmcq is not None:
        base = _collect_baseline(args.merge_rcmcq)
        if not base.empty:
            base["source"] = "rcmcq_baseline"
            wide["source"] = "rcmcq_variants"
            # Drop baseline rows that are already covered by variants for the
            # same (repo_safe, revision, prompt, cefr).
            key = ["repo_safe", "revision", "prompt", "cefr"]
            covered = set(map(tuple, wide[key].itertuples(index=False, name=None)))
            base = base[~base[key].apply(tuple, axis=1).isin(covered)]
            wide = pd.concat([wide, base], ignore_index=True, sort=False)
        else:
            wide["source"] = "rcmcq_variants"
    else:
        wide["source"] = "rcmcq_variants"

    if wide.empty:
        print(f"[rcmcq_rescore] no inputs found under {args.root}")
        return 1

    args.root.mkdir(parents=True, exist_ok=True)
    wide_csv = args.root / "_aggregated_wide.csv"
    wide.to_csv(wide_csv, index=False)
    print(f"[rcmcq_rescore] wrote {wide_csv}  ({len(wide)} rows)")

    long_rows: list[dict] = []
    for _, r in wide.iterrows():
        for rule in RULE_COLUMNS:
            v = r.get(rule)
            if pd.isna(v):
                continue
            long_rows.append({
                "model": r["model"], "model_dir": r["model_dir"],
                "repo_safe": r["repo_safe"], "revision": r["revision"],
                "prompt": r["prompt"], "cefr": r["cefr"],
                "rule": rule, "accuracy": float(v),
                "n_questions": int(r["n_questions"]),
            })
    long = pd.DataFrame(long_rows)
    long_csv = args.root / "_aggregated_long.csv"
    long.to_csv(long_csv, index=False)
    print(f"[rcmcq_rescore] wrote {long_csv}  ({len(long)} rows)")

    # Per-(model, revision) means + deltas vs acc_mean
    per_rows: list[dict] = []
    grouped = wide.groupby(["repo_safe", "revision"], dropna=False)
    for (repo, rev), g in grouped:
        row = {"repo_safe": repo, "revision": rev,
               "n_cells": len(g)}
        for rule in RULE_COLUMNS:
            row[f"{rule}_mean"] = g[rule].mean()
        base_mean = row["acc_mean_mean"]
        for rule in RULE_COLUMNS:
            row[f"{rule}_delta_vs_mean"] = row[f"{rule}_mean"] - base_mean
        per_rows.append(row)
    per = pd.DataFrame(per_rows).sort_values(["repo_safe", "revision"])
    per_csv = args.root / "_per_model_means.csv"
    per.to_csv(per_csv, index=False)
    print(f"[rcmcq_rescore] wrote {per_csv}  ({len(per)} rows)")

    if not per.empty:
        with pd.option_context("display.max_columns", None,
                               "display.width", 200,
                               "display.float_format", lambda v: f"{v:.4f}"):
            print()
            print("Per-model mean accuracy across cells:")
            print(per.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
