#!/usr/bin/env python3
"""scripts/b3_checkpoints_postprocess.py

Post-processing for results/evals_b3_checkpoints/.

(1) Reads every per-checkpoint MECO surprisal CSV
    (results/evals_b3_checkpoints/meco/surprisals_*__step-*.csv),
    rewrites the ``model`` column to ``<regime>@<step>`` so each
    checkpoint is treated as a distinct pseudo-model, then calls the
    canonical MECO ΔlogLik pipeline (analyze.meco_canonical) to fit
    full + baseline lmer-style models on the joined L2 wave-1+2 reading-time
    data and emit one row per (regime, ckpt, lang).

(2) Reads results/evals_b3_checkpoints/bliss/bliss_results.csv (already long
    by (model, branch, l1_language, auto_cefr)) and adds a ``regime`` and
    integer ``step`` column for plotting.

(3) Joins MECO ΔlogLik and BLiSS at the (regime, step) granularity and
    writes a single comparison summary at
    results/evals_b3_checkpoints/regime_summary.csv.

Output files:
  results/evals_b3_checkpoints/meco_canonical_deltalogl.csv
  results/evals_b3_checkpoints/bliss_long.csv
  results/evals_b3_checkpoints/regime_summary.csv
"""
from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make analyze.* importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyze.meco_canonical import (
    compute_delta_loglik_all_models,
    CANONICAL_LANGS,
)

OUT_ROOT = ROOT / "results/evals_b3_checkpoints"
MECO_DIR = OUT_ROOT / "meco"
BLISS_CSV = OUT_ROOT / "bliss" / "bliss_results.csv"
BLIMP_CSV = OUT_ROOT / "blimp" / "blimp_results.csv"

REGIME_PREFIX = (
    "beetle-nld-eng-variants/"
    "beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-2b-nld-eng-"
)
REGIME_PREFIX_HF = (
    "Beetle-nld-eng-Variants/"
    "beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-2b-nld-eng-"
)


def _regime_from_model(m: str) -> str:
    s = m.lower()
    if s.startswith(REGIME_PREFIX):
        return s[len(REGIME_PREFIX):]
    return s.split("/")[-1]


_STEP_RE = re.compile(r"step-?(\d+)$", re.I)


def _step_to_int(b: str) -> int:
    if b is None or pd.isna(b):
        return -1
    s = str(b).strip()
    if s == "main":
        return 10**9   # treat as final
    m = _STEP_RE.match(s)
    return int(m.group(1)) if m else -1


def _load_surprisal_long() -> pd.DataFrame:
    """Concatenate per-checkpoint MECO surprisal CSVs, with model='<regime>@<step>'."""
    files = sorted(glob.glob(str(MECO_DIR / "surprisals_*__step-*.csv")))
    if not files:
        raise SystemExit(f"no MECO surprisal CSVs under {MECO_DIR}")
    parts = []
    for f in files:
        df = pd.read_csv(f)
        if df.empty:
            continue
        # Each CSV has a single (model, branch). Use the file's model+branch
        # to ensure a clean key (df-level columns may have lowercased names
        # from minicons).
        m = re.match(r"surprisals_(.+)__step-(\d+)\.csv$", os.path.basename(f))
        if not m:
            continue
        repo_slug, step = m.group(1), int(m.group(2))
        regime = _regime_from_model(repo_slug.replace("_", "/", 1))
        df = df.copy()
        df["regime"] = regime
        df["step"] = step
        df["model"] = f"{regime}@step-{step}"
        parts.append(df)
    surp = pd.concat(parts, ignore_index=True)
    print(f"[postprocess] loaded {len(surp):,} surprisal rows "
          f"across {surp['model'].nunique()} (regime,step) pseudo-models")
    return surp


def run_meco_deltalogl(surp: pd.DataFrame) -> pd.DataFrame:
    # nld-eng regimes are most diagnostic on Dutch L2 readers (lang='du');
    # keep the canonical 3-lang set behind LANGS=ALL but default to 'du' to
    # keep wall time bounded (statsmodels fits each (model, lang) pair).
    langs_env = os.environ.get("LANGS", "du").strip()
    if langs_env == "ALL":
        langs = CANONICAL_LANGS
    else:
        langs = tuple(s for s in langs_env.split() if s)
    print(f"[postprocess] fitting MECO ΔlogLik on langs={list(langs)}")
    cache = OUT_ROOT / "meco_canonical_deltalogl.csv"
    out = compute_delta_loglik_all_models(
        surp,
        cache_path=cache,
        overwrite=True,
        backend="auto",   # statsmodels fallback if pymer4 missing
        verbose=True,
        langs=langs,
    )
    # Split pseudo-model back into (regime, step).
    rs = out["model"].str.extract(r"^(?P<regime>[^@]+)@step-(?P<step>\d+)$")
    out["regime"] = rs["regime"]
    out["step"] = pd.to_numeric(rs["step"], errors="coerce").astype("Int64")
    out = out[["regime", "step", "lang", "delta_loglik",
               "loglik_full", "loglik_baseline",
               "beta_surprisal", "beta_surprisal_se",
               "n_obs", "n_subjects",
               "fit_backend", "fit_backend_baseline", "notes"]]
    out = out.sort_values(["regime", "lang", "step"]).reset_index(drop=True)
    print(f"[postprocess] wrote MECO ΔlogLik → {cache} ({len(out)} rows)")
    return out


def load_bliss_long() -> pd.DataFrame:
    if not BLISS_CSV.exists():
        print(f"[postprocess] WARN no BLiSS CSV at {BLISS_CSV}")
        return pd.DataFrame()
    bl = pd.read_csv(BLISS_CSV)
    bl = bl.dropna(subset=["model", "branch"])
    bl["regime"] = bl["model"].astype(str).map(_regime_from_model)
    bl["step"] = bl["branch"].astype(str).map(_step_to_int)
    bl = bl[bl["step"] >= 0].copy()
    out = OUT_ROOT / "bliss_long.csv"
    bl.to_csv(out, index=False)
    print(f"[postprocess] wrote BLiSS long → {out} ({len(bl)} rows)")
    return bl


def load_blimp_long() -> pd.DataFrame:
    """Read blimp_results.csv and tag with regime + integer step.

    Schema in: benchmark, model, branch, model_family, eval_language, iso_code,
               accuracy, n_correct, n_total.
    Returns the same with added (regime, step) columns and any rows whose
    branch is not parseable as an integer step dropped (we don't plot 'main').
    """
    if not BLIMP_CSV.exists():
        print(f"[postprocess] WARN no BLiMP CSV at {BLIMP_CSV}")
        return pd.DataFrame()
    bp = pd.read_csv(BLIMP_CSV)
    bp = bp.dropna(subset=["model", "branch", "benchmark"])
    bp["regime"] = bp["model"].astype(str).map(_regime_from_model)
    bp["step"] = bp["branch"].astype(str).map(_step_to_int)
    bp = bp[bp["step"] >= 0].copy()
    out = OUT_ROOT / "blimp_long.csv"
    bp.to_csv(out, index=False)
    print(f"[postprocess] wrote BLiMP long → {out} ({len(bp)} rows)")
    return bp


def aggregate_summary(meco_dl: pd.DataFrame, bl: pd.DataFrame, bp: pd.DataFrame) -> pd.DataFrame:
    # MECO: average ΔlogLik across canonical langs (du, ch_s, ge) per (regime, step).
    mec = (meco_dl.groupby(["regime", "step"], as_index=False)
                  .agg(
                       meco_delta_loglik_du=("delta_loglik",
                            lambda s: s[meco_dl.loc[s.index, "lang"] == "du"].mean()),
                       meco_delta_loglik_ge=("delta_loglik",
                            lambda s: s[meco_dl.loc[s.index, "lang"] == "ge"].mean()),
                       meco_delta_loglik_ch_s=("delta_loglik",
                            lambda s: s[meco_dl.loc[s.index, "lang"] == "ch_s"].mean()),
                       meco_delta_loglik_mean=("delta_loglik", "mean"),
                  ))

    # BLiSS: weighted mean of RP@0 / RP@tau / NGS / CPS over strata, per (regime, step).
    if not bl.empty:
        # Limit to nld+eng-relevant strata (already filtered upstream). Compute
        # per-(regime, step, l1_language) means weighted by n_items, then average
        # across L1 languages.
        per_lang = (
            bl.groupby(["regime", "step", "l1_language"], as_index=False)
              .apply(lambda d: pd.Series({
                  "RP_at_0": np.average(d["RP_at_0"], weights=d["n_items"]),
                  "RP_at_tau": np.average(d["RP_at_tau"], weights=d["n_items"]),
                  "NGS": np.average(d["NGS"], weights=d["n_items"]),
                  "CPS": np.average(d["CPS"], weights=d["n_items"]),
                  "n_items_total": d["n_items"].sum(),
              }), include_groups=False)
              .reset_index(drop=True)
        )
        bliss_overall = (
            per_lang.groupby(["regime", "step"], as_index=False)
                    [["RP_at_0", "RP_at_tau", "NGS", "CPS"]]
                    .mean()
                    .rename(columns={
                        "RP_at_0": "bliss_rp0",
                        "RP_at_tau": "bliss_rp_tau",
                        "NGS": "bliss_ngs",
                        "CPS": "bliss_cps",
                    })
        )
        # Wide per-language as well.
        wide = (per_lang.pivot_table(
                    index=["regime", "step"],
                    columns="l1_language",
                    values=["RP_at_0", "RP_at_tau", "NGS", "CPS"],
                ))
        wide.columns = [f"bliss_{m.lower()}_{lang.lower().replace(' ', '_')}"
                        for m, lang in wide.columns]
        wide = wide.reset_index()
    else:
        bliss_overall = pd.DataFrame(columns=["regime", "step",
                                              "bliss_rp0", "bliss_rp_tau",
                                              "bliss_ngs", "bliss_cps"])
        wide = pd.DataFrame(columns=["regime", "step"])

    # BLiMP: pivot benchmark → wide column per (regime, step).
    if not bp.empty:
        blimp_wide = (
            bp.pivot_table(
                index=["regime", "step"],
                columns="benchmark",
                values="accuracy",
                aggfunc="mean",
            )
            .reset_index()
        )
        blimp_wide = blimp_wide.rename(columns={
            "blimp_eng": "blimp_eng_acc",
            "blimp_nl":  "blimp_nl_acc",
        })
    else:
        blimp_wide = pd.DataFrame(columns=["regime", "step",
                                           "blimp_eng_acc", "blimp_nl_acc"])

    summary = mec.merge(bliss_overall, on=["regime", "step"], how="outer") \
                 .merge(wide, on=["regime", "step"], how="outer") \
                 .merge(blimp_wide, on=["regime", "step"], how="outer")
    summary = summary.sort_values(["regime", "step"]).reset_index(drop=True)
    out = OUT_ROOT / "regime_summary.csv"
    summary.to_csv(out, index=False)
    print(f"[postprocess] wrote regime summary → {out} ({len(summary)} rows)")
    return summary


def main() -> None:
    # If a ΔlogLik cache already exists (e.g., produced by
    # scripts/b3_meco_deltalogl_parallel.py), skip the slow refit and use it.
    cache = OUT_ROOT / "meco_canonical_deltalogl.csv"
    if cache.exists() and os.environ.get("REFIT_MECO", "0") != "1":
        print(f"[postprocess] using existing ΔlogLik cache {cache}")
        meco_dl = pd.read_csv(cache)
        rs = meco_dl["model"].astype(str).str.extract(
            r"^(?P<regime>[^@]+)@step-(?P<step>\d+)$"
        )
        meco_dl["regime"] = rs["regime"]
        meco_dl["step"] = pd.to_numeric(rs["step"], errors="coerce").astype("Int64")
    else:
        surp = _load_surprisal_long()
        meco_dl = run_meco_deltalogl(surp)
    bl = load_bliss_long()
    bp = load_blimp_long()
    summary = aggregate_summary(meco_dl, bl, bp)
    print()
    print("=== regime summary head ===")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
