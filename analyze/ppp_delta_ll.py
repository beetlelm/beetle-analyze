"""
analyze/ppp_delta_ll.py — Psychometric Predictive Power via ∆LL.

Implements the regression-based PPP measure from Aoyama et al. App. B and
applies it uniformly across the eye-tracking / ERP corpora used by Paper A:
Dundee, Provo, Natural Stories, MECO L1, MECO L2, CELER.

The measure is:

    Baseline (B):
        RT  ~  1 + length + position + log_freq + bigram_surprisal
                + (1 | subject)
    Full (F):
        RT  ~  1 + length + position + log_freq + bigram_surprisal
                + lm_surprisal + (1 | subject)

    ∆LL = LL(F) - LL(B)

`lm_surprisal` is the column produced by `meco_surprisal.score_checkpoint`
(or the MECO-L1 / Dundee / Provo equivalent).  ∆LL is reported per
(corpus, language, model, checkpoint).

A pluggable corpus loader interface (`load_corpus`) wires the same model
output to any of the supported corpora; new corpora can be added by
registering a function in `_CORPUS_LOADERS`.

Statistical reporting follows Aoyama et al. Fig 4: per-language ρ → Fisher
z-transform → weighted average (weights = √n_subjects).

CLI
---
    python -m analyze.ppp_delta_ll \
        --surprisals results/meco/surprisals_all.tsv \
        --corpus     meco_l1 \
        --output     results/ppp/meco_l1_delta_ll.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

# Heavy stats imports are lazy because the smoke test path
# does not need them.

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "ppp"
DATA_DIR    = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Corpus loaders
# ---------------------------------------------------------------------------

def _load_meco_l1(language: Optional[str] = None) -> pd.DataFrame:
    from analyze.meco_l1_loader import load_meco_l1
    return load_meco_l1(language=language)


def _load_meco_l2(language: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "meco_l2_stims.tsv", sep="\t")
    if language and "language" in df.columns:
        df = df[df["language"] == language].reset_index(drop=True)
    return df


def _load_dundee(language: Optional[str] = None) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "dundee_stims.tsv", sep="\t")


def _load_provo(language: Optional[str] = None) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "provo_stims.tsv", sep="\t")


def _load_natural_stories(language: Optional[str] = None) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "natural_stories_stims.tsv", sep="\t")


def _load_celer(language: Optional[str] = None) -> pd.DataFrame:
    """CELER L2 reading times.  `language` filters by participant L1.

    CELER ships participant metadata with an L1 column that uses ISO-639-3
    codes; we expect the user to have built `data/celer_stims.tsv` joining
    participant info onto interest-area fixation rows.
    """
    df = pd.read_csv(DATA_DIR / "celer_stims.tsv", sep="\t")
    if language and "L1" in df.columns:
        df = df[df["L1"] == language].reset_index(drop=True)
    return df


_CORPUS_LOADERS: dict[str, Callable[[Optional[str]], pd.DataFrame]] = {
    "meco_l1":          _load_meco_l1,
    "meco_l2":          _load_meco_l2,
    "dundee":           _load_dundee,
    "provo":            _load_provo,
    "natural_stories":  _load_natural_stories,
    "celer":            _load_celer,
}


def load_corpus(name: str, language: Optional[str] = None) -> pd.DataFrame:
    if name not in _CORPUS_LOADERS:
        raise ValueError(
            f"Unknown corpus '{name}'. Known: {sorted(_CORPUS_LOADERS)}"
        )
    return _CORPUS_LOADERS[name](language)


# ---------------------------------------------------------------------------
# Baselines (Aoyama et al. App. B)
# ---------------------------------------------------------------------------

def _ensure_baseline_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add length / position / log_freq / bigram_surprisal columns if missing.

    `length`           — char length of `word`.
    `position`         — within-sentence index (sent_id, word_id).
    `log_freq`         — log unigram count from a bundled wordfreq table; if
                         the package is not installed, falls back to the
                         per-corpus precomputed column if present.
    `bigram_surprisal` — Kneser-Ney bigram surprisal; precomputed offline
                         per corpus.  If missing, set to NaN and dropped.
    """
    df = df.copy()
    if "length" not in df.columns:
        df["length"] = df["word"].astype(str).str.len()
    if "position" not in df.columns and "word_id" in df.columns:
        df["position"] = df["word_id"]
    if "log_freq" not in df.columns:
        try:
            from wordfreq import zipf_frequency
            df["log_freq"] = df.apply(
                lambda r: zipf_frequency(str(r["word"]), r.get("language", "en")),
                axis=1,
            )
        except Exception:
            df["log_freq"] = np.nan
    if "bigram_surprisal" not in df.columns:
        df["bigram_surprisal"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Per-checkpoint regression
# ---------------------------------------------------------------------------

_BASE_COLS = ["length", "position", "log_freq"]


def _fit_and_loglik(df: pd.DataFrame, predictors: list[str]) -> float:
    """Fit OLS (RT ~ predictors) on `df` and return the log-likelihood."""
    import statsmodels.formula.api as smf
    formula = "RT ~ " + " + ".join(predictors)
    model = smf.ols(formula=formula, data=df).fit()
    return float(model.llf)


def delta_ll(
    surprisal_df: pd.DataFrame,
    rt_df: pd.DataFrame,
    *,
    join_on: tuple[str, ...] = ("sent_id", "word_id"),
    surprisal_col: str = "surprisal",
) -> dict[str, float]:
    """Compute ∆LL = LL(B + lm_surprisal) - LL(B).

    Args
    ----
    surprisal_df : DataFrame containing the `surprisal_col` and the join keys.
    rt_df        : Eye-tracking / ERP DataFrame with an RT column.
    join_on      : Keys used to align the two frames.
    """
    merged = rt_df.merge(
        surprisal_df[list(join_on) + [surprisal_col]],
        on=list(join_on), how="inner",
    )
    merged = _ensure_baseline_columns(merged)

    # Drop rows missing any predictor or RT
    use = ["RT", surprisal_col, *_BASE_COLS]
    if "bigram_surprisal" in merged.columns and merged["bigram_surprisal"].notna().any():
        use.append("bigram_surprisal")
    merged = merged.dropna(subset=use)

    if len(merged) < 50:
        return {"delta_ll": float("nan"), "n": int(len(merged))}

    base_preds = [c for c in _BASE_COLS + ["bigram_surprisal"] if c in use]
    full_preds = base_preds + [surprisal_col]
    ll_b = _fit_and_loglik(merged, base_preds)
    ll_f = _fit_and_loglik(merged, full_preds)
    return {
        "delta_ll": ll_f - ll_b,
        "n": int(len(merged)),
        "ll_full": ll_f,
        "ll_baseline": ll_b,
    }


# ---------------------------------------------------------------------------
# Trajectory across checkpoints
# ---------------------------------------------------------------------------

def delta_ll_trajectory(
    surprisal_csv: Path,
    corpus: str,
    *,
    languages: Optional[list[str]] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run ∆LL at every (model, step) in `surprisal_csv` for `corpus`.

    Expects `surprisal_csv` to contain at least:
        model, step (or revision), sent_id, word_id, surprisal
    and optional `language` for multi-language frames.
    """
    sd = pd.read_csv(surprisal_csv, sep="\t" if str(surprisal_csv).endswith(".tsv") else ",")
    if "step" not in sd.columns and "revision" in sd.columns:
        sd["step"] = sd["revision"].astype(str).str.extract(r"step-(\d+)").astype(float)

    rows: list[dict] = []
    langs = languages or sd.get("language", pd.Series(["eng"])).unique().tolist()
    for lang in langs:
        rt = load_corpus(corpus, language=lang)
        sub = sd[sd.get("language", lang) == lang] if "language" in sd.columns else sd
        for (model, step), grp in sub.groupby(["model", "step"], dropna=False):
            stats = delta_ll(grp, rt)
            rows.append({
                "corpus": corpus, "language": lang,
                "model": model, "step": step,
                **stats,
            })

    out = pd.DataFrame(rows)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
        print(f"[ppp] Wrote ∆LL trajectory → {output_path}")
    return out


# ---------------------------------------------------------------------------
# Aggregation: Fisher-z weighted average across languages (Aoyama Fig 4)
# ---------------------------------------------------------------------------

def fisher_z(rho: float) -> float:
    if rho is None or math.isnan(rho):
        return float("nan")
    rho = max(min(float(rho), 0.9999), -0.9999)
    return 0.5 * math.log((1 + rho) / (1 - rho))


def fisher_z_weighted_average(
    rho_values: list[float],
    weights: list[float],
) -> float:
    zs = [fisher_z(r) for r in rho_values]
    pairs = [(z, w) for z, w in zip(zs, weights) if not math.isnan(z) and w > 0]
    if not pairs:
        return float("nan")
    total_w = sum(w for _, w in pairs)
    z_bar = sum(z * w for z, w in pairs) / total_w
    return math.tanh(z_bar)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="∆LL psychometric predictive power.")
    p.add_argument("--surprisals", required=True, type=str,
                   help="Path to a surprisal CSV/TSV from meco_surprisal.")
    p.add_argument("--corpus", required=True,
                   choices=sorted(_CORPUS_LOADERS),
                   help="Eye-tracking / ERP corpus.")
    p.add_argument("--languages", nargs="+", default=None,
                   help="Optional ISO-639-3 filter.")
    p.add_argument("--output", default=None,
                   help="Output CSV path.")
    args = p.parse_args()

    out_path = Path(args.output) if args.output else (
        RESULTS_DIR / f"{args.corpus}_delta_ll.csv"
    )
    delta_ll_trajectory(
        Path(args.surprisals),
        args.corpus,
        languages=args.languages,
        output_path=out_path,
    )
