"""
analyze/n400_alignment.py — N400 ERP alignment via ∆LL.

For Paper-A RQ-A4 we test the dissociation reported by Michaelov & Bergen
(2026): better LMs predict the **N400** ERP *better*, but RT alignment
worsens past a quality threshold.  This module reuses the same regression
machinery as `ppp_delta_ll.py` but swaps reading-time for N400 amplitude.

Supported corpora
-----------------
    michaelov_bergen_2026 — meta-corpus of 9 N400 datasets curated by M&B
    frank_etal            — Frank et al. naturalistic ERP corpus

Each loader returns a DataFrame with the same join keys as the eye-tracking
loaders (sent_id, word_id) plus an ``N400`` column (mean amplitude in µV
within the 300–500 ms window, residualised for word position, length, and
log-frequency offline).

CLI
---
    python -m analyze.n400_alignment \
        --surprisals results/meco/surprisals_all.tsv \
        --corpus     michaelov_bergen_2026 \
        --output     results/n400/mb2026_delta_ll.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from analyze.ppp_delta_ll import _ensure_baseline_columns


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "n400"
DATA_DIR    = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Corpus loaders
# ---------------------------------------------------------------------------

def _load_michaelov_bergen_2026() -> pd.DataFrame:
    p = DATA_DIR / "n400" / "michaelov_bergen_2026.tsv"
    df = pd.read_csv(p, sep="\t")
    if "RT" not in df.columns:
        df = df.rename(columns={"N400": "RT"})
    return df


def _load_frank_etal() -> pd.DataFrame:
    p = DATA_DIR / "n400" / "frank_etal.tsv"
    df = pd.read_csv(p, sep="\t")
    if "RT" not in df.columns:
        df = df.rename(columns={"N400": "RT"})
    return df


_N400_LOADERS: dict[str, Callable[[], pd.DataFrame]] = {
    "michaelov_bergen_2026": _load_michaelov_bergen_2026,
    "frank_etal":            _load_frank_etal,
}


def load_n400(name: str) -> pd.DataFrame:
    if name not in _N400_LOADERS:
        raise ValueError(f"Unknown N400 corpus '{name}'. "
                         f"Known: {sorted(_N400_LOADERS)}")
    return _N400_LOADERS[name]()


# ---------------------------------------------------------------------------
# Per-checkpoint ∆LL (RT column == residualised N400)
# ---------------------------------------------------------------------------

def n400_delta_ll(
    surprisal_df: pd.DataFrame,
    n400_df: pd.DataFrame,
    *,
    join_on: tuple[str, ...] = ("sent_id", "word_id"),
    surprisal_col: str = "surprisal",
) -> dict[str, float]:
    """Same ∆LL machinery as `ppp_delta_ll.delta_ll`, but `RT` ≡ N400 amplitude."""
    from analyze.ppp_delta_ll import delta_ll
    return delta_ll(surprisal_df, n400_df, join_on=join_on, surprisal_col=surprisal_col)


def n400_trajectory(
    surprisal_csv: Path,
    corpus: str = "michaelov_bergen_2026",
    *,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    sd = pd.read_csv(surprisal_csv, sep="\t" if str(surprisal_csv).endswith(".tsv") else ",")
    if "step" not in sd.columns and "revision" in sd.columns:
        sd["step"] = sd["revision"].astype(str).str.extract(r"step-(\d+)").astype(float)
    n400 = load_n400(corpus)

    rows = []
    for (model, step), grp in sd.groupby(["model", "step"], dropna=False):
        st = n400_delta_ll(grp, n400)
        rows.append({"corpus": corpus, "model": model, "step": step, **st})
    out = pd.DataFrame(rows)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
        print(f"[n400] wrote → {output_path}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="N400 ∆LL alignment.")
    p.add_argument("--surprisals", required=True)
    p.add_argument("--corpus", choices=sorted(_N400_LOADERS),
                   default="michaelov_bergen_2026")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    out = Path(args.output) if args.output else (
        RESULTS_DIR / f"{args.corpus}_delta_ll.csv"
    )
    n400_trajectory(Path(args.surprisals), args.corpus, output_path=out)
