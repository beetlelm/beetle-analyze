"""
analyze/multiple_comparisons.py — pre-registered vs FDR-corrected cell tracker.

The EMNLP plan splits comparisons into:

  - **Theory-driven (pre-registered)** — α = 0.05, no correction.
  - **Exploratory** — Benjamini-Hochberg FDR at q = 0.05, computed within
    each analysis section (not pooled across §A/§B/§E/...).

This module provides a single source of truth so every plot caption can
say "(pre-registered, p = …)" or "(BH-FDR-corrected, q = …)" without
ambiguity.

Pre-registration is recorded in ``docs/interp_preregistration.md``. The
``--prereg-yaml`` flag takes a frozen YAML listing of (section, cell-name,
direction, threshold) so any cell that does not appear is automatically
treated as exploratory.

Outputs
-------
results/evals_paper/multiple_comparisons.csv
  One row per comparison cell with the chosen correction and decision.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from analyze._interp_helpers import RESULTS_PAPER, write_csv_atomic


def benjamini_hochberg(pvals: Sequence[float], q: float = 0.05
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rejected, q_values)`` under BH-FDR at level ``q``."""
    p = np.asarray(list(pvals), dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    qvals = np.empty(n, dtype=float)
    # standard BH adjustment: q_i = min_{j>=i}( p_j * n / j )
    running_min = float("inf")
    for i in range(n - 1, -1, -1):
        adj = ranked[i] * n / max(i + 1, 1)
        running_min = min(running_min, adj)
        qvals[order[i]] = min(running_min, 1.0)
    rejected = qvals <= q
    return rejected, qvals


def annotate(df: pd.DataFrame, *, prereg: set[str], section_col: str = "section",
              cell_col: str = "cell", p_col: str = "p_value", q: float = 0.05,
              alpha: float = 0.05) -> pd.DataFrame:
    """Annotate ``df`` with a ``correction`` column and a ``significant``
    decision per the pre-registered-vs-exploratory rule.

    ``prereg`` is the set of ``"<section>::<cell>"`` keys that count as
    pre-registered. Anything else is exploratory and gets BH-FDR within
    its section.
    """
    df = df.copy()
    df["is_prereg"] = df.apply(
        lambda r: f"{r[section_col]}::{r[cell_col]}" in prereg, axis=1
    )
    df["correction"] = np.where(df["is_prereg"], "prereg", "fdr")
    df["q_value"] = np.nan
    df["significant"] = False

    # Pre-registered: simple α threshold.
    pre_mask = df["is_prereg"]
    df.loc[pre_mask, "significant"] = df.loc[pre_mask, p_col] <= alpha

    # Exploratory: BH within each section.
    for sec, sub in df[~pre_mask].groupby(section_col):
        rej, qv = benjamini_hochberg(sub[p_col].fillna(1.0).values, q=q)
        df.loc[sub.index, "q_value"] = qv
        df.loc[sub.index, "significant"] = rej
    return df


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True,
                    help="Long-format CSV with at least: section, cell, p_value.")
    p.add_argument("--prereg-yaml", default=None,
                    help="Frozen pre-reg YAML; entries become 'section::cell' "
                         "keys treated as theory-driven.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--q", type=float, default=0.05)
    p.add_argument("--out", default=str(RESULTS_PAPER / "multiple_comparisons.csv"))
    args = p.parse_args(list(argv) if argv is not None else None)

    df = pd.read_csv(args.input_csv)

    prereg: set[str] = set()
    if args.prereg_yaml and Path(args.prereg_yaml).exists():
        try:
            import yaml  # type: ignore[import-not-found]
            with open(args.prereg_yaml) as f:
                spec = yaml.safe_load(f)
            for sec, cells in (spec or {}).items():
                for c in cells or []:
                    prereg.add(f"{sec}::{c}")
        except Exception:
            pass

    out = annotate(df, prereg=prereg, alpha=args.alpha, q=args.q)
    write_csv_atomic(out, args.out)
    print(f"[multiple_comparisons] {out['significant'].sum()} significant of "
          f"{len(out)} cells → {args.out}")


if __name__ == "__main__":
    main()
