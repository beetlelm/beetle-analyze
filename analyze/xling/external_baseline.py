"""
analyze/xling/external_baseline.py — §I framing-gate run.

Runs the §A geometry pipeline on **non-curriculum** bilingual baselines
(XLM-R-base, mBERT). If those models sit at b5-like CKA, the curriculum-
specificity claim survives; if they sit at b1-like CKA, the framing must
change before submission. Run early — this is a 1-GPU-hour result that
gates everything else (per the EMNLP plan, decision point #5).

Outputs
-------
results/evals_paper/13_external_baseline/geometry_xlmr_mbert.csv
  Same schema as 05_geometry/cka_per_layer.csv, with model column.

Implementation note
-------------------
This module is a thin wrapper over ``analyze.xling.cka_curriculum`` so the
CKA + bootstrap-CI logic stays single-sourced.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from analyze._interp_helpers import (
    EXTERNAL_BASELINE_REPOS, RESULTS_PAPER, write_csv_atomic,
)

logger = logging.getLogger(__name__)


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repos", nargs="+",
                    default=list(EXTERNAL_BASELINE_REPOS.values()))
    p.add_argument("--names", nargs="+",
                    default=list(EXTERNAL_BASELINE_REPOS.keys()))
    p.add_argument("--l1", default="nld"); p.add_argument("--l2", default="eng")
    p.add_argument("--n-pairs", type=int, default=1000)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", default=str(RESULTS_PAPER / "13_external_baseline"
                                         / "geometry_xlmr_mbert.csv"))
    args = p.parse_args(list(argv) if argv is not None else None)

    from analyze.ppl_utils import load_flores_sentences

    from analyze.xling.cka_curriculum import run_one_model

    l1_sents = load_flores_sentences(args.l1)[:args.n_pairs]
    l2_sents = load_flores_sentences(args.l2)[:args.n_pairs]

    rows: list[dict] = []
    for repo, name in zip(args.repos, args.names):
        rows.extend(run_one_model(
            repo, l1_iso=args.l1, l2_iso=args.l2,
            curriculum=name, lang_pair=f"{args.l1}-{args.l2}",
            l1_sentences=l1_sents, l2_sentences=l2_sents,
            stratum="external_baseline", n_boot=args.n_boot,
        ))
    df = pd.DataFrame(rows)
    write_csv_atomic(df, args.out)
    print(f"[external_baseline] wrote {len(df)} rows → {args.out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
