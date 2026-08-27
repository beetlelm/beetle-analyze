"""
analyze/forgetting_pkg/l1_attrition.py — §F catastrophic-interference retrace.

Re-runs L1 evals (Dutch FLORES BPB, Dutch BLiSS) on every checkpoint of a
given curriculum (default: b5 nld-eng), to detect language attrition during
late heavy L2 exposure. Either result is publishable per the EMNLP plan.

Outputs
-------
results/evals_paper/10_attrition/b5_l1_metrics_trace.csv
  Columns: model_id, curriculum, revision, step, l1_iso, mean_bpb, n_pairs
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from analyze._interp_helpers import RESULTS_PAPER, write_csv_atomic

logger = logging.getLogger(__name__)


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="Beetle-FineWeb/beetle-bilingual-b5-nld-eng")
    p.add_argument("--curriculum", default="b5")
    p.add_argument("--l1", default="nld")
    p.add_argument("--n-pairs", type=int, default=512)
    p.add_argument("--max-revisions", type=int, default=14)
    p.add_argument("--out", default=str(RESULTS_PAPER / "10_attrition"
                                         / "b5_l1_metrics_trace.csv"))
    args = p.parse_args(list(argv) if argv is not None else None)

    from analyze.byte_normalize import bpt_to_bpb
    from analyze.ppl_utils import (
        load_flores_sentences, load_model_and_tokenizer, sentence_log_likelihood,
    )
    from analyze.xling.dev_trajectory import list_checkpoint_revisions

    sents = load_flores_sentences(args.l1)[:args.n_pairs]
    revs = list_checkpoint_revisions(args.repo)
    if len(revs) > args.max_revisions:
        idx = np.linspace(0, len(revs) - 1, args.max_revisions).round().astype(int)
        revs = [revs[i] for i in idx]

    rows = []
    for rev in revs:
        pair = load_model_and_tokenizer(args.repo, revision=rev)
        if pair is None:
            continue
        model, tokenizer = pair
        nlls = [sentence_log_likelihood(s, model, tokenizer) for s in sents]
        nlls = [n for n in nlls if np.isfinite(n)]
        if not nlls:
            continue
        mean_bpt = float(np.mean(nlls)) / np.log(2)         # nats → bits/token
        mean_bpb = bpt_to_bpb(mean_bpt, args.repo, args.l1, revision=rev)
        step = 10 ** 18 if rev == "main" else int(rev.split("-")[1])
        rows.append({
            "model_id": args.repo, "curriculum": args.curriculum,
            "revision": rev, "step": step, "l1_iso": args.l1,
            "mean_bpb": mean_bpb, "n_pairs": len(nlls),
        })

    write_csv_atomic(pd.DataFrame(rows), args.out)
    print(f"[l1_attrition] {len(rows)} ckpts → {args.out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
