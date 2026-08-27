"""
analyze/learner_corpus.py — §K learner-corpus validation.

Score real interlanguage from EFCAMDAT / ICLE under each curriculum's LM
and report mean surprisal per (curriculum, L1 section, error type). The
"L2-student" framing is demonstrated rather than asserted only if the
matched-L1 curriculum (b3 nld-eng) assigns systematically lower surprisal
to canonical Dutch-L1-learner errors than b1 nld-eng or mono-eng.

Outputs
-------
results/evals_paper/15_learner_corpus/learner_error_surprisals.csv
  Columns: model_id, curriculum, l1_section, error_type, sentence_id,
           mean_nll, n_tokens
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from analyze._interp_helpers import RESULTS_PAPER, device_for, write_csv_atomic

logger = logging.getLogger(__name__)


# Canonical interlanguage examples — used as a fallback when no EFCAMDAT
# CSV is supplied. These are NOT a substitute for the real corpus; they
# exist so the smoke path runs end-to-end.
SMOKE_LEARNER_ITEMS = pd.DataFrame([
    {"sentence": "Yesterday went I to the supermarket.",
     "l1_section": "nld", "error_type": "v2_leakage"},
    {"sentence": "Tomorrow will I see my friends.",
     "l1_section": "nld", "error_type": "v2_leakage"},
    {"sentence": "She gave me a sleep pill before the flight.",
     "l1_section": "nld", "error_type": "false_friend"},
    {"sentence": "I want save you on the test.",
     "l1_section": "nld", "error_type": "false_friend"},
    {"sentence": "He live in big house with two dogs.",
     "l1_section": "deu", "error_type": "article_drop"},
    {"sentence": "The teacher gave us homework that we should make tonight.",
     "l1_section": "nld", "error_type": "make_do_confusion"},
])


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--curriculum", required=True)
    p.add_argument("--learner-csv", default=None,
                    help="EFCAMDAT or ICLE slice with columns ``sentence, "
                         "l1_section, error_type``. Falls back to a tiny "
                         "smoke set if absent.")
    p.add_argument("--out", default=str(RESULTS_PAPER / "15_learner_corpus"
                                         / "learner_error_surprisals.csv"))
    args = p.parse_args(list(argv) if argv is not None else None)

    from analyze.ppl_utils import load_model_and_tokenizer, sentence_log_likelihood

    items = pd.read_csv(args.learner_csv) if args.learner_csv else SMOKE_LEARNER_ITEMS

    pair = load_model_and_tokenizer(args.repo)
    if pair is None:
        raise SystemExit(f"could not load {args.repo}")
    model, tokenizer = pair
    model.to(device_for()).eval()

    rows = []
    for sid, r in items.reset_index(drop=True).iterrows():
        nll = sentence_log_likelihood(str(r["sentence"]), model, tokenizer)
        n_tok = len(tokenizer.encode(str(r["sentence"]),
                                       add_special_tokens=False))
        rows.append({
            "model_id": args.repo, "curriculum": args.curriculum,
            "l1_section": r.get("l1_section", "unknown"),
            "error_type": r.get("error_type", "unknown"),
            "sentence_id": int(sid),
            "mean_nll": float(nll), "n_tokens": int(n_tok),
        })

    df = pd.DataFrame(rows)
    out = Path(args.out)
    if out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
        df = df.drop_duplicates(["model_id", "sentence_id"], keep="last")
    write_csv_atomic(df, out)
    print(f"[learner_corpus] wrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
