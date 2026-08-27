"""Generate the tiny synthetic BLiMP-style eval-output bundled with the package.

This produces ``src/beetle_analyze/data/blimp_toy.jsonl`` — a SYNTHETIC stand-in
for the kind of eval output ``beetle-analyze`` consumes: one minimal pair per
line, each carrying the total token log-probability that a language model
assigned to the grammatical ("good") and ungrammatical ("bad") sentence.

Nothing here is a real model output or a real research result. The numbers are
drawn from a fixed-seed RNG so the file is fully reproducible. Re-run with::

    python scripts/make_toy_data.py

Sentence text uses trivial synthetic templates purely so the file is readable;
the analysis only ever looks at the log-probabilities and token counts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "src" / "beetle_analyze" / "data" / "blimp_toy.jsonl"

# Each phenomenon gets a "difficulty": the mean advantage (in log-prob) that the
# grammatical sentence has over the ungrammatical one. Larger => easier => the
# model gets it right more often. These are made-up numbers.
PHENOMENA = {
    "subject_verb_agreement": 2.2,
    "determiner_noun_agreement": 1.6,
    "anaphor_agreement": 1.1,
    "npi_licensing": 0.4,
    "ellipsis": -0.2,  # a deliberately hard phenomenon the toy model fails on
}

PAIRS_PER_PHENOMENON = 12
GOOD_TEMPLATE = "The {n} {v} the {n2}."
BAD_TEMPLATE = "The {n} {v2} the {n2}."


def main() -> None:
    rng = np.random.default_rng(20240517)
    records = []
    for phenomenon, advantage in PHENOMENA.items():
        for i in range(PAIRS_PER_PHENOMENON):
            n_tok_good = int(rng.integers(6, 15))
            # bad sentence is usually the same length, occasionally off by one
            n_tok_bad = max(4, n_tok_good + int(rng.integers(-1, 2)))
            # per-token log-prob floor, then a per-pair grammaticality advantage
            base = float(rng.normal(-3.0, 0.3))
            lp_good = base * n_tok_good + float(rng.normal(advantage, 1.4))
            lp_bad = base * n_tok_bad
            records.append(
                {
                    "pair_id": f"{phenomenon}_{i:02d}",
                    "phenomenon": phenomenon,
                    "sentence_good": GOOD_TEMPLATE.format(n="cat", v="chased", n2="dog"),
                    "sentence_bad": BAD_TEMPLATE.format(n="cat", v2="chase", n2="dog"),
                    "logprob_good": round(lp_good, 4),
                    "logprob_bad": round(lp_bad, 4),
                    "n_tokens_good": n_tok_good,
                    "n_tokens_bad": n_tok_bad,
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} pairs -> {OUT}")


if __name__ == "__main__":
    main()
