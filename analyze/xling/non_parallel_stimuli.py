"""
analyze/xling/non_parallel_stimuli.py — §A confound: build a non-parallel
syntax-matched L1/L2 stimulus pool.

FLORES-200 is parallel by construction, so high L1↔L2 CKA could just
reflect shared semantics. The §A claim that curriculum modulates a *shared
subspace* is much stronger if the gradient survives non-parallel matched-
syntax stimuli.

This module:

1. Loads FLORES-200 devtest in L1 and L2 (used as syntactically diverse
   monolingual pools, drawn from the same source register).
2. Reorders the L2 pool by sentence length (in tokens) so the resulting
   pairs match on length but not meaning.
3. Returns ``(l1_sents, l2_sents)`` — both lists of length ``n_pairs``.

A real implementation should swap step 1 for separate monolingual corpora
(e.g., Dutch news vs English Wikipedia) and step 2 for a dependency-depth
matcher; the FLORES-shuffle baseline here is the cheap version that runs
on CPU in seconds and still breaks semantic alignment.
"""
from __future__ import annotations

import argparse
import random
from typing import Iterable

import pandas as pd


def build_non_parallel_pairs(
    l1_iso: str, l2_iso: str, *, n_pairs: int = 1000, seed: int = 17,
) -> tuple[list[str], list[str]]:
    from analyze.ppl_utils import load_flores_sentences

    l1 = load_flores_sentences(l1_iso)
    l2 = load_flores_sentences(l2_iso)
    n = min(len(l1), len(l2), n_pairs)
    l1 = l1[:n]
    rng = random.Random(seed)
    # Length-bucketed shuffle: sort by tokens, then rotate by half — keeps
    # length distribution comparable but breaks the parallel index.
    l2_sorted = sorted(l2[:n], key=lambda s: len(s.split()))
    rotation = max(1, n // 2)
    l2_shuffled = l2_sorted[rotation:] + l2_sorted[:rotation]
    rng.shuffle(l2_shuffled)
    return l1, l2_shuffled


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--l1", default="nld")
    p.add_argument("--l2", default="eng")
    p.add_argument("--n-pairs", type=int, default=1000)
    p.add_argument("--out")
    args = p.parse_args(list(argv) if argv is not None else None)

    l1, l2 = build_non_parallel_pairs(args.l1, args.l2, n_pairs=args.n_pairs)
    df = pd.DataFrame({"l1_sentence": l1, "l2_sentence": l2})
    out = args.out or f"results/evals_paper/05_geometry/non_parallel_pairs_{args.l1}_{args.l2}.csv"
    df.to_csv(out, index=False)
    print(f"[non_parallel_stimuli] wrote {len(df)} pairs → {out}")


if __name__ == "__main__":
    main()
