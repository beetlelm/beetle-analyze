"""
analyze/xling/cognate_filter.py — §A confound: split FLORES pairs into
cognate vs non-cognate strata.

A pair is flagged "cognate" if any content-word pair across the two
sentences shares a normalised string with edit-distance ≤ ``edit_thresh``
relative to length. The default heuristic is intentionally conservative —
the goal is to remove pairs whose CKA could plausibly be inflated by
shared surface form, not to enforce a linguistic definition of cognate.

Public API
----------
- ``partition_cognates(l1_sents, l2_sents, *, l1_iso, l2_iso)`` →
  (cognate_indices, non_cognate_indices).
- ``cognate_pair_share(s1, s2, ...)`` → fraction of content tokens that
  match across the pair.

CLI: ``python -m analyze.xling.cognate_filter --l1 nld --l2 eng`` writes a
``results/evals_paper/05_geometry/cognate_strata.csv`` with one row per
sentence pair.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from analyze._interp_helpers import RESULTS_PAPER, write_csv_atomic

_WORD_RE = re.compile(r"[\w']+", flags=re.UNICODE)
_STOP_LIKE = {  # cross-lingual stoplist — extend as needed
    "the", "a", "an", "of", "in", "to", "and", "or", "for",
    "de", "het", "een", "van", "is", "en", "te",
    "der", "die", "das", "und", "ist", "ein",
}


def _content_tokens(s: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(s) if w.lower() not in _STOP_LIKE
            and len(w) >= 3]


def _ratio_dist(a: str, b: str) -> float:
    """Length-normalised edit distance ∈ [0, 1]; 0 = identical."""
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    # Quick prefilter — if length differs a lot they're not cognate.
    if abs(la - lb) > max(la, lb) // 2:
        return 1.0
    # Levenshtein DP — small strings, no need for an external lib.
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb] / max(la, lb)


def cognate_pair_share(s1: str, s2: str, *, edit_thresh: float = 0.34) -> float:
    """Fraction of content-token *types* with at least one match across
    the two sentences whose normalised edit distance is ≤ ``edit_thresh``.
    """
    t1 = set(_content_tokens(s1))
    t2 = set(_content_tokens(s2))
    if not t1:
        return 0.0
    matches = 0
    for w1 in t1:
        for w2 in t2:
            if _ratio_dist(w1, w2) <= edit_thresh:
                matches += 1
                break
    return matches / len(t1)


def partition_cognates(l1_sents: list[str], l2_sents: list[str],
                        *, l1_iso: str = "nld", l2_iso: str = "eng",
                        share_thresh: float = 0.20) -> tuple[list[int], list[int]]:
    """Return ``(cognate_idx, non_cognate_idx)`` over the parallel index
    set. A pair is "cognate" if ≥ ``share_thresh`` of its content-token
    types match across languages under the edit-distance heuristic.

    The thresholds are calibrated for Germanic ↔ English; for typologically
    distant pairs (zho-eng) the heuristic returns near-empty cognate
    strata, which is the right behaviour — the §E zho-eng cell is a
    built-in negative control.
    """
    cog: list[int] = []
    non: list[int] = []
    for i, (s1, s2) in enumerate(zip(l1_sents, l2_sents)):
        share = cognate_pair_share(s1, s2)
        (cog if share >= share_thresh else non).append(i)
    return cog, non


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--l1", default="nld")
    p.add_argument("--l2", default="eng")
    p.add_argument("--share-thresh", type=float, default=0.20)
    p.add_argument("--out", default=str(RESULTS_PAPER / "05_geometry"
                                         / "cognate_strata.csv"))
    args = p.parse_args(list(argv) if argv is not None else None)

    from analyze.ppl_utils import load_flores_sentences

    l1 = load_flores_sentences(args.l1)
    l2 = load_flores_sentences(args.l2)
    n = min(len(l1), len(l2))
    rows = []
    for i in range(n):
        share = cognate_pair_share(l1[i], l2[i])
        rows.append({
            "sentence_id": i, "l1_iso": args.l1, "l2_iso": args.l2,
            "cognate_share": share,
            "stratum": "cognate" if share >= args.share_thresh else "non_cognate",
        })
    df = pd.DataFrame(rows)
    write_csv_atomic(df, args.out)
    print(f"[cognate_filter] {df['stratum'].value_counts().to_dict()} → {args.out}")


if __name__ == "__main__":
    main()
