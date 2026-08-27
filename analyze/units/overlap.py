"""
analyze/units/overlap.py — Cross-phenomenon and cross-validation overlap.

Wraps ``cross_overlap.py`` from the source repos. The paper's central claim —
"different types of syntactic agreement recruit the *same* units" — is exactly
a statement about the overlap between the localized ``UnitMask``s of different
phenomena. This module quantifies that overlap and its statistical baseline.

  * ``jaccard`` / ``dice`` / ``overlap_coefficient`` — pairwise set similarity.
  * ``overlap_matrix`` — an N×N similarity table over named masks.
  * ``expected_overlap`` — the chance Jaccard for two random masks of the same
    sizes, so an observed value can be read as "k× above chance".
  * ``cross_validation_stability`` — split a phenomenon's stimuli into folds,
    localize each, and report mean pairwise overlap: how *reliable* the mask is.

These functions take anything with an ``.indices()`` method returning a set of
``(layer, unit)`` tuples (i.e. a ``UnitMask``), so there is no import cycle with
``localize``.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np


def _idx(mask) -> set:
    return mask.indices() if hasattr(mask, "indices") else set(mask)


def intersection(a, b) -> int:
    """|A ∩ B| — the raw shared-unit count.

    This is the source paper's headline metric (``cross_overlap`` returns
    ``np.sum(mask_1 & mask_2)``): how many units two phenomena's localizers
    share. ``jaccard``/``dice`` below normalize it for cross-model comparison.
    """
    return len(_idx(a) & _idx(b))


def jaccard(a, b) -> float:
    """|A ∩ B| / |A ∪ B|; 0 when both are empty."""
    A, B = _idx(a), _idx(b)
    union = len(A | B)
    return (len(A & B) / union) if union else 0.0


def dice(a, b) -> float:
    """2|A ∩ B| / (|A| + |B|)."""
    A, B = _idx(a), _idx(b)
    denom = len(A) + len(B)
    return (2 * len(A & B) / denom) if denom else 0.0


def overlap_coefficient(a, b) -> float:
    """|A ∩ B| / min(|A|, |B|) — the Szymkiewicz–Simpson coefficient."""
    A, B = _idx(a), _idx(b)
    denom = min(len(A), len(B))
    return (len(A & B) / denom) if denom else 0.0


def expected_overlap(a, b, total_units: int, *, metric: str = "jaccard") -> float:
    """Chance value of ``metric`` for two random masks of the same cardinality.

    Uses the hypergeometric expectation ``E[|A∩B|] = |A||B| / N``. Handy as a
    denominator: ``observed / expected_overlap(...)`` is the fold-enrichment.
    """
    A, B = _idx(a), _idx(b)
    na, nb = len(A), len(B)
    if total_units <= 0 or na == 0 or nb == 0:
        return 0.0
    exp_inter = na * nb / total_units
    if metric == "jaccard":
        exp_union = na + nb - exp_inter
        return exp_inter / exp_union if exp_union else 0.0
    if metric == "dice":
        return 2 * exp_inter / (na + nb)
    if metric == "overlap":
        return exp_inter / min(na, nb)
    raise ValueError(f"unknown metric {metric!r}")


def random_overlap_expected(n_total: int, k: int, n_folds: int = 2) -> float:
    """Expected size of the intersection of ``n_folds`` independent random
    size-``k`` masks drawn from ``n_total`` units: ``n_total * (k/n_total)**n_folds``.

    Verbatim port of the reference ``random_overlap_expected`` — the chance
    baseline the paper compares cross-validation / cross-overlap counts against.
    """
    if n_total <= 0:
        return 0.0
    return n_total * (k / n_total) ** n_folds


def enrichment(a, b, total_units: int, *, metric: str = "jaccard") -> float:
    """Observed overlap divided by its chance expectation (>1 = above chance)."""
    obs = {"jaccard": jaccard, "dice": dice, "overlap": overlap_coefficient}[metric](a, b)
    exp = expected_overlap(a, b, total_units, metric=metric)
    return obs / exp if exp > 0 else float("nan")


_METRICS = {
    "jaccard": jaccard,
    "dice": dice,
    "overlap": overlap_coefficient,
    "intersection": intersection,     # raw shared-unit count (paper default)
}


def overlap_matrix(masks: Dict[str, object], *, metric: str = "jaccard"):
    """Symmetric overlap ``DataFrame`` over named masks.

    The diagonal is each mask's own size (``|A|``) for ``metric="intersection"``
    and ``1`` for the normalized metrics.
    """
    import pandas as pd

    fn = _METRICS[metric]
    names = list(masks)
    n = len(names)
    M = np.zeros((n, n))
    for i in range(n):
        M[i, i] = fn(masks[names[i]], masks[names[i]])
    for i, j in combinations(range(n), 2):
        v = fn(masks[names[i]], masks[names[j]])
        M[i, j] = M[j, i] = v
    return pd.DataFrame(M, index=names, columns=names)


def mean_pairwise(masks: Sequence[object], *, metric: str = "jaccard") -> float:
    """Average overlap over all unordered pairs (single scalar summary)."""
    fn = _METRICS[metric]
    vals = [fn(a, b) for a, b in combinations(masks, 2)]
    return float(np.mean(vals)) if vals else float("nan")


def intersection_all(masks: Sequence[object]) -> int:
    """|A₁ ∩ A₂ ∩ … ∩ Aₖ| — units selected in *every* mask."""
    if not masks:
        return 0
    common = _idx(masks[0])
    for m in masks[1:]:
        common &= _idx(m)
    return len(common)


def cross_validation_stability(
    model,
    tokenizer,
    pair_set,
    *,
    n_folds: int = 2,
    top_frac: float = 0.01,
    method: str = "ttest",
    site: str = "resid",
    metric: str = "intersection",
    seed: int = 0,
    **localize_kwargs,
) -> Dict[str, float]:
    """Localize on each of ``n_folds`` disjoint stimulus folds and report how
    consistently the same units are selected — the paper's cross-validation.

    The reference method intersects the fold masks and reports the count as a
    fraction of the per-fold mask size (``metric="intersection"``, default).
    ``metric`` may instead be a pairwise similarity (``jaccard``/``dice``/
    ``overlap``), in which case the mean over fold pairs is returned in ``mean``.

    Returns ``{"mean", "std", "n_folds", "mean_units", "intersection",
    "intersection_frac", "expected_frac"}``.
    """
    from .localize import localize  # local import avoids a cycle

    folds = pair_set.folds(n_folds, seed=seed)
    masks = [
        localize(model, tokenizer, f, top_frac=top_frac, method=method,
                 site=site, **localize_kwargs)
        for f in folds if len(f) > 1
    ]
    if not masks:
        return {"mean": float("nan"), "std": float("nan"), "n_folds": 0,
                "mean_units": 0.0, "intersection": 0,
                "intersection_frac": float("nan"), "expected_frac": float("nan")}

    inter = intersection_all(masks)
    mean_units = float(np.mean([m.count() for m in masks]))
    total = int(np.mean([m.total_units() for m in masks]))
    k = int(round(mean_units))
    inter_frac = inter / k if k else float("nan")
    exp = random_overlap_expected(total, k, len(masks))
    exp_frac = exp / k if k else float("nan")

    if metric == "intersection":
        mean = inter_frac
        std = float("nan")
    else:
        fn = _METRICS[metric]
        vals = [fn(a, b) for a, b in combinations(masks, 2)]
        mean = float(np.mean(vals)) if vals else float("nan")
        std = float(np.std(vals)) if vals else float("nan")

    return {
        "mean": mean,
        "std": std,
        "n_folds": len(masks),
        "mean_units": mean_units,
        "intersection": inter,
        "intersection_frac": inter_frac,
        "expected_frac": exp_frac,   # chance baseline for intersection_frac
    }
