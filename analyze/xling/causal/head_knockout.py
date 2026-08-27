"""
analyze/xling/causal/head_knockout.py
─────────────────────────────────────
Set D1 — head knockout for causal validation of cross-lingual transfer.

Pipeline
--------
1. Rank attention heads by per-head cross-lingual CKA from
   :func:`xling.cka_full.per_head_cka`.
2. For each candidate head, install a forward-pre-hook that zeroes the
   head's contribution (re-using the mechanism in
   ``analyze/head_ablation.py``) and re-evaluate a downstream
   cross-lingual task — by default FLORES-200 perplexity, optionally
   XNLI accuracy if the user supplies an XNLI repo.
3. Compare the ablated metric to the un-ablated baseline; report the
   per-head Δ.

A "universal" cross-lingual head is one whose ablation decreases
transfer accuracy / increases parallel-FLORES PPL by significantly
more than the median random head.

This module is an orchestrator, not a re-implementation: the heavy
lifting is in :mod:`analyze.head_ablation`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

@dataclass
class KnockoutResult:
    layer: int
    head: int
    cka_score: float            # per-head cross-lingual CKA used to rank
    baseline_metric: float      # un-ablated value
    ablated_metric: float       # value with the head zeroed
    delta: float                # ablated − baseline


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def rank_universal_heads(
    head_cka: Dict[Tuple[int, int], float],
    *,
    top_k: int = 16,
) -> List[Tuple[int, int]]:
    """Return the ``top_k`` (layer, head) pairs by descending CKA.

    Ranking the candidates *before* running ablations avoids the
    expensive ablation eval for every head.  Voita et al. (2019) found
    <15% of heads carry most of the cross-lingual signal, so ``top_k``
    near ``0.15 × n_layers × n_heads`` is the natural default.
    """
    return sorted(head_cka, key=lambda k: -head_cka[k])[:top_k]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(
    *,
    repo: str,
    revision: str,
    head_cka: Dict[Tuple[int, int], float],
    metric_fn,
    top_k: int = 16,
    include_random_baseline: bool = True,
    seed: int = 0,
) -> List[KnockoutResult]:
    """Run the full D1 sweep and return per-head deltas.

    ``metric_fn`` is a callable ``(model, tokenizer) → float`` that
    returns the metric of interest (lower-is-worse for ΔPPL, higher is
    better for accuracy).  We do NOT prescribe which metric to use:
    the proposal mentions XNLI accuracy and FLORES PPL; both fit this
    interface.

    ``include_random_baseline`` adds a single zero-out of an unrelated
    randomly-chosen head as a control (its delta should be near zero
    if the ranking is meaningful).
    """
    import torch

    from ppl_utils import load_model_and_tokenizer  # type: ignore

    # Lazy import: head_ablation.py installs the actual hooks.
    from head_ablation import zero_head  # type: ignore

    model, tokenizer = load_model_and_tokenizer(repo, revision)
    model.eval()

    baseline = float(metric_fn(model, tokenizer))
    logger.info("[D1] %s @ %s baseline=%g", repo, revision, baseline)

    candidates = rank_universal_heads(head_cka, top_k=top_k)

    results: List[KnockoutResult] = []
    for layer, head in candidates:
        with zero_head(model, layer=layer, head=head):
            ablated = float(metric_fn(model, tokenizer))
        results.append(KnockoutResult(
            layer=layer, head=head,
            cka_score=head_cka[(layer, head)],
            baseline_metric=baseline,
            ablated_metric=ablated,
            delta=ablated - baseline,
        ))

    if include_random_baseline:
        import random
        rng = random.Random(seed)
        all_heads = list(head_cka)
        candidate_set = set(candidates)
        non_top = [k for k in all_heads if k not in candidate_set]
        if non_top:
            l, h = rng.choice(non_top)
            with zero_head(model, layer=l, head=h):
                ablated = float(metric_fn(model, tokenizer))
            results.append(KnockoutResult(
                layer=l, head=h,
                cka_score=head_cka[(l, h)],
                baseline_metric=baseline,
                ablated_metric=ablated,
                delta=ablated - baseline,
            ))

    return results
