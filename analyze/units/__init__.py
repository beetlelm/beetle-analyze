"""
analyze.units — Developmental interpretability of syntactic *units*.

A general dev-interp library that wraps, generalizes, and makes checkpoint-aware
the functional-localization methodology of:

  * Kryvosheieva, Preece et al., "Different types of syntactic agreement recruit
    the same units within large language models" (2026, arXiv:2512.03676);
  * the replication + extension release
    ``anonymous.4open.science/r/syntax-interpretability-F4C3``; and
  * the localization/ablation code at ``github.com/elliepreed/syntax-units``.

Those repos answer a static question — *which* units in a fixed model carry
syntactic agreement, and do different agreement types share them. This package
keeps that pipeline intact but adds the axis ``beetle-analyze`` cares about:
**training time**. You can localize the units in any single checkpoint, or trace
their emergence, consolidation and overlap across a Beetle model's full
checkpoint trajectory.

Pipeline
--------
    stimuli   →  localize  →  { ablate , overlap }  →  trajectory
    (good/bad)   (UnitMask)    (causal)  (same-units)   (developmental)

Quick start
-----------
    from analyze.units import stimuli, localize, ablate, overlap

    sets  = stimuli.load_blimp(categories=stimuli.AGREEMENT_CATEGORIES)
    model, tok = ...                       # any HF causal LM
    masks = {s.phenomenon: localize.localize(model, tok, s, top_frac=0.01)
             for s in sets}

    overlap.overlap_matrix(masks)          # do agreement types share units?
    ablate.ablation_effect(model, tok, sets[0], masks[sets[1].phenomenon])

    # …and the developmental version, across checkpoints:
    from analyze.units.trajectory import localize_trajectory
    df = localize_trajectory(repo, sets, all_checkpoints=True, log_spaced=True)

Sub-modules
-----------
* ``stimuli``     — ``MinimalPairSet`` + BLiMP / MultiBLiMP / JSONL loaders.
* ``localize``    — activation collection, selectivity, ``UnitMask``, ``localize``.
* ``ablate``      — zero/mean ablation hooks + minimal-pair accuracy.
* ``overlap``     — Jaccard/Dice overlap, enrichment, cross-validation stability.
* ``trajectory``  — across-checkpoint developmental driver.

Everything is model-agnostic (works on the custom Beetle/Pico decoders) and
dependency-light (numpy, pandas, torch, transformers, scipy).
"""
from __future__ import annotations

from . import stimuli, localize, overlap, ablate, trajectory
from .stimuli import MinimalPairSet, AGREEMENT_CATEGORIES
from .localize import UnitMask, localize as localize_units, selectivity, fdr_control
from .overlap import (
    jaccard, dice, overlap_coefficient, intersection, overlap_matrix,
    enrichment, random_overlap_expected, cross_validation_stability,
)
from .ablate import (
    minimal_pair_accuracy, ablation_effect, ablation_grid, run_ablation,
    ablated, random_mask_from_remaining,
)
from .trajectory import localize_trajectory

__all__ = [
    "stimuli", "localize", "overlap", "ablate", "trajectory",
    "MinimalPairSet", "AGREEMENT_CATEGORIES",
    "UnitMask", "localize_units", "selectivity", "fdr_control",
    "jaccard", "dice", "overlap_coefficient", "intersection", "overlap_matrix",
    "enrichment", "random_overlap_expected", "cross_validation_stability",
    "minimal_pair_accuracy", "ablation_effect", "ablation_grid", "run_ablation",
    "ablated", "random_mask_from_remaining",
    "localize_trajectory",
]
