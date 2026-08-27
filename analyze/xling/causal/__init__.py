"""
analyze/xling/causal/
─────────────────────
Causal validation experiments for the cross-lingual analysis stack
(Set D of the proposal).

Interpretability findings on top of correlational analyses (SVCCA, CKA,
probes, CrossEncoder) must be causally validated to rule out spurious
correlation — see Elhage et al. (2021).  The two default-on
interventions in this iteration are:

  * :mod:`head_knockout` — Set D1.  Zero out attention heads that
    rank highest under per-head cross-lingual CKA; measure the change
    in zero-shot transfer accuracy.  Reuses
    ``analyze/head_ablation.py`` for the actual hook plumbing.

  * :mod:`subspace_patch` — Set D2.  Replace target-language
    activations at a chosen layer with their projection onto the
    SVCCA-identified shared cross-lingual subspace; measure whether
    transfer accuracy improves without any fine-tuning.

D3 (early-checkpoint fine-tuning) and D4 (forgetting recovery) are
deferred behind the ``--causal-d3`` / ``--causal-d4`` CLI flags; they
require model fine-tuning runs and are not included in the default
analysis pipeline.
"""

__all__ = ["head_knockout", "subspace_patch"]
