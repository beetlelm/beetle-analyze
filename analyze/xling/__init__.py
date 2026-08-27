"""
analyze/xling/
──────────────
Mechanistic cross-lingual analysis stack for Beetle bilingual / multilingual
checkpoints.

This package implements the SVCCA + CKA + CrossEncoder + Probing trajectory
analysis described in the project plan
(``.claude/plans/binary-bouncing-quokka.md``) and corresponds to RQ1–RQ6 /
H1–H6 of the proposal:

* ``svcca``           — SVCCA + PWCCA (Sets A1, A3, C1)
* ``cka_full``        — full-activation CKA, layer × head (Sets B1, B2)
* ``crossencoder``    — CrossEncoder semantic alignment probe (Sets B3, C2)
* ``probes``          — linear probes with control tasks (Sets A2, B4)
* ``tcav``            — TCAV cross-lingual concept vectors (Set B4)
* ``activation_buffer`` — streaming activation extractor + on-disk cache
* ``datasets``        — FLORES-200 / UD treebank loaders, parallel pairs
* ``registry``        — model-agnostic language / repo registry
* ``causal/``         — Set D causal validation (head knockout, subspace patching)

The framework is model-agnostic: it works for the current Beetle bilingual
NLD-ENG checkpoints out of the box and supports any future N-way
multilingual configuration via ``registry.register``.
"""

__all__ = [
    "registry",
    "datasets",
    "activation_buffer",
    "svcca",
    "cka_full",
    "crossencoder",
    "probes",
    "tcav",
    "causal",
]
