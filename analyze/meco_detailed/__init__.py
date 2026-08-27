"""
analyze.meco_detailed — Enriched MECO L2 analyses beyond the headline ΔlogLik.

The EMNLP main-body MECO figures (F4, F5, P1, P2, P7) all reduce the model
to one number per (model, L1): ΔlogLik on first-pass duration. This package
adds eight complementary analyses that pass any beetle model through the
canonical mixed-effects machinery and read out *different* cognitive
signatures.

  D1  measure_robustness   ΔlogLik across 5 eye-tracking measures
  D2  spillover_decay      β(Surprisal_t-k) profile for k ∈ {0,1,2}
  D3  bilingual_decomp     nested L1-mono / L2-mono / bilingual decomposition
  D4  reader_caterpillar   subject-level BLUPs + ICC reduction
  D5  item_granularity     freq × cognate × position interactions
  D6  proficiency_mod      LEAP-Q proficiency moderation
  D7  l1_control           L1-native reader counterfactual
  D8  comprehension_link   downstream comprehension prediction

Each module:
  • takes ``(surprisal_df, langs)`` so it runs on any beetle model
  • writes a single CSV under ``results/meco_detailed/dN_*/``
  • writes a single figure PDF under ``plots/meco_detailed/dN_*.pdf``
  • exposes a CLI: ``python -m analyze.meco_detailed.dN_<slug> [--overwrite]``

Shared loaders for the OSF MECO L2 release (``meco-l2-complete/``) live in
``_common.py``.
"""

from __future__ import annotations
