"""beetle-analyze — a minimal, reproducible toolkit for analysing language-model
evaluation output.

The headline analysis scores a BLiMP-style set of grammatical minimal pairs from
precomputed sentence log-probabilities and reports per-phenomenon and overall
accuracy. It runs on CPU in well under a second with no model downloads.

Quick start::

    from beetle_analyze import analyze_file, toy_data_path
    result = analyze_file(toy_data_path())
    print(result.overall_accuracy)
"""
from __future__ import annotations

from .blimp import BlimpResult, analyze_file, evaluate, load_pairs, toy_data_path

__version__ = "0.1.0"

__all__ = [
    "BlimpResult",
    "analyze_file",
    "evaluate",
    "load_pairs",
    "toy_data_path",
    "__version__",
]
