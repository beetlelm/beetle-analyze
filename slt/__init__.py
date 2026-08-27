"""SLT geometry analysis for Beetle / Pico language models.

Modules in this package implement the experimental pipeline for
"Singular Geometry is Shared Structure" (RLCT, Fisher spectroscopy, and
component-localised susceptibilities across mono / bi / multi regimes).

Public entry points are exposed via ``python -m analyze.run`` flags
``--llc``, ``--fisher``, ``--susceptibilities``, ``--bridge``,
``--falsification``, ``--per-lang-llc``, ``--beta-sweep``.
"""

from __future__ import annotations

__all__ = [
    "components",
    "probe_data",
    "run_llc",
    "run_fisher",
    "run_susceptibilities",
    "bridge",
    "falsification",
    "per_language_llc",
    "beta_sweep",
    "aggregate",
    "plot_slt",
]
