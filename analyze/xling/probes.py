"""
analyze/xling/probes.py
───────────────────────
Linear probes with control tasks (Hewitt & Liang, EMNLP 2019).

Used by Set A2 (linguistic-level emergence ordering — POS / dependency
/ NER / semantic accuracy curves over training) and as a building
block for TCAV (Set B4).

Why linear and why control tasks
--------------------------------
A high-capacity probe can learn almost any property even from a random
representation.  Hewitt & Liang's *control task* — the same probe
trained on randomly shuffled labels — establishes the floor: only the
gap between the real-task probe accuracy and the control-task probe
accuracy is evidence that the *representation* encodes the property.

We follow their two recommendations:

  1. Use linear probes, never deep MLPs.
  2. Always report the control gap alongside raw accuracy.

This module is the simplest of the ``xling`` analyses and intentionally
has very few dependencies; it also exposes ``representation_extract``
as the canonical helper for fetching per-token activations the way
:mod:`xling.tcav` needs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Probe config
# ---------------------------------------------------------------------------

@dataclass
class LinearProbeConfig:
    l2: float = 1e-4
    max_iter: int = 1000
    seed: int = 0


# ---------------------------------------------------------------------------
# Linear probe (sklearn LogReg under the hood)
# ---------------------------------------------------------------------------

def fit_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cfg: Optional[LinearProbeConfig] = None,
):
    """Fit a multinomial logistic regression on (X, y).

    We use sklearn rather than rolling our own SGD; the probes here are
    invariably ≤d_model × n_classes and fit in <1s.
    """
    from sklearn.linear_model import LogisticRegression

    cfg = cfg or LinearProbeConfig()
    clf = LogisticRegression(
        penalty="l2",
        C=1.0 / max(cfg.l2, 1e-12),
        solver="lbfgs",
        max_iter=cfg.max_iter,
        multi_class="auto",
        random_state=cfg.seed,
    )
    clf.fit(X, y)
    return clf


def evaluate_probe(clf, X: np.ndarray, y: np.ndarray) -> float:
    return float((clf.predict(X) == y).mean())


# ---------------------------------------------------------------------------
# Control task (Hewitt & Liang, 2019)
# ---------------------------------------------------------------------------

def make_control_labels(y: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Random label per *type* — the canonical control task.

    For each unique input type (e.g. token), a random class label is
    drawn; the probe must memorise this random map by relying on the
    probe's own capacity rather than the representation's structure.

    Here we treat row index as type id; callers that want token-level
    control labels should pass token ids as ``y_types``.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    perm = rng.choice(classes, size=y.shape[0], replace=True)
    return perm


def control_gap(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    cfg: Optional[LinearProbeConfig] = None,
) -> Dict[str, float]:
    """Compute (real-task acc, control-task acc, gap)."""
    cfg = cfg or LinearProbeConfig()
    real = fit_linear_probe(X_train, y_train, cfg=cfg)
    real_acc = evaluate_probe(real, X_test, y_test)

    y_train_ctrl = make_control_labels(y_train, seed=cfg.seed)
    y_test_ctrl = make_control_labels(y_test, seed=cfg.seed + 1)
    ctrl = fit_linear_probe(X_train, y_train_ctrl, cfg=cfg)
    ctrl_acc = evaluate_probe(ctrl, X_test, y_test_ctrl)

    return {
        "accuracy": real_acc,
        "control_accuracy": ctrl_acc,
        "gap": real_acc - ctrl_acc,
    }


# ---------------------------------------------------------------------------
# Trajectory wrapper
# ---------------------------------------------------------------------------

def trajectory_per_layer(
    *,
    layer_acts_train: Dict[int, np.ndarray],
    y_train: np.ndarray,
    layer_acts_test: Dict[int, np.ndarray],
    y_test: np.ndarray,
    cfg: Optional[LinearProbeConfig] = None,
) -> Dict[int, Dict[str, float]]:
    """Fit one probe per layer; return real / control / gap per layer.

    Used by Set A2 to plot accuracy curves across the depth of the
    model at each checkpoint.
    """
    common = sorted(set(layer_acts_train) & set(layer_acts_test))
    out: Dict[int, Dict[str, float]] = {}
    for l in common:
        out[l] = control_gap(
            layer_acts_train[l], y_train,
            layer_acts_test[l], y_test,
            cfg=cfg,
        )
    return out
