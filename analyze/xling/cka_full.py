"""
analyze/xling/cka_full.py
─────────────────────────
Full-activation Centered Kernel Alignment (CKA).

This is the activation-level CKA used by Sets B1 (layer × pair heatmap)
and B2 (universal head identification).  It generalises the
embedding-only ``_linear_cka`` already in
``analyze/convergence.py:1015-1024`` to:

  * arbitrary (N, D) activation matrices (token-level or pooled),
  * RBF (Gaussian) kernels in addition to the linear kernel,
  * the Nguyen et al. (ICLR 2021) unbiased *minibatch* estimator that
    makes layer × head heatmaps tractable at corpus scale, and
  * per-attention-head decomposition by slicing the head dimension out
    of value-projection outputs (caller controls how the head split is
    exposed; see :func:`per_head_cka`).

We deliberately avoid pulling ``torch`` at module import time — CKA is
a NumPy-friendly operation and the heaviest analyses in this file are
already O(N²) on N≈few-thousand sentences.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core kernels
# ---------------------------------------------------------------------------

def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _center_kernel(K: np.ndarray) -> np.ndarray:
    """H K H, where H = I − 1/n is the centering matrix."""
    n = K.shape[0]
    unit = np.ones((n, n)) / n
    return K - unit @ K - K @ unit + unit @ K @ unit


def linear_kernel(x: np.ndarray) -> np.ndarray:
    return x @ x.T


def rbf_kernel(x: np.ndarray, *, sigma: Optional[float] = None) -> np.ndarray:
    """RBF (Gaussian) kernel.  ``sigma`` defaults to the median pairwise
    distance, the canonical heuristic from Kornblith et al. 2019.
    """
    sq = np.sum(x ** 2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * x @ x.T
    d2 = np.clip(d2, 0.0, None)
    if sigma is None:
        # Median pairwise distance over upper-triangle only.
        iu = np.triu_indices_from(d2, k=1)
        med = np.sqrt(np.median(d2[iu]) + 1e-12)
        sigma = max(med, 1e-6)
    return np.exp(-d2 / (2.0 * sigma ** 2))


# ---------------------------------------------------------------------------
# CKA (biased and unbiased)
# ---------------------------------------------------------------------------

def cka_score(
    x,
    y,
    *,
    kernel: str = "linear",
    rbf_sigma: Optional[float] = None,
) -> float:
    """Standard (biased) CKA.

    Parameters
    ----------
    kernel : {"linear", "rbf"}
        Kernel function applied to each activation matrix before
        centering.  Linear is by far the most common in interpretability
        work; RBF is useful when you suspect non-linear feature reuse.
    """
    x = _to_numpy(x)
    y = _to_numpy(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"[cka] sample mismatch: x has {x.shape[0]}, y has {y.shape[0]}"
        )

    if kernel == "linear":
        Kx = linear_kernel(x)
        Ky = linear_kernel(y)
    elif kernel == "rbf":
        Kx = rbf_kernel(x, sigma=rbf_sigma)
        Ky = rbf_kernel(y, sigma=rbf_sigma)
    else:
        raise ValueError(f"[cka] unknown kernel {kernel!r}")

    Kx_c = _center_kernel(Kx)
    Ky_c = _center_kernel(Ky)

    num = float((Kx_c * Ky_c).sum())
    den = float(np.linalg.norm(Kx_c, "fro") * np.linalg.norm(Ky_c, "fro"))
    if den == 0:
        return 0.0
    return num / den


# ---------------------------------------------------------------------------
# Minibatch unbiased estimator (Nguyen et al., 2021)
# ---------------------------------------------------------------------------

def _hsic_unbiased(K: np.ndarray, L: np.ndarray) -> float:
    """Unbiased HSIC estimator from Song et al. 2012 / Nguyen et al. 2021.

    Both ``K`` and ``L`` should be square Gram matrices with their
    diagonals zeroed.  The estimator is unbiased w.r.t. the population
    HSIC even for finite ``n``.
    """
    n = K.shape[0]
    if n < 4:
        # Estimator is undefined for n<4; fall back to biased value.
        return float((K * L).sum() / max(1, n) ** 2)
    K0 = K - np.diag(np.diag(K))
    L0 = L - np.diag(np.diag(L))
    sumK = K0.sum()
    sumL = L0.sum()
    a = float((K0 * L0).sum())
    b = sumK * sumL / ((n - 1) * (n - 2))
    c = 2 * float((K0.sum(axis=0) * L0.sum(axis=0)).sum()) / (n - 2)
    return (a + b - c) / (n * (n - 3))


def cka_minibatch_unbiased(
    batches_x: Iterable[np.ndarray],
    batches_y: Iterable[np.ndarray],
    *,
    kernel: str = "linear",
    rbf_sigma: Optional[float] = None,
) -> float:
    """Unbiased CKA estimated by averaging HSIC over equal-sized
    minibatches.

    Pairs of corresponding batches must be aligned (same input order).
    Use this for layer × head heatmaps where loading every layer's
    activations into a single ``(N, D)`` block would exhaust memory.
    """
    hsics_xy: List[float] = []
    hsics_xx: List[float] = []
    hsics_yy: List[float] = []
    for bx, by in zip(batches_x, batches_y):
        bx = _to_numpy(bx)
        by = _to_numpy(by)
        if kernel == "linear":
            Kx = linear_kernel(bx)
            Ky = linear_kernel(by)
        elif kernel == "rbf":
            Kx = rbf_kernel(bx, sigma=rbf_sigma)
            Ky = rbf_kernel(by, sigma=rbf_sigma)
        else:
            raise ValueError(f"[cka] unknown kernel {kernel!r}")
        hsics_xy.append(_hsic_unbiased(Kx, Ky))
        hsics_xx.append(_hsic_unbiased(Kx, Kx))
        hsics_yy.append(_hsic_unbiased(Ky, Ky))
    if not hsics_xy:
        raise ValueError("[cka] empty minibatch iterators")
    num = float(np.mean(hsics_xy))
    den = float(np.sqrt(max(np.mean(hsics_xx), 0.0) * max(np.mean(hsics_yy), 0.0)))
    if den == 0:
        return 0.0
    return num / den


# ---------------------------------------------------------------------------
# Layer × pair heatmap (Set B1)
# ---------------------------------------------------------------------------

def layerwise_heatmap(
    acts_a: Dict[int, "np.ndarray"],
    acts_b: Dict[int, "np.ndarray"],
    *,
    kernel: str = "linear",
) -> np.ndarray:
    """Build a square (L × L) CKA heatmap between every pair of layers
    in ``acts_a`` and ``acts_b``.

    The ``[i, j]`` entry is the CKA between layer-i of representation A
    and layer-j of representation B.  The diagonal is the
    "corresponding-layer" similarity that drives the Libovický et al.
    (2020) middle-layer language-neutrality finding.
    """
    layers_a = sorted(acts_a)
    layers_b = sorted(acts_b)
    H = np.zeros((len(layers_a), len(layers_b)))
    for i, la in enumerate(layers_a):
        for j, lb in enumerate(layers_b):
            H[i, j] = cka_score(acts_a[la], acts_b[lb], kernel=kernel)
    return H


# ---------------------------------------------------------------------------
# Per-head CKA (Set B2)
# ---------------------------------------------------------------------------

def per_head_cka(
    head_acts_a: Dict[tuple, "np.ndarray"],
    head_acts_b: Dict[tuple, "np.ndarray"],
    *,
    kernel: str = "linear",
) -> Dict[tuple, float]:
    """CKA over per-head activations.

    Both inputs are dicts keyed by ``(layer_idx, head_idx)`` and
    valued by ``(N, d_head)`` matrices, e.g. as produced by slicing
    each block's value-projection output along the head dimension.
    Returns the same key-set with scalar CKA values.
    """
    common = set(head_acts_a) & set(head_acts_b)
    out: Dict[tuple, float] = {}
    for key in sorted(common):
        out[key] = cka_score(head_acts_a[key], head_acts_b[key], kernel=kernel)
    return out


# ---------------------------------------------------------------------------
# Re-export under the legacy name used by ``analyze/convergence.py``
# ---------------------------------------------------------------------------

# ``convergence.py:1015`` had a private helper ``_linear_cka(X, Y)``;
# the new public name is ``cka_score(X, Y, kernel="linear")``.  Keep an
# alias so the convergence module can import either.
def linear_cka(x, y) -> float:
    return cka_score(x, y, kernel="linear")
