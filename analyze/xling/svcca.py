"""
analyze/xling/svcca.py
──────────────────────
SVCCA (Raghu et al., NeurIPS 2017) and PWCCA (Morcos et al., NeurIPS
2018) layer-wise representation similarity — the workhorse for the
emergence and forgetting analyses (Sets A1, A3, C1).

Why SVCCA + PWCCA over plain CCA
--------------------------------
SVCCA first compresses each (N × D) activation matrix via SVD, retaining
the top-k singular directions that capture 99% of variance.  The
canonical correlations between the two compressed bases then form a
geometry-aware similarity measure that is invariant to rotation and
scale, but not to rank-deficiency artefacts that plain CCA suffers from.

PWCCA further weights canonical correlations by how much input variance
each direction explains, avoiding the spurious near-1.0 alignments that
SVCCA can return when ``k`` is small relative to the population
diversity.  PWCCA is recommended when one of the two compared
representations comes from a low-resource language (small N) where
activation variance is concentrated in a few directions.

This module exposes:

  * :func:`svcca_score` — global similarity in [0, 1]
  * :func:`pwcca_score` — projection-weighted variant
  * :func:`layerwise_trajectory` — convenience wrapper that compares
    two language representations across all model layers and returns
    a 1-D similarity vector (one score per layer).

All functions accept either NumPy arrays or torch tensors and operate
on CPU by default — these analyses are O(N²) at worst, fast on CPU,
and usually run in offline post-processing.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core SVCCA / PWCCA
# ---------------------------------------------------------------------------

def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _center(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def _svd_compress(x: np.ndarray, *, var_keep: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compress columns of ``x`` (N × D) via SVD; keep enough directions
    to retain ``var_keep`` of cumulative variance.

    Returns ``(x_compressed, V_used)`` where ``x_compressed`` is the
    projected matrix and ``V_used`` are the right singular vectors used.
    """
    # SVD on the centered matrix.
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    if s.sum() == 0:
        return x.copy(), np.eye(x.shape[1])
    var = (s ** 2)
    cumvar = np.cumsum(var) / var.sum()
    # +1 because we want the first index whose cum-var ≥ threshold,
    # inclusive.
    k = int(np.searchsorted(cumvar, var_keep) + 1)
    k = max(1, min(k, len(s)))
    return x @ vt[:k].T, vt[:k]


def _canonical_correlations(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute canonical correlations between ``x`` (N × p) and ``y`` (N × q).

    Returns ``(rho, A, B)`` where ``rho`` are the canonical correlation
    coefficients (length ``min(p, q)``) and ``A``, ``B`` are the
    canonical-direction matrices (so that ``X @ A`` are the canonical
    variates on the X side).

    We use the standard SVD-of-whitened-cross-covariance route, which
    is numerically more stable than solving the generalized eigenvalue
    problem directly.
    """
    n = x.shape[0]
    if n != y.shape[0]:
        raise ValueError(f"sample-count mismatch: {n} vs {y.shape[0]}")

    xc = _center(x)
    yc = _center(y)

    # Whitening: Cxx^{-1/2}, Cyy^{-1/2}.  Use SVD-based pseudo-inverses
    # so we tolerate near-singular covariances.
    def _whitener(z):
        u, s, vt = np.linalg.svd(z, full_matrices=False)
        # Drop directions with negligible variance.
        thresh = 1e-10 * s.max() if s.size else 0.0
        keep = s > thresh
        s_inv = np.zeros_like(s)
        s_inv[keep] = 1.0 / np.sqrt(s[keep] * (n - 1))
        return vt.T * s_inv  # shape (D, k_keep)

    Wx = _whitener(xc)
    Wy = _whitener(yc)

    cross = (xc.T @ yc) / (n - 1)
    M = Wx.T @ cross @ Wy
    u, rho, vt = np.linalg.svd(M, full_matrices=False)
    rho = np.clip(rho, 0.0, 1.0)
    A = Wx @ u
    B = Wy @ vt.T
    return rho, A, B


def svcca_score(
    x,
    y,
    *,
    var_keep: float = 0.99,
) -> float:
    """SVCCA mean canonical correlation in [0, 1].

    Higher = more similar subspaces.
    """
    x = _center(_to_numpy(x))
    y = _center(_to_numpy(y))
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"[svcca] sample mismatch: x has {x.shape[0]}, y has {y.shape[0]}"
        )

    x_c, _ = _svd_compress(x, var_keep=var_keep)
    y_c, _ = _svd_compress(y, var_keep=var_keep)

    rho, _, _ = _canonical_correlations(x_c, y_c)
    return float(rho.mean())


def pwcca_score(
    x,
    y,
    *,
    var_keep: float = 0.99,
) -> float:
    """Projection-weighted CCA mean correlation (Morcos et al., 2018).

    Each canonical correlation is weighted by how much of the original
    ``x`` variance the corresponding canonical variate explains.  This
    is the recommended variant when the two activations have very
    different effective ranks (typical for low-resource vs
    high-resource language pairs).
    """
    x_np = _center(_to_numpy(x))
    y_np = _center(_to_numpy(y))
    if x_np.shape[0] != y_np.shape[0]:
        raise ValueError(
            f"[pwcca] sample mismatch: x has {x_np.shape[0]}, y has {y_np.shape[0]}"
        )

    x_c, _ = _svd_compress(x_np, var_keep=var_keep)
    y_c, _ = _svd_compress(y_np, var_keep=var_keep)

    rho, A, _ = _canonical_correlations(x_c, y_c)
    # Canonical variates on the X side.
    H = x_c @ A
    # Project x_c onto each variate and measure the absolute correlation.
    weights = np.abs(H.T @ x_c).sum(axis=1)   # shape (k,)
    if weights.sum() == 0:
        return float(rho.mean())
    weights = weights / weights.sum()
    return float((weights * rho).sum())


# ---------------------------------------------------------------------------
# Layer-wise trajectory wrapper
# ---------------------------------------------------------------------------

def layerwise_trajectory(
    acts_a: Dict[int, "np.ndarray"],
    acts_b: Dict[int, "np.ndarray"],
    *,
    method: str = "svcca",
    var_keep: float = 0.99,
) -> Dict[int, float]:
    """Compute per-layer similarity between two activation banks.

    ``acts_a`` and ``acts_b`` are layer → matrix dicts as returned by
    :class:`xling.activation_buffer.ActivationBuffer`.  The two banks
    must have been computed on the *same* set of inputs (parallel pairs
    for cross-lingual analysis); we do not align inputs across banks.

    Returns ``{layer_idx: score}`` for layers present in both inputs.
    """
    score_fn = {"svcca": svcca_score, "pwcca": pwcca_score}[method]
    common = sorted(set(acts_a) & set(acts_b))
    return {
        l: score_fn(acts_a[l], acts_b[l], var_keep=var_keep)
        for l in common
    }
