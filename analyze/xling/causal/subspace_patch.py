"""
analyze/xling/causal/subspace_patch.py
──────────────────────────────────────
Set D2 — subspace steering / activation patching for causal validation.

Hypothesis under test
---------------------
SVCCA / CKA identify a *shared cross-lingual subspace* in mid-network
layers (Sets A1, B1).  If that subspace is causally responsible for
cross-lingual transfer, then *replacing* a target-language activation
with its projection onto this subspace should *not* hurt downstream
performance — and may improve it for low-resource pairs where
out-of-subspace components are noisy.

Implementation
--------------
Given a fitted ``(D × r)`` projection matrix ``P`` (computed from the
shared SVCCA basis), we install a forward hook on the chosen layer
that replaces ``h`` with ``P @ P.T @ h``.  We provide the projection
matrix in the constructor; how it is computed (from canonical
correlations, from CKA-derived subspaces, or from a custom recipe) is
the caller's responsibility.

A useful sanity check: when ``P = I``, the patch must be a no-op —
verified in :func:`assert_identity_is_noop`.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch installer
# ---------------------------------------------------------------------------

@contextmanager
def project_layer(
    model,
    *,
    layer_idx: int,
    projection: "np.ndarray",
) -> Iterator[None]:
    """Context manager that installs a forward-hook on the given layer
    that replaces hidden states ``h`` with ``P P^T h``.

    ``projection`` is shape ``(D, r)`` — the orthonormal basis of the
    shared subspace.  Hidden states from the layer's output (whether
    a tensor or a tuple whose first element is the tensor) are
    projected; everything else is passed through.

    Removes the hook on exit.
    """
    import torch

    from analyze.xling.activation_buffer import _find_transformer_blocks

    blocks = _find_transformer_blocks(model)
    if layer_idx < 0 or layer_idx >= len(blocks):
        raise IndexError(
            f"[D2] layer_idx {layer_idx} out of range; model has "
            f"{len(blocks)} blocks"
        )

    P = torch.as_tensor(projection, dtype=torch.float32)
    PPt = (P @ P.T).contiguous()

    def _hook(module, inputs, output):
        if isinstance(output, (tuple, list)):
            hs = output[0]
            tail = output[1:]
        else:
            hs, tail = output, None

        proj = PPt.to(device=hs.device, dtype=hs.dtype)
        new_hs = hs @ proj
        if tail is not None:
            return (new_hs,) + tuple(tail)
        return new_hs

    handle = blocks[layer_idx].register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def assert_identity_is_noop(
    model,
    *,
    layer_idx: int,
    sample_input,
    atol: float = 1e-5,
) -> None:
    """Run the model with and without an identity projection and assert
    the outputs match.  Raises ``AssertionError`` on mismatch.

    Used by tests to lock down the contract that ``project_layer(P=I)``
    is a no-op.
    """
    import torch

    model.eval()
    with torch.no_grad():
        baseline = model(sample_input)
        if hasattr(baseline, "logits"):
            baseline = baseline.logits

        # Determine D from the model's first parameter dim — caller may
        # pass any layer; we infer D from the layer's output via a
        # pilot forward.  Simpler: take the trailing dim of the
        # baseline logits.  But we want the *hidden* dim, not vocab.
        # Pull D from the model config when present.
        cfg = getattr(model, "config", None) or getattr(
            getattr(model, "model_config", None), "__dict__", {}
        )
        D = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "d_model", None)
            or baseline.shape[-1]
        )
        I = np.eye(int(D), dtype=np.float32)

        with project_layer(model, layer_idx=layer_idx, projection=I):
            patched = model(sample_input)
            if hasattr(patched, "logits"):
                patched = patched.logits

        diff = (baseline - patched).abs().max().item()
        if diff > atol:
            raise AssertionError(
                f"[D2] identity projection should be a no-op; max diff = {diff:g}"
            )
