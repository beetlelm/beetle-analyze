"""
analyze/xling/activation_buffer.py
──────────────────────────────────
Streaming, layer-selective activation extractor for the SVCCA / CKA /
CrossEncoder / probing analyses.

Activations dominate disk and memory budgets for cross-lingual analysis
across many checkpoints.  This module:

  * extracts per-layer hidden states with PyTorch forward hooks (no
    ``transformer_lens`` dependency required for the basic case),
  * caches them on disk under a deterministic key
    ``{cache_root}/{repo}/{revision}/{dataset_id}/layer_{l}.pt``
    so consecutive analyses in a single ``run.py`` invocation re-use
    the same activations,
  * yields them as torch tensors of shape ``(N, D)`` for whatever ``N``
    tokens / sentences the caller requested.

The implementation intentionally re-uses
``analyze.ppl_utils.load_model_and_tokenizer`` (the canonical loader in
beetle-analyze) so revision / cache-mode behaviour matches every other
stage.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def _slug(s: str) -> str:
    """Filesystem-safe slug.  Hash long strings so paths stay <255 chars."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in s)
    if len(safe) > 80:
        h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
        safe = safe[:60] + "_" + h
    return safe


def cache_path_for(
    cache_root: Path,
    repo: str,
    revision: str,
    dataset_id: str,
    layer: int,
    *,
    pooling: str = "mean",
) -> Path:
    return (
        Path(cache_root)
        / _slug(repo)
        / _slug(revision)
        / _slug(dataset_id)
        / pooling
        / f"layer_{layer}.pt"
    )


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class BufferConfig:
    """How to extract activations.

    ``layers`` selects which transformer-block outputs to capture; an
    empty list means "all layers, including embeddings".  ``pooling``
    controls how token-level activations are reduced to a per-sentence
    vector for SVCCA/CKA — for sub-sentence analyses (probes) callers
    can request ``token`` to keep the full ``(N_tokens, D)`` matrix.
    """
    cache_root: Path
    layers: Sequence[int] = ()
    pooling: str = "mean"      # "mean" | "cls" | "last" | "token"
    max_seq_len: int = 256
    batch_size: int = 8
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Hook-based extractor
# ---------------------------------------------------------------------------

class ActivationBuffer:
    """Per-(repo, revision) extractor with disk-cache backing.

    Lifecycle:

        ab = ActivationBuffer(cfg)
        ab.attach(model, tokenizer)
        for batch in batches:
            ab.feed(batch)              # one forward pass; caches results
        layer_acts = ab.collect()
        ab.detach()
    """

    def __init__(self, cfg: BufferConfig):
        self.cfg = cfg
        self._handles: List = []
        self._captured: Dict[int, List] = {}
        self._model = None
        self._tokenizer = None

    # ── Hook plumbing ────────────────────────────────────────────────────

    def attach(self, model, tokenizer) -> "ActivationBuffer":
        """Install forward hooks on the requested layers."""
        import torch

        self._model = model
        self._tokenizer = tokenizer
        self._handles.clear()
        self._captured = {l: [] for l in self.cfg.layers} or {0: []}

        # Walk the model to find transformer blocks.  Beetle pico exposes
        # them at ``model.layers``; HF causal-LM models at
        # ``model.model.layers`` (LLaMA-style) or ``transformer.h``
        # (GPT-2-style).  We handle the three common shapes; callers can
        # also pass a pre-located list via ``cfg.layer_modules``.
        blocks = _find_transformer_blocks(model)

        target_idxs = list(self.cfg.layers) if self.cfg.layers else list(range(len(blocks)))
        for idx in target_idxs:
            if idx < 0 or idx >= len(blocks):
                raise IndexError(
                    f"[xling.activation_buffer] layer {idx} out of range "
                    f"(model has {len(blocks)} blocks)"
                )
            self._captured.setdefault(idx, [])
            handle = blocks[idx].register_forward_hook(self._make_hook(idx))
            self._handles.append(handle)

        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._captured.clear()
        self._model = None
        self._tokenizer = None

    def _make_hook(self, layer_idx: int):
        def _hook(module, inputs, output):
            import torch

            # HF blocks return a tuple; pico returns a tensor.
            hs = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(hs, torch.Tensor):
                return
            self._captured[layer_idx].append(hs.detach().cpu())

        return _hook

    # ── Public extraction API ────────────────────────────────────────────

    def feed_texts(self, texts: Sequence[str]) -> None:
        """Run the model on a list of plaintext sentences; capture acts."""
        import torch

        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "[xling.activation_buffer] feed_texts requires .attach() first"
            )
        for chunk_start in range(0, len(texts), self.cfg.batch_size):
            chunk = texts[chunk_start : chunk_start + self.cfg.batch_size]
            enc = self._tokenizer(
                list(chunk),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.cfg.max_seq_len,
            )
            enc = {k: v.to(self.cfg.device) for k, v in enc.items()}
            with torch.no_grad():
                self._model(**enc)

    def collect(self) -> Dict[int, "torch.Tensor"]:
        """Pool the captured activations per layer.

        Returns a dict of ``{layer_idx: Tensor(N, D)}`` for "mean" / "cls"
        / "last", or ``{layer_idx: Tensor(N_total_tokens, D)}`` for
        ``"token"``.
        """
        import torch

        out: Dict[int, torch.Tensor] = {}
        for layer_idx, parts in self._captured.items():
            if not parts:
                continue
            # parts: list of (B, T, D) — concat along the batch dim, then
            # apply pooling.
            big = torch.cat(parts, dim=0)
            out[layer_idx] = _pool(big, self.cfg.pooling)
        return out


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------

def _pool(x: "torch.Tensor", how: str) -> "torch.Tensor":
    import torch

    if x.dim() == 2:
        # Already pooled; nothing to do.
        return x
    if how == "mean":
        return x.mean(dim=1)
    if how == "cls":
        return x[:, 0, :]
    if how == "last":
        return x[:, -1, :]
    if how == "token":
        return x.reshape(-1, x.shape[-1])
    raise ValueError(f"[xling.activation_buffer] unknown pooling: {how!r}")


# ---------------------------------------------------------------------------
# Block discovery
# ---------------------------------------------------------------------------

def _find_transformer_blocks(model) -> list:
    """Return the list of transformer blocks for the given model.

    Tries common layouts:
      * ``model.layers``           — pico decoder, LLaMA-style
      * ``model.model.layers``     — HF AutoModelForCausalLM (LLaMA wrap)
      * ``model.transformer.h``    — GPT-2 / GPT-NeoX style
      * ``model.gpt_neox.layers``  — Pythia
    """
    candidates = [
        ("layers", lambda m: getattr(m, "layers", None)),
        ("model.layers", lambda m: getattr(getattr(m, "model", None), "layers", None)),
        ("transformer.h", lambda m: getattr(getattr(m, "transformer", None), "h", None)),
        ("gpt_neox.layers",
         lambda m: getattr(getattr(m, "gpt_neox", None), "layers", None)),
    ]
    for _, getter in candidates:
        blocks = getter(model)
        if blocks is not None and hasattr(blocks, "__len__") and len(blocks) > 0:
            return list(blocks)
    raise RuntimeError(
        "[xling.activation_buffer] could not locate transformer blocks on model "
        f"of type {type(model).__name__}; pass them explicitly."
    )


# ---------------------------------------------------------------------------
# Cached extraction front-end
# ---------------------------------------------------------------------------

def extract_or_load(
    *,
    repo: str,
    revision: str,
    dataset_id: str,
    texts: Sequence[str],
    cfg: BufferConfig,
    load_model_fn=None,
    overwrite: bool = False,
) -> Dict[int, "torch.Tensor"]:
    """Extract activations or return them from disk cache.

    ``load_model_fn(repo, revision)`` is the model-loader callback; if
    None, we use ``ppl_utils.load_model_and_tokenizer`` which is the
    canonical loader in beetle-analyze.
    """
    import torch

    # Cache lookup.
    layers = list(cfg.layers) if cfg.layers else None
    cache_files = {
        l: cache_path_for(cfg.cache_root, repo, revision, dataset_id, l,
                          pooling=cfg.pooling)
        for l in (layers or range(64))   # over-provision; we trim below
    }
    if not overwrite:
        existing = {l: p for l, p in cache_files.items() if p.exists()}
        if layers is not None and all(l in existing for l in layers):
            logger.info(
                "[xling.activation_buffer] cache hit (%d layers) for %s @ %s",
                len(layers), repo, revision,
            )
            return {l: torch.load(p, weights_only=False) for l, p in existing.items()}

    # Cache miss — run extraction.
    if load_model_fn is None:
        from ppl_utils import load_model_and_tokenizer  # type: ignore
        load_model_fn = lambda r, rev: load_model_and_tokenizer(r, rev)

    model, tokenizer = load_model_fn(repo, revision)
    model = model.to(cfg.device)
    model.eval()

    ab = ActivationBuffer(cfg).attach(model, tokenizer)
    try:
        ab.feed_texts(texts)
        acts = ab.collect()
    finally:
        ab.detach()

    # Persist.
    for l, t in acts.items():
        p = cache_path_for(cfg.cache_root, repo, revision, dataset_id, l,
                           pooling=cfg.pooling)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(t, p)

    return acts
