"""
analyze/_interp_helpers.py — Small shared helpers for the EMNLP-interp suite.

Used by every new analyze/{xling,sae,causal,lens,brain,forgetting_pkg,aoa}
module. Kept dependency-light (numpy, pandas, torch, transformers) so the
modules run on CPU notebooks for prototyping and on GPU for the real eval.

Public API
----------
- ``per_layer_mean_activations(model, tokenizer, sentences, layers=None)``
  Returns a dict ``layer_idx -> np.ndarray (n_sents, hidden_dim)`` of
  sentence-mean residual-stream activations.
- ``RESULTS_PAPER`` — canonical results path.
- ``bootstrap_ci(values, *, n=1000, alpha=0.05)`` — percentile bootstrap CI.
- ``write_csv_atomic(df, path)`` — write CSV with mkdir + .tmp swap so
  long-running pipelines don't leave half-written files on a SIGINT.
- ``device_for(allow_gpu=True)`` — picks ``cuda`` if available and not
  disabled by env var ``BEETLE_FORCE_CPU=1``.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _results_paper_root() -> Path:
    """Resolve the results-paper output directory.

    Order of precedence:
      1. ``$BEETLE_RESULTS_ROOT`` if set — overrides everything (Colab path).
      2. ``$PWD/results/evals_paper`` if the package is installed
         (``dist-packages/...``) — write to the user's CWD, not into the
         site-packages tree where the wheel lives.
      3. ``REPO_ROOT/results/evals_paper`` for in-tree dev runs.
    """
    env = os.environ.get("BEETLE_RESULTS_ROOT")
    if env:
        return Path(env)
    if "site-packages" in str(REPO_ROOT) or "dist-packages" in str(REPO_ROOT):
        return Path.cwd() / "results" / "evals_paper"
    return REPO_ROOT / "results" / "evals_paper"


RESULTS_PAPER = _results_paper_root()


def device_for(allow_gpu: bool = True) -> str:
    """Pick ``cuda`` if available and not blocked by ``BEETLE_FORCE_CPU=1``."""
    import torch

    if os.environ.get("BEETLE_FORCE_CPU") == "1":
        return "cpu"
    if allow_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def write_csv_atomic(df: pd.DataFrame, path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(p)


def bootstrap_ci(values: Sequence[float], *,
                 n: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI over ``values``."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

@contextmanager
def _eval_no_grad():
    import torch
    with torch.no_grad():
        yield


def _find_layer_module_list(model):
    """Best-effort search for the transformer-block ``ModuleList``.

    Custom causal-LM models (PicoDecoderHF, etc.) do not honour
    ``output_hidden_states=True`` because their forward signature ignores
    the kwarg. Fall back to forward hooks on the per-layer ModuleList.
    Returns the ModuleList or None.
    """
    import torch.nn as nn

    candidates = ["layers", "h", "blocks", "decoder.layers", "model.layers",
                  "transformer.h", "transformer.layers"]
    for path in candidates:
        m = model
        try:
            for part in path.split("."):
                m = getattr(m, part)
            if isinstance(m, nn.ModuleList) and len(m) > 0:
                return m
        except AttributeError:
            continue
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 4:
            return mod
    return None


def per_layer_mean_activations(
    model,
    tokenizer,
    sentences: Iterable[str],
    *,
    layers: Sequence[int] | None = None,
    batch_size: int = 16,
    max_length: int = 256,
) -> dict[int, np.ndarray]:
    """Per-layer sentence-mean residual-stream activations.

    Tries ``model(..., output_hidden_states=True)`` first, falling back to
    forward hooks on the layer ModuleList if the model returns no
    ``hidden_states`` (e.g., custom decoder classes whose forward ignores
    the kwarg). Returns ``{layer_idx: array(n, d)}``.

    The returned arrays live on CPU as ``float32``. Use ``layers=[k]`` to
    bound memory when only one layer is needed (e.g., SAE @ layer 6).
    """
    import torch
    import torch.nn as nn

    device = next(model.parameters()).device
    sents = list(sentences)
    out: dict[int, list[np.ndarray]] = {}

    layer_list = None
    use_hooks = False

    for i in range(0, len(sents), batch_size):
        batch = sents[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device).float()

        captured: list = []
        handles = []
        if use_hooks:
            def _make_hook(idx):
                def _hook(_mod, _inp, output):
                    h = output[0] if isinstance(output, tuple) else output
                    captured.append((idx, h.detach()))
                return _hook
            for idx, blk in enumerate(layer_list):
                handles.append(blk.register_forward_hook(_make_hook(idx)))

        with _eval_no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)

        for h in handles:
            h.remove()

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None and not use_hooks:
            layer_list = _find_layer_module_list(model)
            if layer_list is None:
                raise RuntimeError(
                    "model returned no hidden_states and no layer ModuleList "
                    "could be found for hook-based extraction"
                )
            use_hooks = True
            captured = []
            handles = []
            def _make_hook2(idx):
                def _hook(_mod, _inp, output):
                    h = output[0] if isinstance(output, tuple) else output
                    captured.append((idx, h.detach()))
                return _hook
            for idx, blk in enumerate(layer_list):
                handles.append(blk.register_forward_hook(_make_hook2(idx)))
            with _eval_no_grad():
                outputs = model(input_ids=input_ids)
            for h in handles:
                h.remove()

        if use_hooks:
            captured.sort(key=lambda x: x[0])
            block_outs = [h for _, h in captured]
            try:
                emb = model.embedding_proj(input_ids)
            except Exception:
                emb = None
            hidden_states = ([emb] if emb is not None else []) + block_outs

        if layers is None:
            layer_iter = range(len(hidden_states))
        else:
            layer_iter = [l for l in layers if 0 <= l < len(hidden_states)]

        for l in layer_iter:
            h = hidden_states[l]                       # (b, s, d)
            if h is None:
                continue
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            sent_mean = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
            arr = sent_mean.detach().to("cpu", dtype=torch.float32).numpy()
            out.setdefault(l, []).append(arr)

    return {l: np.concatenate(parts, axis=0) for l, parts in out.items()}


# ---------------------------------------------------------------------------
# Common defaults
# ---------------------------------------------------------------------------

# Canonical HF repo names — match the manifest at
# analyze/fineweb_model_manifest.json. Earlier drafts used short
# placeholder slugs that did not exist on the Hub.
ALL_BEETLE_REPOS = {
    "b1":       "Beetle-FineWeb/beetle-bilingual-balanced-b1-fineweb-nld-eng",
    "b2":       "Beetle-FineWeb/beetle-bilingual-l2-50-simultaneous-b2-fineweb-nld-eng",
    "b3":       "Beetle-FineWeb/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-nld-eng",
    "b4":       "Beetle-FineWeb/beetle-bilingual-balanced-b4-fineweb-nld-eng",
    "b5":       "Beetle-FineWeb/beetle-bilingual-balanced-b5-fineweb-nld-eng",
    "mono-eng": "Beetle-FineWeb/beetle-monolingual-fineweb-eng",
}

# Default 3-curriculum sweep used by the EMNLP-interp pipeline. Key order
# aligned with --curricula on the CLI so the `zip(args.repos,
# args.curricula)` pairing stays correct.
DEFAULT_BEETLE_REPOS = {
    "b1": ALL_BEETLE_REPOS["b1"],
    "b3": ALL_BEETLE_REPOS["b3"],
    "b5": ALL_BEETLE_REPOS["b5"],
}

EXTERNAL_BASELINE_REPOS = {
    "xlm-r-base": "FacebookAI/xlm-roberta-base",
    "mbert":      "google-bert/bert-base-multilingual-cased",
}
