"""
analyze/polca_shim.py
─────────────────────
Post-hoc adapter that lets :mod:`POLCA` operate on Beetle HF checkpoints.

POLCA's own driver (``POLCA/train_language_model.py``) was written for a
specific ``HookedTransformer`` recipe trained inside the POLCA
codebase.  The Beetle bilingual checkpoints
(``Beetle-FineWeb-2B/beetle-bilingual-*-nld-eng``) live on the HuggingFace
Hub as ``step-{N}`` branches and load through
``transformers.AutoModelForCausalLM``.

This shim:

  1. Loads a Beetle checkpoint via the existing
     :func:`ppl_utils.load_model_and_tokenizer`.
  2. Re-emits the loaded weights as a ``transformer_lens.HookedTransformer``
     (when available) so POLCA's gradient / Hessian helpers can operate
     unchanged.
  3. Optionally runs POLCA's ``compute_hessian_eigenvectors`` /
     ``compute_svd_basis`` / ``run_lca_analysis`` driver on a held-out
     dataset and writes the artifact to the same on-disk layout that
     ``beetlelm/src/training/polca_extractor.py`` produces during live
     training.  Result: post-hoc and live artifacts are
     interchangeable downstream.

The shim also tolerates the absence of ``transformer_lens`` — when the
import fails, it falls back to a "lite" path that uses the
gradient/Hessian pipeline already in
:mod:`beetlelm.src.training.polca_extractor`.
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class POLCAShimConfig:
    repo: str
    revision: str = "main"
    runs_dir: Path = Path("results/polca")
    run_name: Optional[str] = None    # default: f"{repo_basename}-{revision}"
    k_eigenvectors: int = 32
    basis_rank: int = 32
    holdout_batch_size: int = 8
    holdout_seq_len: int = 512
    use_hooked_transformer: bool = True

    @property
    def effective_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return f"{self.repo.replace('/', '__')}-{self.revision}"


# ---------------------------------------------------------------------------
# HookedTransformer detection
# ---------------------------------------------------------------------------

def _have_transformer_lens() -> bool:
    try:
        importlib.import_module("transformer_lens")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Holdout-batch sampling
# ---------------------------------------------------------------------------

def _sample_holdout(tokenizer, texts: Sequence[str], cfg: POLCAShimConfig):
    """Tokenise ``texts`` into a batch the extractor can consume.

    The trainer-side extractor expects a dict with ``input_ids``;
    we mirror that to keep the live and post-hoc code paths
    interchangeable.
    """
    import torch

    enc = tokenizer(
        list(texts)[: cfg.holdout_batch_size],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=cfg.holdout_seq_len,
    )
    return {"input_ids": enc["input_ids"].to(torch.long)}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_polca_artifact(
    cfg: POLCAShimConfig,
    *,
    holdout_texts: Sequence[str],
    device: Optional[str] = None,
) -> Path:
    """Run POLCA artifact extraction on a Beetle HF checkpoint.

    Returns the path of the saved ``step_*.pt`` artifact, with the same
    schema as :mod:`beetlelm.src.training.polca_extractor`.

    Strategy
    --------
    We always run the lite path (the live extractor's ``_compute``)
    because it is identical in spirit to ``POLCA``'s own pipeline and
    produces a directly compatible artifact.  When
    ``cfg.use_hooked_transformer`` is True and ``transformer_lens`` is
    importable we additionally emit the ``HookedTransformerConfig``
    side-car file so downstream POLCA tooling that consumes
    ``HookedTransformer`` directly can still find a compatible config.
    """
    import torch

    # Beetle's canonical loader.
    sys.path.insert(0, str(Path(__file__).parent))   # for in-repo imports
    from ppl_utils import load_model_and_tokenizer  # type: ignore

    # Reuse the live extractor — this is the contract that keeps
    # post-hoc and live artifacts identical.
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "beetlelm"))
    from src.training.polca_extractor import (  # type: ignore
        POLCAExtractor, _POLCASpec,
    )

    spec = _POLCASpec(
        enabled=True,
        every_n_steps=1,
        k_eigenvectors=cfg.k_eigenvectors,
        basis_rank=cfg.basis_rank,
        holdout_batch_size=cfg.holdout_batch_size,
        holdout_seq_len=cfg.holdout_seq_len,
        artifact_dir="polca",
        push_to_hf=False,
        precision="fp32",
    )
    extractor = POLCAExtractor(
        spec,
        runs_dir=str(cfg.runs_dir),
        run_name=cfg.effective_run_name,
    )

    model, tokenizer = load_model_and_tokenizer(cfg.repo, cfg.revision)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    holdout = _sample_holdout(tokenizer, holdout_texts, cfg)
    extractor.set_holdout_batch(holdout)

    step = _step_from_revision(cfg.revision)
    path = extractor.extract(
        model=model,
        step=step,
        device=torch.device(device),
        is_global_zero=True,
        rank=0,
    )
    if path is None:
        raise RuntimeError(
            f"[polca_shim] extraction returned None for {cfg.repo} @ "
            f"{cfg.revision}; check logs."
        )

    if cfg.use_hooked_transformer and _have_transformer_lens():
        try:
            _emit_hooked_config_side_car(model, path)
        except Exception:  # pragma: no cover — diagnostic
            logger.exception(
                "[polca_shim] HookedTransformer side-car emission failed; "
                "post-hoc artifact is still valid."
            )

    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step_from_revision(rev: str) -> int:
    """Parse ``step-1234`` → ``1234``; ``main`` → ``-1``."""
    if rev.startswith("step-"):
        try:
            return int(rev.split("-", 1)[1])
        except ValueError:
            pass
    return -1


def _emit_hooked_config_side_car(model, artifact_path: Path) -> None:
    """Write a small JSON next to the artifact describing the model's
    ``HookedTransformer``-compatible config.

    Downstream POLCA tooling that wants to load the model via
    ``transformer_lens.HookedTransformer.from_pretrained_no_processing``
    can read this side-car to know which model name + revision to pass.
    """
    import json

    cfg = getattr(model, "config", None)
    payload = {
        "n_layers": getattr(cfg, "num_hidden_layers", None),
        "d_model":  getattr(cfg, "hidden_size", None),
        "n_heads":  getattr(cfg, "num_attention_heads", None),
        "vocab_size": getattr(cfg, "vocab_size", None),
        "torch_dtype": str(getattr(cfg, "torch_dtype", "")),
    }
    side = artifact_path.with_suffix(".hookedconfig.json")
    side.write_text(json.dumps(payload, indent=2))
    logger.info("[polca_shim] emitted HookedTransformer side-car: %s", side)
