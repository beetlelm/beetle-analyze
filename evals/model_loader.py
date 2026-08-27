"""
evals/model_loader.py — Shared A100-optimised model loader for eval scripts.

Provides:
  load_model(repo, device)     — (model, tokenizer) with bfloat16 + Flash Attn 2 + TF32
  unload_model(model, tok)     — explicit GPU memory release
  resolve_vocab_size(model)    — robust vocab-size lookup (handles Gemma 3 etc.)
  log_gpu_mem(logger, prefix)  — log allocated / reserved VRAM
  setup_logging(stage, root, rank) — stdout + rotating file handler

A100 optimisations applied automatically when CUDA is available:
  • torch.bfloat16 (Ampere natively supports BF16 with same throughput as FP16,
    better numerical range — avoids overflow in large-vocab logit distributions)
  • attn_implementation="flash_attention_2" (requires transformers >= 4.36 and
    flash-attn package; falls back silently if unavailable)
  • torch.backends.cuda.matmul.allow_tf32 = True  (2× speedup on Ampere GEMMs)
"""

from __future__ import annotations

import atexit
import gc
import logging
import signal
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# A100 / Ampere global precision flags — set once at import time
# ---------------------------------------------------------------------------

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(stage: str, output_root: Path | str, rank: int) -> None:
    """Configure root logger to write to stdout + a per-rank log file.

    Matches the format used in analyze/run.py:_setup_logging().
    Safe to call multiple times — clears prior handlers before adding new ones.
    """
    output_root = Path(output_root)
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{stage}_{date.today()}_rank{rank}.log"

    fmt = f"%(asctime)s [%(levelname)s] [rank{rank}] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(sh)

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    # Line-buffer the underlying stream so each record is flushed to disk on
    # newline. Without this, a SIGHUP/SIGABRT (e.g. parent shell exit, CUDA
    # device-side abort) leaves the most recent log lines — including any
    # traceback — stuck in the FileHandler's buffer and lost.
    try:
        fh.stream.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    root.addHandler(fh)

    # Silence noisy HF / network libraries
    for lib in ("datasets", "huggingface_hub", "huggingface_hub.file_download",
                "fsspec", "httpx", "httpcore", "urllib3"):
        logging.getLogger(lib).setLevel(logging.ERROR)

    # Ensure logs are flushed on any clean Python exit and on SIGTERM
    # (the launcher's `wait` + Ctrl+C / kill propagation path).
    _install_log_flush_hooks()

    logger.info(f"Logging to {log_file}")


_HOOKS_INSTALLED = False


def _install_log_flush_hooks() -> None:
    """Register atexit + SIGTERM handlers that flush all logging handlers.

    Idempotent — safe to call from setup_logging() on every rank.
    """
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return
    _HOOKS_INSTALLED = True

    atexit.register(logging.shutdown)

    def _flush_and_exit(signum, frame):
        try:
            for h in logging.getLogger().handlers:
                try:
                    h.flush()
                except Exception:
                    pass
            logging.shutdown()
        finally:
            # Re-raise default behaviour so the process actually exits.
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)

    # SIGTERM is the polite kill from the launcher / orchestrator.
    # SIGHUP is what a closing parent shell sends; we install setsid in the
    # launcher, but install a handler too in case ranks are run directly.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _flush_and_exit)
        except (ValueError, OSError):
            # Not in main thread or signal unsupported on this platform.
            pass


# ---------------------------------------------------------------------------
# GPU memory reporting
# ---------------------------------------------------------------------------

def log_gpu_mem(log: logging.Logger, prefix: str = "") -> None:
    """Log current and peak GPU memory allocation (no-op on CPU)."""
    if not torch.cuda.is_available():
        return
    alloc   = torch.cuda.memory_allocated()  / 1e9
    reserv  = torch.cuda.memory_reserved()   / 1e9
    peak    = torch.cuda.max_memory_allocated() / 1e9
    tag = f"{prefix} " if prefix else ""
    log.info(f"  {tag}GPU mem: {alloc:.1f} GB alloc / {reserv:.1f} GB reserved / {peak:.1f} GB peak")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    repo: str,
    device: str = DEVICE,
    revision: str = "main",
    device_map: Optional[object] = None,
) -> Optional[tuple]:
    """Load (model, tokenizer) for *repo* with A100-optimised settings.

    Returns None and logs a warning on any failure so callers can skip
    gracefully without crashing the evaluation loop.

    Dtype:
        bfloat16 on CUDA (Ampere), float32 on CPU.

    Attention:
        Tries flash_attention_2 first (requires transformers >= 4.36 and
        flash-attn wheel); falls back to the default scaled-dot-product
        attention if unavailable or unsupported by the model architecture.

    Tokenizer:
        Loads from the model repo (all Beetle model repos and baselines
        bundle their own tokenizer.json). Falls back from fast → slow
        tokenizer if the fast variant raises.

    device_map:
        If given (e.g. ``"auto"`` or a device-map dict), passed straight
        through to ``from_pretrained`` and the explicit ``.to(device)``
        is suppressed. Use this for xlarge models (>=30 B params) that
        must shard across multiple GPUs. Default ``None`` keeps the
        original single-device behaviour for all rank-parallel callers.
    """
    logger.info(f"Loading {repo} @ {revision} …")
    load_kw = dict(revision=revision, trust_remote_code=True)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    extra_kw: dict = {}
    # When the caller has not chosen an explicit placement strategy, route
    # through device_map={"": device}: it places parameters on the target
    # device during from_pretrained, eliminating the host-RAM peak that
    # the prior CPU-load + .to(device) path produced (one of the OOM
    # failure modes that has crashed GPU nodes during sweeps).
    _auto_device_map = device_map is None
    if device_map is not None:
        extra_kw["device_map"] = device_map
    elif device == "cuda":
        extra_kw["device_map"] = {"": device}

    try:
        # Tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo, **load_kw, use_fast=True)
        except (ValueError, ImportError):
            logger.debug(f"  Fast tokenizer unavailable for {repo}, using slow tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(repo, **load_kw, use_fast=False)

        # Model — try Flash Attention 2 first
        try:
            model = AutoModelForCausalLM.from_pretrained(
                repo, **load_kw,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
                **extra_kw,
            )
            logger.info(f"  Loaded with Flash Attention 2")
        except (ValueError, ImportError, NotImplementedError):
            model = AutoModelForCausalLM.from_pretrained(
                repo, **load_kw,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                **extra_kw,
            )

        model.eval()
        # When device_map is set, accelerate has already placed the modules;
        # calling .to(device) would either error or copy the whole model
        # back to one GPU, defeating the purpose.
        if device_map is None and not _auto_device_map:
            model.to(device)
        elif _auto_device_map and device != "cuda":
            # CPU path: device_map was not applied; fall back to .to(device).
            model.to(device)

        # Ensure a pad token exists so the downstream batched scorers
        # (eval_blimp / eval_meco / eval_bliss / eval_flores_ppl) can pad with
        # attention masks. Models like Llama-3.1, Qwen3-Base and Apertus ship
        # without a pad token by default; we fall back to eos, matching the
        # ad-hoc logic already present in eval_bliss.py / legacy/eval_model.py.
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        log_gpu_mem(logger, prefix=repo.split("/")[-1])
        return model, tokenizer

    except Exception as exc:
        logger.warning(f"SKIP {repo}: {str(exc).split(chr(10))[0]}")
        return None


def resolve_vocab_size(model) -> int:
    """Return the language-model vocabulary size for *model*.

    Handles three layouts:

    1. Standard text-only config — ``model.config.vocab_size`` is set.
    2. Composite multimodal configs (e.g. Gemma 3, PaliGemma) where the
       text config is nested under ``config.text_config``.
    3. Last-resort fallback: the input-embedding layer's ``num_embeddings``.

    We clamp input ids against the returned value in every scoring loop to
    guard against tokenizer-model vocabulary drift (some model repos pin a
    tokenizer whose top ids exceed the model's embedding rows).
    """
    cfg = getattr(model, "config", None)
    if cfg is not None:
        v = getattr(cfg, "vocab_size", None)
        if v:
            return int(v)
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            v = getattr(text_cfg, "vocab_size", None)
            if v:
                return int(v)
    try:
        emb = model.get_input_embeddings()
        return int(emb.num_embeddings)
    except Exception as exc:
        raise AttributeError(
            f"Cannot determine vocab_size for {type(model).__name__}: {exc}"
        )


def unload_model(model, tokenizer) -> None:
    """Delete model/tokenizer objects and release GPU memory."""
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
