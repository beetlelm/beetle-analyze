"""
cache_manager.py — Responsible HF cache management for 8xA100 clusters.

The BeetleLM analysis pipeline downloads many large models into HF's cache.
Without rotation, cache size explodes (>500 GB), swapping kicks in, and RAM
fills. This module provides three lifecycle policies:

    keep    — never delete (default for dev)
    rotate  — after each model completes, keep only the latest revision per repo
    clear   — evict everything not currently pinned

The rotation policy is disk-aware: if free disk drops below 30 %, it
auto-escalates to ``clear``.

Usage
-----
    from analyze.cache_manager import with_model_cache

    for repo in repos:
        with with_model_cache(repo, mode="rotate"):
            run_my_analysis(repo)
"""

from __future__ import annotations

import logging
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal

log = logging.getLogger(__name__)

CacheMode = Literal["keep", "rotate", "clear"]

DEFAULT_CACHE_DIR = Path(os.environ.get(
    "BEETLE_HF_CACHE", Path(__file__).resolve().parent.parent / "results" / ".hf_cache"
))

DISK_FREE_ESCALATION_THRESHOLD = 0.30  # escalate rotate→clear below 30% free
DISK_FREE_HARD_STOP_THRESHOLD  = 0.10  # refuse to proceed below 10% free


def _set_hf_env(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))


def _disk_free_fraction(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / usage.total if usage.total else 1.0


def _scan_cache():
    """Return a HFCacheInfo or None if huggingface_hub cannot be imported."""
    try:
        from huggingface_hub import scan_cache_dir  # type: ignore
        return scan_cache_dir()
    except ImportError:
        log.warning("[cache_manager] huggingface_hub not installed; skip scan")
        return None
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("[cache_manager] scan_cache_dir failed: %s", exc)
        return None


def summarise_cache(cache_dir: Path = DEFAULT_CACHE_DIR) -> str:
    """Return a human-readable one-liner for logs."""
    info = _scan_cache()
    if info is None:
        return f"[cache_manager] scan unavailable ({cache_dir})"
    gb = info.size_on_disk / (1024 ** 3)
    free = _disk_free_fraction(cache_dir) * 100
    return (
        f"[cache_manager] {len(info.repos)} repos, "
        f"{gb:.1f} GB on disk, {free:.1f}% disk free"
    )


def _evict_old_revisions(keep_repo: str | None = None) -> int:
    """Keep only the latest revision per repo. Return bytes freed."""
    info = _scan_cache()
    if info is None:
        return 0
    delete_hashes: list[str] = []
    for repo in info.repos:
        # Sort revisions newest-first by last_modified, keep the first.
        revs = sorted(
            repo.revisions, key=lambda r: r.last_modified, reverse=True
        )
        for r in revs[1:]:
            # Never delete the in-flight repo's revisions mid-run.
            if keep_repo and repo.repo_id == keep_repo:
                continue
            delete_hashes.append(r.commit_hash)
    if not delete_hashes:
        return 0
    strat = info.delete_revisions(*delete_hashes)
    freed = strat.expected_freed_size
    strat.execute()
    log.info("[cache_manager] rotate: freed %.1f GB across %d revisions",
             freed / (1024 ** 3), len(delete_hashes))
    return freed


def _clear_cache(keep_repo: str | None = None) -> int:
    """Evict everything not belonging to ``keep_repo``."""
    info = _scan_cache()
    if info is None:
        return 0
    delete_hashes: list[str] = []
    for repo in info.repos:
        if keep_repo and repo.repo_id == keep_repo:
            continue
        for r in repo.revisions:
            delete_hashes.append(r.commit_hash)
    if not delete_hashes:
        return 0
    strat = info.delete_revisions(*delete_hashes)
    freed = strat.expected_freed_size
    strat.execute()
    log.info("[cache_manager] clear: freed %.1f GB", freed / (1024 ** 3))
    return freed


@contextmanager
def with_model_cache(
    repo: str,
    mode: CacheMode = "rotate",
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Iterator[None]:
    """Context manager wrapping a per-model analysis step.

    On enter: ensures HF env vars point at ``cache_dir``.
    On exit: applies the requested eviction policy, with a disk-aware
    escalation from ``rotate`` to ``clear`` when free space is tight.
    """
    _set_hf_env(cache_dir)
    frac = _disk_free_fraction(cache_dir)
    if frac < DISK_FREE_HARD_STOP_THRESHOLD:
        log.error(
            "[cache_manager] disk free fraction %.2f below hard stop %.2f — aborting",
            frac, DISK_FREE_HARD_STOP_THRESHOLD,
        )
        raise RuntimeError("cache_manager: disk full")

    try:
        yield
    finally:
        if mode == "keep":
            return
        frac = _disk_free_fraction(cache_dir)
        effective: CacheMode = mode
        if mode == "rotate" and frac < DISK_FREE_ESCALATION_THRESHOLD:
            log.warning(
                "[cache_manager] disk free %.1f%% below %.0f%% — "
                "escalating rotate→clear",
                frac * 100, DISK_FREE_ESCALATION_THRESHOLD * 100,
            )
            effective = "clear"

        if effective == "rotate":
            _evict_old_revisions(keep_repo=None)
        elif effective == "clear":
            _clear_cache(keep_repo=repo)


@contextmanager
def cache_session(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    mode: CacheMode = "keep",
) -> Iterator[None]:
    """Outer session wrapper for a whole stage. Logs cache summary on exit."""
    _set_hf_env(cache_dir)
    log.info(summarise_cache(cache_dir))
    try:
        yield
    finally:
        if mode == "clear":
            _clear_cache(keep_repo=None)
        log.info(summarise_cache(cache_dir))


__all__ = [
    "CacheMode",
    "DEFAULT_CACHE_DIR",
    "with_model_cache",
    "cache_session",
    "summarise_cache",
]
