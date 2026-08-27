"""
utils.py — Shared utilities for BeetleLM evaluation.

  list_checkpoints()   enumerate step-N branches + main from HF Hub
  append_result()      thread-safe CSV append (flock); header written inside lock
  already_done()       in-memory resume cache — reads CSV at most once per process
"""

import csv
import fcntl
import gc
import logging
import re
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

# ── HF API singleton ──────────────────────────────────────────────────────────
_api: Optional[HfApi] = None

def _get_api(token: Optional[str] = None) -> HfApi:
    global _api
    if _api is None:
        _api = HfApi(token=token)
    return _api


# ── Checkpoint enumeration ────────────────────────────────────────────────────

def list_checkpoints(repo: str, token: Optional[str] = None) -> List[str]:
    """
    Return [step-100, step-200, …, main] sorted numerically.
    Falls back to ['main'] on any error.
    """
    try:
        api   = _get_api(token)
        refs  = api.list_repo_refs(repo_id=repo, token=token)
        names = [b.name for b in refs.branches]
        steps = sorted(
            [n for n in names if re.match(r"^step-\d+$", n)],
            key=lambda n: int(n.split("-")[1]),
        )
        ordered = steps + (["main"] if "main" in names else [])
        return ordered or ["main"]
    except Exception as e:
        logger.warning(f"Could not list branches for {repo}: {e}. Using 'main'.")
        return ["main"]


# ── Thread-safe CSV append ────────────────────────────────────────────────────

RESULT_FIELDS = [
    "benchmark", "model", "lang_pair", "bilingual_type",
    "checkpoint", "eval_language", "accuracy", "n_correct", "n_total",
]

def append_result(csv_path: str, row: dict) -> None:
    """
    Append one row to a CSV file.

    Thread-safety: flock (exclusive) is held for the entire open→write→close
    sequence. The 'write header?' decision (is_new) is made INSIDE the lock
    using f.tell()==0 so two processes racing on an empty file cannot both
    write a header row.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
            if f.tell() == 0:          # empty file → write header first
                writer.writeheader()
            writer.writerow(row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    # Keep the in-memory cache consistent so subsequent already_done() calls
    # reflect this write without re-reading the file.
    _mark_done(
        benchmark=row.get("benchmark", ""),
        model=row["model"],
        checkpoint=row["checkpoint"],
        eval_language=row["eval_language"],
        csv_path=csv_path,
    )


# ── In-memory resume cache ────────────────────────────────────────────────────
# Keyed by csv_path so the cache is per-file.
# Value: set of (benchmark, model, checkpoint, eval_language) tuples.

_done_cache: dict = {}   # csv_path -> set[tuple]


def _load_cache(csv_path: str) -> None:
    """Read the CSV once and populate _done_cache[csv_path]."""
    path = Path(csv_path)
    _done_cache[csv_path] = set()
    if not path.exists():
        return
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            _done_cache[csv_path].add((
                row.get("benchmark", ""),
                row.get("model", ""),
                row.get("checkpoint", ""),
                row.get("eval_language", ""),
            ))


def already_done(csv_path: str, model: str, checkpoint: str,
                 eval_language: str, benchmark: str) -> bool:
    """
    Return True if this (benchmark, model, checkpoint, eval_language)
    combination is already recorded in the CSV.

    The CSV is read at most ONCE per process per file; all subsequent lookups
    hit the in-memory set. New rows written via append_result() update the
    cache immediately, so no re-reads are ever needed.
    """
    if csv_path not in _done_cache:
        _load_cache(csv_path)
    return (benchmark, model, checkpoint, eval_language) in _done_cache[csv_path]


def _mark_done(benchmark: str, model: str, checkpoint: str,
               eval_language: str, csv_path: str) -> None:
    if csv_path not in _done_cache:
        _load_cache(csv_path)
    _done_cache[csv_path].add((benchmark, model, checkpoint, eval_language))
