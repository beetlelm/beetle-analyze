"""
_rank_barrier.py — File-based barrier for multi-rank analyze.run fan-out.

When analyze.run is launched as N independent processes (one per GPU) sharing
an output tree, each per-stage dispatcher writes its own per-repo CSVs and
then calls ``mark_done(stage, rank, world_size)`` before returning. Rank 0
calls ``wait_for_all(stage, world_size)`` before building the per-stage
summary from the per-repo files on disk.

Sentinel files live under ``<output_root>/.barrier/``.

The barrier is intentionally dumb:
  - No inter-process synchronisation (no locks, no MPI).
  - No partial-failure recovery — a dead rank blocks the summary forever.
  - Timeout is generous (default 4 h) so slow ranks don't cause false aborts.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

__all__ = [
    "rank_and_world",
    "shard",
    "is_rank0",
    "mark_done",
    "wait_for_all",
    "barrier_dir",
]


def rank_and_world() -> tuple[int, int]:
    """Read (rank, world_size) from env. Fall back to (0, 1) for single-process."""
    try:
        rank = int(os.environ.get("RANK", "0"))
    except ValueError:
        rank = 0
    try:
        world = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        world = 1
    if world < 1:
        world = 1
    if rank < 0 or rank >= world:
        rank = 0
    return rank, world


def shard(items: list, rank: int, world_size: int) -> list:
    """Return the contiguous ``items[rank::world_size]`` slice."""
    if world_size <= 1:
        return list(items)
    return list(items)[rank::world_size]


def is_rank0(rank: int | None = None) -> bool:
    if rank is None:
        rank, _ = rank_and_world()
    return rank == 0


def barrier_dir(output_root: Path | str = "results") -> Path:
    d = Path(output_root) / ".barrier"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sentinel(stage: str, rank: int, output_root: Path | str) -> Path:
    return barrier_dir(output_root) / f"stage_{stage}_rank_{rank}.done"


def mark_done(stage: str, rank: int, world_size: int,
              output_root: Path | str = "results") -> None:
    """Touch this rank's sentinel for *stage*. No-op when world_size == 1."""
    if world_size <= 1:
        return
    path = _sentinel(stage, rank, output_root)
    path.write_text(f"{time.time()}\n")


def wait_for_all(stage: str, world_size: int,
                 output_root: Path | str = "results",
                 timeout_s: float = 4 * 3600.0,
                 poll_s: float = 2.0) -> bool:
    """Block until every rank has touched its sentinel. Returns False on timeout."""
    if world_size <= 1:
        return True
    deadline = time.time() + timeout_s
    expected = [_sentinel(stage, r, output_root) for r in range(world_size)]
    while time.time() < deadline:
        if all(p.exists() for p in expected):
            return True
        time.sleep(poll_s)
    missing = [p.name for p in expected if not p.exists()]
    raise TimeoutError(
        f"[barrier] stage={stage!r} timed out after {timeout_s:.0f}s; "
        f"missing sentinels: {missing}"
    )


def reset(stage: str, world_size: int,
          output_root: Path | str = "results") -> None:
    """Remove sentinels for a stage. Call on rank 0 before launching work."""
    if world_size <= 1:
        return
    for r in range(world_size):
        p = _sentinel(stage, r, output_root)
        try:
            p.unlink()
        except FileNotFoundError:
            pass
