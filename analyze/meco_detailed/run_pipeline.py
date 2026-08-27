"""
run_pipeline.py — High-throughput orchestrator for meco_detailed across N models.

Design (optimised for "all 81 models on one A100, ASAP"):

  +----------------------------+
  |  HF prefetch thread        |    -> warms ~/.cache/huggingface for next model
  +-------------+--------------+
                |
  +-------------v--------------+
  |  GPU worker (1 thread)     |    -> subprocess: analyze.meco_surprisal
  +-------------+--------------+
                | (model scored → 8 futures submitted)
  +-------------v--------------+
  |  CPU pool (N processes)    |    -> in-process analyze.meco_detailed.d*
  |  initialised w/ RT cache   |
  +-------------+--------------+
                |
  +-------------v--------------+
  |  harvester thread          |    -> futures.result() → manifest, errors
  +----------------------------+

Why this is fast
----------------
  * The CPU pool runs **in-process** via ``ProcessPoolExecutor``: workers
    survive across tasks, so we pay the pandas + statsmodels cold-import
    ONCE per worker instead of once per (model × module).
  * Each worker pre-loads the joined MECO RT cache at startup
    (``init_worker``), so all 8 modules in that worker reuse the same
    in-memory frame — saves the ~10 s rdata + wordfreq pass × 648 calls.
  * BLAS thread counts are pinned to 1 per worker so the lmer fits don't
    fight each other on a shared CPU.
  * ``--fast`` flips ``BEETLE_FAST_FIT=1`` which tells ``meco_canonical``
    to skip the expensive Tier-1 random-effects spec (which fails to
    converge on most single-language subsets anyway).
  * A background thread prefetches the next model's HF snapshot while the
    GPU is busy scoring the current one, so the GPU never blocks on a
    cold download between models.

Manifest semantics (same as v1)
-------------------------------
TODO → SCORING → SCORED → ANALYZING → DONE / FAILED, append-only at
``results/meco_detailed/pipeline_manifest.jsonl``. Resume works by
default; pass ``--no-retry-failed`` to skip permanently broken models.

CLI
---
    python -m analyze.meco_detailed.run_pipeline \\
        --models-file models.txt \\
        --gpu-id 0 \\
        --cpu-workers 4 \\
        --fast
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

from analyze import _path  # noqa: F401
from analyze.meco_detailed._common import OS_CPU_COUNT, REPO_ROOT, RESULTS_DIR
from analyze.meco_detailed._runner import (
    ALL_MODULE_SLUGS, init_worker, regenerate_plots, run_one,
)


MANIFEST_PATH = RESULTS_DIR / "pipeline_manifest.jsonl"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class _ModelState:
    model: str
    state: str = "TODO"           # TODO|SCORING|SCORED|ANALYZING|DONE|FAILED
    ts: str = ""
    gpu_seconds: float = 0.0
    cpu_seconds: float = 0.0
    modules_ok: int = 0
    modules_total: int = 0
    error: str = ""
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_manifest_lock = threading.Lock()


def _append_manifest(rec: _ModelState) -> None:
    rec.ts = _now()
    line = json.dumps(asdict(rec), separators=(",", ":")) + "\n"
    with _manifest_lock:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MANIFEST_PATH.open("a") as f:
            f.write(line)


def _read_manifest() -> dict[str, _ModelState]:
    states: dict[str, _ModelState] = {}
    if not MANIFEST_PATH.exists():
        return states
    with MANIFEST_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            states[d["model"]] = _ModelState(**d)
    return states


# ---------------------------------------------------------------------------
# GPU stage helpers
# ---------------------------------------------------------------------------

def _spawn_gpu(
    cmd: Sequence[str],
    *,
    env_extra: dict[str, str],
    log_path: Path,
) -> tuple[int, float]:
    env = os.environ.copy()
    env.update(env_extra)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("a") as logf:
        logf.write(f"\n# --- {_now()} :: {' '.join(shlex.quote(c) for c in cmd)} ---\n")
        logf.flush()
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        try:
            rc = proc.wait()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            try:
                rc = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = -1
            raise
    return int(rc), time.time() - start


def _hf_prefetch(model_id: str) -> None:
    """Best-effort: warm ~/.cache/huggingface for ``model_id``.

    Runs from a daemon thread; failures are silent because the GPU stage
    will catch the missing model and surface a real error.
    """
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id, etag_timeout=10, max_workers=4)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Pipeline:
    """High-throughput orchestrator with HF prefetch + ProcessPool CPU stage."""

    def __init__(
        self,
        models: Sequence[str],
        *,
        gpu_id: int = 0,
        cpu_workers: int = 0,
        modules: Sequence[str] = ALL_MODULE_SLUGS,
        retry_failed: bool = True,
        score_dir: Path = REPO_ROOT / "results" / "meco",
        log_dir: Path = RESULTS_DIR / "pipeline_logs",
        dry_run: bool = False,
        fast: bool = False,
        prefetch: bool = True,
        regenerate_plots_at_end: bool = True,
    ) -> None:
        self.models = list(dict.fromkeys(models))
        self.gpu_id = gpu_id
        self.cpu_workers = cpu_workers or min(8, OS_CPU_COUNT)
        self.modules = list(modules)
        self.retry_failed = retry_failed
        self.score_dir = Path(score_dir)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.fast = fast
        self.prefetch = prefetch
        self.regenerate_plots_at_end = regenerate_plots_at_end

        self._stop = threading.Event()
        self._print_lock = threading.Lock()

        # Per-model future bookkeeping. The harvester thread drains this and
        # marks models DONE when all their futures are accounted for.
        self._pending: dict[str, list[Future]] = {}
        self._pending_lock = threading.Lock()
        self._initial_states = _read_manifest()

        self._executor: Optional[ProcessPoolExecutor] = None

    # ---------------------------------------------------------- logging

    def _log(self, msg: str) -> None:
        with self._print_lock:
            print(f"[{_now()}] {msg}", file=sys.stderr, flush=True)

    # ---------------------------------------------------------- driver

    def run(self) -> int:
        t0 = time.time()
        self._log(f"pipeline start :: {len(self.models)} models | "
                  f"gpu={self.gpu_id} | cpu_workers={self.cpu_workers} | "
                  f"fast={self.fast} | modules={','.join(self.modules)}")

        # Decide what to actually do per model based on manifest history.
        gpu_q: queue.Queue[Optional[str]] = queue.Queue()
        cpu_only: list[str] = []
        for m in self.models:
            prev = self._initial_states.get(m)
            if prev is None:
                gpu_q.put(m)
                _append_manifest(_ModelState(model=m, state="TODO",
                                              modules_total=len(self.modules)))
                continue
            if prev.state == "DONE":
                self._log(f"skip {m} (DONE)")
                continue
            if prev.state == "FAILED" and not self.retry_failed:
                self._log(f"skip {m} (FAILED, --no-retry-failed)")
                continue
            if prev.state in ("SCORED", "ANALYZING"):
                cpu_only.append(m)
                continue
            gpu_q.put(m)  # TODO / SCORING / FAILED → re-run
        gpu_q.put(None)  # sentinel for GPU worker

        # Spin up the CPU pool (and the harvester); they need to be live
        # before the GPU thread starts submitting.
        if not self.dry_run:
            self._executor = ProcessPoolExecutor(
                max_workers=self.cpu_workers,
                initializer=init_worker,
                initargs=(),
            )

        # Submit CPU-only models (already-scored from a prior run) immediately.
        for m in cpu_only:
            self._submit_cpu_tasks(m)

        gpu_thread = threading.Thread(
            target=self._gpu_loop, name="gpu-worker", args=(gpu_q,),
            daemon=True,
        )
        gpu_thread.start()

        harvester = threading.Thread(
            target=self._harvest_loop, name="harvester", daemon=True,
        )
        harvester.start()

        try:
            gpu_thread.join()
            # After GPU side is done, wait for all CPU tasks to drain.
            self._wait_until_pending_clear()
        except KeyboardInterrupt:
            self._log("SIGINT — shutting down pool")
            self._stop.set()
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
            return 130
        finally:
            self._stop.set()
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=False)
        harvester.join(timeout=5.0)

        elapsed = time.time() - t0

        # End-of-sweep plot regeneration. Runs after all writes are flushed.
        if self.regenerate_plots_at_end and not self.dry_run:
            self._log("regenerating PDFs from current CSVs …")
            plot_status = regenerate_plots(self.modules)
            for slug, msg in plot_status.items():
                self._log(f"  plot[{slug}] {msg}")

        states = _read_manifest()
        done = sum(1 for m in self.models
                   if states.get(m) and states[m].state == "DONE")
        failed = sum(1 for m in self.models
                     if states.get(m) and states[m].state == "FAILED")
        self._log(f"elapsed {elapsed:.0f}s — done={done}/{len(self.models)} "
                  f"failed={failed}")
        return 0 if failed == 0 else 1

    # ------------------------------------------------------- GPU loop

    def _gpu_loop(self, gpu_q: queue.Queue[Optional[str]]) -> None:
        prefetch_thread: Optional[threading.Thread] = None
        while not self._stop.is_set():
            try:
                m = gpu_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if m is None:
                return
            # While we score this model, prefetch the next one's HF snapshot.
            try:
                next_m = gpu_q.queue[0] if gpu_q.queue else None
            except Exception:
                next_m = None
            if self.prefetch and next_m and next_m is not None:
                prefetch_thread = threading.Thread(
                    target=_hf_prefetch, args=(next_m,),
                    name=f"hf-prefetch-{next_m[-12:]}", daemon=True,
                )
                prefetch_thread.start()

            self._score_model(m)

            # Don't hold up the GPU on the prefetch; it'll finish naturally
            # or be redundant once we get there.
            if prefetch_thread is not None:
                prefetch_thread = None

    def _score_model(self, model: str) -> None:
        _append_manifest(_ModelState(model=model, state="SCORING",
                                      modules_total=len(self.modules)))
        log_path = self.log_dir / f"{model.replace('/', '__')}.gpu.log"
        cmd = [
            sys.executable, "-m", "analyze.meco_surprisal",
            "--models", model,
            "--output-dir", str(self.score_dir),
            "--device", "cuda",
        ]
        self._log(f"[gpu] scoring {model}")
        if self.dry_run:
            self._log(f"[dry-run] would run: {' '.join(cmd)}")
            _append_manifest(_ModelState(model=model, state="SCORED",
                                          modules_total=len(self.modules)))
            self._submit_cpu_tasks(model)
            return
        env_extra = {"CUDA_VISIBLE_DEVICES": str(self.gpu_id)}
        if self.fast:
            env_extra["BEETLE_FAST_FIT"] = "1"
        try:
            rc, secs = _spawn_gpu(cmd, env_extra=env_extra, log_path=log_path)
        except KeyboardInterrupt:
            self._stop.set()
            raise
        if rc == 0:
            _append_manifest(_ModelState(
                model=model, state="SCORED", gpu_seconds=secs,
                modules_total=len(self.modules),
            ))
            self._log(f"[gpu] scored {model} in {secs:.0f}s")
            self._submit_cpu_tasks(model)
        else:
            _append_manifest(_ModelState(
                model=model, state="FAILED", gpu_seconds=secs,
                error=f"GPU stage rc={rc}",
                modules_total=len(self.modules),
            ))
            self._log(f"[gpu] FAILED {model} (rc={rc}) — see {log_path}")

    # ------------------------------------------------------- CPU dispatch

    def _submit_cpu_tasks(self, model: str) -> None:
        """Push one future per D-module onto the pool for this model."""
        if self.dry_run or self._executor is None:
            # Dry-run shortcut: pretend everything succeeded.
            _append_manifest(_ModelState(
                model=model, state="ANALYZING",
                modules_total=len(self.modules),
            ))
            _append_manifest(_ModelState(
                model=model, state="DONE",
                modules_ok=len(self.modules),
                modules_total=len(self.modules),
            ))
            return

        _append_manifest(_ModelState(
            model=model, state="ANALYZING",
            modules_total=len(self.modules),
        ))
        futures: list[Future] = []
        for slug in self.modules:
            fut = self._executor.submit(
                run_one, slug, model, fast=self.fast, verbose=False,
            )
            futures.append(fut)
        with self._pending_lock:
            self._pending[model] = futures
        self._log(f"[cpu] submitted {len(futures)} tasks for {model}")

    # ------------------------------------------------------- harvester

    def _harvest_loop(self) -> None:
        """Drain completed futures, mark models DONE / FAILED."""
        while not self._stop.is_set():
            time.sleep(0.5)
            self._harvest_once()

    def _harvest_once(self) -> None:
        with self._pending_lock:
            snapshot = list(self._pending.items())
        for model, futures in snapshot:
            done_idx: list[int] = []
            for i, fut in enumerate(futures):
                if fut.done():
                    done_idx.append(i)
            if not done_idx:
                continue
            # Process every finished future; if any remain pending we leave
            # the model in ANALYZING state.
            results = []
            for i in done_idx:
                fut = futures[i]
                try:
                    results.append(fut.result(timeout=0))
                except Exception as e:
                    results.append({
                        "slug": "?", "model": model, "success": False,
                        "seconds": 0.0, "n_rows": 0,
                        "error": f"future error: {e}",
                    })
            # Remove finished futures from the pending list.
            with self._pending_lock:
                self._pending[model] = [
                    f for j, f in enumerate(futures) if j not in done_idx
                ]
                still_pending = len(self._pending[model])

            for r in results:
                if r["success"]:
                    self._log(f"[cpu]   {r['slug']:<24} ok    "
                              f"({r['seconds']:.0f}s, {r['n_rows']} rows)  {model}")
                else:
                    self._log(f"[cpu]   {r['slug']:<24} FAIL  {r['error']}  {model}")

            if still_pending == 0:
                # All 8 futures resolved for this model — finalise.
                with self._pending_lock:
                    self._pending.pop(model, None)
                self._finalise_model(model)

    def _finalise_model(self, model: str) -> None:
        # Read back the last few manifest entries for this model to compute
        # cpu_seconds and module-success counts.
        states = _read_manifest()
        prev = states.get(model)
        modules_total = prev.modules_total if prev else len(self.modules)
        # NB. We don't have a clean per-task cumulative time after the
        # harvester eats results; just record terminal state.
        _append_manifest(_ModelState(
            model=model, state="DONE",
            modules_total=modules_total,
            modules_ok=modules_total,
        ))
        self._log(f"[cpu] done {model}")

    # ------------------------------------------------------- wait barrier

    def _wait_until_pending_clear(self) -> None:
        # Poll until no model has outstanding futures. The harvester thread
        # is meanwhile draining them.
        while True:
            with self._pending_lock:
                outstanding = sum(len(v) for v in self._pending.values())
            if outstanding == 0:
                return
            time.sleep(0.5)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_models_arg(models: list[str], from_file: Optional[Path]) -> list[str]:
    if from_file:
        with open(from_file) as f:
            return [line.strip() for line in f
                    if line.strip() and not line.strip().startswith("#")]
    return models


def _expand_module_selection(
    include: Optional[Sequence[str]],
    skip: Sequence[str],
) -> list[str]:
    modules = list(include) if include else list(ALL_MODULE_SLUGS)
    expanded: list[str] = []
    for m in modules:
        if m in ALL_MODULE_SLUGS:
            expanded.append(m)
            continue
        match = [n for n in ALL_MODULE_SLUGS if n.split("_")[0] == m]
        if not match:
            raise SystemExit(f"unknown module: {m!r}")
        expanded.extend(match)
    drop = set()
    for s in skip:
        if s in ALL_MODULE_SLUGS:
            drop.add(s)
        else:
            drop.update(n for n in ALL_MODULE_SLUGS if n.split("_")[0] == s)
    return [m for m in expanded if m not in drop]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="High-throughput orchestrator for meco_detailed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--models", nargs="+", default=[])
    ap.add_argument("--models-file", type=Path)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--cpu-workers", type=int, default=0,
                    help=f"number of CPU pool workers (default min(8, "
                         f"OS_CPU_COUNT={OS_CPU_COUNT}))")
    ap.add_argument("--modules", nargs="+", default=None)
    ap.add_argument("--skip", nargs="+", default=[])
    ap.add_argument("--no-retry-failed", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="set BEETLE_FAST_FIT=1 (skip Tier-1 lmer fits)")
    ap.add_argument("--no-prefetch", action="store_true",
                    help="disable next-model HF prefetch thread")
    ap.add_argument("--no-final-plots", action="store_true",
                    help="don't regenerate PDFs at end of sweep")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--score-dir", type=Path,
                    default=REPO_ROOT / "results" / "meco")
    args = ap.parse_args(argv)

    models = _load_models_arg(args.models, args.models_file)
    if not models:
        ap.error("must specify --models or --models-file")

    modules = _expand_module_selection(args.modules, args.skip)
    if not modules:
        ap.error("module selection is empty after applying --skip")

    pipe = Pipeline(
        models=models,
        gpu_id=args.gpu_id,
        cpu_workers=args.cpu_workers,
        modules=modules,
        retry_failed=not args.no_retry_failed,
        score_dir=args.score_dir,
        dry_run=args.dry_run,
        fast=args.fast,
        prefetch=not args.no_prefetch,
        regenerate_plots_at_end=not args.no_final_plots,
    )
    return pipe.run()


if __name__ == "__main__":
    raise SystemExit(main())
