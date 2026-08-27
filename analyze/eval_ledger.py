"""
analyze/eval_ledger.py — "what has been evaluated" ledger.

Each evaluation stage (PPL, BLiMP, MECO, convergence, baseline, …) records a
single JSONL line to ``results/ledger/rank_{rank}.jsonl`` after a successful
write. A merger consolidates the per-rank shards into a single
``results/ledger/ledger.jsonl`` and a deduped ``results/ledger/ledger.json``
keyed by ``stage/repo/checkpoint/benchmark/lang``.

Design choices
--------------
* Per-rank JSONL with ``O_APPEND`` means many ranks can write concurrently
  without locking: POSIX guarantees atomic appends for writes up to
  ``PIPE_BUF`` (≥ 512 bytes on all real filesystems), which comfortably
  covers one ledger record.
* The CSVs written by each stage remain the ground truth. The ledger is a
  *fast index* — ``is_done()`` short-circuits expensive model loads that the
  existing CSV-existence gate would otherwise detect only after data prep.
* ``backfill()`` scans the existing ``results/ppl/``, ``results/blimp_*/``,
  ``results/meco*/`` trees and emits ledger rows so work done before this
  module existed is not re-computed.

Usage
-----
From Python (inside a stage)::

    from analyze.eval_ledger import record, is_done
    if is_done("ppl", repo, ckpt, "flores", "nld"):
        continue
    ...  # do work
    record("ppl", repo, ckpt, "flores", "nld", "ok")

From the shell::

    python -m analyze.eval_ledger merge   --output results
    python -m analyze.eval_ledger backfill --output results
    python -m analyze.eval_ledger report   --output results
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

__all__ = [
    "record",
    "is_done",
    "merge",
    "backfill",
    "report",
    "ledger_key",
    "reset_cache",
]

_log = logging.getLogger("eval_ledger")


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

def _ledger_dir(output_root: Path | str = "results") -> Path:
    p = Path(output_root) / "ledger"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _rank_file(output_root: Path | str, rank: int) -> Path:
    return _ledger_dir(output_root) / f"rank_{rank}.jsonl"


def _merged_jsonl(output_root: Path | str) -> Path:
    return _ledger_dir(output_root) / "ledger.jsonl"


def _merged_json(output_root: Path | str) -> Path:
    return _ledger_dir(output_root) / "ledger.json"


# ─────────────────────────────────────────────────────────────────────────────
# Key helpers
# ─────────────────────────────────────────────────────────────────────────────

def ledger_key(
    stage: str,
    repo: str,
    checkpoint: Optional[str] = None,
    benchmark: Optional[str] = None,
    lang: Optional[str] = None,
) -> str:
    """Canonical string key used to dedupe and look up ledger rows."""
    return "/".join([
        stage or "",
        repo or "",
        checkpoint or "",
        benchmark or "",
        lang or "",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Writer
# ─────────────────────────────────────────────────────────────────────────────

def record(
    stage: str,
    repo: str,
    checkpoint: Optional[str] = None,
    benchmark: Optional[str] = None,
    lang: Optional[str] = None,
    status: str = "ok",
    extra: Optional[dict] = None,
    rank: int = 0,
    output_root: Path | str = "results",
) -> None:
    """Append a single JSONL record. Safe under concurrent ranks."""
    row = {
        "ts":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage":      stage,
        "repo":       repo,
        "checkpoint": checkpoint,
        "benchmark":  benchmark,
        "lang":       lang,
        "status":     status,
        "key":        ledger_key(stage, repo, checkpoint, benchmark, lang),
    }
    if extra:
        row["extra"] = extra

    path = _rank_file(output_root, rank)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    # O_APPEND gives atomic appends for small writes across ranks.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


# ─────────────────────────────────────────────────────────────────────────────
# Reader / cache
# ─────────────────────────────────────────────────────────────────────────────

_READ_CACHE: dict[str, set[str]] = {}


def reset_cache() -> None:
    """Drop the in-process ledger cache (use after merge/backfill)."""
    _READ_CACHE.clear()


def _load_done_keys(output_root: Path | str = "results") -> set[str]:
    """Return the set of ``ledger_key`` strings that are recorded as ok.

    Reads the merged JSONL if present; otherwise unions every per-rank shard.
    Cached per ``output_root`` for the life of the process.
    """
    key = str(Path(output_root).resolve())
    cached = _READ_CACHE.get(key)
    if cached is not None:
        return cached

    done: set[str] = set()
    merged = _merged_jsonl(output_root)
    files: list[Path] = []
    if merged.exists():
        files.append(merged)
    files.extend(sorted(_ledger_dir(output_root).glob("rank_*.jsonl")))

    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("status", "ok") != "ok":
                        continue
                    k = row.get("key") or ledger_key(
                        row.get("stage", ""),
                        row.get("repo", ""),
                        row.get("checkpoint"),
                        row.get("benchmark"),
                        row.get("lang"),
                    )
                    done.add(k)
        except FileNotFoundError:
            continue

    _READ_CACHE[key] = done
    return done


def is_done(
    stage: str,
    repo: str,
    checkpoint: Optional[str] = None,
    benchmark: Optional[str] = None,
    lang: Optional[str] = None,
    output_root: Path | str = "results",
) -> bool:
    """Check whether a given eval entry is already recorded as ok."""
    return ledger_key(stage, repo, checkpoint, benchmark, lang) in _load_done_keys(output_root)


# ─────────────────────────────────────────────────────────────────────────────
# Merge
# ─────────────────────────────────────────────────────────────────────────────

def merge(output_root: Path | str = "results") -> Path:
    """Concatenate ``rank_*.jsonl`` → ``ledger.jsonl`` + deduped ``ledger.json``.

    Later records for the same key win (timestamp-ordered). Always safe to
    re-run.
    """
    ld = _ledger_dir(output_root)
    shards = sorted(ld.glob("rank_*.jsonl"))

    by_key: dict[str, dict] = {}
    for shard in shards:
        with shard.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = row.get("key") or ledger_key(
                    row.get("stage", ""),
                    row.get("repo", ""),
                    row.get("checkpoint"),
                    row.get("benchmark"),
                    row.get("lang"),
                )
                prev = by_key.get(k)
                if prev is None or row.get("ts", "") >= prev.get("ts", ""):
                    by_key[k] = row

    merged_jsonl = _merged_jsonl(output_root)
    with merged_jsonl.open("w", encoding="utf-8") as fh:
        for k in sorted(by_key):
            fh.write(json.dumps(by_key[k], ensure_ascii=False) + "\n")

    merged_json = _merged_json(output_root)
    with merged_json.open("w", encoding="utf-8") as fh:
        json.dump(by_key, fh, ensure_ascii=False, indent=2, sort_keys=True)

    reset_cache()
    _log.info(
        "ledger merged: %d unique rows from %d rank shards → %s",
        len(by_key), len(shards), merged_jsonl,
    )
    return merged_jsonl


# ─────────────────────────────────────────────────────────────────────────────
# Backfill from existing CSVs
# ─────────────────────────────────────────────────────────────────────────────

def _safe_read_csv_rows(csv_path: Path) -> Iterable[dict]:
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                yield row
    except Exception:
        return


def backfill(output_root: Path | str = "results", rank: int = 0) -> int:
    """Emit ledger rows for work already present on disk.

    Scans:
      * ``results/ppl/ppl_<lang>_<slug>.csv``     → stage="ppl"
      * ``results/blimp_*/**/*.csv`` and          → stage="blimp"
        ``results/*_results.csv`` (legacy layout)
      * ``results/meco/surprisals_<slug>.csv``    → stage="meco"
      * ``results/meco/checkpoint_trajectory_<slug>.csv``
      * ``results/meco_beetle/**/*.csv``
      * ``results/convergence/ppl_traj_<slug>.csv``

    Returns the number of records written.
    """
    root = Path(output_root)
    written = 0

    # ── PPL ────────────────────────────────────────────────────────────────
    ppl_dir = root / "ppl"
    for csv_path in sorted(ppl_dir.glob("ppl_*.csv")):
        name = csv_path.stem  # ppl_<lang>_<slug>
        if name == "ppl_summary":
            continue
        parts = name.split("_", 2)
        if len(parts) < 3:
            continue
        _, lang, slug = parts
        # Prefer the 'repo' column if present; fall back to slug→repo.
        repo = None
        revisions: set[str] = set()
        for row in _safe_read_csv_rows(csv_path):
            if repo is None:
                repo = row.get("repo")
            rev = row.get("revision") or row.get("checkpoint")
            if rev:
                revisions.add(rev)
        if repo is None:
            repo = slug.replace("__", "/", 1)
        if not revisions:
            revisions = {""}
        for rev in revisions:
            record(
                stage="ppl", repo=repo, checkpoint=rev or None,
                benchmark="flores", lang=lang, status="ok",
                extra={"source": str(csv_path)},
                rank=rank, output_root=root,
            )
            written += 1

    # ── BLiMP / XNLI / XCOMPS (legacy layout: results/<bm>_results.csv) ────
    for csv_path in sorted(root.glob("*_results.csv")):
        bm = csv_path.stem.replace("_results", "")
        for row in _safe_read_csv_rows(csv_path):
            repo = row.get("model")
            if not repo:
                continue
            ckpt = row.get("checkpoint")
            lang = row.get("lang_code") or row.get("eval_language")
            record(
                stage="blimp", repo=repo, checkpoint=ckpt,
                benchmark=bm, lang=lang, status="ok",
                extra={"source": str(csv_path),
                       "accuracy": row.get("accuracy")},
                rank=rank, output_root=root,
            )
            written += 1

    # ── BLiMP (per-benchmark dir layout: results/blimp_*/<slug>.csv) ───────
    for blimp_dir in sorted(root.glob("blimp_*")):
        if not blimp_dir.is_dir():
            continue
        bm = blimp_dir.name
        for csv_path in sorted(blimp_dir.glob("*.csv")):
            for row in _safe_read_csv_rows(csv_path):
                repo = row.get("model") or row.get("repo")
                if not repo:
                    continue
                ckpt = row.get("checkpoint") or row.get("revision")
                lang = row.get("lang_code") or row.get("eval_language")
                record(
                    stage="blimp", repo=repo, checkpoint=ckpt,
                    benchmark=bm, lang=lang, status="ok",
                    extra={"source": str(csv_path),
                           "accuracy": row.get("accuracy")},
                    rank=rank, output_root=root,
                )
                written += 1

    # ── MECO (top-level and beetle subdir) ─────────────────────────────────
    for meco_dir_name in ("meco", "meco_beetle"):
        meco_dir = root / meco_dir_name
        if not meco_dir.exists():
            continue
        for csv_path in sorted(meco_dir.rglob("*.csv")):
            seen_keys: set[tuple[str, str | None]] = set()
            for row in _safe_read_csv_rows(csv_path):
                repo = row.get("model") or row.get("repo")
                if not repo:
                    # Some MECO CSVs encode the model in the filename only.
                    slug = csv_path.stem
                    for prefix in ("surprisals_", "checkpoint_trajectory_"):
                        if slug.startswith(prefix):
                            slug = slug[len(prefix):]
                            break
                    repo = slug.replace("__", "/", 1)
                ckpt = row.get("checkpoint") or row.get("revision")
                k = (repo, ckpt)
                if k in seen_keys:
                    continue
                seen_keys.add(k)
                record(
                    stage="meco", repo=repo, checkpoint=ckpt,
                    benchmark="meco_l2", lang=row.get("lang") or row.get("eval_language"),
                    status="ok", extra={"source": str(csv_path)},
                    rank=rank, output_root=root,
                )
                written += 1

    # ── Convergence PPL trajectories ───────────────────────────────────────
    conv_dir = root / "convergence"
    for csv_path in sorted(conv_dir.glob("ppl_traj_*.csv")):
        slug = csv_path.stem[len("ppl_traj_"):]
        repo = slug.replace("__", "/", 1)
        seen_keys: set[tuple[str, str | None, str | None]] = set()
        for row in _safe_read_csv_rows(csv_path):
            ckpt = row.get("revision") or row.get("step")
            lang = row.get("lang")
            k = (repo, ckpt, lang)
            if k in seen_keys:
                continue
            seen_keys.add(k)
            record(
                stage="convergence", repo=repo, checkpoint=str(ckpt) if ckpt else None,
                benchmark="ppl_traj", lang=lang, status="ok",
                extra={"source": str(csv_path)},
                rank=rank, output_root=root,
            )
            written += 1

    _log.info("backfill wrote %d ledger rows from existing CSVs", written)
    merge(root)
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def report(output_root: Path | str = "results") -> str:
    """Print a (model × stage/benchmark) done-count table.

    Returns the formatted string (also printed to stdout) for test use.
    """
    # Ensure merged view exists.
    merge(output_root)
    rows: list[dict] = []
    with _merged_jsonl(output_root).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # Count by (repo, stage[/benchmark]) pair.
    counts: dict[str, dict[str, int]] = {}
    by_stage_total: dict[str, int] = {}
    by_org_counts: dict[str, int] = {}
    for row in rows:
        repo = row.get("repo") or "(unknown)"
        stage = row.get("stage", "?")
        bm = row.get("benchmark")
        col = f"{stage}:{bm}" if bm else stage
        counts.setdefault(repo, {}).setdefault(col, 0)
        counts[repo][col] += 1
        by_stage_total[col] = by_stage_total.get(col, 0) + 1
        org = repo.split("/", 1)[0] if "/" in repo else "(no-org)"
        by_org_counts[org] = by_org_counts.get(org, 0) + 1

    stages_sorted = sorted(by_stage_total)
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"Evaluation ledger report ({output_root})")
    lines.append(f"Total records: {len(rows)} across {len(counts)} models")
    lines.append("")
    lines.append("Records per HF org:")
    for org in sorted(by_org_counts):
        lines.append(f"  {org:<30s} {by_org_counts[org]:>6d}")
    lines.append("")
    lines.append("Records per stage/benchmark:")
    for col in stages_sorted:
        lines.append(f"  {col:<30s} {by_stage_total[col]:>6d}")
    lines.append("")
    lines.append(f"{'repo':<70s}  " + "  ".join(f"{c:>10s}" for c in stages_sorted))
    lines.append("-" * (72 + len(stages_sorted) * 12))
    for repo in sorted(counts):
        cells = "  ".join(f"{counts[repo].get(c, 0):>10d}" for c in stages_sorted)
        lines.append(f"{repo:<70s}  {cells}")
    lines.append("=" * 80)
    out = "\n".join(lines)
    print(out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analyze.eval_ledger",
        description="Manage the beetle-analyze evaluation ledger.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("merge", "backfill", "report"):
        sp = sub.add_parser(name, help=f"{name} ledger")
        sp.add_argument("--output", default="results",
                        help="Output root (default: results)")
        if name == "backfill":
            sp.add_argument("--rank", type=int,
                            default=int(os.environ.get("RANK", "0")),
                            help="Rank tag for backfill shard (default: $RANK or 0)")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)
    root = Path(args.output)
    if args.cmd == "merge":
        merge(root)
    elif args.cmd == "backfill":
        backfill(root, rank=args.rank)
    elif args.cmd == "report":
        report(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
