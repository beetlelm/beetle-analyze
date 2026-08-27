"""
analyze/results_push.py — Continuous-push helper for Colab notebooks.

The Colab pipeline writes CSVs into ``$BEETLE_RESULTS_ROOT`` (a working
checkout of the repo) and PDFs into ``$BEETLE_PLOTS_ROOT``. Between stages
we want to commit and push so that:

  - if Colab times out, partial results are not lost;
  - the user can pull the figures locally without copying files manually;
  - the plot-regenerator always sees fresh CSVs.

This module provides a single entry point ``push(stage, paths=None)`` that:

  1. ``git add`` the listed paths (or the entire results-paper + plots tree).
  2. Creates a commit with a stage-tagged message; if there is nothing to
     commit, returns silently.
  3. ``git push`` to the configured remote / branch (default
     ``origin emnlp-interp``).
  4. Logs to stdout so the notebook cell shows what landed.

Everything is best-effort: a failed push does not crash the pipeline (the
next stage retries, and the final push is a hard guarantee anyway).

Set ``BEETLE_PUSH_DISABLED=1`` to no-op (useful for offline runs).
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Iterable


def _run(cmd: str, cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        shlex.split(cmd), cwd=str(cwd),
        capture_output=True, text=True,
    )
    out = (proc.stdout + proc.stderr).strip()
    return proc.returncode, out


def _repo_root() -> Path:
    """Return the working git checkout to push from.

    Order:
      1. ``$BEETLE_REPO_ROOT`` if set.
      2. CWD if it is a git repo.
      3. raise — no fallback because we don't want to commit into the
         site-packages tree where the wheel was installed.
    """
    env = os.environ.get("BEETLE_REPO_ROOT")
    if env and (Path(env) / ".git").exists():
        return Path(env)
    cwd = Path.cwd()
    if (cwd / ".git").exists():
        return cwd
    raise RuntimeError(
        "no git checkout found — set $BEETLE_REPO_ROOT to your "
        "beetle-analyze clone before calling push()"
    )


def _default_paths(repo_root: Path) -> list[str]:
    out: list[str] = []
    for rel in ("results/evals_paper", "plots"):
        if (repo_root / rel).exists():
            out.append(rel)
    return out


def push(stage: str,
         paths: Iterable[str | Path] | None = None,
         *,
         remote: str = "origin",
         branch: str | None = None,
         retries: int = 2) -> bool:
    """Stage, commit, and push results for ``stage``.

    Returns True on success, False if nothing committed / push failed.
    """
    if os.environ.get("BEETLE_PUSH_DISABLED") == "1":
        print(f"[push] disabled — skipping {stage}")
        return False

    root = _repo_root()
    branch = branch or os.environ.get("BEETLE_PUSH_BRANCH") or "emnlp-interp"
    rels = [str(p) for p in (paths or _default_paths(root))]
    if not rels:
        print(f"[push] {stage}: nothing to push (no results/plots dirs)")
        return False

    add_cmd = "git add -- " + " ".join(shlex.quote(r) for r in rels)
    rc, out = _run(add_cmd, root)
    if rc != 0:
        print(f"[push] {stage}: git add failed → {out}")
        return False

    rc, out = _run("git diff --cached --quiet", root)
    if rc == 0:
        print(f"[push] {stage}: nothing changed since last push")
        return False

    msg = f"emnlp-interp: {stage} results @ {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"
    rc, out = _run(f"git commit -m {shlex.quote(msg)}", root)
    if rc != 0:
        print(f"[push] {stage}: commit failed → {out}")
        return False
    print(f"[push] {stage}: committed — {msg}")

    for attempt in range(1, retries + 1):
        rc, out = _run(f"git push {remote} HEAD:{branch}", root)
        if rc == 0:
            print(f"[push] {stage}: pushed to {remote}/{branch}")
            return True
        # Non-fast-forward: another machine (e.g. GPU node) pushed in
        # parallel. Rebase our new commit onto the remote tip and retry.
        last = out.splitlines()[-1] if out else "unknown"
        print(f"[push] {stage}: push attempt {attempt}/{retries} failed → {last}")
        if "fast-forward" in out or "rejected" in out:
            rc2, out2 = _run(f"git fetch {remote} {branch}", root)
            if rc2 == 0:
                rc3, out3 = _run(f"git rebase --autostash FETCH_HEAD", root)
                if rc3 == 0:
                    print(f"[push] {stage}: rebased onto {remote}/{branch}; retrying push")
                    continue
                print(f"[push] {stage}: rebase failed → "
                      f"{out3.splitlines()[-1] if out3 else 'unknown'}; aborting")
                _run("git rebase --abort", root)
                return False
        time.sleep(2)
    return False
