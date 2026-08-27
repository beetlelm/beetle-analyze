#!/usr/bin/env python3
"""evals/eval_cefr_perplexity.py

§4.4 CEFR-graded text perplexity gradient.

For each (model, CEFR level) pair compute mean per-token bits-per-token (BPT)
on UniversalCEFR/cefr_sp_en (sentence-level CEFR labels). Output one CSV per
model with one row per CEFR level + bootstrap CI, plus a global summary CSV.

Usage:
    python -m evals.eval_cefr_perplexity \
        --models Beetle-FineWeb-2B/...:step-5862 Beetle-HumanScale/...:main \
        --output-root results/evals_paper/04d_cefr_ppl \
        --batch-size 32
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402

from evals.model_loader import (  # noqa: E402
    DEVICE, load_model, unload_model, setup_logging,
)
from evals.eval_bliss import score_sentences_bpt  # noqa: E402

logger = logging.getLogger(__name__)
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


def load_cefr_sp(min_chars: int = 8, max_per_level: int = 0) -> dict[str, list[str]]:
    ds = load_dataset("UniversalCEFR/cefr_sp_en", split="train")
    by_level: dict[str, list[str]] = defaultdict(list)
    for r in ds:
        lvl = (r.get("cefr_level") or "").strip().upper()
        text = (r.get("text") or "").strip()
        if lvl in CEFR_LEVELS and len(text) >= min_chars:
            by_level[lvl].append(text)
    if max_per_level > 0:
        for k in by_level:
            by_level[k] = by_level[k][:max_per_level]
    return by_level


def bootstrap_mean_ci(xs: list[float], n_boot: int = 1000, alpha: float = 0.05,
                     seed: int = 0) -> tuple[float, float, float]:
    if not xs:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(xs, dtype=np.float64)
    n = len(arr)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = arr[idx].mean()
    mean = float(arr.mean())
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return mean, lo, hi


def evaluate_pair(repo: str, revision: str, by_level: dict[str, list[str]],
                  output_root: Path, batch_size: int) -> None:
    pair = load_model(repo, DEVICE, revision=revision)
    if pair is None:
        logger.error("[%s @ %s] load failed", repo, revision)
        return
    model, tok = pair
    repo_safe = repo.replace("/", "_")
    rev_tag = "" if revision in ("main", "", None) else f"__rev={revision}"
    out_dir = output_root / f"{repo_safe}{rev_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for lvl in CEFR_LEVELS:
        sents = by_level.get(lvl, [])
        if not sents:
            continue
        t0 = time.time()
        bpts = score_sentences_bpt(model, tok, sents, DEVICE, batch_size)
        bpts = [x for x in bpts if math.isfinite(x)]
        m, lo, hi = bootstrap_mean_ci(bpts)
        ppl = float(np.exp(np.array(bpts) * math.log(2)).mean()) if bpts else float("nan")
        rows.append({
            "model": repo, "revision": revision, "cefr": lvl,
            "n": len(bpts), "mean_bpt": m, "lo_bpt": lo, "hi_bpt": hi,
            "mean_ppl": ppl,
        })
        logger.info("[%s @ %s] %s n=%d mean_bpt=%.3f ppl=%.2f (%.1fs)",
                    repo, revision, lvl, len(bpts), m, ppl, time.time() - t0)
    out_csv = out_dir / "cefr_ppl.csv"
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        logger.info("wrote %s", out_csv)
    unload_model(model, tok)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True,
                   help="<repo>:<revision> pairs; revision defaults to main")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-per-level", type=int, default=400)
    p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    p.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    args = p.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    setup_logging("cefr_ppl", args.output_root, args.rank)
    logger.info("cefr_ppl rank %d/%d", args.rank, args.world_size)

    by_level = load_cefr_sp(max_per_level=args.max_per_level)
    logger.info("loaded cefr_sp: %s", {k: len(v) for k, v in by_level.items()})

    pairs = []
    for spec in args.models:
        if ":" in spec:
            repo, rev = spec.rsplit(":", 1)
        else:
            repo, rev = spec, "main"
        pairs.append((repo, rev))
    pairs = [p for i, p in enumerate(pairs) if i % args.world_size == args.rank]
    for repo, rev in pairs:
        try:
            evaluate_pair(repo, rev, by_level, args.output_root, args.batch_size)
        except Exception:
            logger.exception("[%s @ %s] failed", repo, rev)
    logger.info("cefr_ppl done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
