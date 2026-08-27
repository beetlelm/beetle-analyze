#!/usr/bin/env python3
"""scripts/b3_meco_deltalogl_parallel.py

Parallel version of the MECO ΔlogLik fit step for the b3 checkpoint sweep.

The canonical pipeline (analyze.meco_canonical) runs sequentially with a
shared CSV cache, which is too slow when we have ~45 (regime, ckpt) ×
~1 lang fits and each MixedLM fit on n=30K obs takes ~1 min on this box.

This script:
  1. Loads MECO L2 RT (joined wave-1 + wave-2) once.
  2. Walks every per-checkpoint surprisal CSV under
     results/evals_b3_checkpoints/meco/, sets ``model = '<regime>@step-<N>'``.
  3. Builds a worklist of (pseudo-model, lang) pairs not yet in the
     ΔlogLik cache.
  4. Fits in a process pool (default 8 workers), writing each row to the
     cache as it returns.

Output: appends to results/evals_b3_checkpoints/meco_canonical_deltalogl.csv
(same schema as analyze.meco_canonical writes).

Env:
  LANGS="du"        # space-separated subset of (du, ch_s, ge); default du
  WORKERS=8         # process-pool size
"""
from __future__ import annotations

import glob
import os
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyze.meco_canonical import (  # noqa: E402
    CANONICAL_LANGS,
    add_word_features,
    attach_spillover,
    drop_sentence_boundaries,
    fit_canonical_pair,
    join_model_surprisal,
    load_joined_rt,
)

OUT_ROOT = ROOT / "results/evals_b3_checkpoints"
MECO_DIR = OUT_ROOT / "meco"
CACHE = OUT_ROOT / "meco_canonical_deltalogl.csv"

REGIME_PREFIX = "beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-2b-nld-eng-"


def _regime(repo_slug: str) -> str:
    s = repo_slug.lower()
    if REGIME_PREFIX in s:
        return s.split(REGIME_PREFIX, 1)[1]
    return s.split("/")[-1]


def _surprisal_file_iter():
    for f in sorted(glob.glob(str(MECO_DIR / "surprisals_*__step-*.csv"))):
        m = re.match(r"surprisals_(.+)__step-(\d+)\.csv$", os.path.basename(f))
        if not m:
            continue
        repo_slug, step = m.group(1), int(m.group(2))
        yield f, _regime(repo_slug), step


def _build_surprisal_long() -> pd.DataFrame:
    parts = []
    for f, regime, step in _surprisal_file_iter():
        df = pd.read_csv(f)
        if df.empty:
            continue
        df = df.copy()
        df["regime"] = regime
        df["step"] = step
        df["model"] = f"{regime}@step-{step}"
        parts.append(df)
    if not parts:
        raise SystemExit(f"no MECO surprisal CSVs under {MECO_DIR}")
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# One-fit worker. Self-contained: pickles the merged & spillover-attached
# subset for a single (model, lang), so the child process does no global setup
# beyond importing fit_canonical_pair.
# ---------------------------------------------------------------------------

def _fit_one(payload: tuple[str, str, bytes]) -> dict:
    model, lang, sub_pkl = payload
    import io, pickle  # noqa: E401

    sub = pickle.loads(sub_pkl)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = fit_canonical_pair(sub, backend="statsmodels")
            row = {
                "model": model, "lang": lang, **fit,
                "n_obs": int(len(sub)),
                "n_subjects": int(sub["subid"].nunique()),
                "notes": "",
            }
        except Exception as e:
            row = {
                "model": model, "lang": lang,
                "delta_loglik": np.nan,
                "loglik_full": np.nan, "loglik_baseline": np.nan,
                "beta_surprisal": np.nan, "beta_surprisal_se": np.nan,
                "n_obs": int(len(sub)),
                "n_subjects": int(sub["subid"].nunique()),
                "fit_backend": "error",
                "fit_backend_baseline": "error",
                "notes": f"{e.__class__.__name__}: {e}",
            }
    row["_seconds"] = time.time() - t0
    return row


def main() -> None:
    langs_env = os.environ.get("LANGS", "du").strip()
    langs = CANONICAL_LANGS if langs_env == "ALL" else tuple(s for s in langs_env.split() if s)
    workers = int(os.environ.get("WORKERS", "8"))
    print(f"[parallel-deltalogl] langs={list(langs)} workers={workers}")

    surp = _build_surprisal_long()
    print(f"[parallel-deltalogl] loaded {len(surp):,} surprisal rows / "
          f"{surp['model'].nunique()} pseudo-models")

    rt = add_word_features(load_joined_rt())

    # Determine work to do (skip already-cached rows).
    cached_pairs = set()
    if CACHE.exists():
        old = pd.read_csv(CACHE)
        cached_pairs = set(zip(old.get("model", []), old.get("lang", [])))
        print(f"[parallel-deltalogl] {len(cached_pairs)} already-cached rows; "
              f"resuming")
    rows = list(old.to_dict("records")) if cached_pairs else []

    # Build per-(model, lang) subsets in main, pickle and ship to workers.
    import pickle
    payloads: list[tuple[str, str, bytes]] = []
    for model in sorted(surp["model"].unique()):
        merged = join_model_surprisal(rt, surp, model)
        if merged.empty:
            continue
        for lang in langs:
            if (model, lang) in cached_pairs:
                continue
            sub = merged[merged["lang"] == lang].dropna(
                subset=["firstrun_dur", "Surprisal", "wlen", "freq", "subid"]
            )
            sub = attach_spillover(sub)
            sub = drop_sentence_boundaries(sub)
            if len(sub) < 200:
                rows.append({
                    "model": model, "lang": lang,
                    "delta_loglik": np.nan,
                    "loglik_full": np.nan, "loglik_baseline": np.nan,
                    "beta_surprisal": np.nan, "beta_surprisal_se": np.nan,
                    "n_obs": int(len(sub)),
                    "n_subjects": int(sub["subid"].nunique()),
                    "fit_backend": "skipped",
                    "fit_backend_baseline": "skipped",
                    "notes": f"n={len(sub)} < 200",
                })
                continue
            payloads.append((model, lang, pickle.dumps(sub)))

    print(f"[parallel-deltalogl] dispatching {len(payloads)} fits to {workers} workers")

    def _flush() -> None:
        df = pd.DataFrame(rows).drop_duplicates(subset=["model", "lang"], keep="last")
        tmp = CACHE.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(CACHE)

    if not payloads:
        print("[parallel-deltalogl] nothing to do")
        _flush()
        return

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fit_one, p): (p[0], p[1]) for p in payloads}
        n_done = 0
        for fut in as_completed(futures):
            model, lang = futures[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {
                    "model": model, "lang": lang,
                    "delta_loglik": np.nan,
                    "loglik_full": np.nan, "loglik_baseline": np.nan,
                    "beta_surprisal": np.nan, "beta_surprisal_se": np.nan,
                    "n_obs": -1, "n_subjects": -1,
                    "fit_backend": "error",
                    "fit_backend_baseline": "error",
                    "notes": f"worker-exception: {e.__class__.__name__}: {e}",
                }
            secs = row.pop("_seconds", float("nan"))
            rows.append(row)
            n_done += 1
            print(
                f"[parallel-deltalogl] {n_done:3d}/{len(payloads)} {model} @ {lang} "
                f"ΔLL={row.get('delta_loglik'):.2f} ({secs:.0f}s, "
                f"backend={row.get('fit_backend')})",
                flush=True,
            )
            _flush()

    print(f"[parallel-deltalogl] done in {time.time() - t_start:.0f}s; "
          f"cache → {CACHE}")


if __name__ == "__main__":
    main()
