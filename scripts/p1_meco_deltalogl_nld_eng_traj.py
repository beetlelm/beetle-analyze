#!/usr/bin/env python3
"""scripts/p1_meco_deltalogl_nld_eng_traj.py

P1 driver: fit MECO ΔlogLik per (model, step, lang) for **all** nld-eng
Beetle models, using the per-checkpoint surprisal CSVs that already exist
under:
    results/evals_checkpoints/meco/surprisals_*nld-eng*__step-*.csv
    results/evals_b3_checkpoints/meco/surprisals_*nld-eng*__step-*.csv

Generalises scripts/b3_meco_deltalogl_parallel.py (which only walks the
b3-fineweb-2b nld-eng sweep) to the full nld-eng population that
analyze/meco_dutch_grid.py expects, and writes one unified cache used by
the appendix figure regeneration step:

    results/meco/canonical_deltalogl_nld_eng_traj.csv

Schema mirrors analyze.meco_canonical writes (model, lang, loglik_full,
loglik_baseline, delta_loglik, beta_surprisal, beta_surprisal_se,
fit_backend, fit_backend_baseline, n_obs, n_subjects, notes) plus a
synthetic step column for downstream join.

The cache is flushed after every fit so a tmux autopush sidekick can
commit incremental progress to git.

Env
---
LANGS="du"        # space-separated subset of (du, ch_s, ge); default "du"
                  # (the headline appendix figure is Dutch-L1 readers only)
                  # use "ALL" for all three.
WORKERS=8         # process-pool size

Resumability
------------
On startup, any (model, lang) row already in the cache is skipped, so the
script can be killed and restarted without recomputing.
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

# Two source dirs for nld-eng surprisals (main checkpoint sweep + the b3
# fineweb-2b regime sweep that lives in its own subtree).
SURPR_DIRS = [
    ROOT / "results" / "evals_checkpoints" / "meco",
    ROOT / "results" / "evals_b3_checkpoints" / "meco",
]
CACHE = ROOT / "results" / "meco" / "canonical_deltalogl_nld_eng_traj.csv"

_FNAME_RE = re.compile(
    r"^surprisals_(?P<org>[a-z0-9-]+)_(?P<name>[a-z0-9-]+?)__step-(?P<step>\d+)\.csv$"
)


def _surprisal_file_iter():
    for d in SURPR_DIRS:
        if not d.exists():
            continue
        for f in sorted(glob.glob(str(d / "surprisals_*nld-eng*__step-*.csv"))):
            base = os.path.basename(f)
            m = _FNAME_RE.match(base)
            if not m:
                continue
            name = m.group("name")
            # Restrict to nld-eng (Dutch L1, English L2). Skip eng-nld and others.
            if "-nld-eng" not in name:
                continue
            step = int(m.group("step"))
            # Pseudo-model id = "<org>/<name>@step-<N>" — matches the form the
            # appendix figure code expects when joining cache → trajectory.
            pseudo_model = f"{m.group('org')}/{name}@step-{step}"
            yield f, pseudo_model, name, step


def _build_surprisal_long() -> pd.DataFrame:
    parts = []
    for f, pseudo_model, name, step in _surprisal_file_iter():
        df = pd.read_csv(f)
        if df.empty:
            continue
        df = df.copy()
        df["model"] = pseudo_model
        df["regime"] = name
        df["step"] = step
        parts.append(df)
    if not parts:
        raise SystemExit(
            "no nld-eng MECO surprisal CSVs under "
            f"{[str(d) for d in SURPR_DIRS]}"
        )
    return pd.concat(parts, ignore_index=True)


# One-fit worker — pickles the merged subset for one (model, lang).
def _fit_one(payload):
    model, lang, sub_pkl = payload
    import pickle

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
    print(f"[p1-deltalogl] langs={list(langs)} workers={workers}")
    print(f"[p1-deltalogl] cache={CACHE}")

    surp = _build_surprisal_long()
    print(f"[p1-deltalogl] loaded {len(surp):,} surprisal rows / "
          f"{surp['model'].nunique()} pseudo-models / "
          f"{surp['regime'].nunique()} regimes")

    rt = add_word_features(load_joined_rt())

    cached_pairs = set()
    rows = []
    if CACHE.exists():
        old = pd.read_csv(CACHE)
        cached_pairs = set(zip(old.get("model", []), old.get("lang", [])))
        rows = list(old.to_dict("records"))
        print(f"[p1-deltalogl] {len(cached_pairs)} already-cached rows; resuming")

    import pickle
    payloads = []
    skipped_short = 0
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
                skipped_short += 1
                continue
            payloads.append((model, lang, pickle.dumps(sub)))

    if skipped_short:
        print(f"[p1-deltalogl] {skipped_short} (model,lang) pairs skipped (n<200)")
    print(f"[p1-deltalogl] dispatching {len(payloads)} fits to {workers} workers")

    def _flush() -> None:
        df = pd.DataFrame(rows).drop_duplicates(subset=["model", "lang"], keep="last")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(CACHE)

    if not payloads:
        print("[p1-deltalogl] nothing to do")
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
            dll = row.get("delta_loglik")
            dll_str = f"{dll:.2f}" if isinstance(dll, (int, float)) and not np.isnan(dll) else "nan"
            print(
                f"[p1-deltalogl] {n_done:3d}/{len(payloads)} {model} @ {lang} "
                f"ΔLL={dll_str} ({secs:.0f}s, backend={row.get('fit_backend')})",
                flush=True,
            )
            _flush()

    print(f"[p1-deltalogl] done in {time.time() - t_start:.0f}s; cache → {CACHE}")


if __name__ == "__main__":
    main()
