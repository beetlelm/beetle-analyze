"""Fit canonical MECO ΔlogLik for baseline LLMs that aren't in the cache yet.

The existing canonical_deltalogl.csv uses lowercased model ids; the
filename-based surprisal loader produces mixed-case ids (Beetle-FineWeb/…,
google/gemma-2-9b, etc). To avoid accidentally re-fitting all Beetle models,
we lowercase model ids in the loaded surprisal frame so the (model, lang)
cache key matches the existing rows. compute_delta_loglik_all_models then
only fits the new (model, lang) pairs.

Run:
  venv/bin/python scripts/fit_meco_missing_baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from analyze.meco_canonical import compute_delta_loglik_all_models
from plots.checkpoint_loaders import load_meco_checkpoint_files


def main() -> None:
    surp = load_meco_checkpoint_files()
    if surp.empty:
        print("[fit-missing] no surprisal CSVs found", file=sys.stderr)
        sys.exit(2)

    # Lowercase model ids so they match the existing cache keys.
    surp["model"] = surp["model"].str.lower()

    # Keep only the final-step row per model (canonical pipeline convention).
    last_step = surp.groupby("model")["step"].max().rename("_last")
    surp = surp.join(last_step, on="model")
    surp = surp[surp["step"] == surp["_last"]].drop(columns=["_last"])

    cache = ROOT / "results" / "meco" / "canonical_deltalogl.csv"
    existing = pd.read_csv(cache) if cache.exists() else pd.DataFrame()
    have = set(zip(existing.get("model", []), existing.get("lang", [])))
    todo = []
    for m in surp["model"].unique():
        for lang in ("du", "ge", "ch_s"):
            if (m, lang) not in have:
                todo.append((m, lang))
    print(f"[fit-missing] {len(todo)} (model, lang) pairs to fit; "
          f"{len(have)} already cached")
    for m, lang in todo:
        print(f"  pending: {m} @ {lang}")

    if not todo:
        print("[fit-missing] nothing to do; cache is complete")
        return

    compute_delta_loglik_all_models(
        surp,
        cache_path=cache,
        overwrite=False,
        backend="statsmodels",
        verbose=True,
    )
    print("[fit-missing] done")


if __name__ == "__main__":
    main()
