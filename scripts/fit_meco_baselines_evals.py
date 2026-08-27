"""Fit canonical MECO ΔlogLik for the baseline-cohort surprisal CSVs that live
under ``results/evals/meco/`` (frontier + Chinese-cohort + llm-jp baselines).

The original ``fit_meco_missing_baselines.py`` only scans
``results/evals_checkpoints/meco/`` so the new ``evals/meco/`` outputs are
invisible to it. This script fills that gap: it loads every
``surprisals_*.csv`` under ``results/evals/meco/`` whose schema carries the
canonical columns and feeds them into ``compute_delta_loglik_all_models``.

Output: appends new (model, lang) rows to
``results/meco/canonical_deltalogl.csv`` (the file the paper figures read).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyze.meco_canonical import compute_delta_loglik_all_models, CANONICAL_LANGS

# Filename convention from evals/eval_meco.py:
#   surprisals_<org>_<repo>.csv  (org and repo separated by underscore)
_FNAME_RE = re.compile(r"^surprisals_(?P<org>[^/]+?)_(?P<repo>.+)\.csv$")


def _repo_from_filename(p: Path) -> str | None:
    m = _FNAME_RE.match(p.name)
    if not m:
        return None
    return f"{m.group('org')}/{m.group('repo')}"


def load_evals_meco(evals_dir: Path) -> pd.DataFrame:
    rows = []
    for csv in sorted(evals_dir.glob("surprisals_*.csv")):
        repo = _repo_from_filename(csv)
        if repo is None:
            continue
        try:
            df = pd.read_csv(csv)
        except Exception as e:
            print(f"  skip {csv.name}: {e}", file=sys.stderr)
            continue
        if df.empty or "surprisal" not in df.columns:
            continue
        # Some files already carry a 'model' column — overwrite to ensure
        # consistency with the org/repo derived from the filename.
        df = df.copy()
        df["model"] = repo
        if "step" not in df.columns:
            df["step"] = 0  # final/main checkpoint
        df["itemid"] = df["itemid"].astype(str)
        df["wordnum"] = pd.to_numeric(df["wordnum"], errors="coerce")
        df["surprisal"] = pd.to_numeric(df["surprisal"], errors="coerce")
        df = df.dropna(subset=["wordnum", "surprisal"])
        df["wordnum"] = df["wordnum"].astype(int)
        rows.append(df[["model", "step", "itemid", "wordnum", "surprisal"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evals-dir", type=Path,
                    default=ROOT / "results" / "evals" / "meco")
    ap.add_argument("--cache",     type=Path,
                    default=ROOT / "results" / "meco" / "canonical_deltalogl.csv")
    ap.add_argument("--lowercase", action="store_true", default=True,
                    help="Lowercase model ids to match existing cache keys.")
    ap.add_argument("--no-lowercase", action="store_false", dest="lowercase")
    args = ap.parse_args()

    surp = load_evals_meco(args.evals_dir)
    if surp.empty:
        print(f"[fit-baselines] no surprisal CSVs under {args.evals_dir}", file=sys.stderr)
        sys.exit(2)

    if args.lowercase:
        surp["model"] = surp["model"].str.lower()

    cache = pd.read_csv(args.cache) if args.cache.exists() else pd.DataFrame()
    have = set(zip(cache.get("model", []), cache.get("lang", [])))

    todo = []
    for m in surp["model"].unique():
        for lang in CANONICAL_LANGS:
            if (m, lang) not in have:
                todo.append((m, lang))
    print(f"[fit-baselines] {surp['model'].nunique()} surprisal models loaded; "
          f"{len(todo)} (model, lang) pairs missing from cache; "
          f"{len(have)} already cached")
    for m, lang in todo[:30]:
        print(f"  pending: {m} @ {lang}")
    if len(todo) > 30:
        print(f"  … and {len(todo) - 30} more")

    if not todo:
        print("[fit-baselines] nothing to do")
        return

    compute_delta_loglik_all_models(
        surp,
        cache_path=args.cache,
        overwrite=False,
        backend="statsmodels",
        verbose=True,
    )
    print("[fit-baselines] done")


if __name__ == "__main__":
    main()
