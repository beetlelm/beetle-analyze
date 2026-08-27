"""
analyze/xling/cka_curriculum.py — §A geometry driver.

Per-(curriculum, layer) L1↔L2 linear-CKA on FLORES-200 dev pairs, with
percentile-bootstrap CIs over sentence pairs (Nguyen 2020 stability concern).

Outputs
-------
results/evals_paper/05_geometry/cka_per_layer.csv
  Columns: model_id, curriculum, lang_pair, layer, cka, cka_lo, cka_hi,
           n_pairs, stratum  (stratum ∈ {all, cognate, non_cognate,
                                          parallel, non_parallel})

Run modes
---------
- ``--mode parallel``      FLORES-200 parallel pairs (headline §A panel).
- ``--mode cognate-split`` cognate vs non-cognate strata (§A confound).
- ``--mode non-parallel``  syntax-matched non-parallel stimuli (§A confound).

Compute
-------
~4 GPU-hours base + ~1 hr cognate split + ~1 hr non-parallel ≈ 6 GPU-h
per the EMNLP plan. CPU-runnable on a small ``--n-pairs 32`` smoke test.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from analyze._interp_helpers import (
    DEFAULT_BEETLE_REPOS, RESULTS_PAPER, bootstrap_ci,
    device_for, per_layer_mean_activations, write_csv_atomic,
)
from analyze.xling.cka_full import linear_cka

logger = logging.getLogger(__name__)


def _bootstrap_cka(X: np.ndarray, Y: np.ndarray, *,
                   n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = float(linear_cka(X, Y))
    n_obs = X.shape[0]
    samples = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, size=n_obs)
        samples.append(float(linear_cka(X[idx], Y[idx])))
    arr = np.asarray(samples, dtype=float)
    lo, hi = np.quantile(arr, [0.025, 0.975])
    return point, float(lo), float(hi)


def run_one_model(repo: str, *, l1_iso: str, l2_iso: str,
                  curriculum: str, lang_pair: str,
                  l1_sentences: list[str], l2_sentences: list[str],
                  stratum: str = "all",
                  n_boot: int = 1000) -> list[dict]:
    """Compute per-layer L1↔L2 CKA for one model, with bootstrap CIs."""
    from analyze.ppl_utils import load_model_and_tokenizer

    device = device_for(allow_gpu=True)
    pair = load_model_and_tokenizer(repo)
    if pair is None:
        logger.warning("[cka_curriculum] %s failed to load — skipping", repo)
        return []
    model, tokenizer = pair
    model.to(device).eval()

    l1_acts = per_layer_mean_activations(model, tokenizer, l1_sentences)
    l2_acts = per_layer_mean_activations(model, tokenizer, l2_sentences)

    rows: list[dict] = []
    for layer in sorted(l1_acts.keys() & l2_acts.keys()):
        X = l1_acts[layer]
        Y = l2_acts[layer]
        if X.shape[0] != Y.shape[0]:
            n = min(X.shape[0], Y.shape[0])
            X, Y = X[:n], Y[:n]
        point, lo, hi = _bootstrap_cka(X, Y, n=n_boot, seed=42 + layer)
        rows.append({
            "model_id": repo, "curriculum": curriculum,
            "lang_pair": lang_pair, "layer": layer,
            "cka": point, "cka_lo": lo, "cka_hi": hi,
            "n_pairs": X.shape[0], "stratum": stratum,
        })
    return rows


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos", nargs="+",
                        default=list(DEFAULT_BEETLE_REPOS.values()))
    parser.add_argument("--curricula", nargs="+",
                        default=list(DEFAULT_BEETLE_REPOS.keys()))
    parser.add_argument("--l1", default="nld")
    parser.add_argument("--l2", default="eng")
    parser.add_argument("--mode", choices=["parallel", "cognate-split",
                                            "non-parallel"],
                        default="parallel")
    parser.add_argument("--n-pairs", type=int, default=1000,
                        help="Cap FLORES dev to this many pairs (CPU smoke "
                             "tests use 32; full run uses all 1012).")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out", default=str(RESULTS_PAPER / "05_geometry"
                                              / "cka_per_layer.csv"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    from analyze.ppl_utils import load_flores_sentences

    l1_sents_full = load_flores_sentences(args.l1)
    l2_sents_full = load_flores_sentences(args.l2)
    n = min(args.n_pairs, len(l1_sents_full), len(l2_sents_full))
    l1_sents = l1_sents_full[:n]
    l2_sents = l2_sents_full[:n]

    if args.mode == "parallel":
        strata: list[tuple[str, list[str], list[str]]] = [
            ("all", l1_sents, l2_sents),
        ]
    elif args.mode == "cognate-split":
        from analyze.xling.cognate_filter import partition_cognates
        cog_idx, non_idx = partition_cognates(l1_sents, l2_sents,
                                              l1_iso=args.l1, l2_iso=args.l2)
        strata = [
            ("cognate", [l1_sents[i] for i in cog_idx],
                        [l2_sents[i] for i in cog_idx]),
            ("non_cognate", [l1_sents[i] for i in non_idx],
                            [l2_sents[i] for i in non_idx]),
        ]
    else:
        from analyze.xling.non_parallel_stimuli import build_non_parallel_pairs
        l1_np, l2_np = build_non_parallel_pairs(args.l1, args.l2, n_pairs=n)
        strata = [("non_parallel", l1_np, l2_np)]

    rows: list[dict] = []
    for repo, cur in zip(args.repos, args.curricula):
        for stratum, l1s, l2s in strata:
            if not l1s or not l2s:
                continue
            rows.extend(run_one_model(
                repo, l1_iso=args.l1, l2_iso=args.l2,
                curriculum=cur, lang_pair=f"{args.l1}-{args.l2}",
                l1_sentences=l1s, l2_sentences=l2s,
                stratum=stratum, n_boot=args.n_boot,
            ))

    df_new = pd.DataFrame(rows)
    out_path = Path(args.out)
    if out_path.exists():
        df_old = pd.read_csv(out_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(["model_id", "layer", "stratum", "lang_pair"],
                                keep="last")
    else:
        df = df_new
    write_csv_atomic(df, out_path)
    print(f"[cka_curriculum] wrote {len(df_new)} rows → {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
