"""
analyze/brain/encoder_curriculum.py — §C neural alignment × curriculum
(appendix-bound pending power-calc pilot).

Fits a held-out ridge encoder from per-(curriculum, layer) activations to
brain responses for each subject, stratified by subject L1. The pre-
registered pattern: native English readers best predicted by mono-eng /
b1; L2 readers (L1 = nld / deu / zho) best predicted by matched-L1 b3-b5.

ZuCo 2.0 has ~18 subjects total (~6-8 L2). The EMNLP plan demotes §C to
the appendix unless a power-calc pilot on the L2 subset shows usable
effect sizes. This module supports the pilot (small-n bootstrap) and the
full run with the same code path.

Outputs
-------
results/evals_paper/07_brain/encoder_r2.csv
  Columns: model_id, curriculum, layer, dataset, subject_id, subject_l1,
           r2, ci_lo, ci_hi, n_train, n_test
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from analyze._interp_helpers import (
    DEFAULT_BEETLE_REPOS, RESULTS_PAPER, bootstrap_ci, device_for,
    per_layer_mean_activations, write_csv_atomic,
)

logger = logging.getLogger(__name__)


def _ridge_r2(X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray,
                alphas=(0.1, 1.0, 10.0, 100.0)) -> float:
    """Held-out R² for a small ridge with α picked by leave-one-out CV.
    Uses ``sklearn.linear_model.RidgeCV`` if available; falls back to NumPy."""
    try:
        from sklearn.linear_model import RidgeCV
        clf = RidgeCV(alphas=alphas, cv=3)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
    except Exception:
        # NumPy fallback — simple closed-form ridge with fixed α.
        alpha = alphas[0]
        A = X_train.T @ X_train + alpha * np.eye(X_train.shape[1])
        beta = np.linalg.solve(A, X_train.T @ y_train)
        pred = X_test @ beta
    ss_res = ((y_test - pred) ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum() or 1.0
    return float(1.0 - ss_res / ss_tot)


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repos", nargs="+",
                    default=list(DEFAULT_BEETLE_REPOS.values()))
    p.add_argument("--curricula", nargs="+",
                    default=list(DEFAULT_BEETLE_REPOS.keys()))
    p.add_argument("--brain-csv", required=True,
                    help="Long CSV: subject_id, subject_l1, dataset, sentence_id, "
                         "sentence, response.")
    p.add_argument("--n-test", type=int, default=10)
    p.add_argument("--out", default=str(RESULTS_PAPER / "07_brain" / "encoder_r2.csv"))
    args = p.parse_args(list(argv) if argv is not None else None)

    from analyze.ppl_utils import load_model_and_tokenizer

    brain = pd.read_csv(args.brain_csv)
    rows = []
    rng = np.random.default_rng(0)

    for repo, cur in zip(args.repos, args.curricula):
        pair = load_model_and_tokenizer(repo)
        if pair is None:
            continue
        model, tokenizer = pair
        model.to(device_for()).eval()
        # Activations per unique sentence.
        unique_sents = brain[["sentence_id", "sentence"]].drop_duplicates(
            "sentence_id"
        ).reset_index(drop=True)
        acts_per_layer = per_layer_mean_activations(
            model, tokenizer, unique_sents["sentence"].astype(str).tolist()
        )
        sid_to_pos = {sid: i for i, sid in enumerate(unique_sents["sentence_id"])}

        for (subject, dataset, l1), sub in brain.groupby(["subject_id", "dataset", "subject_l1"]):
            sub = sub.dropna(subset=["response"])
            if len(sub) < (args.n_test + 5):
                continue
            idx = sub["sentence_id"].map(sid_to_pos).dropna().astype(int).values
            y = sub["response"].astype(float).values[: len(idx)]
            test_idx = rng.choice(len(idx), size=args.n_test, replace=False)
            train_mask = np.ones(len(idx), dtype=bool); train_mask[test_idx] = False
            for layer, X in acts_per_layer.items():
                Xs = X[idx]
                r2 = _ridge_r2(Xs[train_mask], y[train_mask],
                                Xs[~train_mask], y[~train_mask])
                ci_lo, ci_hi = bootstrap_ci([r2])
                rows.append({
                    "model_id": repo, "curriculum": cur, "layer": layer,
                    "dataset": dataset, "subject_id": subject,
                    "subject_l1": l1, "r2": r2,
                    "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "n_train": int(train_mask.sum()),
                    "n_test": int((~train_mask).sum()),
                })

    write_csv_atomic(pd.DataFrame(rows), args.out)
    print(f"[encoder_curriculum] wrote {len(rows)} rows → {args.out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
