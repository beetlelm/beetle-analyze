"""End-to-end BLiMP-style analysis over an eval-output file.

Input: a JSONL file, one minimal pair per line, with at least these fields::

    {"phenomenon": "subject_verb_agreement",
     "logprob_good": -21.3, "logprob_bad": -24.8,
     "n_tokens_good": 8, "n_tokens_bad": 8}

Output: a per-phenomenon accuracy table plus an overall accuracy and a
bootstrap confidence interval — the headline number of the analysis.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .scoring import Scoring, accuracy, bootstrap_ci, pair_correct, sentence_scores

REQUIRED_FIELDS = ("phenomenon", "logprob_good", "logprob_bad")


def toy_data_path() -> Path:
    """Absolute path to the bundled synthetic eval output."""
    with resources.as_file(
        resources.files("beetle_analyze.data") / "blimp_toy.jsonl"
    ) as p:
        return Path(p)


def load_pairs(path: str | Path) -> pd.DataFrame:
    """Load a JSONL eval-output file into a DataFrame, validating fields."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            missing = [f for f in REQUIRED_FIELDS if f not in rec]
            if missing:
                raise ValueError(f"line {i}: missing fields {missing}")
            rows.append(rec)
    if not rows:
        raise ValueError(f"no records found in {path}")
    df = pd.DataFrame(rows)
    for col in ("n_tokens_good", "n_tokens_bad"):
        if col not in df.columns:
            df[col] = 1
    return df


@dataclass
class BlimpResult:
    """Result of a BLiMP-style analysis run."""

    scoring: Scoring
    n_pairs: int
    overall_accuracy: float
    ci_low: float
    ci_high: float
    per_phenomenon: pd.DataFrame  # columns: phenomenon, n_pairs, accuracy
    _correct: np.ndarray = field(repr=False, default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scoring": self.scoring,
            "n_pairs": int(self.n_pairs),
            "overall_accuracy": round(float(self.overall_accuracy), 6),
            "ci_low": round(float(self.ci_low), 6),
            "ci_high": round(float(self.ci_high), 6),
            "per_phenomenon": {
                row.phenomenon: round(float(row.accuracy), 6)
                for row in self.per_phenomenon.itertuples()
            },
        }


def evaluate(
    df: pd.DataFrame,
    scoring: Scoring = "sum",
    bootstrap_resamples: int = 2000,
    seed: int = 0,
) -> BlimpResult:
    """Compute per-phenomenon and overall BLiMP accuracy for one eval output."""
    s_good = sentence_scores(df["logprob_good"].to_numpy(), df["n_tokens_good"].to_numpy(), scoring)
    s_bad = sentence_scores(df["logprob_bad"].to_numpy(), df["n_tokens_bad"].to_numpy(), scoring)
    correct = pair_correct(s_good, s_bad)

    work = df.assign(correct=correct)
    per = (
        work.groupby("phenomenon")["correct"]
        .agg(n_pairs="size", accuracy="mean")
        .reset_index()
        .sort_values("phenomenon")
        .reset_index(drop=True)
    )

    lo, hi = bootstrap_ci(correct, n_resamples=bootstrap_resamples, seed=seed)
    return BlimpResult(
        scoring=scoring,
        n_pairs=len(df),
        overall_accuracy=accuracy(correct),
        ci_low=lo,
        ci_high=hi,
        per_phenomenon=per,
        _correct=np.asarray(correct),
    )


def analyze_file(
    path: str | Path,
    scoring: Scoring = "sum",
    bootstrap_resamples: int = 2000,
    seed: int = 0,
) -> BlimpResult:
    """Convenience: load an eval-output file and evaluate it in one call."""
    return evaluate(load_pairs(path), scoring=scoring, bootstrap_resamples=bootstrap_resamples, seed=seed)
