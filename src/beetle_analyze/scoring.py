"""Core minimal-pair scoring.

A BLiMP-style benchmark is a set of *minimal pairs*: a grammatical sentence and
a near-identical ungrammatical one. A language model "gets a pair right" when it
assigns higher probability to the grammatical sentence. The headline metric is
the fraction of pairs the model gets right (accuracy).

This module works on *precomputed* sentence log-probabilities, i.e. eval output,
so it needs neither a GPU nor any model download. The comparison logic mirrors
the ``run_minimal_pairs`` routine used in the full Beetle evaluation pipeline
(``n_correct = sum(score_good > score_bad)``).
"""
from __future__ import annotations

from typing import Literal

import numpy as np

Scoring = Literal["sum", "mean"]


def sentence_scores(
    logprobs: np.ndarray,
    n_tokens: np.ndarray,
    scoring: Scoring = "sum",
) -> np.ndarray:
    """Turn raw total log-probabilities into comparison scores.

    Parameters
    ----------
    logprobs:
        Total token log-probability of each sentence (summed over tokens), as
        produced by the evaluation pass.
    n_tokens:
        Number of tokens in each sentence. Only used when ``scoring="mean"``.
    scoring:
        ``"sum"`` compares raw total log-probabilities (the BLiMP default).
        ``"mean"`` compares per-token log-probabilities (``logprob / n_tokens``),
        which removes the bias toward shorter sentences.
    """
    logprobs = np.asarray(logprobs, dtype=float)
    if scoring == "sum":
        return logprobs
    if scoring == "mean":
        n = np.asarray(n_tokens, dtype=float)
        if np.any(n <= 0):
            raise ValueError("n_tokens must be positive for mean scoring")
        return logprobs / n
    raise ValueError(f"unknown scoring mode: {scoring!r}")


def pair_correct(
    score_good: np.ndarray,
    score_bad: np.ndarray,
) -> np.ndarray:
    """Boolean array: True where the grammatical sentence outscores the bad one."""
    return np.asarray(score_good) > np.asarray(score_bad)


def accuracy(correct: np.ndarray) -> float:
    """Fraction of pairs scored correctly."""
    correct = np.asarray(correct, dtype=bool)
    if correct.size == 0:
        return 0.0
    return float(correct.mean())


def bootstrap_ci(
    correct: np.ndarray,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the accuracy.

    Returns the ``(low, high)`` bounds of the ``1 - alpha`` interval.
    """
    correct = np.asarray(correct, dtype=float)
    n = correct.size
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = correct[rng.integers(0, n, size=(n_resamples, n))].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)
