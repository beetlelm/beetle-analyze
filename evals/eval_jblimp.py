#!/usr/bin/env python3
"""
evals/eval_jblimp.py — JBLiMP minimal-pair evaluation with SLOR scoring.

JBLiMP (Someya & Oseki, 2023; polm-stability/jblimp on HuggingFace) is a
Japanese minimal-pair grammaticality benchmark. Each row contains:
  - good_sentence : grammatical sentence
  - bad_sentence  : minimally-different ungrammatical sentence
  - type, phenomenon, phenomenon-2, paradigm (metadata)

Following the JBLiMP paper (§4.2), we score sentences with SLOR:

    SLOR(X) = (log p_m(X) - log p_u(X)) / |X|

  - log p_m(X)  : sum of token log-probabilities under the model
  - log p_u(X)  : sum of subword unigram log-probabilities
  - |X|         : number of subwords in X

Unigram probabilities are estimated by MLE over the JBLiMP corpus itself
(all good+bad sentences) under the *model's own* tokenizer, so the unigram
model is per-tokenizer and self-contained.

For each minimal pair we count it correct iff SLOR(good) > SLOR(bad).
"""

from __future__ import annotations

import logging
import math
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import torch
from datasets import load_dataset

logger = logging.getLogger(__name__)

JBLIMP_HF_ID = "polm-stability/jblimp"
JBLIMP_GOOD_COL = "good_sentence"
JBLIMP_BAD_COL = "bad_sentence"


# ---------------------------------------------------------------------------
# Dataset loading (with pkl cache)
# ---------------------------------------------------------------------------

def load_jblimp_data(pkl_dir: str, split: str = "train") -> List[Tuple[str, str]]:
    """Load JBLiMP minimal pairs; cache to pickle for fast re-runs."""
    pkl_name = f"jblimp_{split}.pkl"
    pkl_path = os.path.join(pkl_dir, pkl_name) if pkl_dir else ""

    if pkl_path and os.path.exists(pkl_path):
        with open(pkl_path, "rb") as fh:
            pairs = pickle.load(fh)
        logger.info(f"  pkl cache hit: {len(pairs):,} pairs ({pkl_name})")
        return pairs

    logger.info(f"  Downloading {JBLIMP_HF_ID} (split={split}) …")
    try:
        ds = load_dataset(JBLIMP_HF_ID, split=split)
    except Exception:
        # Fallback: dataset may only have a "test" or default split.
        ds_dict = load_dataset(JBLIMP_HF_ID)
        ds = ds_dict[list(ds_dict.keys())[0]]

    pairs: List[Tuple[str, str]] = []
    for row in ds:
        g = row.get(JBLIMP_GOOD_COL)
        b = row.get(JBLIMP_BAD_COL)
        if isinstance(g, str) and g and isinstance(b, str) and b:
            pairs.append((g, b))
    del ds

    logger.info(f"  Loaded {len(pairs):,} JBLiMP pairs")

    if pkl_path:
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        with open(pkl_path, "wb") as fh:
            pickle.dump(pairs, fh)
        logger.info(f"  Cached → {pkl_name}")

    return pairs


# ---------------------------------------------------------------------------
# Per-tokenizer subword unigram model (MLE over the JBLiMP corpus itself)
# ---------------------------------------------------------------------------

def build_unigram_logprobs(
    tokenizer, sentences: List[str], smoothing: float = 1.0,
) -> dict[int, float]:
    """Estimate log p_u(token_id) by add-`smoothing` MLE over the given corpus.

    Returns a dict mapping token_id → log_prob. Unknown ids fall back to the
    smoothed log-prob of an unseen token (handled at lookup time).
    """
    counts: Counter[int] = Counter()
    for s in sentences:
        ids = tokenizer.encode(s, add_special_tokens=False)
        counts.update(ids)

    vocab_size = max(counts) + 1 if counts else 1
    # Use a vocabulary-size-aware denominator so unseen ids get a small mass.
    total = sum(counts.values()) + smoothing * vocab_size
    log_total = math.log(total) if total > 0 else 0.0
    return {
        tid: math.log(c + smoothing) - log_total
        for tid, c in counts.items()
    }


def _sentence_unigram_logprob(
    tokenizer, sentence: str, log_unigram: dict[int, float],
    fallback_log: float,
) -> Tuple[float, int]:
    """Return (sum log p_u(X), |X|) for a sentence under the unigram model."""
    ids = tokenizer.encode(sentence, add_special_tokens=False)
    if not ids:
        return 0.0, 0
    total = 0.0
    for tid in ids:
        total += log_unigram.get(tid, fallback_log)
    return total, len(ids)


# ---------------------------------------------------------------------------
# SLOR scoring
# ---------------------------------------------------------------------------

def run_jblimp_slor(
    model, tokenizer, pairs: List[Tuple[str, str]],
    device: str, batch_size: int,
) -> Tuple[float, int, int]:
    """Score JBLiMP pairs with SLOR. Returns (accuracy, n_correct, n_total).

    Reuses ``score_sentences`` from eval_blimp for the model log-prob term,
    so the autocast / OOM-fallback / heartbeat machinery is shared.
    """
    if not pairs:
        return 0.0, 0, 0

    # Late import to avoid circular import at module load time.
    from evals.eval_blimp import score_sentences

    good_sents = [p[0] for p in pairs]
    bad_sents = [p[1] for p in pairs]

    # 1) Subword unigram model over the JBLiMP corpus itself.
    logger.info(f"  JBLiMP: estimating subword unigram MLE over "
                f"{len(good_sents) + len(bad_sents):,} sentences …")
    log_unigram = build_unigram_logprobs(tokenizer, good_sents + bad_sents)
    # Fallback log-prob for tokens unseen in the JBLiMP corpus (rare; uses
    # smoothing mass in build_unigram_logprobs).
    if log_unigram:
        # Smoothing mass / total ≈ 1 / total (when count = 0).
        # Use the minimum observed log-prob as a conservative fallback.
        fallback_log = min(log_unigram.values())
    else:
        fallback_log = -20.0  # effectively zero probability

    # 2) Model log-prob (sum log P) for each sentence.
    s_good = score_sentences(model, tokenizer, good_sents, device, batch_size)
    s_bad = score_sentences(model, tokenizer, bad_sents, device, batch_size)

    # 3) Combine into SLOR per sentence and compare per pair.
    n = len(pairs)
    n_correct = 0
    for i in range(n):
        log_pm_good = float(s_good[i].item())
        log_pm_bad = float(s_bad[i].item())
        log_pu_good, len_good = _sentence_unigram_logprob(
            tokenizer, good_sents[i], log_unigram, fallback_log,
        )
        log_pu_bad, len_bad = _sentence_unigram_logprob(
            tokenizer, bad_sents[i], log_unigram, fallback_log,
        )
        if len_good == 0 or len_bad == 0:
            continue
        slor_good = (log_pm_good - log_pu_good) / len_good
        slor_bad = (log_pm_bad - log_pu_bad) / len_bad
        if slor_good > slor_bad:
            n_correct += 1

    return n_correct / n, n_correct, n
