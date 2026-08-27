#!/usr/bin/env python3
"""
evals/eval_xcomps.py — xCOMPs minimal-pair evaluation.

xCOMPs (fpadovani/xcomps-dataset on HuggingFace) is a multilingual
minimal-pair dataset for complement-clause acceptability. Each row has:
  - acceptable_sent   : grammatical sentence
  - unacceptable_sent : minimally-different ungrammatical sentence

Configs of interest for our two cohorts:
  comps_ja  → Japanese
  comps_zh  → Chinese

Scoring uses the same BLiMP-style sum-log-probability comparison as
evals/eval_blimp.py: pair is correct iff log p_m(good) > log p_m(bad).
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import List, Tuple

from datasets import load_dataset

logger = logging.getLogger(__name__)

XCOMPS_HF_ID = "fpadovani/xcomps-dataset"
XCOMPS_GOOD_COL = "acceptable_sent"
XCOMPS_BAD_COL = "unacceptable_sent"

# ISO-639-1 → (eval_language label, ISO-639-3) mapping for langs we support.
XCOMPS_LANGS: dict[str, Tuple[str, str]] = {
    "ja": ("Japanese", "jpn"),
    "zh": ("Chinese", "zho"),
    "fr": ("French", "fra"),
    "de": ("German", "deu"),
    "uk": ("Ukrainian", "ukr"),
    "fa": ("Persian", "fas"),
}


def _split_name(lang_iso1: str) -> str:
    return f"comps_{lang_iso1}"


def load_xcomps_data(
    lang_iso1: str, pkl_dir: str,
) -> List[Tuple[str, str]]:
    """Load xCOMPs minimal pairs for one language; cache to pickle.

    Tries split=`comps_<iso1>` first (the dataset stores languages as named
    splits, per legacy/bilingual_eval.py:454).
    """
    pkl_name = f"xcomps_{lang_iso1}.pkl"
    pkl_path = os.path.join(pkl_dir, pkl_name) if pkl_dir else ""

    if pkl_path and os.path.exists(pkl_path):
        with open(pkl_path, "rb") as fh:
            pairs = pickle.load(fh)
        logger.info(f"  pkl cache hit: {len(pairs):,} pairs ({pkl_name})")
        return pairs

    split = _split_name(lang_iso1)
    logger.info(f"  Downloading {XCOMPS_HF_ID} (split={split}) …")
    ds = load_dataset(XCOMPS_HF_ID, split=split)

    pairs: List[Tuple[str, str]] = []
    for row in ds:
        g = row.get(XCOMPS_GOOD_COL)
        b = row.get(XCOMPS_BAD_COL)
        if isinstance(g, str) and g and isinstance(b, str) and b:
            pairs.append((g, b))
    del ds

    logger.info(f"  Loaded {len(pairs):,} xCOMPs pairs ({split})")

    if pkl_path:
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        with open(pkl_path, "wb") as fh:
            pickle.dump(pairs, fh)
        logger.info(f"  Cached → {pkl_name}")

    return pairs


def run_xcomps(
    model, tokenizer, pairs: List[Tuple[str, str]],
    device: str, batch_size: int,
) -> Tuple[float, int, int]:
    """Score xCOMPs pairs with BLiMP-style sum-logprob comparison."""
    if not pairs:
        return 0.0, 0, 0
    # Late import: same scoring path as eval_blimp.run_minimal_pairs.
    from evals.eval_blimp import run_minimal_pairs
    return run_minimal_pairs(model, tokenizer, pairs, device, batch_size)
