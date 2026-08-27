"""
finetune/learner_datasets.py — load + format lilygoulder/learner_tuning_* corpora.

Each lilygoulder/learner_tuning_<LANG> dataset has the schema
    {"split", "Instruction", "Output", "Unnamed: 2"}
with `Instruction` = essay prompt, `Output` = student response. The unnamed
float column is a CSV-import artifact and is dropped.

We render every example into a single `text` field using a minimal
completion template — Beetle base models have no chat template baked in,
so a plain ``### Instruction: … ### Response: …`` body is the safest
option. The same template must be applied at inference time when
prompting the SFT'd model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

# Mapping from Beetle L1 ISO-639-3 → lilygoulder language code. Beetle's
# Dutch L1 has no matching learner-tuning corpus; we leave it absent so
# the caller can decide (skip, use English, or pick a related L1).
BEETLE_L1_TO_LEARNER: Dict[str, str] = {
    "deu": "GER",
    "zho": "CHI",
    "fra": "FRE",
    "spa": "SPA",
    "ara": "ARA",
    "tur": "TUR",
    "pol": "POL",
}

LEARNER_DATASET_PREFIX = "lilygoulder/learner_tuning_"

INSTRUCTION_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n### Response:\n{response}"
)

# EOS gets appended by SFTTrainer's collator; do not include it in the
# template body or we double-count it during loss computation.


@dataclass
class LearnerExample:
    instruction: str
    response: str


def resolve_dataset_name(lang_or_dataset: str) -> str:
    """Accept either a Beetle ISO-639-3 ('deu') or a full HF id and return the HF id."""
    s = lang_or_dataset.strip()
    if "/" in s:  # already an HF dataset id
        return s
    s_lower = s.lower()
    if s_lower in BEETLE_L1_TO_LEARNER:
        return f"{LEARNER_DATASET_PREFIX}{BEETLE_L1_TO_LEARNER[s_lower]}"
    # Bare 3-letter lilygoulder code (GER/CHI/…) — pass through
    return f"{LEARNER_DATASET_PREFIX}{s.upper()}"


def format_example(example: dict) -> dict:
    """Convert one raw lilygoulder row into ``{"text": ...}`` for SFTTrainer."""
    instruction = (example.get("Instruction") or "").strip()
    response    = (example.get("Output") or "").strip()
    text = INSTRUCTION_TEMPLATE.format(instruction=instruction, response=response)
    return {"text": text}


def load_learner_dataset(
    name: str,
    split: str = "train",
    min_chars: int = 1,
    max_chars: Optional[int] = None,
    eval_fraction: float = 0.0,
    seed: int = 42,
) -> Dict[str, Dataset]:
    """Load + format a lilygoulder learner_tuning corpus for SFTTrainer.

    Parameters
    ----------
    name
        Either an ISO code understood by ``BEETLE_L1_TO_LEARNER`` (e.g. ``deu``),
        a bare lilygoulder code (e.g. ``GER``), or a full HF dataset id.
    split
        HF split to load (lilygoulder corpora ship a single ``train``).
    min_chars, max_chars
        Filter rows whose response length lies outside the window. The
        upstream data has empty / multi-thousand-char rows.
    eval_fraction
        If > 0, splits the loaded data into train/eval by ``train_test_split``
        on the post-filter Dataset. Returned dict has keys ``train`` and ``eval``.

    Returns
    -------
    {"train": Dataset, "eval": Dataset?}
    """
    hf_id = resolve_dataset_name(name)
    logger.info(f"Loading {hf_id} (split={split}) …")
    raw = load_dataset(hf_id, split=split)
    n_raw = len(raw)

    # Drop empty / oversized examples
    def _keep(ex):
        out = (ex.get("Output") or "").strip()
        if len(out) < min_chars:
            return False
        if max_chars is not None and len(out) > max_chars:
            return False
        instr = (ex.get("Instruction") or "").strip()
        return bool(instr)

    filtered = raw.filter(_keep)
    n_filt = len(filtered)
    logger.info(f"  {hf_id}: kept {n_filt:,}/{n_raw:,} rows after length filter")

    formatted = filtered.map(
        format_example,
        remove_columns=[c for c in filtered.column_names if c != "Output"],
    )
    # Keep only the text column for SFTTrainer
    formatted = formatted.remove_columns([c for c in formatted.column_names if c != "text"])

    if eval_fraction and eval_fraction > 0.0:
        split_ds = formatted.train_test_split(test_size=eval_fraction, seed=seed)
        return {"train": split_ds["train"], "eval": split_ds["test"]}
    return {"train": formatted}


def render_inference_prompt(instruction: str) -> str:
    """Render the same template at inference time, leaving ``Response:`` open.

    Use this when you want to query the SFT'd model with a CEFR-style prompt
    so that the formatting matches what the model saw during training.
    """
    return INSTRUCTION_TEMPLATE.format(instruction=instruction.strip(), response="").rstrip()
