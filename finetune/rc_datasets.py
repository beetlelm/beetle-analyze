"""
finetune/rc_datasets.py — RC-MCQ training corpora for RACE→Cambridge transfer SFT.

Provides ``load_rc_dataset(name, split, ...)`` which formats RACE (or any
HF MCQ dataset with the same schema) into a single ``text`` column whose
template matches the prompts used by ``evals/eval_rcmcq_variants.py`` /
``evals/eval_rcmcq.py``. The point is that downstream SFT teaches the model
to put high LL on the gold option string when conditioned on the same
prompt the eval uses, so PMI / mean-LL ranking at eval time picks the
gold option.

Supported datasets:
  - "race"          → ``EleutherAI/race`` (high subset by default)
  - "race_high"     → ``EleutherAI/race`` (high)
  - "race_middle"   → ``EleutherAI/race`` (middle)
  - "race_combined" → high + middle concatenated
  - "boolq"         → ``google/boolq`` (binary-choice; useful for fast smoke runs)
  - any "<org>/<name>" string is passed through verbatim (must have RACE schema).

Persona dropout: at training time we randomly attach a CEFR persona prefix
matching one of the four eval prompt templates (or no persona). The model
therefore sees all eval-time prompt distributions during SFT and won't
collapse to a single template.
"""
from __future__ import annotations

import logging
import random
from typing import Optional

from datasets import Dataset, concatenate_datasets, load_dataset

from evals.eval_rcmcq import (
    PROMPT_TEMPLATES, CEFR_DESCRIPTORS, OPTION_KEYS, render_prompt,
)

logger = logging.getLogger(__name__)

CEFR_LEVELS = ["A2", "B1", "B2", "C1", "C2"]


def _race_split(name: str, split: str) -> Dataset:
    sub = "high"
    if name == "race_middle":
        sub = "middle"
    return load_dataset("EleutherAI/race", sub, split=split)


def _load_race_combined(split: str) -> Dataset:
    h = load_dataset("EleutherAI/race", "high", split=split)
    m = load_dataset("EleutherAI/race", "middle", split=split)
    return concatenate_datasets([h, m])


def _load_boolq(split: str) -> Dataset:
    # BoolQ schema: passage, question, answer (bool).
    # We project to RACE schema: options=["No","Yes"], answer="A"/"B".
    raw = load_dataset("google/boolq", split=split)

    def _to_race(row):
        return {
            "article": row["passage"],
            "question": row["question"],
            "options": ["No", "Yes"],
            "answer": "B" if row["answer"] else "A",
        }
    return raw.map(_to_race, remove_columns=raw.column_names)


def _format_example(article: str, question: str, options: list[str],
                    answer_letter: str, persona_dropout: float,
                    rng: random.Random) -> str:
    """Render one (passage, question, options, gold) into the `text` field.

    With probability ``persona_dropout`` we drop the persona and use a plain
    'Read and answer' prompt. Otherwise we sample a uniformly random
    (prompt-template, CEFR) pair from the eval-time grid.
    """
    answer_idx = OPTION_KEYS.index(answer_letter.lower())
    gold = options[answer_idx]
    if rng.random() < persona_dropout:
        prompt = (
            "Read the passage and answer the question.\n\n"
            f"Passage:\n{article}\n\nQuestion: {question}\nAnswer:"
        )
    else:
        tmpl_key = rng.choice(list(PROMPT_TEMPLATES.keys()))
        cefr = rng.choice(CEFR_LEVELS)
        prompt = render_prompt(tmpl_key, cefr, article, question)
    return prompt + " " + gold


def load_rc_dataset(
    name: str,
    split: str = "train",
    *,
    eval_fraction: float = 0.0,
    persona_dropout: float = 0.25,
    min_chars: int = 64,
    max_chars: int = 8000,
    seed: int = 42,
    max_examples: Optional[int] = None,
) -> dict[str, Dataset]:
    """Return ``{"train": Dataset, "eval": Dataset|None}`` with a single ``text`` column."""
    name = name.strip().lower()
    if name in ("race", "race_high"):
        ds = _race_split(name, split)
    elif name == "race_middle":
        ds = _race_split(name, split)
    elif name == "race_combined":
        ds = _load_race_combined(split)
    elif name == "boolq":
        ds = _load_boolq(split)
    elif "/" in name:
        ds = load_dataset(name, split=split)
    else:
        raise ValueError(f"Unknown rc dataset name: {name}")

    rng = random.Random(seed)

    def _row_to_text(row):
        article = (row.get("article") or row.get("passage") or "").strip()
        question = (row.get("question") or row.get("query") or "").strip()
        options = row.get("options") or []
        if isinstance(options, dict):
            options = [options.get(k) for k in ("A", "B", "C", "D") if options.get(k)]
        answer = (row.get("answer") or "").strip()
        if not article or not question or not options or not answer:
            return {"text": ""}
        if not (min_chars <= len(article) <= max_chars):
            return {"text": ""}
        if answer not in ("A", "B", "C", "D"):
            return {"text": ""}
        return {"text": _format_example(article, question, options, answer,
                                        persona_dropout, rng)}

    ds = ds.map(_row_to_text, remove_columns=[c for c in ds.column_names])
    ds = ds.filter(lambda r: bool(r["text"]))
    if max_examples is not None and len(ds) > max_examples:
        ds = ds.shuffle(seed=seed).select(range(max_examples))

    if eval_fraction and eval_fraction > 0:
        n_eval = max(1, int(len(ds) * eval_fraction))
        ds_shuf = ds.shuffle(seed=seed)
        eval_ds = ds_shuf.select(range(n_eval))
        train_ds = ds_shuf.select(range(n_eval, len(ds_shuf)))
        return {"train": train_ds, "eval": eval_ds}
    return {"train": ds, "eval": None}
