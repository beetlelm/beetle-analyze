"""Probe-distribution construction for susceptibility analysis.

We build four probe distributions (rows of the response matrix X):

  ``eng``        : FLORES English-only batch
  ``nld``        : FLORES Dutch-only batch
  ``mix_code``   : 50/50 English / code (BabyLM-style)
  ``multi``      : equal mixture across all FLORES languages the
                   model's tokenizer supports

A "probe" is a HuggingFace ``Dataset`` of ``input_ids`` whose construction
is deterministic given the seed and the model's tokenizer. The
sampling-task distribution (the unperturbed q) is built from the same
multilingual mixture that the model was trained on, approximated by an
equal mix of available FLORES languages plus code.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Sequence

import torch
from datasets import Dataset, load_dataset

# Reuse the FLORES loader from ppl_utils.
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze.ppl_utils import FLORES_LANG_MAP, load_flores_sentences  # noqa: E402

log = logging.getLogger(__name__)


PROBE_NAMES = ("eng", "nld", "mix_code", "multi")


@dataclass(frozen=True)
class ProbeBatch:
    """A probe-distribution batch ready for ``devinterp.sample()``."""

    name: str
    dataset: Dataset
    description: str


def _tokenize(
    tokenizer,
    sentences: Sequence[str],
    *,
    max_length: int,
) -> Dataset:
    """Tokenize, pad/truncate, and wrap into an HF Dataset of input_ids."""
    enc = tokenizer(
        list(sentences),
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    return Dataset.from_dict({"input_ids": enc["input_ids"]})


def _supported_flores_langs(tokenizer) -> list[str]:
    """Heuristic: any FLORES language whose ASCII-ish sample tokenises
    without too many ``unk`` tokens is considered supported.

    For Beetle bilingual models this returns ``["nld", "eng"]``;
    for FineWeb multilingual it includes the FineWeb-2 set.
    """
    supported: list[str] = []
    candidates = list(FLORES_LANG_MAP.keys())
    unk_id = getattr(tokenizer, "unk_token_id", None)

    for iso in candidates:
        try:
            sents = load_flores_sentences(iso)[:8]
        except Exception:  # pragma: no cover — network or column missing
            continue
        if not sents:
            continue
        ids = tokenizer(" ".join(sents), truncation=True, max_length=128)["input_ids"]
        if unk_id is not None and ids.count(unk_id) / max(len(ids), 1) > 0.30:
            continue
        supported.append(iso)

    if not supported:
        # Conservative fallback.
        supported = ["eng"]
    return supported


def _english_code_mix(tokenizer, max_length: int, n: int, seed: int) -> Dataset:
    """50/50 English / code mix.

    Code from ``codeparrot/github-code-clean`` (1k stream sample); English
    from FLORES devtest. Used as the ``mix_code`` probe.
    """
    rng = random.Random(seed)
    eng = load_flores_sentences("eng")[:n // 2]
    try:
        ds = load_dataset(
            "codeparrot/github-code-clean",
            "all-all",
            split="train",
            streaming=True,
        )
        code_iter = iter(ds)
        code_lines: list[str] = []
        for _ in range(n // 2):
            try:
                row = next(code_iter)
            except StopIteration:
                break
            txt = row.get("code") or row.get("content") or ""
            if txt:
                code_lines.append(txt[: max_length * 4])
    except Exception as exc:  # pragma: no cover
        log.warning("github-code-clean unavailable (%s); using FLORES english only", exc)
        code_lines = eng

    mix = list(eng) + list(code_lines)
    rng.shuffle(mix)
    return _tokenize(tokenizer, mix[:n], max_length=max_length)


def build_probes(
    tokenizer,
    *,
    n_per_probe: int = 256,
    max_length: int = 256,
    seed: int = 1337,
) -> dict[str, ProbeBatch]:
    """Construct the four probe distributions for a given tokenizer.

    Output keys: ``"eng"``, ``"nld"``, ``"mix_code"``, ``"multi"``.
    """
    rng = random.Random(seed)
    probes: dict[str, ProbeBatch] = {}

    # Single-language probes
    for iso, name in (("eng", "eng"), ("nld", "nld")):
        sents = load_flores_sentences(iso)
        rng.shuffle(sents)
        probes[name] = ProbeBatch(
            name=name,
            dataset=_tokenize(tokenizer, sents[:n_per_probe], max_length=max_length),
            description=f"FLORES {iso} only",
        )

    # Mix probe
    probes["mix_code"] = ProbeBatch(
        name="mix_code",
        dataset=_english_code_mix(tokenizer, max_length, n_per_probe, seed),
        description="50/50 English / code",
    )

    # Multilingual mixture
    langs = _supported_flores_langs(tokenizer)
    per_lang = max(n_per_probe // len(langs), 1)
    multi: list[str] = []
    for iso in langs:
        sents = load_flores_sentences(iso)
        rng.shuffle(sents)
        multi.extend(sents[:per_lang])
    rng.shuffle(multi)
    probes["multi"] = ProbeBatch(
        name="multi",
        dataset=_tokenize(tokenizer, multi[:n_per_probe], max_length=max_length),
        description=f"Equal-mix {','.join(langs)}",
    )

    return probes


def build_sampling_task(
    tokenizer,
    *,
    n: int = 1024,
    max_length: int = 256,
    seed: int = 1337,
) -> Dataset:
    """Build the unperturbed q used as ``sampling_task`` in
    ``devinterp.susceptibilities``.

    Uses the same multilingual mixture as the ``multi`` probe but at a
    larger sample count, since this defines the local posterior.
    """
    langs = _supported_flores_langs(tokenizer)
    per_lang = max(n // len(langs), 1)
    rng = random.Random(seed)
    out: list[str] = []
    for iso in langs:
        sents = load_flores_sentences(iso)
        rng.shuffle(sents)
        out.extend(sents[:per_lang])
    rng.shuffle(out)
    return _tokenize(tokenizer, out[:n], max_length=max_length)


def per_language_dataset(
    tokenizer,
    iso_code: str,
    *,
    n: int = 512,
    max_length: int = 256,
) -> Dataset:
    """Per-language FLORES dataset for S3 (per-language LLC decomposition)."""
    sents = load_flores_sentences(iso_code)[:n]
    return _tokenize(tokenizer, sents, max_length=max_length)
