"""
analyze/xling/datasets.py
─────────────────────────
Parallel-corpus and treebank loaders used by the SVCCA / CKA / probe /
CrossEncoder analyses.

Two corpora matter for the cross-lingual programme:

* **FLORES-200** (Costa-jussà et al., 2022) — sentence-aligned across
  200+ languages.  Used for cross-lingual subspace alignment (SVCCA),
  layer-wise CKA on parallel pairs, and CrossEncoder semantic probing.

* **Universal Dependencies treebanks** — used for the dependency /
  POS / NER probes (Set A2, B4).

We keep this module thin: it returns iterables of dicts, so individual
analyses can stream rather than materialising everything into memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParallelPair:
    """A single sentence pair from a parallel corpus."""
    src_text: str
    tgt_text: str
    src_code: str       # FLORES-style language identifier, e.g. "nld_Latn"
    tgt_code: str
    pair_id: int


@dataclass(frozen=True)
class TaggedSentence:
    """A token-tagged sentence for probe training."""
    tokens: List[str]
    tags: List[str]
    lang_code: str      # ISO 639-3
    sent_id: str


# ---------------------------------------------------------------------------
# FLORES-200
# ---------------------------------------------------------------------------

# Default HF dataset id.  ``facebook/flores`` is the canonical mirror.
FLORES_DATASET = "facebook/flores"


def load_flores_pairs(
    src_code: str,
    tgt_code: str,
    *,
    split: str = "dev",
    max_pairs: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[ParallelPair]:
    """Load aligned sentence pairs from FLORES-200.

    ``src_code`` / ``tgt_code`` are FLORES-style identifiers (e.g.
    ``nld_Latn``, ``eng_Latn``).

    Returns an in-memory list since FLORES dev is ~1k pairs (small).
    """
    from datasets import load_dataset  # type: ignore

    src = load_dataset(FLORES_DATASET, src_code, split=split, cache_dir=cache_dir)
    tgt = load_dataset(FLORES_DATASET, tgt_code, split=split, cache_dir=cache_dir)
    if len(src) != len(tgt):
        raise RuntimeError(
            f"[xling.datasets] FLORES alignment mismatch: "
            f"{src_code} has {len(src)} rows, {tgt_code} has {len(tgt)}."
        )
    n = len(src) if max_pairs is None else min(max_pairs, len(src))
    pairs: List[ParallelPair] = []
    for i in range(n):
        pairs.append(ParallelPair(
            src_text=src[i]["sentence"],
            tgt_text=tgt[i]["sentence"],
            src_code=src_code,
            tgt_code=tgt_code,
            pair_id=i,
        ))
    return pairs


def iter_negatives(
    pairs: Sequence[ParallelPair],
    *,
    shift: int = 1,
) -> Iterator[ParallelPair]:
    """Generate paraphrase-negative pairs by cyclically shifting targets.

    Used by the CrossEncoder probe to compute a contrastive baseline:
    a genuine cross-lingual encoder must distinguish the true pair
    from the shifted (mismatched) one.
    """
    n = len(pairs)
    if n < 2:
        return
    for i, p in enumerate(pairs):
        j = (i + shift) % n
        yield ParallelPair(
            src_text=p.src_text,
            tgt_text=pairs[j].tgt_text,
            src_code=p.src_code,
            tgt_code=p.tgt_code,
            pair_id=-(p.pair_id + 1),  # negative tag
        )


# ---------------------------------------------------------------------------
# Universal Dependencies
# ---------------------------------------------------------------------------

# Map ISO 639-3 → UD treebank short id.  Extend as multilingual Beetle
# adds languages.
UD_TREEBANKS = {
    "eng": "en_ewt",
    "nld": "nl_alpino",
    "deu": "de_gsd",
    "ara": "ar_padt",
    "fin": "fi_tdt",
    "tur": "tr_imst",
    "jpn": "ja_gsd",
    "zho": "zh_gsd",
}


def load_ud_pos(
    iso: str,
    *,
    split: str = "train",
    cache_dir: Optional[str] = None,
    max_sentences: Optional[int] = None,
) -> List[TaggedSentence]:
    """Load POS-tagged sentences from a UD treebank.

    Returns an in-memory list; UD treebank splits are typically <50k
    sentences each.
    """
    from datasets import load_dataset  # type: ignore

    if iso not in UD_TREEBANKS:
        raise KeyError(
            f"[xling.datasets] no UD treebank registered for {iso!r}; "
            f"add it to UD_TREEBANKS."
        )
    name = f"universal_dependencies/{UD_TREEBANKS[iso]}"
    ds = load_dataset(name, split=split, cache_dir=cache_dir)
    out: List[TaggedSentence] = []
    n = len(ds) if max_sentences is None else min(max_sentences, len(ds))
    for i in range(n):
        row = ds[i]
        out.append(TaggedSentence(
            tokens=list(row["tokens"]),
            tags=[str(t) for t in row["upos"]],
            lang_code=iso,
            sent_id=str(row.get("idx", i)),
        ))
    return out


# ---------------------------------------------------------------------------
# Subsampling utility — for training the on-distribution CrossEncoder.
# ---------------------------------------------------------------------------

def subsample_token_budget(
    pretok_repos: Sequence[str],
    *,
    target_tokens: int = 400_000_000,
    seq_len: int = 512,
    seed: int = 0,
    cache_dir: Optional[str] = None,
) -> Iterator[List[int]]:
    """Yield pretokenised sequences from one or more HF repos until
    ``target_tokens`` cumulative tokens have been emitted.

    The McCoy-et-al. principle (cited in the user spec): a probe trained
    on the same distribution as the model under study reflects the
    model's behaviour, not the probe's.  We use this to sample 400M
    tokens out of FineWeb-2B for CrossEncoder training, mirroring the
    Beetle bilingual training distribution.

    Each yielded item is a list of ``seq_len`` token IDs.
    """
    import random

    from datasets import load_dataset  # type: ignore

    rng = random.Random(seed)
    emitted = 0
    streams = [
        load_dataset(r, split="train", streaming=True, cache_dir=cache_dir)
        for r in pretok_repos
    ]
    iters = [iter(s) for s in streams]

    while emitted < target_tokens and any(it is not None for it in iters):
        # Round-robin with a small random jitter so we don't deterministic-
        # ally exhaust one stream first.
        for k in rng.sample(range(len(iters)), len(iters)):
            it = iters[k]
            if it is None:
                continue
            try:
                row = next(it)
            except StopIteration:
                iters[k] = None
                continue
            ids = row.get("input_ids") or row.get("ids") or row.get("tokens")
            if ids is None:
                continue
            ids = list(ids)[:seq_len]
            if len(ids) < 2:
                continue
            yield ids
            emitted += len(ids)
            if emitted >= target_tokens:
                return
