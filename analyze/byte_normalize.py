"""
analyze/byte_normalize.py — Cross-tokenizer-fair perplexity conversion.

Per-token perplexity (PPL) and bits-per-token (BPT) are NOT comparable across
models that use different tokenizers (Beetle vs B-GPT vs XLM-R vs mBERT all
have different vocabularies). The standard fix in the LM-evaluation literature
is **bits-per-byte (BPB)**: total cross-entropy in bits divided by the number
of UTF-8 bytes in the corpus. BPB is invariant to the tokenizer.

This module provides:

- ``flores_bytes(iso_code)``        UTF-8 byte count of FLORES-200 devtest.
- ``flores_tokens(repo, iso_code)`` token count of the same corpus under the
  ``repo`` tokenizer (with caching).
- ``bpt_to_bpb(bpt, repo, iso_code)`` / ``bpt_to_bpb_series(...)``
  Convert per-token BPT to per-byte BPB:
      BPB = BPT * (n_tokens / n_bytes)
  This is exact whenever ``bpt`` is the corpus-mean BPT for the same corpus
  whose token / byte counts are returned here.

If the tokenizer cannot be loaded (offline / private repo), the conversion
falls back to returning ``bpt`` unchanged and emits a warning so plots remain
runnable without GPU/network at the cost of a non-fair comparison.

Cache lives at ``results/cache/bpt_bpb_ratios.csv`` so subsequent plot runs
are fast and reproducible.
"""
from __future__ import annotations

import logging
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE_PATH = _REPO_ROOT / "results" / "cache" / "bpt_bpb_ratios.csv"


# ---------------------------------------------------------------------------
# Byte counts
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def flores_bytes(iso_code: str) -> int:
    """Total UTF-8 byte count of the FLORES-200 devtest corpus for ``iso_code``."""
    from analyze.ppl_utils import load_flores_sentences

    sents = load_flores_sentences(iso_code)
    return sum(len(s.encode("utf-8")) for s in sents)


@lru_cache(maxsize=128)
def flores_tokens(repo: str, iso_code: str, revision: str = "main") -> int | None:
    """Total token count of the FLORES-200 devtest corpus under ``repo``'s tokenizer.

    Returns None if the tokenizer cannot be loaded (offline / private repo) so
    callers can fall back to BPT without crashing.
    """
    cached = _read_cache()
    key = (repo, iso_code, revision)
    if cached is not None and key in cached.index:
        return int(cached.loc[key, "n_tokens"])

    try:
        from transformers import AutoTokenizer

        from analyze.ppl_utils import load_flores_sentences

        tok = AutoTokenizer.from_pretrained(
            repo, revision=revision, trust_remote_code=True, use_fast=True
        )
    except Exception as exc:
        try:
            tok = AutoTokenizer.from_pretrained(
                repo, revision=revision, trust_remote_code=True, use_fast=False
            )
        except Exception as exc2:  # noqa: BLE001
            warnings.warn(
                f"[byte_normalize] tokenizer load failed for {repo}@{revision}: {exc2}; "
                f"BPT→BPB falls back to identity",
                stacklevel=2,
            )
            return None

    sents = load_flores_sentences(iso_code)
    n_tokens = sum(len(tok.encode(s, add_special_tokens=False)) for s in sents)
    _write_cache_row(repo, iso_code, revision, n_tokens, flores_bytes(iso_code))
    return n_tokens


def _read_cache() -> pd.DataFrame | None:
    if not _CACHE_PATH.exists():
        return None
    df = pd.read_csv(_CACHE_PATH)
    if df.empty:
        return None
    return df.set_index(["repo", "iso_code", "revision"])


def _write_cache_row(repo: str, iso_code: str, revision: str,
                     n_tokens: int, n_bytes: int) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "repo": repo, "iso_code": iso_code, "revision": revision,
        "n_tokens": n_tokens, "n_bytes": n_bytes,
        "tokens_per_byte": n_tokens / max(n_bytes, 1),
    }])
    if _CACHE_PATH.exists():
        existing = pd.read_csv(_CACHE_PATH)
        merged = pd.concat([existing, row], ignore_index=True)
        merged = merged.drop_duplicates(["repo", "iso_code", "revision"], keep="last")
    else:
        merged = row
    merged.to_csv(_CACHE_PATH, index=False)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def tokens_per_byte(repo: str, iso_code: str, revision: str = "main") -> float | None:
    """Return n_tokens / n_bytes for the FLORES devtest. None if uncomputable."""
    n_tok = flores_tokens(repo, iso_code, revision=revision)
    if n_tok is None:
        return None
    n_byte = flores_bytes(iso_code)
    if n_byte <= 0:
        return None
    return n_tok / n_byte


def bpt_to_bpb(bpt: float, repo: str, iso_code: str,
               revision: str = "main") -> float:
    """Convert mean bits-per-token to bits-per-byte for ``repo`` on FLORES ``iso_code``.

    BPB = BPT * (n_tokens / n_bytes).

    If the tokenizer cannot be loaded, returns ``bpt`` unchanged so plots
    remain runnable without GPU/network — emit a warning when this happens.
    """
    ratio = tokens_per_byte(repo, iso_code, revision=revision)
    if ratio is None:
        return float(bpt)
    return float(bpt) * ratio


def bpt_to_bpb_series(df: pd.DataFrame,
                      *, bpt_col: str = "mean_bpt",
                      repo_col: str = "repo",
                      lang_col: str = "lang",
                      out_col: str = "mean_bpb") -> pd.DataFrame:
    """Vectorised BPT→BPB conversion. Mutates a copy of ``df``.

    The lang column may use 2-letter (``nl``, ``en``) or 3-letter ISO codes;
    ``LANG_2_TO_3`` normalises to 3-letter for FLORES lookup.
    """
    df = df.copy()
    df[out_col] = df.apply(
        lambda r: bpt_to_bpb(
            float(r[bpt_col]),
            str(r[repo_col]),
            _normalize_iso(str(r[lang_col])),
            revision=str(r.get("revision", "main")),
        ) if pd.notna(r[bpt_col]) else float("nan"),
        axis=1,
    )
    return df


LANG_2_TO_3 = {
    "en": "eng", "nl": "nld", "de": "deu", "zh": "zho",
    "fr": "fra", "es": "spa", "ja": "jpn", "ko": "kor", "ru": "rus",
}


def _normalize_iso(lang: str) -> str:
    lang = lang.strip().lower()
    return LANG_2_TO_3.get(lang, lang)


# ---------------------------------------------------------------------------
# CLI: precompute the cache for a list of (repo, lang) pairs
# ---------------------------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repos", nargs="+", required=True,
                   help="HF repos to tokenise FLORES under")
    p.add_argument("--langs", nargs="+", default=["nld", "eng"])
    p.add_argument("--revision", default="main")
    args = p.parse_args(list(argv) if argv is not None else None)

    for repo in args.repos:
        for lang in args.langs:
            ratio = tokens_per_byte(repo, lang, revision=args.revision)
            print(f"{repo:60s}  {lang}  tokens/byte = {ratio}")


if __name__ == "__main__":
    main()
