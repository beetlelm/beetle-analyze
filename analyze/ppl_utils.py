"""
ppl_utils.py — Shared utilities for the BeetleLM analysis pipeline.
(Named ppl_utils to avoid collision with the repo-root utils.py used by eval_model.py.)

Provides:
  - Model / tokenizer loading with caching
  - Sentence-level log-likelihood (token-mean NLL → PPL)
  - FLORES-200 sentence loader per ISO-639-3 language code
  - Goldfish truncation (first-N-token filtering)
"""

from __future__ import annotations

import math
import os
import re
from functools import lru_cache
from typing import Iterator, TYPE_CHECKING

import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FLORES_SPLIT = "devtest"   # 1012 sentences per language

# Goldfish: only score the first GOLDFISH_TOKENS tokens of each sentence.
# Set to None to disable (score full sentence).
GOLDFISH_TOKENS: int | None = 512

# Map BeetleLM ISO codes → FLORES-200 BCP-47 language tags
FLORES_LANG_MAP: dict[str, str] = {
    # Core languages
    "nld": "nld_Latn",
    "deu": "deu_Latn",
    "zho": "zho_Hans",
    "fra": "fra_Latn",
    "fas": "pes_Arab",
    "bul": "bul_Cyrl",
    "eng": "eng_Latn",
    "ukr": "ukr_Cyrl",
    "ind": "ind_Latn",
    "pol": "pol_Latn",
    "spa": "spa_Latn",
    "ell": "ell_Grek",
    "jpn": "jpn_Jpan",
    "ita": "ita_Latn",
    "eus": "eus_Latn",
    "tur": "tur_Latn",
    "tgl": "fil_Latn",
    "hin": "hin_Deva",
    "tam": "tam_Taml",
    "swe": "swe_Latn",
    "rus": "rus_Cyrl",
    "cat": "cat_Latn",
    "arb": "arb_Arab",
    # Extension languages
    "urd": "urd_Arab",
    "ben": "ben_Beng",
    "ces": "ces_Latn",
    "guj": "guj_Gujr",
    "tha": "tha_Thai",
    "vie": "vie_Latn",
    "kor": "kor_Hang",
    "dan": "dan_Latn",
    # Low-resource
    "hun": "hun_Latn",
    "hrv": "hrv_Latn",
    "slv": "slv_Latn",
    "som": "som_Latn",
    "amh": "amh_Ethi",
    "yor": "yor_Latn",
    "wol": "wol_Latn",
}


# ---------------------------------------------------------------------------
# FLORES sentence loading — direct parquet, no dataset scripts
# ---------------------------------------------------------------------------

def load_flores_sentences(iso_code: str) -> list[str]:
    """
    Return FLORES-200 devtest sentences for *iso_code*.

    Uses crystina-z/flores200 — a parquet-native dataset with all 200
    languages as columns named sentence_{tag} (e.g. sentence_nld_Latn).
    No legacy dataset script; loads cleanly with any datasets version.
    """
    flores_tag = FLORES_LANG_MAP.get(iso_code)
    if flores_tag is None:
        raise ValueError(
            f"No FLORES tag for '{iso_code}'. Add to FLORES_LANG_MAP in ppl_utils.py."
        )

    col = f"sentence_{flores_tag}"
    ds  = load_dataset("crystina-z/flores200", "all", split="devtest",
                       trust_remote_code=False)

    if col not in ds.column_names:
        raise ValueError(
            f"Column '{col}' not in crystina-z/flores200. "
            f"Available: {[c for c in ds.column_names if c.startswith('sentence_')][:10]}"
        )

    sentences = [s for s in ds[col] if isinstance(s, str) and s.strip()]
    print(f"[flores] {iso_code}: {len(sentences)} sentences (col={col})")
    return sentences



# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_best_revision(repo: str, hf_token: str | None = None) -> str:
    """Final-checkpoint mode: every eval stage pins ``revision="main"``.

    The project has exactly two checkpoint policies:
      * final (default) — score ``main`` once; this function is the
        single source of truth.
      * trajectory (opt-in via ``--ckpt-scope all``) — handled by
        :func:`_list_checkpoints` in ``ppl_eval.py`` and by the
        corresponding ``all_checkpoints`` branches in
        ``convergence.list_checkpoints`` /
        ``aoa.get_beetle_surprisals.get_checkpoints``.

    Accepting ``hf_token`` keeps the signature backwards-compatible for
    callers that used to pass it.
    """
    del hf_token  # unused; kept for signature compatibility
    return "main"


def final_revision_row_mask(df: "pd.DataFrame") -> "pd.Series":
    """Return a boolean mask selecting the final revision per repo.

    "Final" is defined as ``main`` when present in the repo's rows,
    otherwise the numerically-largest ``step-N`` branch. Used by
    final-snapshot figure readers (paper_figures, ppl_heatmap, …) to
    collapse an all-checkpoints ``ppl_summary.csv`` back to one row
    per (repo, iso_code).
    """
    import pandas as pd

    if df.empty or "repo" not in df.columns or "revision" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)

    def _rank(rev: object) -> int:
        if not isinstance(rev, str):
            return -1
        if rev == "main":
            return 10 ** 18          # main beats any step-N
        m = re.match(r"^step-(\d+)$", rev)
        return int(m.group(1)) if m else -1

    ranks = df["revision"].map(_rank)
    # For each repo, select the index of the max-ranked revision.
    idx = ranks.groupby(df["repo"]).idxmax()
    mask = pd.Series(False, index=df.index)
    mask.loc[idx.values] = True
    return mask


# Registry of repos that failed to load — populated by load_model_and_tokenizer,
# read by ppl_eval / forgetting / embedding_drift to print a skip summary.
# Keyed on ``f"{repo}@{revision}"`` so failures on one checkpoint do not
# blacklist the whole repo.
LOAD_FAILURES: dict[str, str] = {}


def clear_failures() -> None:
    """Clear the in-process load-failure registry.

    Multi-repo dispatchers should call this at stage boundaries so the
    dict cannot grow unbounded across long-running sweeps.
    """
    LOAD_FAILURES.clear()


# Module-level (repo, revision) -> (model, tokenizer) cache with bounded size.
# We keep size small because in practice we iterate one checkpoint at a time
# per rank/process and call ``clear_model_cache()`` between models.
_MODEL_CACHE: dict[tuple[str, str], tuple] = {}
_MODEL_CACHE_MAX = 1


def load_model_and_tokenizer(
    repo: str,
    device: str = DEVICE,
    revision: str | None = None,
) -> tuple | None:
    """
    Load (model, tokenizer) for *repo* at *revision*.

    If *revision* is None, falls back to :func:`get_best_revision` for
    backwards compatibility with callers that pre-date all-checkpoints mode.
    Cached on ``(repo, revision)`` so iterating over checkpoints reuses the
    currently loaded one without reloading the whole pipeline.

    Returns None (and records the error in LOAD_FAILURES) if the model
    cannot be loaded, so callers can skip gracefully rather than crashing.
    """
    if revision is None:
        revision = get_best_revision(repo)

    key = (repo, revision)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    print(f"[utils] Loading {repo} @ {revision} …")
    load_kw = dict(revision=revision, trust_remote_code=True)
    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo, **load_kw, use_fast=True)
        except Exception as tok_exc:
            # B-GPT checkpoints ship a custom fast-tokenizer config that
            # PyO3 rejects with a bare TypeError, so we must catch broadly.
            print(f"[utils] Fast tokenizer unavailable for {repo} ({tok_exc}); using slow tokenizer.")
            tokenizer = AutoTokenizer.from_pretrained(repo, **load_kw, use_fast=False)
        # device_map={"": device} materialises weights directly on the
        # target device, avoiding the 2x host-RAM peak from
        # `from_pretrained(...).to(device)`.  Falls back to the legacy path
        # for any custom config that rejects accelerate placement.
        dtype = torch.float16 if device == "cuda" else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(
                repo,
                **load_kw,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map={"": device},
            )
        except (ValueError, TypeError, NotImplementedError) as dm_exc:
            print(f"[utils] device_map load failed for {repo}@{revision} "
                  f"({dm_exc}); falling back to .to({device})")
            model = AutoModelForCausalLM.from_pretrained(
                repo,
                **load_kw,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            model.to(device)
        model.eval()
    except Exception as exc:
        msg = str(exc).split("\n")[0]          # first line only — keep logs tidy
        print(f"[utils] SKIP {repo}@{revision}: {msg}")
        LOAD_FAILURES[f"{repo}@{revision}"] = msg
        return None

    # Evict older entries to bound RAM.
    while len(_MODEL_CACHE) >= _MODEL_CACHE_MAX:
        _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
    _MODEL_CACHE[key] = (model, tokenizer)
    return model, tokenizer


def clear_model_cache():
    """Release all cached models and free GPU memory.

    Call this between model evaluations to prevent RAM explosion when
    iterating over many models.
    """
    import gc
    _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Log-likelihood / perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def sentence_log_likelihood(
    sentence: str,
    model,
    tokenizer,
    goldfish_tokens: int | None = GOLDFISH_TOKENS,
    stride: int = 512,
) -> float:
    """
    Return the mean-token negative log-likelihood (NLL) for *sentence*.

    Lower is better (more likely under the model).
    Perplexity = exp(NLL).

    Args:
        sentence:        The raw string to score.
        model:           A loaded CausalLM.
        tokenizer:       Matching tokenizer.
        goldfish_tokens: If set, only score the first N tokens (Goldfish
                         evaluation protocol). Set to None for full sentence.
        stride:          Sliding-window stride for long sequences.

    Returns:
        Mean NLL (float). Returns float("inf") on tokenisation error.
    """
    encodings = tokenizer(sentence, return_tensors="pt")
    input_ids: torch.Tensor = encodings.input_ids.to(model.device)
    seq_len = input_ids.size(1)

    # Goldfish truncation
    if goldfish_tokens is not None and seq_len > goldfish_tokens:
        input_ids = input_ids[:, :goldfish_tokens]
        seq_len = goldfish_tokens

    if seq_len < 2:
        return float("inf")

    loss_fn = CrossEntropyLoss(reduction="sum")
    nlls: list[float] = []
    token_count = 0

    max_len = getattr(model.config, "max_position_embeddings", 2048)
    window = min(max_len, 1024)

    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + window, seq_len)
        target_len = end - prev_end

        chunk = input_ids[:, begin:end]
        labels = chunk.clone()
        # Mask out the context tokens that were already scored
        labels[:, :-target_len] = -100

        with torch.amp.autocast("cuda", enabled=(model.device.type == "cuda")):
            outputs = model(chunk, labels=labels)

        # outputs.loss is mean NLL over non-masked tokens; recover sum
        num_scored = (labels != -100).sum().item() - 1  # shift by 1
        if num_scored <= 0:
            prev_end = end
            continue

        nlls.append(outputs.loss.item() * num_scored)
        token_count += num_scored
        prev_end = end

        if end == seq_len:
            break

    if token_count == 0:
        return float("inf")

    return sum(nlls) / token_count  # mean NLL across all scored tokens


def nll_to_ppl(nll: float) -> float:
    """Convert mean NLL to perplexity."""
    return math.exp(nll) if nll < 700 else float("inf")


# ---------------------------------------------------------------------------
# Batch scoring helper
# ---------------------------------------------------------------------------

def score_sentences(
    sentences: list[str],
    repo: str,
    goldfish_tokens: int | None = GOLDFISH_TOKENS,
    verbose: bool = True,
    revision: str | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> list[tuple[int, float]] | None:
    """Score *sentences* under *repo* @ *revision*, optionally sharded.

    When *world_size* > 1, this rank scores only
    ``sentences[rank::world_size]``. Return value is a list of
    ``(sentence_id, nll)`` pairs — the ``sentence_id`` is the index into
    the original full *sentences* list, so callers can reassemble after a
    barrier. Returns None if the model failed to load.

    Backwards-compatible: with the default ``world_size=1``, the result
    is equivalent to the old unsharded call (just with ids attached).
    """
    # Clear cache before loading a new model to prevent RAM accumulation
    clear_model_cache()
    result = load_model_and_tokenizer(repo, revision=revision)
    if result is None:
        return None                              # load failed — caller skips
    model, tokenizer = result

    my_indices = list(range(rank, len(sentences), world_size)) \
        if world_size > 1 else list(range(len(sentences)))
    my_sents = [sentences[i] for i in my_indices]

    from tqdm.auto import tqdm
    tag = f"{repo}@{revision}" if revision else repo
    shard_tag = f"[rank {rank}/{world_size}]" if world_size > 1 else ""
    iterator = tqdm(my_sents, desc=f"ppl{shard_tag}[{tag}]", leave=False,
                    disable=not verbose, unit="sent")
    indexed: list[tuple[int, float]] = []
    for i, sent in zip(my_indices, iterator):
        nll = sentence_log_likelihood(sent, model, tokenizer, goldfish_tokens)
        indexed.append((i, nll))
    return indexed