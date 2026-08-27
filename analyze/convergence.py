"""
convergence.py — Checkpoint-level convergence and embedding drift analysis.

Mirrors the checkpoint infrastructure from eval_model.py:
  - Uses `list_checkpoints()` + the step-N branch naming convention
  - Loads one checkpoint at a time, frees it before loading the next
    (same RAM contract as eval_model.py)

Three tracking signals across training steps
--------------------------------------------
1. PPL TRAJECTORY
2. EMBEDDING DRIFT (per checkpoint)
3. REPRESENTATIONAL SIMILARITY ACROSS STEPS (CKA trajectory)

Outputs
-------
results/convergence/
  ppl_traj_<repo_slug>.csv       — step, lang, mean_ppl, mean_nll
  drift_traj_<repo_slug>.csv     — step, word, cosine_dist_from_step0
  cka_traj_<repo_slug>.csv       — step_a, step_b, cka  (consecutive pairs)
  all_ppl_trajectories.csv       — concatenated PPL trajectories, all models
"""

from __future__ import annotations
from analyze import _path  # noqa: F401  — adds repo root to sys.path

import gc
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

from models import MODEL_GROUPS, get_bilingual_type, get_lang_pair
from ppl_utils import (
    FLORES_LANG_MAP,
    GOLDFISH_TOKENS,
    load_flores_sentences,
    nll_to_ppl,
    sentence_log_likelihood,
)
from embedding_drift import PROBE_WORDS, token_ids_for_words

try:
    from eval_ledger import record as _ledger_record
except Exception:  # pragma: no cover — ledger is best-effort
    def _ledger_record(*_a, **_kw) -> None:
        pass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR      = Path("results/convergence")
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
PPL_SAMPLE_SIZE = 200

# Registry of (repo, revision) pairs that failed to load.
CONVERGENCE_FAILURES: dict[str, str] = {}   # key = "repo@revision"


# ---------------------------------------------------------------------------
# Checkpoint listing
# ---------------------------------------------------------------------------

def _step_number(name: str) -> Optional[int]:
    """Parse a step index from a checkpoint branch name.

    Recognises:
      * ``step-<N>``  — Beetle convention.
      * ``<N>``       — B-GPT convention (numeric-only branch names).
      * everything else (including ``main``) → None.
    """
    m = re.match(r"^step-(\d+)$", name)
    if m:
        return int(m.group(1))
    if re.match(r"^\d+$", name):
        return int(name)
    return None


def _log_spaced_subset(steps: list[int], ratio: float = 1.4) -> list[int]:
    """Return a log-spaced subsample of *steps* (assumed already sorted asc).

    Greedy: keep step iff it is >= ratio × last-kept-step (with 0 always
    kept and the last step always kept). This thins the dense linear-
    sampled checkpoints found in the original Beetle-FineWeb 24B repos
    down to a publication-grade trajectory (~25 ckpts) while preserving
    the early dense sampling of newer log-spaced repos."""
    if not steps:
        return []
    kept: list[int] = []
    last = -1
    for s in steps:
        if s == 0 or (last <= 0 and s > 0) or s >= max(1, last) * ratio:
            kept.append(s)
            last = s
    if steps[-1] not in kept:
        kept.append(steps[-1])
    return kept


def list_checkpoints(
    repo: str,
    hf_token: Optional[str] = None,
    *,
    all_checkpoints: bool = False,
    log_spaced: bool = False,
) -> list[str]:
    """Enumerate checkpoints for *repo*.

    Final-checkpoint mode (``all_checkpoints=False``, the project default)
    returns ``["main"]`` without hitting the Hub. Trajectory mode
    (``all_checkpoints=True``, opt-in via ``--ckpt-scope all``) returns
    ``["main"] + sorted(step-N)`` for Beetle-style repos, or the sorted
    numeric branches for B-GPT-style repos (where ``main`` is the alias of
    the largest numeric branch and gets de-duplicated).

    With ``log_spaced=True``, the step branches are thinned to a greedy
    log-spaced subset (~25 per repo) so dense linear-sampled cohorts (the
    original Beetle-FineWeb 24B run, which saved 200+ checkpoints per
    model) do not dominate eval cost.
    """
    if not all_checkpoints:
        return ["main"]
    try:
        from huggingface_hub import list_repo_refs
        refs = list_repo_refs(repo, token=hf_token)
        all_branches = [r.name for r in refs.branches]
        step_branches = sorted(
            [b for b in all_branches if _step_number(b) is not None],
            key=lambda n: _step_number(n),
        )
        # B-GPT: "main" is the alias of the final numeric branch — keep
        # "main" as the canonical label and drop the duplicate numeric
        # branch so step_label() can map main → max_step.
        if "main" in all_branches and step_branches:
            largest = step_branches[-1]
            if re.match(r"^\d+$", largest):
                step_branches = step_branches[:-1]
        if log_spaced and step_branches:
            steps_int = [_step_number(b) for b in step_branches]
            keep_set = set(_log_spaced_subset(steps_int))
            step_branches = [b for b in step_branches
                             if _step_number(b) in keep_set]
        return ["main"] + step_branches
    except Exception as e:
        logger.warning(f"[list_checkpoints] {repo}: {e} — returning ['main'] only")
        return ["main"]


def step_label(revision: str, max_step: Optional[int] = None) -> int:
    """Return the integer step for a revision.

    For ``"main"``, returns *max_step* if supplied (B-GPT case where main
    aliases the largest numeric branch), else 0. Existing callers that pass
    only ``revision`` keep the historical behaviour of mapping main → 0.
    """
    if revision == "main":
        return int(max_step) if max_step is not None else 0
    n = _step_number(revision)
    return n if n is not None else 0


# ---------------------------------------------------------------------------
# Checkpoint loading — returns None on failure, records in CONVERGENCE_FAILURES
# ---------------------------------------------------------------------------

def _load_tokenizer_json_only(
    repo: str,
    revision: str,
    hf_token: Optional[str] = None,
) -> PreTrainedTokenizerFast:
    """Last-resort tokenizer loader for repos whose tokenizer_config.json
    trips the Rust `tokenizers` parser (e.g. B-GPT checkpoints).

    Downloads only ``tokenizer.json`` and instantiates a
    ``PreTrainedTokenizerFast`` directly, then re-attaches special tokens
    read from ``special_tokens_map.json`` if present. Skips the broken
    ``tokenizer_config.json`` entirely.
    """
    import json
    from huggingface_hub import hf_hub_download

    dl_kw = {"repo_id": repo, "revision": revision}
    if hf_token:
        dl_kw["token"] = hf_token

    tok_json = hf_hub_download(filename="tokenizer.json", **dl_kw)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_json)

    # Best-effort: pull special tokens from special_tokens_map.json so
    # eos/pad/bos are populated for downstream scoring code.
    try:
        stm_path = hf_hub_download(filename="special_tokens_map.json", **dl_kw)
        with open(stm_path) as f:
            stm = json.load(f)
        # Values may be plain strings or {"content": "...", ...} dicts.
        def _content(v):
            return v["content"] if isinstance(v, dict) and "content" in v else v
        add = {k: _content(v) for k, v in stm.items() if v}
        if add:
            tokenizer.add_special_tokens(add)
    except Exception as e:
        logger.debug(f"  no special_tokens_map.json for {repo}@{revision} ({e})")
    return tokenizer


def _load_checkpoint(
    repo: str,
    revision: str,
    hf_token: Optional[str] = None,
    device: str = DEVICE,
) -> tuple | None:
    """
    Load (model, tokenizer) for one checkpoint.
    Returns None and records the error if loading fails.
    Caller must del both when done.
    """
    load_kw = dict(
        revision          = revision,
        trust_remote_code = True,
        torch_dtype       = torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage = True,
    )
    if hf_token:
        load_kw["token"] = hf_token

    try:
        # Fast tokenizer first; fall back to slow on ANY failure. B-GPT
        # checkpoints ship a custom fast-tokenizer config that the Rust
        # `tokenizers` lib rejects with a TypeError ("argument
        # 'special_tokens': Expected Union[…]"), which is not a ValueError
        # or ImportError — so we have to catch broadly here.
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo, **load_kw, use_fast=True)
        except Exception as fast_exc:
            logger.debug(f"  fast-tokenizer failed for {repo}@{revision} ({fast_exc}); trying slow")
            try:
                tokenizer = AutoTokenizer.from_pretrained(repo, **load_kw, use_fast=False)
            except Exception as slow_exc:
                # B-GPT only ships tokenizer.json — no slow-tokenizer assets
                # AND its tokenizer_config.json trips the Rust parser. Bypass
                # both by loading tokenizer.json directly, which ignores the
                # config's broken special_tokens field.
                logger.debug(f"  slow-tokenizer failed for {repo}@{revision} ({slow_exc}); "
                             f"loading tokenizer.json directly")
                tokenizer = _load_tokenizer_json_only(repo, revision, hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Use device_map={"": device} so weights are materialised directly on
        # the target device, avoiding the 2x host-RAM peak of
        # `from_pretrained(...).to(device)` (CPU materialise then copy).
        # BGPT custom configs occasionally reject device_map; fall back to
        # the legacy path with a logged warning.
        try:
            model = (
                AutoModelForCausalLM
                .from_pretrained(repo, **load_kw, device_map={"": device})
                .eval()
            )
        except (ValueError, TypeError, NotImplementedError) as dm_exc:
            logger.warning(
                f"  device_map load failed for {repo}@{revision} ({dm_exc}); "
                f"falling back to .to({device})"
            )
            model = (
                AutoModelForCausalLM
                .from_pretrained(repo, **load_kw)
                .to(device)
                .eval()
            )
        return model, tokenizer

    except Exception as exc:
        msg = str(exc).split("\n")[0]
        key = f"{repo}@{revision}"
        logger.error(f"  SKIP {key}: {msg}")
        CONVERGENCE_FAILURES[key] = msg
        return None


def _free(model, tokenizer) -> None:
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def clear_failures() -> None:
    """Clear the in-process failure registry.

    Called at the start of each multi-repo dispatch so memory diagnostics do
    not bleed across stages.  Snapshot is dumped to disk by the dispatcher
    before clearing if it wants persistent diagnostics.
    """
    CONVERGENCE_FAILURES.clear()


# ---------------------------------------------------------------------------
# Signal 1: PPL trajectory
# ---------------------------------------------------------------------------

def _ppl_for_checkpoint(model, tokenizer, sentences, goldfish_tokens=GOLDFISH_TOKENS):
    nlls = [sentence_log_likelihood(s, model, tokenizer, goldfish_tokens) for s in sentences]
    return float(np.mean([n for n in nlls if n < float("inf")]))


def compute_ppl_trajectory(
    repo: str,
    lang_codes: list[str],
    hf_token: Optional[str] = None,
    sample_size: int = PPL_SAMPLE_SIZE,
    overwrite: bool = False,
    *,
    all_checkpoints: bool = False,
) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug     = repo.replace("/", "__")
    out_path = OUTPUT_DIR / f"ppl_traj_{slug}.csv"

    if out_path.exists() and not overwrite:
        logger.info(f"[ppl_traj] Loading cached {out_path}")
        return pd.read_csv(out_path)

    sentences_by_lang: dict[str, list[str]] = {}
    for code in lang_codes:
        if code not in FLORES_LANG_MAP:
            logger.warning(f"[ppl_traj] No FLORES mapping for {code!r} — skipping")
            continue
        all_sents = load_flores_sentences(code)
        step = max(1, len(all_sents) // sample_size)
        sentences_by_lang[code] = all_sents[::step][:sample_size]
        logger.info(f"[ppl_traj] {code}: {len(sentences_by_lang[code])} sentences")

    checkpoints = list_checkpoints(repo, hf_token, all_checkpoints=all_checkpoints)
    logger.info(f"[ppl_traj] {repo}: {len(checkpoints)} checkpoints × {len(sentences_by_lang)} langs")

    # For BGPT-style repos, "main" aliases the largest numeric branch;
    # supply that as the step label so ordering is correct downstream.
    numeric_steps = [_step_number(c) for c in checkpoints]
    numeric_steps = [s for s in numeric_steps if s is not None]
    max_step = max(numeric_steps) if numeric_steps else None

    rows: list[dict] = []
    for revision in checkpoints:
        step   = step_label(revision, max_step=max_step)
        result = _load_checkpoint(repo, revision, hf_token)
        if result is None:
            logger.warning(f"  SKIPPED step {step} ({revision}) — load failed.")
            continue
        model, tokenizer = result

        try:
            for code, sentences in sentences_by_lang.items():
                mean_nll = _ppl_for_checkpoint(model, tokenizer, sentences)
                rows.append({
                    "repo":     repo,
                    "revision": revision,
                    "step":     step,
                    "lang":     code,
                    "mean_nll": round(mean_nll, 4),
                    "mean_ppl": round(nll_to_ppl(mean_nll), 2),
                })
                logger.info(f"    {code}: PPL={nll_to_ppl(mean_nll):.1f}")
        finally:
            _free(model, tokenizer)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info(f"[ppl_traj] saved {out_path}")
    # One ledger row per (repo, checkpoint, language) — mirrors the rows in
    # the CSV so backfill + in-process writes converge on the same keys.
    for row in rows:
        _ledger_record(
            stage="convergence", repo=repo,
            checkpoint=row.get("revision"), benchmark="ppl_traj",
            lang=row.get("lang"), status="ok",
        )
    return df


# ---------------------------------------------------------------------------
# Signal 1b: Held-out NLL trajectory (in-domain validation curve)
# ---------------------------------------------------------------------------

# Sibling beetle-data tree containing FineWeb held-out / decontaminated text.
_BEETLE_DATA_DEFAULT = Path(__file__).resolve().parents[2] / "beetle-data"


def _resolve_heldout_source(
    repo: str,
    lang: str,
    *,
    explicit_dir: Optional[Path] = None,
    beetle_data_root: Optional[Path] = None,
) -> tuple[str, str | Path] | None:
    """Decide where to load held-out sentences for (repo, lang).

    Returns (kind, source) where kind is "hf" (HuggingFace dataset id) or
    "dir" (local parquet directory). Returns None if no source resolves.

    Precedence:
      1. ``explicit_dir`` if provided.
      2. HuggingFace ``Beetle-HumanScale/BabyBabel-{lang}-held-out`` for
         HumanScale repos.
      3. ``$beetle_data_root/pipeline_output/held_out_babybabel/{lang}``.
      4. ``$beetle_data_root/pipeline_output/decontaminated/{ISO_2}``
         (fallback for FineWeb / BGPT proxy).
    """
    if explicit_dir is not None:
        p = Path(explicit_dir)
        if p.exists():
            return ("dir", p)

    # HumanScale published held-out on the Hub.
    try:
        from models import resolve_held_out_repo  # local import — avoids cycles
        repo_l = repo.lower()
        regime = "humanscale" if "humanscale" in repo_l else (
            "fineweb" if "fineweb" in repo_l else None
        )
        if regime is not None:
            hub_repo = resolve_held_out_repo(regime, lang)
            if hub_repo:
                return ("hf", hub_repo)
    except Exception as e:
        logger.warning(f"[heldout] resolve_held_out_repo failed: {e}")

    root = Path(beetle_data_root) if beetle_data_root else _BEETLE_DATA_DEFAULT
    bb = root / "pipeline_output" / "held_out_babybabel" / lang
    if bb.exists() and any(bb.glob("*.parquet")):
        return ("dir", bb)

    # FineWeb / BGPT proxy: pipeline_output/decontaminated/<2-letter>.
    iso2_map = {
        "eng": "en", "nld": "nl", "deu": "de", "zho": "zh",
        "spa": "es", "ell": "el", "pol": "pl", "fra": "fr",
        "ita": "it", "rus": "ru",
    }
    iso2 = iso2_map.get(lang)
    if iso2:
        decon = root / "pipeline_output" / "decontaminated" / iso2
        if decon.exists() and any(decon.glob("*.parquet")):
            return ("dir", decon)

    return None


def _load_heldout_sentences(
    kind: str, source: "str | Path", sample_size: int, seed: int = 0,
) -> list[str]:
    """Load up to *sample_size* sentences from a held-out source."""
    rng = np.random.default_rng(seed)

    if kind == "hf":
        from datasets import load_dataset
        ds = load_dataset(str(source), split="train", streaming=False)
        col = "text" if "text" in ds.column_names else ds.column_names[0]
        texts = [t for t in ds[col] if isinstance(t, str) and t.strip()]
    else:  # "dir" — parquet directory
        import pyarrow.parquet as pq
        files = sorted(Path(source).glob("*.parquet"))
        texts: list[str] = []
        for fp in files:
            tbl = pq.read_table(fp, columns=["text"])
            for t in tbl.column("text").to_pylist():
                if isinstance(t, str) and t.strip():
                    texts.append(t)
                    if len(texts) >= sample_size * 20:
                        break
            if len(texts) >= sample_size * 20:
                break

    if not texts:
        return []
    if len(texts) <= sample_size:
        return texts
    idx = rng.choice(len(texts), size=sample_size, replace=False)
    return [texts[i] for i in idx]


def compute_heldout_trajectory(
    repo: str,
    lang_codes: list[str],
    hf_token: Optional[str] = None,
    sample_size: int = PPL_SAMPLE_SIZE,
    overwrite: bool = False,
    *,
    all_checkpoints: bool = False,
    held_out_dir: Optional[Path] = None,
    beetle_data_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Per-checkpoint mean NLL on the held-out training-data slice.

    Mirrors :func:`compute_ppl_trajectory` but scores sentences sampled
    from the held-out portion of the training corpus rather than FLORES,
    so the curve validates that FLORES NLL tracks in-domain loss.

    Output: ``results/convergence/heldout_traj_{slug}.csv`` with columns
    ``repo, revision, step, lang, mean_nll, mean_ppl``.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug     = repo.replace("/", "__")
    out_path = OUTPUT_DIR / f"heldout_traj_{slug}.csv"

    if out_path.exists() and not overwrite:
        logger.info(f"[heldout_traj] Loading cached {out_path}")
        return pd.read_csv(out_path)

    sentences_by_lang: dict[str, list[str]] = {}
    for code in lang_codes:
        src = _resolve_heldout_source(
            repo, code,
            explicit_dir=held_out_dir,
            beetle_data_root=beetle_data_root,
        )
        if src is None:
            logger.warning(f"[heldout_traj] No held-out source for ({repo}, {code}) — skipping")
            continue
        kind, source = src
        try:
            sents = _load_heldout_sentences(kind, source, sample_size)
        except Exception as e:
            logger.warning(f"[heldout_traj] {repo}/{code}: load failed ({e}) — skipping")
            continue
        if not sents:
            logger.warning(f"[heldout_traj] {repo}/{code}: empty source ({source}) — skipping")
            continue
        sentences_by_lang[code] = sents
        logger.info(f"[heldout_traj] {repo}/{code}: {len(sents)} sentences from {source}")

    if not sentences_by_lang:
        logger.warning(f"[heldout_traj] {repo}: no langs scoreable — skipping")
        return pd.DataFrame()

    checkpoints = list_checkpoints(repo, hf_token, all_checkpoints=all_checkpoints)
    numeric_steps = [s for s in (_step_number(c) for c in checkpoints) if s is not None]
    max_step = max(numeric_steps) if numeric_steps else None
    logger.info(f"[heldout_traj] {repo}: {len(checkpoints)} checkpoints × {len(sentences_by_lang)} langs")

    rows: list[dict] = []
    for revision in checkpoints:
        step = step_label(revision, max_step=max_step)
        result = _load_checkpoint(repo, revision, hf_token)
        if result is None:
            logger.warning(f"  SKIPPED step {step} ({revision}) — load failed.")
            continue
        model, tokenizer = result

        try:
            for code, sentences in sentences_by_lang.items():
                mean_nll = _ppl_for_checkpoint(model, tokenizer, sentences)
                rows.append({
                    "repo":     repo,
                    "revision": revision,
                    "step":     step,
                    "lang":     code,
                    "mean_nll": round(mean_nll, 4),
                    "mean_ppl": round(nll_to_ppl(mean_nll), 2),
                })
                logger.info(f"    [heldout] {code}: PPL={nll_to_ppl(mean_nll):.1f}")
        finally:
            _free(model, tokenizer)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info(f"[heldout_traj] saved {out_path}")
    for row in rows:
        _ledger_record(
            stage="convergence", repo=repo,
            checkpoint=row.get("revision"), benchmark="heldout_traj",
            lang=row.get("lang"), status="ok",
        )
    return df


# ---------------------------------------------------------------------------
# Signal 1c: Bits-per-token (length-normalised NLL) trajectory
# ---------------------------------------------------------------------------
#
# compute_ppl_trajectory / compute_heldout_trajectory above report the mean
# of per-sentence mean-token NLL — already length-normalised in nats. To
# put BGPT and Beetle-FineWeb on the same axis (paper Fig. comparing loss
# curves across architectures + tokenisers), we want BITS PER TOKEN (= NLL
# / ln(2)). This function computes that and returns a separate CSV so the
# legacy mean_nll outputs are unaffected.

import math as _math
_BPC_FACTOR = 1.0 / _math.log(2.0)  # nats → bits


def _bpt_for_checkpoint(model, tokenizer, sentences, goldfish_tokens=GOLDFISH_TOKENS):
    """Mean bits-per-token across *sentences*.

    sentence_log_likelihood already returns per-sentence MEAN-token NLL
    (length-normalised, in nats). We multiply by 1/ln(2) to convert to
    bits and average across sentences. This matches the BLiSS BPT metric
    in evals/eval_bliss.py (BPC_FACTOR), so a model evaluated on FLORES
    here is directly comparable to a BLiSS bits-per-token score.
    """
    nlls = [sentence_log_likelihood(s, model, tokenizer, goldfish_tokens) for s in sentences]
    finite = [n for n in nlls if n < float("inf")]
    if not finite:
        return float("nan")
    return float(np.mean(finite)) * _BPC_FACTOR


def compute_bpt_trajectory(
    repo: str,
    lang_codes: list[str],
    source: str = "flores",                 # "flores" | "heldout"
    hf_token: Optional[str] = None,
    sample_size: int = PPL_SAMPLE_SIZE,
    overwrite: bool = False,
    *,
    all_checkpoints: bool = False,
    held_out_dir: Optional[Path] = None,
    beetle_data_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Per-checkpoint bits-per-token trajectory for *repo*.

    Reuses the same checkpoint enumeration + load/free contract as the
    PPL trajectories. Output:
      ``results/convergence/bpt_traj_{slug}_{source}.csv``
    with columns ``[repo, revision, step, lang, source, n_sents, mean_bpt]``.

    Used by scripts/eval_loss_curves_bgpt_vs_beetle_dutch.sh to overlay
    BGPT and Beetle-FineWeb training curves on a tokeniser-invariant axis.
    """
    if source not in ("flores", "heldout"):
        raise ValueError(f"source must be 'flores' or 'heldout', got {source!r}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug     = repo.replace("/", "__")
    out_path = OUTPUT_DIR / f"bpt_traj_{slug}_{source}.csv"

    if out_path.exists() and not overwrite:
        # Treat 0-byte / unreadable / row-less files as cache misses so a
        # prior failed run cannot poison subsequent runs.
        try:
            cached = pd.read_csv(out_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            cached = None
        if cached is not None and len(cached) > 0:
            logger.info(f"[bpt_traj/{source}] Loading cached {out_path}")
            return cached
        logger.info(f"[bpt_traj/{source}] Cached {out_path} is empty — recomputing")

    # Resolve sentences per language. Identical resolution to the existing
    # FLORES / heldout trajectory functions so the eval data matches.
    sentences_by_lang: dict[str, list[str]] = {}
    if source == "flores":
        for code in lang_codes:
            if code not in FLORES_LANG_MAP:
                logger.warning(f"[bpt_traj] No FLORES mapping for {code!r} — skipping")
                continue
            all_sents = load_flores_sentences(code)
            step = max(1, len(all_sents) // sample_size)
            sentences_by_lang[code] = all_sents[::step][:sample_size]
            logger.info(f"[bpt_traj/flores] {code}: {len(sentences_by_lang[code])} sentences")
    else:  # heldout
        for code in lang_codes:
            src = _resolve_heldout_source(
                repo, code,
                explicit_dir=held_out_dir,
                beetle_data_root=beetle_data_root,
            )
            if src is None:
                logger.warning(f"[bpt_traj/heldout] No source for ({repo}, {code}) — skipping")
                continue
            kind, source_path = src
            try:
                sents = _load_heldout_sentences(kind, source_path, sample_size)
            except Exception as e:
                logger.warning(f"[bpt_traj/heldout] {repo}/{code}: load failed ({e})")
                continue
            if not sents:
                logger.warning(f"[bpt_traj/heldout] {repo}/{code}: empty source — skipping")
                continue
            sentences_by_lang[code] = sents
            logger.info(f"[bpt_traj/heldout] {repo}/{code}: {len(sents)} sentences from {source_path}")

    if not sentences_by_lang:
        logger.warning(f"[bpt_traj/{source}] {repo}: no langs scoreable — skipping")
        return pd.DataFrame()

    checkpoints = list_checkpoints(repo, hf_token, all_checkpoints=all_checkpoints)
    numeric_steps = [s for s in (_step_number(c) for c in checkpoints) if s is not None]
    max_step = max(numeric_steps) if numeric_steps else None
    logger.info(f"[bpt_traj/{source}] {repo}: {len(checkpoints)} checkpoints "
                f"× {len(sentences_by_lang)} langs")

    rows: list[dict] = []
    tmp_path = out_path.with_suffix(".csv.tmp")
    for revision in checkpoints:
        step = step_label(revision, max_step=max_step)
        try:
            result = _load_checkpoint(repo, revision, hf_token)
        except Exception as e:
            logger.warning(f"  SKIPPED step {step} ({revision}) — load raised: {e}")
            continue
        if result is None:
            logger.warning(f"  SKIPPED step {step} ({revision}) — load failed.")
            continue
        model, tokenizer = result

        try:
            for code, sentences in sentences_by_lang.items():
                try:
                    mean_bpt = _bpt_for_checkpoint(model, tokenizer, sentences)
                except Exception as e:
                    logger.warning(f"    [{source}] {code}: BPT failed at step {step} ({e})")
                    continue
                rows.append({
                    "repo":     repo,
                    "revision": revision,
                    "step":     step,
                    "lang":     code,
                    "source":   source,
                    "n_sents":  len(sentences),
                    "mean_bpt": round(mean_bpt, 4),
                })
                logger.info(f"    [{source}] {code}: BPT={mean_bpt:.3f}")
        finally:
            _free(model, tokenizer)

        # Flush after every checkpoint via atomic rename so a crash/Ctrl-C
        # never leaves a half-written CSV at out_path.
        if rows:
            pd.DataFrame(rows).to_csv(tmp_path, index=False)
            tmp_path.replace(out_path)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_path, index=False)
    logger.info(f"[bpt_traj/{source}] saved {out_path} ({len(df)} rows)")
    for row in rows:
        _ledger_record(
            stage="convergence", repo=repo,
            checkpoint=row.get("revision"),
            benchmark=f"bpt_traj_{source}",
            lang=row.get("lang"), status="ok",
        )
    return df


# ---------------------------------------------------------------------------
# Single-checkpoint subprocess mode
# ---------------------------------------------------------------------------
#
# The bash launcher (scripts/_loss_curves_lib.sh) runs one Python subprocess
# per checkpoint so the kernel reclaims model RAM, glibc arenas, and CUDA
# context between checkpoints.  This helper does the work of one such
# subprocess: resolve eval sentences, skip-if-already-scored, load, score,
# atomically append rows, exit.
#
# CSV layout for ppl/heldout (matches existing files):
#   repo, revision, step, lang, mean_nll, mean_ppl
# CSV layout for bpt:
#   repo, revision, step, lang, source, n_sents, mean_bpt

def _append_rows_atomic(out_path: Path, rows: list[dict]) -> None:
    """Append `rows` to `out_path`, file-locked, header-aware.

    Mirrors the row-append pattern from evals/eval_blimp.py:_append_row but
    serialised behind fcntl.flock so concurrent ranks targeting the same
    CSV (rare, but possible) cannot interleave half-written lines.
    """
    if not rows:
        return
    import fcntl
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    df = pd.DataFrame(rows)
    with open(lock_path, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            df.to_csv(
                out_path, mode="a",
                header=not out_path.exists(),
                index=False,
            )
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def _csv_has_step(out_path: Path, revision: str, lang: str) -> bool:
    """True if `out_path` already contains a row for (revision, lang)."""
    if not out_path.exists() or out_path.stat().st_size == 0:
        return False
    try:
        df = pd.read_csv(out_path, usecols=lambda c: c in {"revision", "lang"})
    except Exception:
        return False
    if df.empty or "revision" not in df.columns or "lang" not in df.columns:
        return False
    mask = (df["revision"].astype(str) == str(revision)) & (df["lang"] == lang)
    return bool(mask.any())


def score_single_checkpoint(
    repo: str,
    lang_codes: list[str],
    revision: str,
    *,
    source: str = "flores",                 # "flores" | "heldout"
    metric: str = "ppl",                    # "ppl" | "bpt"
    hf_token: Optional[str] = None,
    sample_size: int = PPL_SAMPLE_SIZE,
    overwrite: bool = False,
    held_out_dir: Optional[Path] = None,
    beetle_data_root: Optional[Path] = None,
) -> None:
    """Score one checkpoint of `repo` and atomically append rows to the CSV.

    Designed to be invoked by a bash launcher in a per-checkpoint subprocess
    loop: each invocation loads ONE model, scores, frees, and exits — so
    cumulative RSS growth across checkpoints is structurally impossible.

    Output paths match the multi-checkpoint trajectory functions:
      metric=ppl,  source=flores  -> ppl_traj_<slug>.csv
      metric=ppl,  source=heldout -> heldout_traj_<slug>.csv
      metric=bpt,  source=*       -> bpt_traj_<slug>_<source>.csv
    """
    if metric not in ("ppl", "bpt"):
        raise ValueError(f"metric must be 'ppl' or 'bpt', got {metric!r}")
    if source not in ("flores", "heldout"):
        raise ValueError(f"source must be 'flores' or 'heldout', got {source!r}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = repo.replace("/", "__")
    if metric == "bpt":
        out_path = OUTPUT_DIR / f"bpt_traj_{slug}_{source}.csv"
    elif source == "heldout":
        out_path = OUTPUT_DIR / f"heldout_traj_{slug}.csv"
    else:
        out_path = OUTPUT_DIR / f"ppl_traj_{slug}.csv"

    # Step 1: per-(revision,lang) resume check.  Any langs already scored
    # for this revision drop out of the work list.
    if not overwrite:
        lang_codes = [
            c for c in lang_codes
            if not _csv_has_step(out_path, revision, c)
        ]
        if not lang_codes:
            logger.info(
                f"[{metric}_traj/{source}] {repo}@{revision}: all langs "
                f"already scored — skip"
            )
            return

    # Step 2: resolve sentences.
    sentences_by_lang: dict[str, list[str]] = {}
    if source == "flores":
        for code in lang_codes:
            if code not in FLORES_LANG_MAP:
                logger.warning(f"  no FLORES mapping for {code!r} — skipping")
                continue
            all_sents = load_flores_sentences(code)
            stride = max(1, len(all_sents) // sample_size)
            sentences_by_lang[code] = all_sents[::stride][:sample_size]
    else:  # heldout
        for code in lang_codes:
            src = _resolve_heldout_source(
                repo, code,
                explicit_dir=held_out_dir,
                beetle_data_root=beetle_data_root,
            )
            if src is None:
                logger.warning(f"  no held-out source for ({repo},{code}) — skipping")
                continue
            kind, source_path = src
            try:
                sents = _load_heldout_sentences(kind, source_path, sample_size)
            except Exception as e:
                logger.warning(f"  held-out load failed for ({repo},{code}): {e}")
                continue
            if sents:
                sentences_by_lang[code] = sents

    if not sentences_by_lang:
        logger.warning(f"[{metric}_traj/{source}] {repo}@{revision}: no langs scoreable")
        return

    # Step 3: figure out the integer step label.
    #   For BGPT, "main" aliases the largest numeric branch — ask the Hub.
    if revision == "main":
        try:
            ckpts = list_checkpoints(repo, hf_token, all_checkpoints=True)
        except Exception:
            ckpts = ["main"]
        numeric_steps = [s for s in (_step_number(c) for c in ckpts) if s is not None]
        max_step = max(numeric_steps) if numeric_steps else None
    else:
        max_step = None
    step = step_label(revision, max_step=max_step)

    # Step 4: load → score → free.  Subprocess exit reclaims everything.
    result = _load_checkpoint(repo, revision, hf_token)
    if result is None:
        logger.error(f"  {repo}@{revision}: load failed — exiting cleanly")
        return
    model, tokenizer = result

    rows: list[dict] = []
    try:
        for code, sentences in sentences_by_lang.items():
            try:
                if metric == "bpt":
                    val = _bpt_for_checkpoint(model, tokenizer, sentences)
                    rows.append({
                        "repo":     repo,
                        "revision": revision,
                        "step":     step,
                        "lang":     code,
                        "source":   source,
                        "n_sents":  len(sentences),
                        "mean_bpt": round(val, 4),
                    })
                    logger.info(f"    [{source}] {code}: BPT={val:.3f}")
                else:
                    mean_nll = _ppl_for_checkpoint(model, tokenizer, sentences)
                    rows.append({
                        "repo":     repo,
                        "revision": revision,
                        "step":     step,
                        "lang":     code,
                        "mean_nll": round(mean_nll, 4),
                        "mean_ppl": round(nll_to_ppl(mean_nll), 2),
                    })
                    logger.info(f"    [{source}] {code}: PPL={nll_to_ppl(mean_nll):.1f}")
            except Exception as e:
                logger.warning(f"    [{source}] {code}: scoring failed at step {step}: {e}")
    finally:
        _free(model, tokenizer)

    _append_rows_atomic(out_path, rows)
    for row in rows:
        _ledger_record(
            stage="convergence", repo=repo,
            checkpoint=revision,
            benchmark=(f"bpt_traj_{source}" if metric == "bpt"
                       else (f"heldout_traj" if source == "heldout"
                             else "ppl_traj")),
            lang=row.get("lang"), status="ok",
        )
    logger.info(f"[{metric}_traj/{source}] {repo}@{revision}: appended {len(rows)} rows")


# ---------------------------------------------------------------------------
# Signal 2: Probe-word embedding drift across checkpoints
# ---------------------------------------------------------------------------

def _probe_vecs_for_checkpoint(model, tokenizer, probe_words):
    emb_weight = model.get_input_embeddings().weight.detach().float().cpu()
    probe_map  = token_ids_for_words(probe_words, tokenizer)
    return {word: emb_weight[ids[0]].numpy()
            for word, ids in probe_map.items() if len(ids) == 1}


def compute_drift_trajectory(
    repo: str,
    iso_code: str,
    hf_token: Optional[str] = None,
    overwrite: bool = False,
    *,
    all_checkpoints: bool = False,
) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug     = repo.replace("/", "__")
    out_path = OUTPUT_DIR / f"drift_traj_{slug}.csv"

    if out_path.exists() and not overwrite:
        logger.info(f"[drift_traj] Loading cached {out_path}")
        return pd.read_csv(out_path)

    probe_words = PROBE_WORDS.get(iso_code, [])
    checkpoints = list_checkpoints(repo, hf_token, all_checkpoints=all_checkpoints)
    logger.info(f"[drift_traj] {repo}: {len(checkpoints)} checkpoints, {len(probe_words)} probe words")

    logger.info("  Loading step-0 anchor (main) …")
    result0 = _load_checkpoint(repo, "main", hf_token)
    if result0 is None:
        logger.error(f"  SKIPPED {repo} drift — step-0 anchor failed to load.")
        return pd.DataFrame()
    model0, tok0 = result0
    try:
        step0_vecs = _probe_vecs_for_checkpoint(model0, tok0, probe_words)
    finally:
        _free(model0, tok0)

    rows: list[dict] = []
    prev_vecs = step0_vecs.copy()

    for revision in checkpoints:
        step   = step_label(revision)
        result = _load_checkpoint(repo, revision, hf_token)
        if result is None:
            logger.warning(f"  SKIPPED step {step} ({revision}) — load failed.")
            continue
        model, tokenizer = result
        try:
            ckpt_vecs = _probe_vecs_for_checkpoint(model, tokenizer, probe_words)
        finally:
            _free(model, tokenizer)

        for word in step0_vecs:
            if word not in ckpt_vecs:
                continue
            v0    = torch.tensor(step0_vecs[word]).unsqueeze(0)
            vt    = torch.tensor(ckpt_vecs[word]).unsqueeze(0)
            vprev = torch.tensor(prev_vecs.get(word, step0_vecs[word])).unsqueeze(0)
            rows.append({
                "repo":              repo,
                "iso_code":          iso_code,
                "revision":          revision,
                "step":              step,
                "word":              word,
                "cosine_dist_step0": round(1.0 - cosine_similarity(v0, vt).item(), 5),
                "cosine_dist_prev":  round(1.0 - cosine_similarity(vprev, vt).item(), 5),
            })
        prev_vecs = ckpt_vecs

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info(f"[drift_traj] saved {out_path}")
    return df


# ---------------------------------------------------------------------------
# Signal 3: Consecutive-checkpoint CKA
# ---------------------------------------------------------------------------

def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    K_X = X @ X.T
    K_Y = Y @ Y.T
    n   = K_X.shape[0]
    H   = np.eye(n) - np.ones((n, n)) / n
    cKX = H @ K_X @ H
    cKY = H @ K_Y @ H
    return float(np.sum(cKX * cKY) / (np.linalg.norm(cKX, "fro") * np.linalg.norm(cKY, "fro") + 1e-10))


def compute_cka_trajectory(
    repo: str,
    iso_code: str,
    hf_token: Optional[str] = None,
    overwrite: bool = False,
    *,
    all_checkpoints: bool = False,
) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug     = repo.replace("/", "__")
    out_path = OUTPUT_DIR / f"cka_traj_{slug}.csv"

    if out_path.exists() and not overwrite:
        logger.info(f"[cka_traj] Loading cached {out_path}")
        return pd.read_csv(out_path)

    probe_words = PROBE_WORDS.get(iso_code, [])
    checkpoints = list_checkpoints(repo, hf_token, all_checkpoints=all_checkpoints)

    rows: list[dict] = []
    prev_vecs: dict[str, np.ndarray] | None = None
    prev_step = None
    prev_rev  = None

    for revision in checkpoints:
        step   = step_label(revision)
        result = _load_checkpoint(repo, revision, hf_token)
        if result is None:
            logger.warning(f"  SKIPPED step {step} ({revision}) — load failed.")
            continue
        model, tokenizer = result
        try:
            ckpt_vecs = _probe_vecs_for_checkpoint(model, tokenizer, probe_words)
        finally:
            _free(model, tokenizer)

        if prev_vecs is not None:
            shared = sorted(set(prev_vecs.keys()) & set(ckpt_vecs.keys()))
            if len(shared) >= 2:
                X   = np.stack([prev_vecs[w]  for w in shared])
                Y   = np.stack([ckpt_vecs[w]  for w in shared])
                cka = _linear_cka(X, Y)
                rows.append({
                    "repo":       repo,
                    "iso_code":   iso_code,
                    "revision_a": prev_rev,
                    "revision_b": revision,
                    "step_a":     prev_step,
                    "step_b":     step,
                    "delta_step": step - prev_step,
                    "cka":        round(cka, 5),
                    "change":     round(1.0 - cka, 5),
                })
                logger.info(f"    CKA({prev_step}→{step}) = {cka:.4f}")

        prev_vecs = ckpt_vecs
        prev_step = step
        prev_rev  = revision

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info(f"[cka_traj] saved {out_path}")
    return df


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def _iso_codes_for_repo(repo: str) -> list[str]:
    return [code for code, models in MODEL_GROUPS.items() if repo in models]


def _lang_pair_codes(repo: str) -> list[str]:
    name  = repo.split("/")[-1].replace("beetlelm_", "")
    for tag in ("_mono", "_balanced", "_simultaneous", "_sequential",
                "_part_time", "_late", "_heritage"):
        name = name.replace(tag, "")
    name  = re.sub(r"_L[12]", "", name)
    codes = re.split(r"[-_]", name)
    known = set(FLORES_LANG_MAP.keys())
    return [c for c in codes if c in known]


def run_convergence_analysis(
    repos: list[str],
    hf_token: Optional[str] = None,
    overwrite: bool = False,
    signals: list[str] = ("ppl", "drift", "cka"),
    rank: int = 0,
    world_size: int = 1,
    all_repos: Optional[list[str]] = None,
    output_root: "Path | str" = "results",
    all_checkpoints: bool = False,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    When ``world_size > 1``, ``repos`` is this rank's shard and ``all_repos``
    is the full list used on rank 0 to assemble ``all_ppl_trajectories.csv``
    after a barrier.
    """
    from _rank_barrier import mark_done, reset, wait_for_all

    if rank == 0:
        reset("convergence", world_size, output_root)

    results: dict[str, dict] = {}

    from tqdm.auto import tqdm
    logger.info(f"Convergence: {len(repos)} repo(s) on this rank")
    for repo in tqdm(repos, desc="convergence", unit="repo"):
        iso_codes   = _iso_codes_for_repo(repo)
        lang_pair   = _lang_pair_codes(repo)
        primary_iso = iso_codes[0] if iso_codes else (lang_pair[0] if lang_pair else None)

        logger.info(f"\n{'='*60}")
        logger.info(f"Convergence: {repo}")
        logger.info(f"  ISO groups : {iso_codes}  |  PPL langs: {lang_pair}  |  Primary: {primary_iso}")

        results[repo] = {}

        if "ppl" in signals and lang_pair:
            results[repo]["ppl"] = compute_ppl_trajectory(
                repo, lang_pair, hf_token=hf_token, overwrite=overwrite,
                all_checkpoints=all_checkpoints,
            )

        if "drift" in signals and primary_iso:
            results[repo]["drift"] = compute_drift_trajectory(
                repo, primary_iso, hf_token=hf_token, overwrite=overwrite,
                all_checkpoints=all_checkpoints,
            )

        if "cka" in signals and primary_iso:
            results[repo]["cka"] = compute_cka_trajectory(
                repo, primary_iso, hf_token=hf_token, overwrite=overwrite,
                all_checkpoints=all_checkpoints,
            )

    # Signal this rank is done; only rank 0 builds the combined summary.
    mark_done("convergence", rank, world_size, output_root)
    if rank != 0:
        return results
    if world_size > 1:
        wait_for_all("convergence", world_size, output_root)

    # Concatenate all PPL trajectories by reading per-repo CSVs from disk
    # (every rank wrote its own slug-namespaced file).
    summary_repos = all_repos if all_repos is not None else repos
    ppl_frames: list[pd.DataFrame] = []
    for r in summary_repos:
        slug = r.replace("/", "__")
        p = OUTPUT_DIR / f"ppl_traj_{slug}.csv"
        if p.exists():
            df = pd.read_csv(p)
            if not df.empty:
                ppl_frames.append(df)
    if ppl_frames:
        combined_path = OUTPUT_DIR / "all_ppl_trajectories.csv"
        pd.concat(ppl_frames, ignore_index=True).to_csv(combined_path, index=False)
        logger.info(f"\n[convergence] All PPL trajectories → {combined_path}")

    if CONVERGENCE_FAILURES:
        logger.info(f"\n[convergence] {'='*56}")
        logger.info(f"[convergence] SKIPPED {len(CONVERGENCE_FAILURES)} checkpoint(s) due to load errors:")
        for key, reason in CONVERGENCE_FAILURES.items():
            logger.info(f"  ✗  {key}")
            logger.info(f"     {reason}")
        logger.info(f"[convergence] {'='*56}")

    return results


# ---------------------------------------------------------------------------
# Focus repo sets for the key research questions
# ---------------------------------------------------------------------------

FOCUS_REPOS = {
    "forgetting_focus": [
        "BeetleLM/beetlelm_nld_mono",
        "BeetleLM/beetlelm_eng-nld_simultaneous",
        "BeetleLM/beetlelm_eng-nld_balanced",
        "BeetleLM/beetlelm_eng-nld_late",
        "BeetleLM/beetlelm_deu_mono",
        "BeetleLM/beetlelm_eng-deu_simultaneous",
        "BeetleLM/beetlelm_bul-deu_sequential",
        "BeetleLM/beetlelm_zho_mono",
        "BeetleLM/beetlelm_zho-eng_balanced",
        "BeetleLM/beetlelm_zho_L1-eng_L2_simultaneous",
    ],
    "rt_focus": [
        "BeetleLM/beetlelm_eng-nld_simultaneous",
        "BeetleLM/beetlelm_nld_L1-eng_L2_simultaneous",
        "BeetleLM/beetlelm_zho-eng_balanced",
        "BeetleLM/beetlelm_zho_L1-eng_L2_balanced",
    ],
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run checkpoint convergence analysis.")
    sub = parser.add_subparsers(dest="cmd")

    # Default (legacy) entry point.
    parser.add_argument("--repos",     nargs="+")
    parser.add_argument("--focus",     choices=list(FOCUS_REPOS.keys()), default="forgetting_focus")
    parser.add_argument("--signals",   nargs="+", choices=["ppl", "drift", "cka"],
                        default=["ppl", "drift", "cka"])
    parser.add_argument("--hf_token",  default=None)
    parser.add_argument("--overwrite", action="store_true")

    # flores-trajectory subcommand
    p_flores = sub.add_parser(
        "flores-trajectory",
        help="Compute FLORES NLL trajectory (per-checkpoint mean NLL) for one repo.",
    )
    p_flores.add_argument("--repo", required=True)
    p_flores.add_argument("--langs", nargs="+", required=True,
                          help="ISO-639-3 codes (e.g. eng nld).")
    p_flores.add_argument("--all-checkpoints", action="store_true")
    p_flores.add_argument(
        "--checkpoint", default=None,
        help="If set, score only this revision; the bash launcher uses this "
             "to run one checkpoint per Python subprocess (kernel reclaims "
             "all memory between checkpoints).",
    )
    p_flores.add_argument("--sample-size", type=int, default=PPL_SAMPLE_SIZE)
    p_flores.add_argument("--overwrite", action="store_true")
    p_flores.add_argument("--hf-token", default=None)

    # heldout-trajectory subcommand
    p_held = sub.add_parser(
        "heldout-trajectory",
        help="Compute held-out training-data NLL trajectory for one repo.",
    )
    p_held.add_argument("--repo", required=True)
    p_held.add_argument("--langs", nargs="+", required=True)
    p_held.add_argument("--all-checkpoints", action="store_true")
    p_held.add_argument("--checkpoint", default=None,
                        help="Score only this revision (subprocess-per-checkpoint mode).")
    p_held.add_argument("--sample-size", type=int, default=PPL_SAMPLE_SIZE)
    p_held.add_argument("--overwrite", action="store_true")
    p_held.add_argument("--hf-token", default=None)
    p_held.add_argument("--held-out-dir", type=Path, default=None,
                        help="Optional explicit parquet directory (overrides resolution).")
    p_held.add_argument("--beetle-data-root", type=Path, default=None,
                        help="Override the sibling beetle-data root.")

    # bpt-trajectory subcommand — bits-per-token (length-normalised NLL).
    # Used to overlay BGPT vs Beetle-FineWeb on a tokeniser-invariant axis.
    p_bpt = sub.add_parser(
        "bpt-trajectory",
        help="Compute mean bits-per-token trajectory for one repo "
             "(length-normalised; comparable across tokenisers).",
    )
    p_bpt.add_argument("--repo", required=True)
    p_bpt.add_argument("--langs", nargs="+", required=True)
    p_bpt.add_argument("--source", choices=["flores", "heldout"], default="flores",
                       help="Eval data: FLORES devtest or held-out training slice.")
    p_bpt.add_argument("--all-checkpoints", action="store_true")
    p_bpt.add_argument("--checkpoint", default=None,
                       help="Score only this revision (subprocess-per-checkpoint mode).")
    p_bpt.add_argument("--sample-size", type=int, default=PPL_SAMPLE_SIZE)
    p_bpt.add_argument("--overwrite", action="store_true")
    p_bpt.add_argument("--hf-token", default=None)
    p_bpt.add_argument("--held-out-dir", type=Path, default=None)
    p_bpt.add_argument("--beetle-data-root", type=Path, default=None)

    # list-checkpoints subcommand — used by the bash launcher to drive the
    # per-checkpoint subprocess loop.
    p_list = sub.add_parser(
        "list-checkpoints",
        help="Print one checkpoint revision per line for the given repo.",
    )
    p_list.add_argument("--repo", required=True)
    p_list.add_argument("--all-checkpoints", action="store_true")
    p_list.add_argument("--log-spaced", action="store_true",
                        help="Thin step-N branches to a log-spaced subset "
                             "(reuses Beetle-FineWeb 24B's 200+ ckpts as "
                             "~25 publication-grade trajectory points).")
    p_list.add_argument("--hf-token", default=None)

    args = parser.parse_args()

    if args.cmd == "list-checkpoints":
        for r in list_checkpoints(args.repo, args.hf_token,
                                  all_checkpoints=args.all_checkpoints,
                                  log_spaced=args.log_spaced):
            print(r)
    elif args.cmd == "flores-trajectory":
        if getattr(args, "checkpoint", None):
            score_single_checkpoint(
                args.repo, args.langs, args.checkpoint, source="flores",
                metric="ppl",
                hf_token=args.hf_token, sample_size=args.sample_size,
                overwrite=args.overwrite,
            )
        else:
            compute_ppl_trajectory(
                args.repo, args.langs,
                hf_token=args.hf_token,
                sample_size=args.sample_size,
                overwrite=args.overwrite,
                all_checkpoints=args.all_checkpoints,
            )
    elif args.cmd == "heldout-trajectory":
        if getattr(args, "checkpoint", None):
            score_single_checkpoint(
                args.repo, args.langs, args.checkpoint, source="heldout",
                metric="ppl",
                hf_token=args.hf_token, sample_size=args.sample_size,
                overwrite=args.overwrite,
                held_out_dir=args.held_out_dir,
                beetle_data_root=args.beetle_data_root,
            )
        else:
            compute_heldout_trajectory(
                args.repo, args.langs,
                hf_token=args.hf_token,
                sample_size=args.sample_size,
                overwrite=args.overwrite,
                all_checkpoints=args.all_checkpoints,
                held_out_dir=args.held_out_dir,
                beetle_data_root=args.beetle_data_root,
            )
    elif args.cmd == "bpt-trajectory":
        if getattr(args, "checkpoint", None):
            score_single_checkpoint(
                args.repo, args.langs, args.checkpoint, source=args.source,
                metric="bpt",
                hf_token=args.hf_token, sample_size=args.sample_size,
                overwrite=args.overwrite,
                held_out_dir=args.held_out_dir,
                beetle_data_root=args.beetle_data_root,
            )
        else:
            compute_bpt_trajectory(
                args.repo, args.langs,
                source=args.source,
                hf_token=args.hf_token,
                sample_size=args.sample_size,
                overwrite=args.overwrite,
                all_checkpoints=args.all_checkpoints,
                held_out_dir=args.held_out_dir,
                beetle_data_root=args.beetle_data_root,
            )
    else:
        repos = args.repos if args.repos else FOCUS_REPOS[args.focus]
        run_convergence_analysis(repos, hf_token=args.hf_token,
                                 overwrite=args.overwrite, signals=args.signals)