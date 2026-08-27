#!/usr/bin/env python3
"""
evals/eval_aoa.py — Final-checkpoint Age-of-Acquisition evaluation.

Scores every model on its final published checkpoint (revision="main") against
the standardized human AoA norms in ``aoa/standardized/{lang}.csv``.

Per (model, language) we compute the Spearman rank correlation between

    * model score   — mean surprisal of the last target subword over up to 512
                      contexts per word, using the jsonl prepared by
                      ``aoa.prepare_eval_data``
    * human score   — ``aoa_value`` column of the standardized CSV

A model is evaluated only on languages whose ISO-639-3 code appears in its
repo name (matches ``scripts/run_beetlelm_blimp.sh`` parsing).  Baselines
(EuroLLM / XGLM / BLOOM) are evaluated on every language in
``models.LANGUAGES_WITH_AOA``.

Completed (repo, lang) pairs are recorded in the eval ledger under stage
``aoa_final`` and skipped on re-runs.

Usage
-----
  # All Beetle orgs + all baselines, final-checkpoint AoA, 8-GPU fan-out
  CUDA_VISIBLE_DEVICES=0 python evals/eval_aoa.py \\
      --models beetle-humanscale beetle-fineweb baselines \\
      --rank 0 --world-size 8 --regime humanscale \\
      --output results/evals/aoa

  # One model + one language (smoke test)
  python evals/eval_aoa.py \\
      --models Beetle-HumanScale/beetlelm_nld_L1-eng_L2_balanced \\
      --langs nld --output /tmp/aoa-smoke

  # Prep only (rank-0 preflight before launching workers)
  python evals/eval_aoa.py --models beetle-humanscale beetle-fineweb \\
      --prep-only --rank 0 --world-size 8 --output results/evals/aoa
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "analyze"))

from evals.baseline_models import resolve_models, BASELINE_MODELS
from evals.model_loader import (
    DEVICE, load_model, unload_model, setup_logging, resolve_vocab_size,
)

import logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ledger (best-effort — never crashes the eval loop)
# ---------------------------------------------------------------------------

try:
    from analyze.eval_ledger import is_done as _ledger_is_done
    from analyze.eval_ledger import record as _ledger_record
except Exception:
    def _ledger_is_done(*_a, **_kw) -> bool: return False  # type: ignore[misc]
    def _ledger_record(*_a, **_kw) -> None: pass            # type: ignore[misc]

try:
    from analyze._rank_barrier import mark_done, wait_for_all, reset
except Exception:
    def mark_done(*_a, **_kw) -> None: pass                 # type: ignore[misc]
    def wait_for_all(*_a, **_kw) -> bool: return True       # type: ignore[misc]
    def reset(*_a, **_kw) -> None: pass                     # type: ignore[misc]


# ---------------------------------------------------------------------------
# Language parsing from repo names (mirrors scripts/run_beetlelm_blimp.sh)
# ---------------------------------------------------------------------------

KNOWN_ISO: set[str] = {
    "eng", "nld", "deu", "zho", "fra", "fas", "bul", "ukr",
    "ind", "srp", "kor",
    # AoA languages beyond the BLiMP set
    "jpn", "spa", "rus", "por", "pol", "tur", "ell", "hun",
    "ces", "cat", "dan", "swe", "ita",
}

_TOKEN_SPLIT = re.compile(r"[^A-Za-z]+")
_L1_MARKER = re.compile(r"([a-z]{3})_L1", re.IGNORECASE)


def parse_model_langs(repo: str) -> list[str]:
    """Return ISO-639-3 codes found in *repo* name, in order of first appearance."""
    name = repo.split("/", 1)[-1]
    seen: set[str] = set()
    out: list[str] = []
    for tok in _TOKEN_SPLIT.split(name):
        t = tok.lower()
        if len(t) == 3 and t in KNOWN_ISO and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def parse_model_l1(repo: str, langs_in_name: list[str]) -> str:
    """L1 = explicit ``{lang}_L1`` marker if present, else first ISO in name.

    Baselines with no language in name (EuroLLM / XGLM / BLOOM) → ``multi``.
    """
    name = repo.split("/", 1)[-1]
    m = _L1_MARKER.search(name)
    if m:
        cand = m.group(1).lower()
        if cand in KNOWN_ISO:
            return cand
    if langs_in_name:
        return langs_in_name[0]
    return "multi"


def _is_baseline(repo: str) -> bool:
    for family, repos in BASELINE_MODELS.items():
        if repo in repos:
            return True
    return False


# ---------------------------------------------------------------------------
# Per-word surprisal scoring (final checkpoint only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _score_word_contexts(
    model,
    tokenizer,
    contexts: list[dict],
    device: str,
    batch_size: int,
) -> float:
    """Return mean surprisal (bits) of the last target subword across *contexts*.

    Uses the same batched autocast forward pass as
    ``evals/eval_blimp.py::score_sentences``. Contexts are already tokenised
    by ``aoa.prepare_eval_data``; we feed them straight as ``input_ids``.
    """
    vocab_size = resolve_vocab_size(model)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
    )

    per_context_bits: list[float] = []
    for start in range(0, len(contexts), batch_size):
        batch = contexts[start: start + batch_size]
        # Filter contexts whose last target position is 0 (no left context).
        usable = []
        for ctx in batch:
            tp = [p for p in ctx.get("target_token_positions", []) if p > 0]
            if not tp:
                continue
            last = max(tp)
            usable.append((ctx["token_ids"], last))
        if not usable:
            continue

        max_len = max(len(ids) for ids, _ in usable)
        max_len = min(max_len, 512)
        ids_tensor = torch.full(
            (len(usable), max_len), pad_id, dtype=torch.long, device=device,
        )
        for i, (ids, _) in enumerate(usable):
            cut = ids[:max_len]
            ids_tensor[i, : len(cut)] = torch.tensor(cut, dtype=torch.long)

        ids_tensor.clamp_(0, vocab_size - 1)
        attn = (ids_tensor != pad_id).long()

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            out = model(input_ids=ids_tensor, attention_mask=attn)
        log_probs = F.log_softmax(out.logits.float(), dim=-1)
        del out

        for i, (ids, last_pos) in enumerate(usable):
            last_pos = min(last_pos, max_len - 1)
            target_id = ids[last_pos]
            # Prediction for position `last_pos` lives at logits[:, last_pos-1]
            log_p = log_probs[i, last_pos - 1, target_id].item()
            per_context_bits.append(-log_p / math.log(2))

    if not per_context_bits:
        return float("nan")
    return float(np.mean(per_context_bits))


def score_jsonl(
    model,
    tokenizer,
    eval_jsonl: Path,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    """Iterate *eval_jsonl* and return a DataFrame with word-level metrics.

    Columns: ``word``, ``aoa_value``, ``n_contexts``, ``mean_surprisal_bits``.
    """
    rows: list[dict] = []
    with open(eval_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            contexts = rec.get("contexts", [])
            if not contexts:
                continue
            mean_bits = _score_word_contexts(
                model, tokenizer, contexts, device, batch_size,
            )
            rows.append({
                "word": rec["word"],
                "aoa_value": rec.get("aoa_value"),
                "n_contexts": len(contexts),
                "mean_surprisal_bits": mean_bits,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Eval-data preparation (rank-0 preflight)
# ---------------------------------------------------------------------------

def _corpus_for_regime(regime: str) -> str:
    return "babybabel" if regime == "humanscale" else "fineweb"


def _ensure_eval_jsonl(
    lang: str,
    tokenizer_repo: str,
    eval_data_dir: Path,
    regime: str,
    held_out_base: Optional[Path],
    n_contexts: int,
    min_tokens: int,
) -> Optional[Path]:
    """Ensure the per-(lang, tokenizer) eval jsonl exists; generate if missing.

    Returns the jsonl path on success, or None if prep failed.
    """
    corpus = _corpus_for_regime(regime)
    tok_tag = tokenizer_repo.split("/")[-1]
    eval_jsonl = eval_data_dir / f"{lang}_{corpus}_{tok_tag}.jsonl"
    if eval_jsonl.exists():
        return eval_jsonl

    # Step 1: standardize human AoA CSV if missing
    from aoa.standardize_aoa import standardize_all, filter_languages_to_lm_vocab
    standardized_csv = _REPO_ROOT / "aoa" / "standardized" / f"{lang}.csv"
    if not standardized_csv.exists():
        try:
            standardize_all(langs=[lang])
        except Exception as exc:
            logger.warning("standardize_all(%s) failed: %s", lang, exc)
            return None

    # Step 2: filter to LM vocab
    lm_vocab_csv = (
        _REPO_ROOT / "aoa" / "standardized"
        / f"{lang}_lm_vocab_{tok_tag}.csv"
    )
    if not lm_vocab_csv.exists():
        try:
            filter_languages_to_lm_vocab([lang], tokenizer_repo)
        except Exception as exc:
            logger.warning(
                "filter_languages_to_lm_vocab(%s, %s) failed: %s",
                lang, tokenizer_repo, exc,
            )
            return None

    # Step 3: prepare contexts from held-out corpus
    from aoa.run_learning_curves import resolve_held_out_dir
    from aoa.prepare_eval_data import prepare_eval_data
    from models import resolve_held_out_repo

    held_out_dir = None
    held_out_repo = None
    if held_out_base is not None:
        held_out_dir = resolve_held_out_dir(regime, lang, held_out_base)
    else:
        held_out_repo = resolve_held_out_repo(regime, lang)

    try:
        prepare_eval_data(
            lang=lang,
            output_dir=eval_data_dir,
            tokenizer_repo=tokenizer_repo,
            n_contexts=n_contexts,
            min_tokens=min_tokens,
            corpus=corpus,
            held_out_dir=held_out_dir,
            held_out_repo=held_out_repo,
            lm_vocab_csv=lm_vocab_csv if lm_vocab_csv.exists() else None,
        )
    except Exception as exc:
        logger.warning("prepare_eval_data(%s, %s) failed: %s",
                       lang, tokenizer_repo, exc)
        return None

    return eval_jsonl if eval_jsonl.exists() else None


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def _append_row(output_file: Path, row: dict) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        output_file, mode="a", header=not output_file.exists(), index=False,
    )


# ---------------------------------------------------------------------------
# Per-model evaluation plan
# ---------------------------------------------------------------------------

def _eligible_languages(
    repo: str,
    family: str,
    lang_filter: Optional[set[str]],
    aoa_languages: set[str],
) -> list[str]:
    """Return AoA languages to evaluate for (repo, family).

    * BeetleLM / Beetle-HumanScale / Beetle-FineWeb / custom: only the ISO
      codes that appear in the repo name.
    * Baselines (EuroLLM / XGLM / BLOOM): every language in AoA set.
    """
    if _is_baseline(repo) or family in {"eurollm", "xglm", "bloom"}:
        langs = list(aoa_languages)
    else:
        langs = parse_model_langs(repo)
    langs = [L for L in langs if L in aoa_languages]
    if lang_filter is not None:
        langs = [L for L in langs if L in lang_filter]
    # Preserve order of appearance in repo name, otherwise sort for baselines
    if _is_baseline(repo):
        langs = sorted(langs)
    return langs


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "aoa_results.csv"
    eval_data_dir = Path(args.eval_data_dir)
    eval_data_dir.mkdir(parents=True, exist_ok=True)

    setup_logging("aoa", output_dir, args.rank)

    from models import LANGUAGES_WITH_AOA, resolve_tokenizer_for_model

    aoa_langs = set(LANGUAGES_WITH_AOA.keys())
    lang_filter: Optional[set[str]] = set(args.langs) if args.langs else None

    all_model_pairs = resolve_models(args.models, output_dir=output_dir.parent)
    logger.info("Resolved %d total models from --models spec", len(all_model_pairs))

    # Build the full (repo, family, lang, tokenizer) plan.
    plan: list[tuple[str, str, str, str]] = []
    for repo, family in all_model_pairs:
        langs = _eligible_languages(repo, family, lang_filter, aoa_langs)
        for lang in langs:
            tok = resolve_tokenizer_for_model(repo, lang)
            if tok is None:
                logger.warning("No tokenizer resolvable for %s (lang=%s); skipping",
                               repo, lang)
                continue
            plan.append((repo, family, lang, tok))
    logger.info("Plan: %d (model, lang) tasks total", len(plan))

    # ------------------------------------------------------------------
    # Prep phase — rank 0 ensures every (lang, tokenizer) jsonl exists,
    # then drops a barrier sentinel so workers can start.
    # ------------------------------------------------------------------
    if not args.skip_prep:
        distinct_pairs: list[tuple[str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for _, _, lang, tok in plan:
            key = (lang, tok)
            if key not in seen_pairs:
                seen_pairs.add(key)
                distinct_pairs.append(key)

        if args.rank == 0:
            logger.info("rank 0: preparing %d distinct (lang, tokenizer) jsonl(s)",
                        len(distinct_pairs))
            for lang, tok in distinct_pairs:
                _ensure_eval_jsonl(
                    lang=lang,
                    tokenizer_repo=tok,
                    eval_data_dir=eval_data_dir,
                    regime=args.regime,
                    held_out_base=args.held_out_base,
                    n_contexts=args.n_contexts,
                    min_tokens=args.min_tokens,
                )
            (eval_data_dir / ".aoa_prep_done").write_text(f"{time.time()}\n")
        else:
            sentinel = eval_data_dir / ".aoa_prep_done"
            logger.info("rank %d: waiting for rank 0's eval-data prep…", args.rank)
            while not sentinel.exists():
                time.sleep(2.0)

    if args.prep_only:
        logger.info("--prep-only set; exiting after prep phase.")
        return

    # ------------------------------------------------------------------
    # Worker phase — round-robin model striping. Same pattern as
    # evals/eval_blimp.py: one rank loads each model once and loops over
    # all its eligible languages.
    # ------------------------------------------------------------------
    unique_models = list({(r, f) for r, f, _, _ in plan})
    unique_models.sort(key=lambda rf: rf[0])
    my_models = [
        rf for i, rf in enumerate(unique_models)
        if i % args.world_size == args.rank
    ]
    logger.info("[rank %d/%d] %d model(s) assigned",
                args.rank, args.world_size, len(my_models))

    corpus = _corpus_for_regime(args.regime)

    for repo, family in my_models:
        langs_in_name = parse_model_langs(repo)
        l1 = parse_model_l1(repo, langs_in_name)

        # Tasks for this model not yet recorded in the ledger.
        tasks: list[tuple[str, str]] = []  # (lang, tok_repo)
        for r, f, lang, tok in plan:
            if r != repo or f != family:
                continue
            if _ledger_is_done("aoa_final", repo, "main", "aoa", lang,
                               output_root=str(output_dir)):
                logger.info("SKIP %s | %s — already in ledger", repo, lang)
                continue
            tasks.append((lang, tok))
        if not tasks:
            logger.info("SKIP %s — all langs already in ledger", repo)
            continue

        result = load_model(repo, DEVICE)
        if result is None:
            continue
        model, tokenizer = result

        try:
            for lang, tok_repo in tasks:
                tok_tag = tok_repo.split("/")[-1]
                eval_jsonl = eval_data_dir / f"{lang}_{corpus}_{tok_tag}.jsonl"
                if not eval_jsonl.exists():
                    logger.warning("No eval jsonl at %s; skipping (%s | %s)",
                                   eval_jsonl, repo, lang)
                    continue

                human_csv = _REPO_ROOT / "aoa" / "standardized" / f"{lang}.csv"
                if not human_csv.exists():
                    logger.warning("No human AoA CSV at %s; skipping", human_csv)
                    continue

                logger.info("  %s | %s  (tok=%s)", repo, lang, tok_tag)
                t0 = time.time()
                try:
                    model_df = score_jsonl(
                        model, tokenizer, eval_jsonl, DEVICE, args.batch_size,
                    )
                    human_df = pd.read_csv(human_csv)
                    merged = pd.merge(
                        model_df.assign(word_lower=model_df["word"].str.lower().str.strip()),
                        human_df.assign(word_lower=human_df["word"].str.lower().str.strip()),
                        on="word_lower",
                        suffixes=("_model", "_human"),
                    )
                    valid = (
                        merged["mean_surprisal_bits"].notna()
                        & merged["aoa_value_human"].notna()
                    )
                    n_matched = int(valid.sum())
                    if n_matched < 5:
                        logger.warning("    too few matched words (%d); skipping rho", n_matched)
                        rho, pval = float("nan"), float("nan")
                    else:
                        rho, pval = spearmanr(
                            merged.loc[valid, "mean_surprisal_bits"],
                            merged.loc[valid, "aoa_value_human"],
                        )
                        rho, pval = float(rho), float(pval)

                    elapsed = time.time() - t0
                    logger.info(
                        "    rho=%.4f  p=%.2e  n_matched=%d  n_words=%d  [%.1fs]",
                        rho, pval, n_matched, len(model_df), elapsed,
                    )
                    _append_row(output_file, {
                        "model":          repo,
                        "model_family":   family,
                        "l1_code":        l1,
                        "eval_language":  lang,
                        "iso_code":       lang,
                        "spearman_rho":   round(rho, 6) if not math.isnan(rho) else "",
                        "p_value":        round(pval, 6) if not math.isnan(pval) else "",
                        "n_matched":      n_matched,
                        "n_words":        len(model_df),
                    })
                    _ledger_record("aoa_final", repo, "main", "aoa", lang, "ok",
                                   rank=args.rank, output_root=str(output_dir))
                except Exception as exc:
                    logger.error("    ERROR %s | %s: %s", repo, lang, exc)
                    _ledger_record("aoa_final", repo, "main", "aoa", lang, "error",
                                   rank=args.rank, output_root=str(output_dir))
        finally:
            # Always unload — even if an inner-loop error escapes the
            # per-lang try/except — so subsequent repos start with a clean
            # GPU + host RAM footprint.
            unload_model(model, tokenizer)
            del model, tokenizer

    logger.info("Done. Results at %s", output_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Final-checkpoint AoA evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--models", nargs="+", required=True, metavar="SPEC",
                   help="beetle-humanscale | beetle-fineweb | baselines | "
                        "eurollm | xglm | bloom | <org/repo>")
    p.add_argument("--langs", nargs="+", default=None, metavar="ISO",
                   help="Restrict evaluation to these ISO-639-3 codes")
    p.add_argument("--regime", choices=["humanscale", "fineweb"],
                   default="humanscale",
                   help="Training regime (picks held-out corpus; default: humanscale)")
    p.add_argument("--output", default="results/evals/aoa")
    p.add_argument("--eval-data-dir", default="results/aoa/eval_data",
                   help="Directory for per-(lang, tokenizer) jsonl files")
    p.add_argument("--held-out-base", type=Path, default=None,
                   help="Base directory of held-out Parquet shards "
                        "(beetle-data pipeline_output/held_out{_babybabel})")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Scoring batch size (default: 32)")
    p.add_argument("--n-contexts", type=int, default=512)
    p.add_argument("--min-tokens", type=int, default=8)
    p.add_argument("--skip-prep", action="store_true",
                   help="Assume eval jsonls already exist (rank != 0 workers)")
    p.add_argument("--prep-only", action="store_true",
                   help="Generate eval jsonls and exit (rank 0 preflight)")
    p.add_argument("--rank",       type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
