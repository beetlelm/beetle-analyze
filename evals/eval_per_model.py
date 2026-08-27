#!/usr/bin/env python3
"""
evals/eval_per_model.py — Per-model orchestrator: BLiMP + MECO + BLiSS.

For each model assigned to this rank:
  1. Load model with HuggingFace AutoModelForCausalLM (one VRAM resident copy).
  2. Run BLiMP suite (blimp_eng, blimp_nl, zhoblimp, multiblimp).
  3. Run BLiSS (if --skip-bliss not set; requires HF_TOKEN for the dataset).
  4. Unload the HF model.
  5. Load model via minicons.scorer.IncrementalLMScorer (separate copy — minicons
     does not accept a pre-loaded HF model, so this is a second load).
  6. Run MECO surprisal scoring.
  7. Unload, garbage-collect, empty CUDA cache.
  8. Move to the next model.

This restructure replaces the four serial "stage by benchmark" passes in
``scripts/eval_frontier_baselines.sh`` with a single "for model in models:
do everything" pass per rank. Net result: only one model is VRAM-resident
at a time, and each model is loaded at most twice (HF + minicons) instead
of three or four times across separate eval-script invocations.

Reuses the existing per-benchmark helpers from eval_blimp / eval_meco /
eval_bliss so output CSVs and the eval ledger remain interchangeable with
the standalone scripts.

Usage
-----
  CUDA_VISIBLE_DEVICES=0 python evals/eval_per_model.py \\
      --models frontier --rank 0 --world-size 8 \\
      --output-root results
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "analyze"))

from evals.baseline_models import (
    resolve_models,
    recommended_batch_sizes,
    needs_device_map_auto,
    model_size_tier,
)
from evals.model_loader import (
    DEVICE, load_model, unload_model, setup_logging, log_gpu_mem,
)

# BLiMP machinery
from evals.eval_blimp import (
    BENCHMARKS as BLIMP_BENCHMARKS,
    load_benchmark_data as blimp_load_benchmark_data,
    run_minimal_pairs as blimp_run_minimal_pairs,
    _append_row as blimp_append_row,
)

# BLiSS machinery
from evals.eval_bliss import (
    load_bliss_data,
    score_sentences_bpt,
    compute_bliss_metrics,
    _append_row as bliss_append_row,
    _normalize_lang as bliss_normalize_lang,
)
from collections import defaultdict

# JBLiMP (Japanese; SLOR-scored)
from evals.eval_jblimp import (
    load_jblimp_data,
    run_jblimp_slor,
)

# xCOMPs (multilingual; BLiMP-style sum-logprob)
from evals.eval_xcomps import (
    load_xcomps_data,
    run_xcomps,
    XCOMPS_LANGS,
)

# MECO machinery (uses minicons internally — cannot share the HF model)
from evals.eval_meco import (
    score_model as meco_score_model,
    _output_path as meco_output_path,
)

# UniversalCEFR machinery (CEFR proficiency-level classification)
from evals.eval_universalcefr import (
    DEFAULT_CORPORA as UCEFR_DEFAULT_CORPORA,
    run_universalcefr,
)
import pandas as pd

logger = logging.getLogger(__name__)

# Ledger (best-effort)
try:
    from analyze.eval_ledger import is_done as _ledger_is_done
    from analyze.eval_ledger import record as _ledger_record
except Exception:
    def _ledger_is_done(*_a, **_kw) -> bool: return False
    def _ledger_record(*_a, **_kw) -> None: pass


# ---------------------------------------------------------------------------
# Per-benchmark runners (each takes an already-loaded model where possible)
# ---------------------------------------------------------------------------

def run_blimp(
    model, tokenizer, repo: str, family: str, revision: str,
    output_dir: Path, batch_size: int, pkl_dir: str,
    benchmarks: list[str],
    lang_filter: set[str] | None = None,
) -> None:
    """Run the BLiMP suite for one already-loaded model.

    *lang_filter*, when provided, restricts evaluation to ISO-639-3 codes in
    the set. For multiblimp this filters per-language configs; for the
    single-language benchmarks (blimp_eng/blimp_nl/zhoblimp) the benchmark is
    skipped if its iso_code is not in the filter.
    """
    output_file = output_dir / "blimp_results.csv"
    bench_data = {}
    for name in benchmarks:
        if name not in BLIMP_BENCHMARKS:
            logger.warning(f"  BLiMP: unknown benchmark '{name}', skipping")
            continue
        bench_data[name] = blimp_load_benchmark_data(
            name, pkl_dir, lang_filter=lang_filter,
        )

    for bname, lang_data in bench_data.items():
        for lang, (pairs, iso) in lang_data.items():
            if _ledger_is_done("blimp", repo, revision, bname, iso,
                               output_root=str(output_dir)):
                logger.info(f"  BLiMP SKIP {bname}|{lang} (ledger)")
                continue
            logger.info(f"  BLiMP {repo}@{revision} | {bname} | {lang}: "
                        f"{len(pairs):,} pairs …")
            t0 = time.time()
            try:
                acc, n_correct, n_total = blimp_run_minimal_pairs(
                    model, tokenizer, pairs, DEVICE, batch_size,
                )
                elapsed = time.time() - t0
                pps = n_total / elapsed if elapsed > 0 else 0
                logger.info(
                    f"    accuracy={acc:.4f} ({n_correct}/{n_total}) "
                    f"{pps:.0f} pairs/s [{elapsed:.1f}s]"
                )
                blimp_append_row(output_file, {
                    "benchmark":     bname,
                    "model":         repo,
                    "branch":        revision,
                    "model_family":  family,
                    "eval_language": lang,
                    "iso_code":      iso,
                    "accuracy":      round(acc, 6),
                    "n_correct":     n_correct,
                    "n_total":       n_total,
                })
                _ledger_record("blimp", repo, revision, bname, iso, "ok",
                               output_root=str(output_dir))
            except Exception:
                logger.error(f"  BLiMP ERROR {repo}|{bname}|{lang}", exc_info=True)
                _ledger_record("blimp", repo, revision, bname, iso, "error",
                               output_root=str(output_dir))


def run_bliss(
    model, tokenizer, repo: str, revision: str,
    output_dir: Path, batch_size: int, pkl_dir: str,
    split: str = "train", tau: float = 0.10,
    l1_filter: set[str] | None = None,
) -> None:
    """Run BLiSS for one already-loaded model.

    *l1_filter*, when provided, restricts evaluation to BLiSS strata whose
    `l1_language` (normalized via bliss_normalize_lang) matches one of the
    given codes (ISO-639-1, ISO-639-3, or English language names).
    """
    output_file = output_dir / "bliss_results.csv"

    if _ledger_is_done("bliss", repo, revision, "bliss", None,
                       output_root=str(output_dir)):
        logger.info(f"  BLiSS SKIP {repo}@{revision} (ledger)")
        return

    try:
        records = load_bliss_data(split, pkl_dir)
    except Exception:
        logger.error(f"  BLiSS: could not load dataset (HF_TOKEN missing?)",
                     exc_info=True)
        _ledger_record("bliss", repo, revision, "bliss", None, "error",
                       output_root=str(output_dir))
        return

    if l1_filter:
        wanted = {bliss_normalize_lang(c) for c in l1_filter}
        wanted.discard("")
        before = len(records)
        records = [r for r in records if bliss_normalize_lang(r.l1) in wanted]
        logger.info(
            f"  BLiSS l1 filter {sorted(wanted)}: kept {len(records):,}/{before:,} "
            f"triplets"
        )
        if not records:
            logger.warning(f"  BLiSS: no records match l1 filter; skipping {repo}")
            _ledger_record("bliss", repo, revision, "bliss", None, "error",
                           output_root=str(output_dir))
            return

    logger.info(f"  BLiSS {repo}: {len(records):,} triplets")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    strata = defaultdict(list)
    for r in records:
        strata[(r.l1, r.cefr)].append(r)

    all_good, all_bad, all_art = [], [], []
    for (l1, cefr), recs in sorted(strata.items()):
        t0 = time.time()
        try:
            bpt_good = score_sentences_bpt(model, tokenizer,
                                           [r.good for r in recs], DEVICE, batch_size)
            bpt_bad  = score_sentences_bpt(model, tokenizer,
                                           [r.bad for r in recs], DEVICE, batch_size)
            bpt_art  = score_sentences_bpt(model, tokenizer,
                                           [r.art for r in recs], DEVICE, batch_size)
            m = compute_bliss_metrics(bpt_good, bpt_bad, bpt_art, tau=tau)
            elapsed = time.time() - t0
            logger.info(f"    {l1}/{cefr} ({len(recs):,}): "
                        f"RP@0={m['RP_at_0']:.2f} NGS={m['NGS']:.2f} "
                        f"[{elapsed:.1f}s]")
            bliss_append_row(output_file, {
                "model":       repo,
                "branch":      revision,
                "split":       split,
                "l1_language": l1,
                "auto_cefr":   cefr,
                **m,
            })
            all_good.extend(bpt_good); all_bad.extend(bpt_bad); all_art.extend(bpt_art)
        except Exception:
            logger.error(f"  BLiSS ERROR {l1}/{cefr}", exc_info=True)

    if all_good:
        overall = compute_bliss_metrics(all_good, all_bad, all_art, tau=tau)
        bliss_append_row(output_file, {
            "model":       repo,
            "branch":      revision,
            "split":       split,
            "l1_language": "ALL",
            "auto_cefr":   "ALL",
            **overall,
        })
        logger.info(f"    OVERALL RP@0={overall['RP_at_0']:.2f} "
                    f"NGS={overall['NGS']:.2f}")
        _ledger_record("bliss", repo, revision, "bliss", None, "ok",
                       output_root=str(output_dir))
    else:
        _ledger_record("bliss", repo, revision, "bliss", None, "error",
                       output_root=str(output_dir))


def run_jblimp(
    model, tokenizer, repo: str, family: str, revision: str,
    output_dir: Path, batch_size: int, pkl_dir: str,
) -> None:
    """Run JBLiMP (Japanese, SLOR-scored) for one already-loaded model."""
    output_file = output_dir / "jblimp_results.csv"

    if _ledger_is_done("jblimp", repo, revision, "jblimp", "jpn",
                       output_root=str(output_dir)):
        logger.info(f"  JBLiMP SKIP {repo}@{revision} (ledger)")
        return

    try:
        pairs = load_jblimp_data(pkl_dir)
    except Exception:
        logger.error(f"  JBLiMP: could not load dataset", exc_info=True)
        _ledger_record("jblimp", repo, revision, "jblimp", "jpn", "error",
                       output_root=str(output_dir))
        return

    logger.info(f"  JBLiMP {repo}@{revision}: {len(pairs):,} pairs (SLOR)")
    t0 = time.time()
    try:
        acc, n_correct, n_total = run_jblimp_slor(
            model, tokenizer, pairs, DEVICE, batch_size,
        )
        elapsed = time.time() - t0
        pps = n_total / elapsed if elapsed > 0 else 0
        logger.info(
            f"    accuracy={acc:.4f} ({n_correct}/{n_total}) "
            f"{pps:.0f} pairs/s [{elapsed:.1f}s]"
        )
        bliss_append_row(output_file, {
            "benchmark":     "jblimp",
            "model":         repo,
            "branch":        revision,
            "model_family":  family,
            "eval_language": "Japanese",
            "iso_code":      "jpn",
            "scorer":        "SLOR",
            "accuracy":      round(acc, 6),
            "n_correct":     n_correct,
            "n_total":       n_total,
        })
        _ledger_record("jblimp", repo, revision, "jblimp", "jpn", "ok",
                       output_root=str(output_dir))
    except Exception:
        logger.error(f"  JBLiMP ERROR {repo}", exc_info=True)
        _ledger_record("jblimp", repo, revision, "jblimp", "jpn", "error",
                       output_root=str(output_dir))


def run_xcomps_one(
    model, tokenizer, repo: str, family: str, revision: str,
    output_dir: Path, batch_size: int, pkl_dir: str,
    lang_iso1: str,
) -> None:
    """Run xCOMPs for a single language (ISO-639-1 code, e.g. 'ja' or 'zh')."""
    output_file = output_dir / "xcomps_results.csv"
    if lang_iso1 not in XCOMPS_LANGS:
        logger.warning(f"  xCOMPs: unknown language '{lang_iso1}', skipping")
        return
    eval_lang, iso3 = XCOMPS_LANGS[lang_iso1]
    bench_name = f"xcomps_{lang_iso1}"

    if _ledger_is_done("xcomps", repo, revision, bench_name, iso3,
                       output_root=str(output_dir)):
        logger.info(f"  xCOMPs SKIP {repo}@{revision} | {bench_name} (ledger)")
        return

    try:
        pairs = load_xcomps_data(lang_iso1, pkl_dir)
    except Exception:
        logger.error(f"  xCOMPs: could not load {bench_name}", exc_info=True)
        _ledger_record("xcomps", repo, revision, bench_name, iso3, "error",
                       output_root=str(output_dir))
        return

    logger.info(f"  xCOMPs {repo}@{revision} | {bench_name} | {eval_lang}: "
                f"{len(pairs):,} pairs …")
    t0 = time.time()
    try:
        acc, n_correct, n_total = run_xcomps(
            model, tokenizer, pairs, DEVICE, batch_size,
        )
        elapsed = time.time() - t0
        pps = n_total / elapsed if elapsed > 0 else 0
        logger.info(
            f"    accuracy={acc:.4f} ({n_correct}/{n_total}) "
            f"{pps:.0f} pairs/s [{elapsed:.1f}s]"
        )
        bliss_append_row(output_file, {
            "benchmark":     bench_name,
            "model":         repo,
            "branch":        revision,
            "model_family":  family,
            "eval_language": eval_lang,
            "iso_code":      iso3,
            "accuracy":      round(acc, 6),
            "n_correct":     n_correct,
            "n_total":       n_total,
        })
        _ledger_record("xcomps", repo, revision, bench_name, iso3, "ok",
                       output_root=str(output_dir))
    except Exception:
        logger.error(f"  xCOMPs ERROR {repo}|{bench_name}", exc_info=True)
        _ledger_record("xcomps", repo, revision, bench_name, iso3, "error",
                       output_root=str(output_dir))


def run_meco(
    repo: str, revision: str, output_dir: Path, stimuli_path: Path,
) -> None:
    """Run MECO surprisal scoring. Loads its own model copy via minicons."""
    if _ledger_is_done("meco", repo, revision, "meco", None,
                       output_root=str(output_dir)):
        logger.info(f"  MECO SKIP {repo}@{revision} (ledger)")
        return

    if not stimuli_path.exists():
        logger.error(f"  MECO: stimuli file not found at {stimuli_path}")
        _ledger_record("meco", repo, revision, "meco", None, "error",
                       output_root=str(output_dir))
        return

    stimuli = pd.read_csv(stimuli_path, sep="\t")
    if "FullTextMarked" not in stimuli.columns:
        logger.error("  MECO: stimuli TSV missing 'FullTextMarked' column")
        _ledger_record("meco", repo, revision, "meco", None, "error",
                       output_root=str(output_dir))
        return

    logger.info(f"  MECO {repo}: scoring {len(stimuli)} stimuli (loads via minicons)")
    try:
        df = meco_score_model(repo, stimuli, DEVICE, revision=revision)
    except Exception:
        logger.error(f"  MECO ERROR {repo}", exc_info=True)
        _ledger_record("meco", repo, revision, "meco", None, "error",
                       output_root=str(output_dir))
        return

    if df is None:
        _ledger_record("meco", repo, revision, "meco", None, "error",
                       output_root=str(output_dir))
        return

    out_path = meco_output_path(output_dir, repo, revision)
    df.to_csv(out_path, index=False)
    logger.info(f"  MECO wrote {len(df)} rows → {out_path}")
    _ledger_record("meco", repo, revision, "meco", None, "ok",
                   output_root=str(output_dir))


def _hard_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    blimp_dir = output_root / "evals" / "blimp"
    bliss_dir = output_root / "evals" / "bliss"
    meco_dir  = output_root / "evals" / "meco"
    ucefr_dir = output_root / "evals" / "universalcefr"
    jblimp_dir = output_root / "evals" / "jblimp"
    xcomps_dir = output_root / "evals" / "xcomps"
    for d in (blimp_dir, bliss_dir, meco_dir, ucefr_dir, jblimp_dir, xcomps_dir):
        d.mkdir(parents=True, exist_ok=True)

    setup_logging("per_model", output_root, args.rank)

    # Resolve.
    all_pairs = resolve_models(args.models, output_dir=output_root / "evals")

    # xlarge models (>= 30 B params) cannot fit on a single GPU and must
    # be loaded with device_map="auto" across all visible GPUs. Rank-parallel
    # launches of an xlarge would 8× the VRAM cost and OOM. The launcher is
    # expected to invoke this script TWICE for cohorts containing xlarge
    # repos: once with --rank/--world-size for the small/medium/large pass,
    # then once with --xlarge-only --world-size 1 (and CUDA_VISIBLE_DEVICES
    # set to all GPUs) for the xlarge pass.
    if args.xlarge_only:
        my_models = [(r, f) for (r, f) in all_pairs if needs_device_map_auto(r)]
        logger.info(f"[xlarge-only] {len(my_models)} xlarge models assigned")
    elif args.skip_xlarge:
        non_xlarge = [(r, f) for (r, f) in all_pairs if not needs_device_map_auto(r)]
        my_models = [p for i, p in enumerate(non_xlarge)
                     if i % args.world_size == args.rank]
        logger.info(
            f"[rank {args.rank}/{args.world_size}] {len(my_models)} non-xlarge models "
            f"assigned (xlarge skipped: {len(all_pairs) - len(non_xlarge)})"
        )
    else:
        my_models = [p for i, p in enumerate(all_pairs)
                     if i % args.world_size == args.rank]
        logger.info(f"[rank {args.rank}/{args.world_size}] {len(my_models)} models assigned")
    for repo, family in my_models:
        logger.info(f"  → {repo} ({family})")

    pkl_blimp = args.pkl_dir_blimp or str(blimp_dir / ".pkl_cache")
    pkl_bliss = args.pkl_dir_bliss or str(bliss_dir / ".pkl_cache")
    pkl_jblimp = str(jblimp_dir / ".pkl_cache")
    pkl_xcomps = str(xcomps_dir / ".pkl_cache")

    blimp_benchmarks = args.blimp_benchmarks or list(BLIMP_BENCHMARKS.keys())
    blimp_lang_filter = set(args.blimp_langs) if args.blimp_langs else None
    bliss_l1_filter = set(args.bliss_langs) if args.bliss_langs else None
    xcomps_langs = list(args.xcomps_langs) if args.xcomps_langs else []

    for repo, family in my_models:
        logger.info("=" * 72)
        logger.info(f"MODEL {repo}@{args.revision}  ({family})")
        logger.info("=" * 72)
        t_model = time.time()

        # Resolve per-model batch sizes from the size-tier registry (enabled
        # by default via --auto-batch-size). Explicit --batch-size-* CLI flags
        # (non-None) always win.
        tier = model_size_tier(repo)
        rec = recommended_batch_sizes(repo)
        if args.batch_size_blimp is not None:
            bs_blimp = args.batch_size_blimp
        elif args.auto_batch_size:
            bs_blimp = rec["blimp"]
        else:
            bs_blimp = 32  # legacy default
        if args.batch_size_bliss is not None:
            bs_bliss = args.batch_size_bliss
        elif args.auto_batch_size:
            bs_bliss = rec["bliss"]
        else:
            bs_bliss = 16  # legacy default

        # xlarge models (Apertus-70B, LLM-jp-172B, ...) need device_map="auto"
        # so a single process spans all visible GPUs. The launcher must set
        # CUDA_VISIBLE_DEVICES accordingly. needs_device_map_auto() encodes
        # this in the registry (see evals/baseline_models.py:MODEL_SIZE_HINT).
        use_device_map = needs_device_map_auto(repo)
        load_kwargs: dict = {"revision": args.revision}
        if use_device_map:
            load_kwargs["device_map"] = "auto"
        logger.info(
            f"  tier={tier}  blimp_bs={bs_blimp}  bliss_bs={bs_bliss}  "
            f"device_map={'auto' if use_device_map else 'single'}"
        )

        # ───────── Phase 1: HF model load → BLiMP + BLiSS + UniversalCEFR + JBLiMP + xCOMPs ─────────
        needs_hf = (
            not args.skip_blimp
            or not args.skip_bliss
            or not args.skip_universalcefr
            or args.run_jblimp
            or args.run_xcomps
        )
        if needs_hf:
            logger.info(f"  Loading HF model {repo} …")
            result = load_model(repo, DEVICE, **load_kwargs)
            if result is None:
                logger.error(f"  HF load failed for {repo}; skipping BLiMP+BLiSS")
            else:
                model, tokenizer = result
                # try/finally guarantees the HF copy is unloaded before we
                # progress to the minicons load below, even when one of the
                # benchmark runners (BLiMP/BLiSS/JBLiMP/xCOMPs/UniversalCEFR)
                # raises. Without this guard a benchmark crash leaves the
                # model resident on GPU + host RAM through the next
                # iteration's load — the failure mode that has crashed
                # nodes via swap thrash.
                try:
                    log_gpu_mem(logger, prefix=f"after-HF-load {repo}")

                    if not args.skip_blimp:
                        logger.info(f"  --- BLiMP for {repo} ---")
                        run_blimp(
                            model, tokenizer, repo, family, args.revision,
                            blimp_dir, bs_blimp, pkl_blimp,
                            blimp_benchmarks,
                            lang_filter=blimp_lang_filter,
                        )

                    if not args.skip_bliss:
                        if not (os.environ.get("HF_TOKEN")
                                or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
                            logger.warning("  BLiSS: no HF_TOKEN; skipping (gated dataset)")
                        else:
                            logger.info(f"  --- BLiSS for {repo} ---")
                            run_bliss(
                                model, tokenizer, repo, args.revision,
                                bliss_dir, bs_bliss, pkl_bliss,
                                split=args.bliss_split, tau=args.bliss_tau,
                                l1_filter=bliss_l1_filter,
                            )

                    if args.run_jblimp:
                        logger.info(f"  --- JBLiMP for {repo} ---")
                        run_jblimp(
                            model, tokenizer, repo, family, args.revision,
                            jblimp_dir, bs_blimp, pkl_jblimp,
                        )

                    if args.run_xcomps and xcomps_langs:
                        logger.info(f"  --- xCOMPs for {repo} ({xcomps_langs}) ---")
                        for lang_iso1 in xcomps_langs:
                            run_xcomps_one(
                                model, tokenizer, repo, family, args.revision,
                                xcomps_dir, bs_blimp, pkl_xcomps, lang_iso1,
                            )

                    if not args.skip_universalcefr:
                        logger.info(f"  --- UniversalCEFR for {repo} ---")
                        if args.universalcefr_corpora:
                            ucefr_corpora = list(args.universalcefr_corpora)
                        elif args.universalcefr_languages:
                            ucefr_corpora = [c for lang in args.universalcefr_languages
                                             for c in UCEFR_DEFAULT_CORPORA.get(lang, [])]
                        else:
                            ucefr_corpora = [c for lang in ("en", "de", "nl")
                                             for c in UCEFR_DEFAULT_CORPORA[lang]]
                        try:
                            run_universalcefr(
                                model=model, tokenizer=tokenizer,
                                repo=repo, revision=args.revision,
                                corpora=ucefr_corpora,
                                output_dir=ucefr_dir,
                                mode=args.universalcefr_mode,
                                task=args.universalcefr_task,
                                split=args.universalcefr_split,
                                limit=args.universalcefr_limit,
                                device=DEVICE,
                            )
                        except Exception:
                            logger.error(f"  UniversalCEFR ERROR {repo}", exc_info=True)
                finally:
                    unload_model(model, tokenizer)
                    del model, tokenizer
                    _hard_cleanup()
                    log_gpu_mem(logger, prefix=f"after-HF-unload {repo}")

        # ───────── Phase 2: minicons load → MECO ─────────
        if not args.skip_meco:
            logger.info(f"  --- MECO for {repo} (separate minicons load) ---")
            try:
                run_meco(repo, args.revision, meco_dir, Path(args.stimuli))
            finally:
                # MECO loads its own minicons copy; ensure CUDA + host caches
                # are reclaimed even if scoring raises mid-stimulus.
                _hard_cleanup()
                log_gpu_mem(logger, prefix=f"after-MECO-unload {repo}")

        elapsed = time.time() - t_model
        logger.info(f"MODEL {repo} done in {elapsed/60:.1f} min")

    logger.info(f"[rank {args.rank}] all assigned models complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-model orchestrator: BLiMP + MECO + BLiSS for each model in turn.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--models", nargs="+", required=True,
                   help="Model spec(s): frontier | baselines | beetle-* | <hf/repo>")
    p.add_argument("--revision", default="main")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--output-root", default="results",
                   help="Root output dir; subdirs evals/{blimp,meco,bliss} created.")

    # BLiMP options
    p.add_argument("--blimp-benchmarks", nargs="*",
                   help="Subset of BLiMP suite (default: all 4).")
    p.add_argument("--blimp-langs", nargs="*", default=None, metavar="ISO",
                   help="Restrict BLiMP evaluation to ISO-639-3 codes "
                        "(filters multiblimp configs and skips single-language "
                        "benchmarks whose iso_code is not listed).")
    # batch-size-blimp / -bliss default to None: when --auto-batch-size is on
    # (the default), the per-repo recommendation from MODEL_SIZE_HINT is
    # used; an explicit integer here always wins. With --no-auto-batch-size,
    # a None falls back to the legacy literal default (32 / 16).
    p.add_argument("--batch-size-blimp", type=int, default=None)
    p.add_argument("--pkl-dir-blimp", default=None)

    # BLiSS options
    p.add_argument("--batch-size-bliss", type=int, default=None)
    p.add_argument("--bliss-split", default="train")
    p.add_argument("--bliss-tau", type=float, default=0.10)
    p.add_argument("--pkl-dir-bliss", default=None)
    p.add_argument("--bliss-langs", nargs="*", default=None, metavar="CODE",
                   help="Restrict BLiSS to strata whose l1_language matches "
                        "one of these codes (ISO-639-1, ISO-639-3, or English "
                        "language names like 'japanese' / 'chinese').")

    # JBLiMP / xCOMPs options
    p.add_argument("--run-jblimp", action="store_true",
                   help="Run JBLiMP (Japanese minimal pairs, SLOR-scored).")
    p.add_argument("--run-xcomps", action="store_true",
                   help="Run xCOMPs (multilingual minimal pairs).")
    p.add_argument("--xcomps-langs", nargs="*", default=None, metavar="ISO1",
                   help="ISO-639-1 codes for xCOMPs configs (e.g. ja, zh).")

    # Auto-batch-size + xlarge handling. See evals/baseline_models.py for
    # the per-repo size tier and recommended batch sizes.
    p.add_argument("--auto-batch-size", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Use the per-repo recommended batch sizes from "
                        "MODEL_SIZE_HINT (default: on). Disable with "
                        "--no-auto-batch-size to honour --batch-size-* literally.")
    p.add_argument("--xlarge-only", action="store_true",
                   help="Only run xlarge models (>=30 B) in this invocation, "
                        "with device_map='auto'. Launcher must expose all GPUs.")
    p.add_argument("--skip-xlarge", action="store_true",
                   help="Skip xlarge models in this invocation. Use this on "
                        "the rank-parallel pass; run a separate --xlarge-only "
                        "pass with all GPUs visible.")

    # MECO options
    p.add_argument("--stimuli", default=str(_REPO_ROOT / "data" / "meco_l2_stims.tsv"))

    # UniversalCEFR options
    p.add_argument("--universalcefr-mode", choices=["surprisal", "generation"],
                   default="surprisal",
                   help="Scoring mode: surprisal (base LMs) or generation (instruction-tuned).")
    p.add_argument("--universalcefr-task", choices=["read", "write"], default="read")
    p.add_argument("--universalcefr-split", default="train")
    p.add_argument("--universalcefr-limit", type=int, default=None,
                   help="Cap samples per corpus (smoke testing).")
    p.add_argument("--universalcefr-corpora", nargs="*", default=None,
                   help="Explicit list of UniversalCEFR/* HF datasets (overrides --universalcefr-languages).")
    p.add_argument("--universalcefr-languages", nargs="*", default=None,
                   help="ISO-639-1 codes (en/de/nl/...) → corpora via DEFAULT_CORPORA.")

    # Skip flags
    p.add_argument("--skip-blimp", action="store_true")
    p.add_argument("--skip-meco",  action="store_true")
    p.add_argument("--skip-bliss", action="store_true")
    p.add_argument("--skip-universalcefr", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
