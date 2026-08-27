#!/usr/bin/env python3
"""
evals/eval_bliss.py — BLiSS evaluation: RP@0, RP@tau, NGS, CPS.

Loads ALTACambridge/BLiSS from HuggingFace, scores a single model on
(sentence_good, sentence_bad, artificial_error) triplets, and reports
headline metrics stratified by l1_language and auto_cefr.

Metrics (0-100 scale, from the BLiSS paper):
  RP@0   — Random-over-Human Preference (ties count as 0.5)
  RP@tau — Random-over-Human Preference with margin threshold τ
  NGS    — Normalized Gap Score
  CPS    — Correct-Preferred Sanity check

BLiSS is a private dataset — authenticate before running:
    huggingface-cli login

Usage
-----
  python evals/eval_bliss.py \\
      --model Beetle-HumanScale/beetle-bilingual-l2-50-classroom-20-b4-humanscale-nld-eng \\
      --output results/evals/bliss \\
      --split train \\
      --batch-size 32
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evals.model_loader import (
    DEVICE, load_model, unload_model, setup_logging, resolve_vocab_size,
)

import logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (matching BLiSS/evaluate_model.py)
# ---------------------------------------------------------------------------

BPC_FACTOR = 1.0 / math.log(2.0)  # nats → bits
DEFAULT_TAU = 0.10
TIE_TOL = 1e-6
EPS = 1e-9

# BLiSS stores `l1_language` as English language names. Canonical mapping to
# ISO 639-3 so that `--langs eng nld` (from the caller's model slug) can be
# compared against the dataset labels. Extend here if a new L1 appears in a
# future BLiSS release — run eval_bliss.py once with no --langs filter and
# grep the "Strata:" log lines for the unmapped label.
_BLISS_L1_TO_ISO3: Dict[str, str] = {
    "english":    "eng",
    "dutch":      "nld",
    "german":     "deu",
    "french":     "fra",
    "spanish":    "spa",
    "italian":    "ita",
    "portuguese": "por",
    "chinese":    "zho",
    "mandarin":   "zho",
    "japanese":   "jpn",
    "korean":     "kor",
    "russian":    "rus",
    "polish":     "pol",
    "turkish":    "tur",
    "arabic":     "ara",
    "persian":    "fas",
    "farsi":      "fas",
    "hindi":      "hin",
    "greek":      "ell",
    "bulgarian":  "bul",
    "ukrainian":  "ukr",
    "serbian":    "srp",
    "indonesian": "ind",
    "vietnamese": "vie",
    "thai":       "tha",
    "czech":      "ces",
    "romanian":   "ron",
    "hungarian":  "hun",
    "swedish":    "swe",
    "norwegian":  "nor",
    "danish":     "dan",
    "finnish":    "fin",
    "hebrew":     "heb",
    "urdu":       "urd",
    "bengali":    "ben",
    "tamil":      "tam",
    "telugu":     "tel",
    "marathi":    "mar",
    "gujarati":   "guj",
    "punjabi":    "pan",
    "malay":      "msa",
    "tagalog":    "tgl",
    "filipino":   "tgl",
    "swahili":    "swa",
    "afrikaans":  "afr",
    "icelandic":  "isl",
    "catalan":    "cat",
    "lithuanian": "lit",
    "latvian":    "lav",
    "estonian":   "est",
    "slovak":     "slk",
    "slovenian":  "slv",
    "croatian":   "hrv",
}

# Accept 2-letter ISO-639-1 on the CLI side and normalize to ISO-639-3.
_ISO1_TO_ISO3: Dict[str, str] = {
    "en": "eng", "nl": "nld", "de": "deu", "fr": "fra", "es": "spa",
    "it": "ita", "pt": "por", "zh": "zho", "ja": "jpn", "ko": "kor",
    "ru": "rus", "pl": "pol", "tr": "tur", "ar": "ara", "fa": "fas",
    "hi": "hin", "el": "ell", "bg": "bul", "uk": "ukr", "sr": "srp",
    "id": "ind", "vi": "vie", "th": "tha", "cs": "ces", "ro": "ron",
    "hu": "hun", "sv": "swe", "no": "nor", "da": "dan", "fi": "fin",
    "he": "heb", "ur": "urd", "bn": "ben", "ta": "tam", "te": "tel",
}


def _normalize_lang(label: str) -> str:
    """Map a BLiSS `l1_language` label or a user-supplied code to ISO 639-3.

    Unrecognised values are returned lowercased so the comparison still works
    if BLiSS happens to already use ISO codes or invents a new label.
    """
    s = (label or "").strip().lower()
    if not s:
        return ""
    if s in _BLISS_L1_TO_ISO3:
        return _BLISS_L1_TO_ISO3[s]
    if len(s) == 2 and s in _ISO1_TO_ISO3:
        return _ISO1_TO_ISO3[s]
    return s  # already 3-letter ISO, or unknown — leave as-is


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BLiSSRecord:
    good: str
    bad: str
    art: str
    l1: str
    cefr: str


# ---------------------------------------------------------------------------
# Dataset loading (with pickle cache)
# ---------------------------------------------------------------------------

def load_bliss_data(split: str, pkl_dir: str) -> List[BLiSSRecord]:
    """Load BLiSS from HuggingFace; cache to pickle for fast re-runs.

    Only records with a non-empty artificial_error field are kept — the
    remaining fields (sentence_good, sentence_bad) are always present.
    """
    pkl_name = f"bliss_{split}.pkl"
    pkl_path = os.path.join(pkl_dir, pkl_name) if pkl_dir else ""

    if pkl_path and os.path.exists(pkl_path):
        with open(pkl_path, "rb") as fh:
            raw = pickle.load(fh)
        # Detect stale cache (old dummy script stored dicts without 'art')
        stale = raw and isinstance(raw[0], dict) and "art" not in raw[0]
        if not stale:
            records = [
                r if isinstance(r, BLiSSRecord) else BLiSSRecord(
                    good=r["good"], bad=r["bad"], art=r["art"],
                    l1=r.get("l1", r.get("l1_language", "unknown")),
                    cefr=r.get("cefr", r.get("auto_cefr", "unknown")),
                )
                for r in raw
            ]
            logger.info(f"pkl cache hit: {len(records):,} records ({pkl_name})")
            return records
        logger.info(f"Stale pkl cache (missing 'art' field) — re-downloading …")
        os.remove(pkl_path)

    logger.info(f"Downloading ALTACambridge/BLiSS split='{split}' …")
    ds = load_dataset("ALTACambridge/BLiSS", split=split)

    records: List[BLiSSRecord] = []
    skipped = 0
    for row in ds:
        g = row.get("sentence_good")
        b = row.get("sentence_bad")
        a = row.get("artificial_error")
        if not (isinstance(g, str) and g and
                isinstance(b, str) and b and
                isinstance(a, str) and a):
            skipped += 1
            continue
        records.append(BLiSSRecord(
            good=g,
            bad=b,
            art=a,
            l1=row.get("l1_language") or "unknown",
            cefr=row.get("auto_cefr") or "unknown",
        ))
    del ds
    gc.collect()

    logger.info(
        f"Loaded {len(records):,} valid triplets "
        f"({skipped:,} skipped — missing artificial_error)"
    )

    if pkl_path:
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        with open(pkl_path, "wb") as fh:
            pickle.dump(records, fh)
        logger.info(f"Cached {len(records):,} records → {pkl_name}")

    return records


# ---------------------------------------------------------------------------
# Scoring — bits-per-token via cross-entropy NLL
# ---------------------------------------------------------------------------

def _is_oom(exc: BaseException) -> bool:
    """Recognise CUDA OOMs across torch versions (mirrors eval_blimp._is_oom)."""
    if torch.cuda.is_available():
        oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


@torch.no_grad()
def score_sentences_bpt(
    model,
    tokenizer,
    sentences: List[str],
    device: str,
    batch_size: int,
) -> List[float]:
    """Return bits-per-token for each sentence (lower = model assigns higher probability).

    Uses per-token average NLL so that sentences of different lengths are
    comparable — this matches the scoring approach in BLiSS/evaluate_model.py.

    Adapts ``batch_size`` downward on CUDA OOM and retries the same chunk;
    minimum batch size is 1. Mirrors the OOM fallback in
    ``evals/eval_blimp.py:score_sentences``. Without this, BLiSS scoring on
    >=10 B baseline models reliably OOMs at the static batch_size=16 used by
    eval_per_model.py and the rank crashes mid-stratum.
    """
    vocab_size = resolve_vocab_size(model)
    all_bpt: List[float] = []
    n_total = len(sentences)
    cur_bs = batch_size
    i = 0
    while i < n_total:
        batch = sentences[i: i + cur_bs]
        try:
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=512,
                return_tensors="pt", add_special_tokens=True,
            )
            input_ids      = enc["input_ids"].to(device).clamp(0, vocab_size - 1)
            attention_mask = enc["attention_mask"].to(device)

            # Mask padding tokens in labels so they don't contribute to NLL
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            del outputs

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous().to(dtype=torch.long)
            shift_mask   = shift_labels != -100

            loss_tok = F.cross_entropy(
                shift_logits.transpose(1, 2), shift_labels,
                ignore_index=-100, reduction="none",
            )
            seq_nll_nats = (loss_tok * shift_mask).sum(dim=1)
            seq_len      = shift_mask.sum(dim=1).clamp_min(1)
            nll_nats_per_tok = (seq_nll_nats / seq_len).cpu().tolist()

            all_bpt.extend(nll * BPC_FACTOR for nll in nll_nats_per_tok)
        except BaseException as exc:
            if _is_oom(exc) and cur_bs > 1:
                new_bs = max(1, cur_bs // 2)
                logger.warning(
                    f"    BLiSS CUDA OOM at batch_size={cur_bs}; "
                    f"halving to {new_bs} and retrying"
                )
                cur_bs = new_bs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            logger.error(
                f"    score_sentences_bpt failed at i={i}, batch_size={cur_bs}",
                exc_info=True,
            )
            raise

        i += cur_bs

    return all_bpt


# ---------------------------------------------------------------------------
# Metrics (RP@0, RP@tau, NGS, CPS) — direct port from BLiSS/evaluate_model.py
# ---------------------------------------------------------------------------

def compute_bliss_metrics(
    bpt_good: List[float],
    bpt_bad: List[float],
    bpt_art: List[float],
    tau: float = DEFAULT_TAU,
) -> Dict:
    """Compute headline BLiSS metrics and significance tests.

    The margin is  bpt_art − bpt_bad:
      positive → model finds the artificial error MORE surprising than the
                 human learner error (desired behaviour).
    """
    n = len(bpt_good)
    if n == 0:
        return {}

    margins = []
    wins_at_0 = 0.0
    wins_at_tau = 0
    correct_best = 0
    gap_scores = []
    # LP and SO additions:
    # LP — Learner Preference: fraction where BPT(learner) < BPT(corrected)
    # SO — Strict Order:       fraction where BPT(corr) < BPT(lrn) < BPT(art)
    learner_preferred = 0
    strict_order      = 0

    for g, b, a in zip(bpt_good, bpt_bad, bpt_art):
        margin = a - b  # bpt_art − bpt_bad

        margins.append(margin)

        if abs(margin) <= TIE_TOL:
            wins_at_0 += 0.5  # tie counts as half-win
        elif margin > TIE_TOL:
            wins_at_0 += 1.0

        if margin >= tau:
            wins_at_tau += 1

        # CPS sanity check: good sentence should have lowest BPT
        if g < min(b, a):
            correct_best += 1

        # LP: learner sentence preferred over corrected
        if b < g:
            learner_preferred += 1

        # SO: full hypothesised ranking corrected < learner < artificial
        if g < b < a:
            strict_order += 1

        # NGS: normalized relative gap
        delta_h = b - g
        delta_r = a - g
        denom = abs(delta_r) + abs(delta_h) + EPS
        gap_scores.append(margin / denom)

    rp_at_0   = 100.0 * wins_at_0 / n
    rp_at_tau = 100.0 * wins_at_tau / n
    ngs       = 100.0 * (0.5 + 0.5 * (sum(gap_scores) / n))
    cps       = 100.0 * correct_best / n
    lp        = 100.0 * learner_preferred / n
    so        = 100.0 * strict_order / n

    # Significance tests vs random baseline (50%)
    sig: Dict[str, float] = {}
    try:
        n_decisive = sum(1 for m in margins if abs(m) > TIE_TOL)
        n_wins     = sum(1 for m in margins if m > TIE_TOL)
        if n_decisive > 0:
            sig["p_rp0"] = stats.binomtest(n_wins, n_decisive, 0.5).pvalue
        sig["p_rp_tau"] = stats.binomtest(wins_at_tau, n, 0.5).pvalue
        sig["p_cps"]    = stats.binomtest(correct_best, n, 0.5).pvalue
        if n > 1:
            _, p_ngs = stats.ttest_1samp(gap_scores, 0.0)
            sig["p_ngs"] = p_ngs
    except Exception as exc:
        logger.warning(f"Significance test failed: {exc}")

    return {
        "n_items":    n,
        "RP_at_0":    round(rp_at_0, 4),
        "RP_at_tau":  round(rp_at_tau, 4),
        "NGS":        round(ngs, 4),
        "CPS":        round(cps, 4),
        "LP":         round(lp, 4),
        "SO":         round(so, 4),
        **{k: round(v, 6) for k, v in sig.items()},
    }


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def _append_row(output_file: Path, row: dict) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        output_file, mode="a", header=not output_file.exists(), index=False,
    )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    output_dir  = Path(args.output)
    output_file = output_dir / "bliss_results.csv"
    pkl_dir = args.pkl_dir or os.environ.get("PKL_DIR", str(output_dir / ".pkl_cache"))

    setup_logging("bliss", output_dir, rank=0)

    # Load + cache dataset
    records = load_bliss_data(args.split, pkl_dir)
    logger.info(f"Total BLiSS triplets (split={args.split}): {len(records):,}")

    # Optional L1 filter — restrict to the languages encoded in the caller's
    # model slug (L1/L2). When omitted, fall back to all-strata behaviour so
    # the script remains usable standalone and for multilingual baselines
    # that don't encode L1/L2 in the repo name.
    if args.langs:
        wanted = {_normalize_lang(c) for c in args.langs}
        wanted.discard("")
        before = len(records)
        records = [r for r in records if _normalize_lang(r.l1) in wanted]
        logger.info(
            f"--langs filter {sorted(wanted)}: kept {len(records):,}/{before:,} "
            f"triplets"
        )
        if not records:
            logger.warning(
                f"No BLiSS records match --langs {sorted(wanted)} for "
                f"{args.model!r}@{args.revision!r} — skipping."
            )
            return

    # Group by (l1, cefr)
    strata: Dict[Tuple[str, str], List[BLiSSRecord]] = defaultdict(list)
    for r in records:
        strata[(r.l1, r.cefr)].append(r)

    logger.info(f"Strata: {len(strata)} (l1_language, auto_cefr) combinations")
    for (l1, cefr), items in sorted(strata.items()):
        logger.info(f"  {l1:20s} {cefr:6s}  {len(items):,} triplets")

    # Load model (reuse beetle-analyze model loader)
    result = load_model(args.model, DEVICE, revision=args.revision)
    if result is None:
        logger.error(f"Could not load model {args.model!r}@{args.revision!r} — aborting.")
        sys.exit(1)
    model, tokenizer = result

    try:
        # Ensure pad token exists (required for batched scoring with attention masks)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def _score_stratum(recs: List[BLiSSRecord]) -> Tuple[List[float], List[float], List[float]]:
            bpt_good = score_sentences_bpt(model, tokenizer, [r.good for r in recs], DEVICE, args.batch_size)
            bpt_bad  = score_sentences_bpt(model, tokenizer, [r.bad  for r in recs], DEVICE, args.batch_size)
            bpt_art  = score_sentences_bpt(model, tokenizer, [r.art  for r in recs], DEVICE, args.batch_size)
            return bpt_good, bpt_bad, bpt_art

        # Evaluate per stratum, accumulate for overall row
        all_good: List[float] = []
        all_bad:  List[float] = []
        all_art:  List[float] = []

        for (l1, cefr), recs in sorted(strata.items()):
            logger.info(f"Evaluating {l1} / {cefr} ({len(recs):,} triplets) …")
            t0 = time.time()
            try:
                bpt_good, bpt_bad, bpt_art = _score_stratum(recs)
                m = compute_bliss_metrics(bpt_good, bpt_bad, bpt_art, tau=args.tau)
                elapsed = time.time() - t0
                logger.info(
                    f"  RP@0={m['RP_at_0']:.2f}  RP@tau={m['RP_at_tau']:.2f}  "
                    f"NGS={m['NGS']:.2f}  CPS={m['CPS']:.2f}  [{elapsed:.1f}s]"
                )
                _append_row(output_file, {
                    "model":       args.model,
                    "branch":      args.revision,
                    "split":       args.split,
                    "l1_language": l1,
                    "auto_cefr":   cefr,
                    **m,
                })
                all_good.extend(bpt_good)
                all_bad.extend(bpt_bad)
                all_art.extend(bpt_art)
            except Exception as exc:
                logger.error(f"  ERROR {l1}/{cefr}: {exc}", exc_info=True)

        # Overall row across all strata
        if all_good:
            logger.info(f"Computing overall metrics across {len(all_good):,} triplets …")
            overall = compute_bliss_metrics(all_good, all_bad, all_art, tau=args.tau)
            logger.info(
                f"  OVERALL RP@0={overall['RP_at_0']:.2f}  RP@tau={overall['RP_at_tau']:.2f}  "
                f"NGS={overall['NGS']:.2f}  CPS={overall['CPS']:.2f}"
            )
            _append_row(output_file, {
                "model":       args.model,
                "branch":      args.revision,
                "split":       args.split,
                "l1_language": "ALL",
                "auto_cefr":   "ALL",
                **overall,
            })
    finally:
        # Always unload the model so a stratum-scoring crash cannot leak
        # GPU + host memory into whatever runs next.
        unload_model(model, tokenizer)
        del model, tokenizer

    # Print summary table
    if output_file.exists():
        df = pd.read_csv(output_file)
        print("\n=== BLiSS Results ===")
        cols = ["l1_language", "auto_cefr", "n_items", "RP_at_0", "RP_at_tau", "NGS", "CPS", "LP", "SO"]
        present = [c for c in cols if c in df.columns]
        print(df[present].to_string(index=False))
        print(f"\nFull results (incl. p-values) at {output_file}")

    logger.info(f"Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BLiSS evaluation: RP@0, RP@tau, NGS, CPS stratified by L1 and CEFR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", required=True, metavar="ORG/REPO",
                   help="HuggingFace model repo to evaluate (e.g. gpt2, org/model)")
    p.add_argument("--split", default="train",
                   help="Dataset split (default: train)")
    p.add_argument("--output", default="results/evals/bliss",
                   help="Output directory for bliss_results.csv (default: results/evals/bliss)")
    p.add_argument("--pkl-dir", default=None,
                   help="Pickle cache dir (default: <output>/.pkl_cache)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Scoring batch size (default: 32)")
    p.add_argument("--tau", type=float, default=DEFAULT_TAU,
                   help=f"Margin threshold for RP@tau (default: {DEFAULT_TAU})")
    p.add_argument("--revision", default="main",
                   help="HuggingFace branch / revision to evaluate (default: main). "
                        "Recorded as the 'branch' column in bliss_results.csv.")
    p.add_argument("--langs", nargs="+", default=None, metavar="CODE",
                   help="Restrict evaluation to BLiSS strata whose l1_language "
                        "matches one of these language codes (ISO 639-3 or -1, "
                        "or English names). When omitted, all strata are "
                        "evaluated. Intended for sweep drivers that pass the "
                        "L1/L2 parsed from the model slug.")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
