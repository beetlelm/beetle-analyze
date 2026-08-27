#!/usr/bin/env python3
"""
eval_alta_blimp.py — Evaluate RA-ALTA models on BLiMP (nyu-mll/blimp).

For each model and each BLiMP phenomenon config, computes:
    accuracy = fraction of pairs where P(sentence_good) > P(sentence_bad)

Results are appended to a single CSV as each model finishes.

Usage (single GPU):
    python eval_alta_blimp.py \
        --output_csv results/alta_blimp.csv \
        --batch_size 32

Usage (multi-GPU, 4 processes):
    for rank in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$rank python eval_alta_blimp.py \
            --rank $rank --world_size 4 \
            --output_csv results/alta_blimp.csv \
            --batch_size 32 &
    done
    wait
"""

import argparse
import csv
import fcntl
import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import get_dataset_config_names, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [rank%(rank)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Models to evaluate ────────────────────────────────────────────────────────
ALL_MODELS = [
    # Polish (phase1-final is new; beginner–fluent already evaluated)
    "RA-ALTA/pl-en-phase1-final",
    # "RA-ALTA/pl-en-beginner",
    # "RA-ALTA/pl-en-intermediate",
    # "RA-ALTA/pl-en-advanced",
    # "RA-ALTA/pl-en-fluent",
    # Turkish (already evaluated)
    # "RA-ALTA/tr-en-phase1-final",
    # "RA-ALTA/tr-en-beginner",
    # "RA-ALTA/tr-en-intermediate",
    # "RA-ALTA/tr-en-advanced",
    # "RA-ALTA/tr-en-fluent",
    # Arabic (already evaluated)
    # "RA-ALTA/ar-en-phase1-final",
    # "RA-ALTA/ar-en-beginner",
    # "RA-ALTA/ar-en-intermediate",
    # "RA-ALTA/ar-en-advanced",
    # "RA-ALTA/ar-en-fluent",
    # Chinese (already evaluated)
    # "RA-ALTA/zh-en-phase1-final",
    # "RA-ALTA/zh-en-beginner",
    # "RA-ALTA/zh-en-intermediate",
    # "RA-ALTA/zh-en-advanced",
    # "RA-ALTA/zh-en-fluent",
    # Spanish
    "RA-ALTA/es-en-phase1-final",
    "RA-ALTA/es-en-beginner",
    "RA-ALTA/es-en-intermediate",
    "RA-ALTA/es-en-advanced",
    "RA-ALTA/es-en-fluent",
    # French
    "RA-ALTA/fr-en-phase1-final",
    "RA-ALTA/fr-en-beginner",
    "RA-ALTA/fr-en-intermediate",
    "RA-ALTA/fr-en-advanced",
    "RA-ALTA/fr-en-fluent",
    # German
    "RA-ALTA/de-en-phase1-final",
    "RA-ALTA/de-en-beginner",
    "RA-ALTA/de-en-intermediate",
    "RA-ALTA/de-en-advanced",
    "RA-ALTA/de-en-fluent",
]

BLIMP_HF_ID   = "nyu-mll/blimp"
GOOD_COL      = "sentence_good"
BAD_COL       = "sentence_bad"

RESULT_FIELDS = [
    "model", "blimp_phenomenon", "accuracy", "n_correct", "n_total",
    "checkpoint",
]

# ═════════════════════════════════════════════════════════════════════════════
# BLiMP dataset loading
# ═════════════════════════════════════════════════════════════════════════════

def load_blimp_pairs(pkl_path: Optional[str] = None) -> dict:
    """
    Returns {phenomenon_name: [(good, bad), ...]} for all BLiMP configs.
    If pkl_path is given and the file exists, loads from pickle (fast, zero HTTP).
    Otherwise downloads from HF and optionally saves to pkl_path.
    """
    import pickle

    if pkl_path and os.path.exists(pkl_path):
        logger.info(f"Loading BLiMP pairs from cache: {pkl_path}")
        with open(pkl_path, "rb") as fh:
            return pickle.load(fh)

    logger.info("Downloading BLiMP from HuggingFace (this happens once) …")
    configs = get_dataset_config_names(BLIMP_HF_ID)
    logger.info(f"  {len(configs)} phenomena found.")

    all_pairs: dict = {}
    for i, cfg in enumerate(configs):
        for attempt in range(4):
            try:
                try:
                    ds = load_dataset(BLIMP_HF_ID, cfg, split="train")
                except Exception:
                    ds_dict = load_dataset(BLIMP_HF_ID, cfg)
                    ds = ds_dict[list(ds_dict.keys())[0]]
                all_pairs[cfg] = list(zip(ds[GOOD_COL], ds[BAD_COL]))
                del ds
                break
            except Exception as e:
                if "429" in str(e) and attempt < 3:
                    wait = 30 * (2 ** attempt)
                    logger.warning(f"  429 on '{cfg}' — waiting {wait}s")
                    time.sleep(wait)
                else:
                    logger.warning(f"  Skipping '{cfg}': {e}")
                    break
        if (i + 1) % 10 == 0:
            logger.info(f"  … {i+1}/{len(configs)} phenomena loaded")

    gc.collect()

    if pkl_path:
        os.makedirs(os.path.dirname(pkl_path) or ".", exist_ok=True)
        with open(pkl_path, "wb") as fh:
            pickle.dump(all_pairs, fh)
        logger.info(f"  Cached to {pkl_path}")

    logger.info(f"BLiMP ready: {len(all_pairs)} phenomena, "
                f"{sum(len(v) for v in all_pairs.values()):,} pairs total.")
    return all_pairs


# ═════════════════════════════════════════════════════════════════════════════
# Scoring
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def score_sentences(
    model, tokenizer, sentences: List[str],
    device: torch.device, batch_size: int
) -> torch.Tensor:
    # Resolve vocab_size robustly — Gemma 3 / other multimodal configs nest it
    # under config.text_config.
    _cfg = getattr(model, "config", None)
    _vs = getattr(_cfg, "vocab_size", None)
    if not _vs and _cfg is not None and getattr(_cfg, "text_config", None):
        _vs = getattr(_cfg.text_config, "vocab_size", None)
    if not _vs:
        _vs = model.get_input_embeddings().num_embeddings
    vocab_size = int(_vs)
    all_scores = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        )
        input_ids      = enc["input_ids"].to(device).clamp(0, vocab_size - 1)
        attention_mask = enc["attention_mask"].to(device)

        outputs   = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = F.log_softmax(outputs.logits, dim=-1)
        del outputs

        lp_shifted   = log_probs[:, :-1, :]
        ids_shifted  = input_ids[:, 1:]
        mask_shifted = attention_mask[:, 1:].float()
        token_lp     = lp_shifted.gather(-1, ids_shifted.unsqueeze(-1)).squeeze(-1)
        sent_lp      = (token_lp * mask_shifted).sum(dim=-1)
        all_scores.append(sent_lp.cpu())

    return torch.cat(all_scores)


def eval_blimp_phenomenon(
    model, tokenizer,
    pairs: List[Tuple[str, str]],
    device: torch.device,
    batch_size: int,
) -> Tuple[float, int, int]:
    s_good = score_sentences(model, tokenizer, [p[0] for p in pairs], device, batch_size)
    s_bad  = score_sentences(model, tokenizer, [p[1] for p in pairs], device, batch_size)
    n_correct = int((s_good > s_bad).sum().item())
    acc = n_correct / len(pairs) if pairs else 0.0
    return acc, n_correct, len(pairs)


# ═════════════════════════════════════════════════════════════════════════════
# CSV helpers
# ═════════════════════════════════════════════════════════════════════════════

def append_row(csv_path: str, row: dict) -> None:
    """Thread/process-safe CSV append via flock."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def already_done(csv_path: str, model: str, phenomenon: str) -> bool:
    """Return True if this (model, phenomenon) pair is already in the CSV."""
    path = Path(csv_path)
    if not path.exists():
        return False
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("model") == model and row.get("blimp_phenomenon") == phenomenon:
                return True
    return False


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate RA-ALTA models on BLiMP")
    p.add_argument("--rank",       type=int, default=0,
                   help="Index of this process (0-indexed, for multi-GPU)")
    p.add_argument("--world_size", type=int, default=1,
                   help="Total number of parallel processes")
    p.add_argument("--output_csv", default="results/blimp_results_alta.csv",
                   help="Path to output CSV (appended, not overwritten)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hf_token",   default=None,
                   help="HuggingFace token (if models are private)")
    p.add_argument("--resume",     action="store_true", default=True,
                   help="Skip (model, phenomenon) pairs already in the CSV")
    p.add_argument("--pkl_cache",  default="results/.blimp_cache.pkl",
                   help="Path to cache pre-fetched BLiMP pairs (saves HF re-downloads)")
    p.add_argument("--no_resume",  action="store_true",
                   help="Disable resume — re-evaluate everything")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    resume = args.resume and not args.no_resume

    # ── Logging with rank ─────────────────────────────────────────────────
    old_factory = logging.getLogRecordFactory()
    rank_val    = args.rank

    def record_factory(*a, **kw):
        rec = old_factory(*a, **kw)
        rec.rank = rank_val
        return rec
    logging.setLogRecordFactory(record_factory)

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Rank {args.rank}/{args.world_size} — device: {device}")

    # ── Model slice for this rank ─────────────────────────────────────────
    my_models = ALL_MODELS[args.rank :: args.world_size]
    logger.info(f"This rank evaluates {len(my_models)} model(s): {my_models}")

    # ── Load all BLiMP pairs once ─────────────────────────────────────────
    # Rank 0 downloads/caches; other ranks wait until the pkl exists.
    if args.rank == 0 or not args.pkl_cache:
        blimp = load_blimp_pairs(pkl_path=args.pkl_cache)
    else:
        logger.info(f"Rank {args.rank}: waiting for rank-0 to write BLiMP cache …")
        for _ in range(120):          # wait up to 2 min
            if args.pkl_cache and os.path.exists(args.pkl_cache):
                break
            time.sleep(1)
        blimp = load_blimp_pairs(pkl_path=args.pkl_cache)

    phenomena = sorted(blimp.keys())
    logger.info(f"Evaluating on {len(phenomena)} BLiMP phenomena.\n")

    # ── Outer loop: one model at a time ───────────────────────────────────
    for repo in my_models:
        logger.info(f"{'='*60}")
        logger.info(f"Model: {repo}")

        # Determine which phenomena still need eval
        needed = [
            ph for ph in phenomena
            if not (resume and already_done(args.output_csv, repo, ph))
        ]
        if not needed:
            logger.info("  All phenomena already done — skipping model load.")
            continue

        logger.info(f"  {len(needed)}/{len(phenomena)} phenomena to evaluate.")

        # ── Load model ────────────────────────────────────────────────────
        try:
            load_kw = dict(
                trust_remote_code = True,
                torch_dtype       = torch.float16,
                low_cpu_mem_usage = True,
                token             = args.hf_token,
            )
            model = (
                AutoModelForCausalLM
                .from_pretrained(repo, **load_kw)
                .to(device)
                .eval()
            )
            tokenizer = AutoTokenizer.from_pretrained(repo, **load_kw)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info("  Model loaded.")
        except Exception as e:
            logger.error(f"  Failed to load {repo}: {e}")
            continue

        # ── Evaluate each phenomenon ──────────────────────────────────────
        phenomenon_accs = []
        for ph in needed:
            pairs = blimp[ph]
            try:
                acc, n_correct, n_total = eval_blimp_phenomenon(
                    model, tokenizer, pairs, device, args.batch_size
                )
                phenomenon_accs.append(acc)
                logger.info(
                    f"  {ph:45s}  acc={acc:.4f}  ({n_correct}/{n_total})"
                )
                append_row(args.output_csv, {
                    "model"            : repo,
                    "blimp_phenomenon" : ph,
                    "accuracy"         : round(acc, 6),
                    "n_correct"        : n_correct,
                    "n_total"          : n_total,
                    "checkpoint"       : "main",
                })
            except Exception as e:
                logger.error(f"  Error on phenomenon '{ph}': {e}")

        # ── Summary for this model ────────────────────────────────────────
        if phenomenon_accs:
            mean_acc = sum(phenomenon_accs) / len(phenomenon_accs)
            logger.info(
                f"  ── Mean BLiMP accuracy across {len(phenomenon_accs)} "
                f"phenomena: {mean_acc:.4f}"
            )

        # ── Free GPU memory before loading next model ─────────────────────
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"  Model freed.\n")

    logger.info("Done. Results written to: " + args.output_csv)


if __name__ == "__main__":
    main()