#!/usr/bin/env python3
"""
eval_xnli.py — Evaluate BeetleLM models on XNLI.

Method (causal LM scoring)
--------------------------
For each example we have:
  premise, hypothesis, label ∈ {entailment=0, neutral=1, contradiction=2}

We score the model's log P of the full string
    "premise. hypothesis."
for three versions of the hypothesis corresponding to the three labels,
then pick argmax and compare to the gold label.

Because XNLI provides a *single* hypothesis (not three), we build
positive / negative pairs with:
    positive = "premise [ENTAIL_TEMPLATE] hypothesis"
    negative = "premise [CONTRA_TEMPLATE] hypothesis"
and treat it like a binary minimal-pair accuracy.

XNLI languages used: English, French, German, Chinese
(Ukrainian and Persian are not in XNLI; those models are scored on the
 closest available language instead — see XNLI_LANG_MAP below.)

Usage
-----
  python eval_xnli.py \\
      --gpu 0 --rank 0 --world_size 8 \\
      --output_dir results/
"""

import argparse
import logging
import os
import sys
from typing import List, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from models import MODEL_GROUPS, get_bilingual_type, get_lang_pair
from utils  import (
    list_checkpoints, load_model_and_tokenizer, release,
    score_sentences, append_result, already_done,
)

logging.basicConfig(
    level  = logging.INFO,
    format = "[%(asctime)s] [GPU%(gpu)s] %(levelname)s %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── XNLI language codes ───────────────────────────────────────────────────────
# Maps BeetleLM lang code → XNLI split language code
# XNLI supports: ar bg de el en es fr hi ru sw th tr ur vi zh
XNLI_LANG_MAP = {
    "eng": "en",
    "fra": "fr",
    "deu": "de",
    "zho": "zh",
    "bul": "bg",
    # ukr / fas / nld / ind not in XNLI — skip
}

# Entailment / contradiction templates (simple concatenation works best for CLMs)
# The model sees: "<premise> <sep> <hypothesis>"
SEP = " [SEP] "   # simple separator; adjust if your tokeniser has a special token

BENCHMARK = "xnli"


def load_xnli(lang_code: str) -> List[Tuple[str, str, int]]:
    """
    Returns list of (premise, hypothesis, label) where
    label: 0=entailment, 1=neutral, 2=contradiction.
    We load the validation split (test split has no labels).
    """
    ds = load_dataset("xnli", lang_code, split="validation")
    return [(r["premise"], r["hypothesis"], r["label"]) for r in ds]


def xnli_accuracy(
    model,
    tokenizer,
    examples: List[Tuple[str, str, int]],
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[float, int, int]:
    """
    Score each (premise, hypothesis) pair twice:
      - positive: premise + SEP + hypothesis   [entailment framing]
      - negative: premise + SEP + "not" + hypothesis  [contradiction framing]
    Predict "entailment" if P(positive) > P(negative).
    Gold label: we treat entailment (0) as the positive class.
    We evaluate only on entailment / contradiction examples, skipping neutral.
    """
    entail_examples = [(p, h) for p, h, l in examples if l in (0, 2)]

    positive_seqs = [p + SEP + h          for p, h in entail_examples]
    negative_seqs = [p + SEP + "not " + h for p, h in entail_examples]
    gold          = [l == 0 for _, __, l in examples if l in (0, 2)]

    scores_pos = score_sentences(model, tokenizer, positive_seqs, device, batch_size)
    scores_neg = score_sentences(model, tokenizer, negative_seqs, device, batch_size)

    correct = int(((scores_pos > scores_neg) == torch.tensor(gold)).sum().item())
    total   = len(gold)
    return correct / total if total > 0 else 0.0, correct, total


def models_for_xnli(rank: int, world_size: int) -> List[str]:
    """All models whose language group appears in XNLI, sliced by rank."""
    seen, repos = set(), []
    for group in MODEL_GROUPS:
        if group in XNLI_LANG_MAP:
            for m in MODEL_GROUPS[group]:
                if m not in seen:
                    seen.add(m)
                    repos.append(m)
    return repos[rank::world_size]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu",        type=int, default=0)
    p.add_argument("--rank",       type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--output_dir", default="results")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Reduce if OOM (XNLI sequences are longer)")
    p.add_argument("--hf_token",   default=None)
    p.add_argument("--resume",     action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        n_visible = torch.cuda.device_count()
        if args.gpu >= n_visible:
            raise RuntimeError(
                f"--gpu {args.gpu} is out of range: only {n_visible} GPU(s) visible "
                f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}). "
                f"If launching via launch_all.sh with CUDA_VISIBLE_DEVICES set per process, "
                f"pass --gpu 0 and let the env var do the device mapping."
            )
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    old_factory = logging.getLogRecordFactory()
    def record_factory(*a, **kw):
        rec = old_factory(*a, **kw)
        rec.gpu = args.gpu
        return rec
    logging.setLogRecordFactory(record_factory)

    csv_path  = os.path.join(args.output_dir, "xnli_results.csv")
    my_models = models_for_xnli(args.rank, args.world_size)

    logger.info(f"Benchmark    : xnli")
    logger.info(f"GPU          : {args.gpu}  rank {args.rank}/{args.world_size}")
    logger.info(f"Models slice : {len(my_models)}")
    logger.info(f"Output CSV   : {csv_path}")

    # Pre-load all XNLI splits
    xnli_data = {}
    for group, xnli_code in XNLI_LANG_MAP.items():
        try:
            logger.info(f"Loading XNLI / {xnli_code} …")
            xnli_data[group] = (xnli_code, load_xnli(xnli_code))
            logger.info(f"  {len(xnli_data[group][1]):,} examples")
        except Exception as e:
            logger.warning(f"  Failed to load XNLI/{xnli_code}: {e}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    for repo in my_models:
        # Determine which XNLI language to use for this model
        model_group = next(
            (g for g, ms in MODEL_GROUPS.items() if repo in ms), None
        )
        if model_group not in xnli_data:
            continue

        xnli_code, examples = xnli_data[model_group]
        lang_name            = xnli_code.upper()
        checkpoints          = list_checkpoints(repo, args.hf_token)
        bil_type             = get_bilingual_type(repo)
        lang_pair            = get_lang_pair(repo)

        logger.info(f"\n{'='*60}")
        logger.info(f"Model : {repo}  ({len(checkpoints)} checkpoints)  XNLI/{xnli_code}")

        for ckpt in checkpoints:
            if args.resume and already_done(csv_path, repo, ckpt, lang_name, BENCHMARK):
                logger.info(f"  SKIP {ckpt} / {lang_name}")
                continue

            logger.info(f"  Loading {repo} @ {ckpt} …")
            try:
                model, tokenizer = load_model_and_tokenizer(
                    repo, ckpt, device, args.hf_token
                )
            except Exception as e:
                logger.error(f"  Load failed: {e}")
                continue

            try:
                acc, n_correct, n_total = xnli_accuracy(
                    model, tokenizer, examples, device, args.batch_size
                )
                logger.info(
                    f"  XNLI/{xnli_code}  acc={acc:.4f}  ({n_correct}/{n_total})"
                )
                append_result(csv_path, {
                    "benchmark"     : BENCHMARK,
                    "model"         : repo,
                    "lang_pair"     : lang_pair,
                    "bilingual_type": bil_type,
                    "checkpoint"    : ckpt,
                    "eval_language" : lang_name,
                    "accuracy"      : round(acc, 6),
                    "n_correct"     : n_correct,
                    "n_total"       : n_total,
                })
            except Exception as e:
                logger.error(f"  Eval failed: {e}")

            release(model, tokenizer, device)

    logger.info(f"\nDone. Results written to {csv_path}")


if __name__ == "__main__":
    main()
