#!/usr/bin/env python3
"""
finetune/sft_learner.py — LoRA SFT of a Beetle base model on a learner-essay corpus.

Single-model entry point. For each (base_model, dataset) pair, this script:
  1. Loads the base Beetle model via the shared A100 loader (bf16 + Flash Attn 2).
  2. Wraps it with a PEFT LoRA adapter (q_proj/k_proj/v_proj/o_proj, r=16).
  3. Loads + formats a lilygoulder/learner_tuning_<LANG> dataset via
     finetune.learner_datasets.
  4. Trains with trl.SFTTrainer (packing on, gradient checkpointing on).
  5. Saves the adapter (and tokenizer) to ``--output-dir``; optionally
     pushes to the HF hub.

Adapter-only saves keep this cheap on disk and let the eval pipeline stack
the adapter on top of the existing base model checkpoint at inference.

Usage
-----
  python -m finetune.sft_learner \\
      --base-model Beetle-HumanScale/beetle-bilingual-...-deu-eng \\
      --dataset deu \\
      --output-dir results/sft/beetle-deu-eng-ger-learner \\
      --epochs 3 --learning-rate 2e-4

Smoke test:
  python -m finetune.sft_learner \\
      --base-model gpt2 --dataset deu --max-steps 20 \\
      --output-dir /tmp/sft_smoke
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evals.model_loader import DEVICE, load_model, setup_logging  # reuse loader

from finetune.learner_datasets import load_learner_dataset

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA SFT of a Beetle base model on a learner-essay corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--base-model", required=True,
                   help="HF repo of the Beetle base model")
    p.add_argument("--revision", default="main",
                   help="HF branch / revision of the base model (default: main)")
    p.add_argument("--dataset", required=True,
                   help="ISO-639-3 (deu/zho/...), lilygoulder code (GER/CHI/...), "
                        "or full HF dataset id (lilygoulder/learner_tuning_<X>).")

    # Output
    p.add_argument("--output-dir", required=True,
                   help="Where to save the LoRA adapter + tokenizer.")
    p.add_argument("--push-to-hub", default=None, metavar="ORG/REPO",
                   help="If set, push the adapter to this HF repo after training.")

    # Training hyperparams
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="If > 0, override --epochs (use for smoke tests).")
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--no-packing", action="store_true",
                   help="Disable example packing (default: packed)")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-fraction", type=float, default=0.0,
                   help="If > 0, hold out this fraction as eval split.")
    p.add_argument("--seed", type=int, default=42)

    # LoRA hyperparams
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj"])

    # Data filters
    p.add_argument("--min-chars", type=int, default=10)
    p.add_argument("--max-chars", type=int, default=8000)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Late imports so `--help` works without TRL/PEFT installed.
    try:
        from peft import LoraConfig, get_peft_model
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        sys.stderr.write(
            f"ERROR: missing dependency ({exc}). "
            "Install via: pip install 'trl>=0.9' 'peft>=0.10'\n"
        )
        sys.exit(2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("sft", output_dir, rank=0)

    logger.info(f"Base model: {args.base_model}@{args.revision}")
    logger.info(f"Dataset:    {args.dataset}")
    logger.info(f"Output dir: {output_dir}")

    # ── Data ────────────────────────────────────────────────────────────────
    splits = load_learner_dataset(
        args.dataset,
        split="train",
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )
    train_ds = splits["train"]
    eval_ds = splits.get("eval")
    logger.info(f"Train rows: {len(train_ds):,}"
                + (f"  Eval rows: {len(eval_ds):,}" if eval_ds is not None else ""))

    # ── Model + LoRA ────────────────────────────────────────────────────────
    result = load_model(args.base_model, DEVICE, revision=args.revision)
    if result is None:
        logger.error(f"Could not load {args.base_model!r}@{args.revision!r}")
        sys.exit(1)
    model, tokenizer = result

    # Gradient checkpointing requires use_cache=False; SFTTrainer normally sets
    # this for us, but doing it here makes the intent explicit.
    if not args.no_gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Trainer ─────────────────────────────────────────────────────────────
    sft_cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=(DEVICE == "cuda"),
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        # SFTConfig-specific
        max_seq_length=args.max_seq_length,
        packing=not args.no_packing,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    logger.info("Starting SFT …")
    trainer.train()

    # ── Save adapter + tokenizer ────────────────────────────────────────────
    logger.info(f"Saving LoRA adapter → {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if args.push_to_hub:
        logger.info(f"Pushing adapter to {args.push_to_hub}")
        try:
            model.push_to_hub(args.push_to_hub, private=True)
            tokenizer.push_to_hub(args.push_to_hub, private=True)
        except Exception as exc:
            logger.error(f"push_to_hub failed: {exc}")

    logger.info("SFT complete.")


if __name__ == "__main__":
    main()
