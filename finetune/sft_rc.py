#!/usr/bin/env python3
"""
finetune/sft_rc.py — LoRA SFT of a Beetle base model on RACE (or BoolQ) for
RC-MCQ transfer to the Cambridge Reading Comprehension MCQ eval.

Adapter-only saves; eval-time the adapter is stacked on top of the base.

Usage
-----
  python -m finetune.sft_rc \
      --base-model Beetle-FineWeb-2B/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-2b-nld-eng \
      --revision step-5862 \
      --dataset race_combined \
      --output-dir results/sft_rc/fineweb2b_b3_5862_race_combined \
      --epochs 1 --max-examples 20000

Then:
  ADAPTER=results/sft_rc/fineweb2b_b3_5862_race_combined \
    bash scripts/eval_sft_on_rcmcq.sh

Smoke:
  python -m finetune.sft_rc --base-model gpt2 --dataset boolq \
      --max-steps 30 --output-dir /tmp/sft_rc_smoke
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evals.model_loader import DEVICE, load_model, setup_logging
from finetune.rc_datasets import load_rc_dataset

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA SFT for RC-MCQ transfer.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--dataset", default="race_combined",
                   help="race | race_high | race_middle | race_combined | boolq | <org>/<name>")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--push-to-hub", default=None)

    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--no-packing", action="store_true")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-fraction", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-examples", type=int, default=0,
                   help="If > 0, cap training set to this many examples after filtering.")
    p.add_argument("--persona-dropout", type=float, default=0.25)

    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from peft import LoraConfig, get_peft_model
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        sys.stderr.write(f"ERROR: missing dependency ({exc}). "
                         "pip install 'trl>=0.9' 'peft>=0.10'\n")
        sys.exit(2)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("sft_rc", out_dir, rank=0)
    logger.info(f"Base model: {args.base_model}@{args.revision}")
    logger.info(f"Dataset:    {args.dataset}")
    logger.info(f"Output dir: {out_dir}")

    splits = load_rc_dataset(
        args.dataset,
        split="train",
        eval_fraction=args.eval_fraction,
        persona_dropout=args.persona_dropout,
        seed=args.seed,
        max_examples=args.max_examples or None,
    )
    train_ds = splits["train"]
    eval_ds = splits.get("eval")
    logger.info(
        f"Train rows: {len(train_ds):,}"
        + (f"  Eval rows: {len(eval_ds):,}" if eval_ds is not None else "")
    )

    result = load_model(args.base_model, DEVICE, revision=args.revision)
    if result is None:
        logger.error(f"Could not load {args.base_model!r}@{args.revision!r}")
        sys.exit(1)
    model, tokenizer = result

    if not args.no_gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
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
        max_seq_length=args.max_seq_length,
        packing=not args.no_packing,
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=model, args=sft_cfg,
        train_dataset=train_ds, eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )
    logger.info("Starting RC-SFT …")
    trainer.train()

    logger.info(f"Saving LoRA adapter → {out_dir}")
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    if args.push_to_hub:
        logger.info(f"Pushing adapter to {args.push_to_hub}")
        try:
            model.push_to_hub(args.push_to_hub, private=True)
            tokenizer.push_to_hub(args.push_to_hub, private=True)
        except Exception as exc:
            logger.error(f"push_to_hub failed: {exc}")

    logger.info("RC-SFT complete.")


if __name__ == "__main__":
    main()
