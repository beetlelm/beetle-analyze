"""
analyze/sae/train_curriculum_sae.py — §B SAE training driver.

Trains one SAE per (curriculum, language-pair) at a chosen residual-stream
layer (~layer 6 for the EMNLP-plan default). Reuses the existing SAE model
and training loop in ``analyze.sae``.

Defaults follow the M+2 pilot in the EMNLP plan: 32k features, top-k=64,
~500M activation tokens. CPU smoke runs use 8k features and 100k tokens.

Outputs
-------
results/evals_paper/06_sae/<curriculum>__<langpair>__layer=<L>/sae.pt
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

from analyze._interp_helpers import RESULTS_PAPER, device_for

logger = logging.getLogger(__name__)


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True,
                    help="HF repo of the curriculum to train an SAE on.")
    p.add_argument("--curriculum", required=True)
    p.add_argument("--lang-pair", default="nld-eng")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--n-features", type=int, default=32_000)
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--n-tokens", type=int, default=500_000_000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--smoke", action="store_true",
                    help="CPU smoke config: 8k features, 100k tokens, 1k batch.")
    p.add_argument("--out-root", default=str(RESULTS_PAPER / "06_sae"))
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.smoke:
        args.n_features = min(args.n_features, 8_000)
        args.n_tokens   = min(args.n_tokens, 100_000)
        args.batch_size = min(args.batch_size, 1024)
        logger.info("[train_curriculum_sae] smoke mode: n_features=%d, n_tokens=%d",
                    args.n_features, args.n_tokens)

    out_dir = Path(args.out_root) / f"{args.curriculum}__{args.lang_pair}__layer={args.layer}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Delegate to the existing SAE infrastructure. ``train_sae_for_checkpoint``
    # handles model load, activation buffering, top-k SAE training, and pt
    # checkpointing. We forward curriculum-shaped hyperparameters into it.
    from analyze.sae.train_sae import train_sae_for_checkpoint

    sae_dir = train_sae_for_checkpoint(
        repo=args.repo,
        revision="main",
        layer=args.layer,
        num_latents=args.n_features,
        k=args.top_k,
        lr=args.lr,
        tokens=args.n_tokens,
        dataset_repo="Beetle-Data/BabyBabel-eng-70M",
        dataset_split="train",
        pretokenized=False,
        ctx_len=512,
        bs_lm=8,
        bs_sae=min(args.batch_size, 16384),
        buffer_size=500_000,
        output_root=out_dir,
        overwrite=False,
    )
    print(f"[train_curriculum_sae] wrote SAE → {sae_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
