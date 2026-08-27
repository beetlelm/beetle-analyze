"""train_sae.py — Train a TopK-SAE on activations from a beetle checkpoint.

Adapted from llm-jp-sae/train.py. Differences:
  * Reuses `analyze.ppl_utils.load_model_and_tokenizer` (fp16, revision, cache).
  * Submodule path defaults to `"layers.{i}"` (PicoDecoderHF), overridable.
  * Supports pretokenized HF datasets directly (skips tokenizer pass).
  * Output paths via `analyze.sae.paths.sae_dir`.

Run as a module:
    python -m analyze.sae.train_sae --repo <repo> --revision <rev> --layer <L> ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup

# Allow `python -m analyze.sae.train_sae` from repo root.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))         # analyze/
sys.path.insert(0, str(_THIS.parent.parent.parent))  # repo root

from analyze.sae.activation_buffer import ActivationBuffer
from analyze.sae.config import TrainConfig
from analyze.sae.model import TopKSAE, TopKSAEConfig
from analyze.sae.paths import sae_dir as resolve_sae_dir


@torch.no_grad()
def geometric_median(points: torch.Tensor, max_iter: int = 100, tol: float = 1e-5) -> torch.Tensor:
    """Geometric median of `points` — used for SAE decoder-bias initialization."""
    guess = points.mean(dim=0)
    for _ in range(max_iter):
        prev = guess
        distances = torch.norm(points - guess, dim=1)
        weights = 1 / (distances + 1e-8)
        weights /= weights.sum()
        guess = (weights.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(guess - prev) < tol:
            break
    return guess


def _train_loop(sae, activation_buffer, train_cfg, device):
    optimizer = torch.optim.Adam(sae.parameters(), lr=train_cfg.lr, eps=6.25e-10)
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=train_cfg.num_warmup_steps
    )
    autocast_context = (
        nullcontext()
        if device.type == "cpu"
        else torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    )
    n_token = 0
    train_loss: list[float] = []
    loss_accum = 0.0

    sae.train()
    pbar = tqdm(total=train_cfg.tokens, desc="sae-train", unit="tok")

    for step, x in enumerate(activation_buffer):
        x = x / x.norm(dim=-1, keepdim=True)
        if step == 0:
            median = geometric_median(x)
            sae.b_dec.data = median.to(sae.dtype)

        with autocast_context:
            x_hat = sae(x)
            e = x - x_hat
            loss = e.pow(2).sum(dim=-1).mean()
            loss.backward()
            loss_accum += loss.item()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            sae.set_decoder_norm_to_unit_norm()

        n_token += x.shape[0]
        pbar.update(x.shape[0])
        if (step + 1) % train_cfg.logging_step == 0:
            train_loss.append(loss_accum / train_cfg.logging_step)
            loss_accum = 0.0

        if n_token >= train_cfg.tokens:
            break

    pbar.close()
    return sae, train_loss


def train_sae_for_checkpoint(
    repo: str,
    revision: str,
    layer: int,
    *,
    num_latents: int = 12288,
    k: int = 32,
    lr: float = 1e-3,
    tokens: int = 100_000_000,
    dataset_repo: str,
    dataset_split: str = "train",
    dataset_name: str | None = None,
    pretokenized: bool = True,
    submodule_template: str = "layers.{i}",
    ctx_len: int = 512,
    bs_lm: int = 32,
    bs_sae: int = 32_768,
    buffer_size: int = 3_000_000,
    output_root: Path | str = Path("results"),
    overwrite: bool = False,
) -> Path:
    """Train a TopK-SAE on `repo @ revision`, layer `layer`.

    Returns the directory the SAE was written to.
    """
    save_dir = resolve_sae_dir(repo, revision, layer, num_latents, k, lr, output_root)
    save_dir.mkdir(parents=True, exist_ok=True)
    sae_pt = save_dir / "sae.pt"
    if sae_pt.exists() and not overwrite:
        print(f"[sae-train] cached {sae_pt} — skipping (use overwrite=True to rebuild)")
        return save_dir

    # Reuse beetle-analyze's loader so dtype, cache, and revision are consistent.
    from analyze.ppl_utils import load_model_and_tokenizer  # noqa: E402

    result = load_model_and_tokenizer(repo, revision=revision)
    if result is None:
        raise RuntimeError(f"Failed to load {repo}@{revision}")
    model, tokenizer = result
    model.eval()
    device = next(model.parameters()).device

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    submodule = submodule_template.format(i=layer)
    print(f"[sae-train] {repo}@{revision} layer={layer} -> submodule={submodule!r}")

    # Build dataset iterator.
    if pretokenized:
        ds = load_dataset(dataset_repo, name=dataset_name, split=dataset_split, streaming=True)
        # Keep only the columns we need so dataset.iter is fast.
        keep_cols = {"input_ids", "attention_mask"}
        first = next(iter(ds))
        drop = [c for c in first.keys() if c not in keep_cols]
        if drop:
            ds = ds.remove_columns(drop)
    else:
        ds = load_dataset(dataset_repo, name=dataset_name, split=dataset_split, streaming=True)
        ds = ds.map(
            lambda x: tokenizer(
                x["text"], truncation=True, padding="max_length", max_length=ctx_len
            ),
            batched=True,
        )

    train_cfg = TrainConfig(
        lr=lr,
        ctx_len=ctx_len,
        bs_lm=bs_lm,
        bs_sae=bs_sae,
        buffer_size=buffer_size,
        tokens=tokens,
    )

    activation_buffer = ActivationBuffer(
        model=model,
        submodule=submodule,
        dataset=ds,
        buffer_size=train_cfg.buffer_size,
        refresh_batch_size=train_cfg.bs_lm,
        batch_size=train_cfg.bs_sae,
        device=device,
        pretokenized=pretokenized,
        padding_side=getattr(tokenizer, "padding_side", "right"),
    )

    sae_cfg = TopKSAEConfig(d_in=activation_buffer.d_submodule, num_latents=num_latents, k=k)
    sae = TopKSAE(sae_cfg).to(device)

    sae, train_loss = _train_loop(sae, activation_buffer, train_cfg, device)

    # Save weights + config + loss.
    torch.save(sae.state_dict(), sae_pt)
    with (save_dir / "config.json").open("w") as f:
        json.dump(
            {
                "d_in": sae_cfg.d_in,
                "num_latents": sae_cfg.num_latents,
                "k": sae_cfg.k,
                "repo": repo,
                "revision": revision,
                "layer": layer,
                "submodule": submodule,
                "tokens": train_cfg.tokens,
                "lr": train_cfg.lr,
                "bs_sae": train_cfg.bs_sae,
                "ctx_len": train_cfg.ctx_len,
                "pretokenized": pretokenized,
                "dataset_repo": dataset_repo,
                "dataset_name": dataset_name,
                "padding_side": getattr(tokenizer, "padding_side", "right"),
            },
            f,
            indent=2,
        )
    with (save_dir / "train_loss.json").open("w") as f:
        json.dump({"train_loss": train_loss}, f, indent=2)

    print(f"[sae-train] saved {sae_pt}")
    return save_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a TopK-SAE on a beetle checkpoint.")
    p.add_argument("--repo", required=True, help="HuggingFace model repo (e.g. Beetle-HumanScale/...).")
    p.add_argument("--revision", default="main", help="HF revision (default: main).")
    p.add_argument("--layer", type=int, required=True, help="Hidden-layer index to hook (0-indexed).")
    p.add_argument("--num-latents", type=int, default=12288)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tokens", type=int, default=100_000_000)
    p.add_argument("--dataset", required=True, dest="dataset_repo",
                   help="HF dataset repo for activation source (e.g. Beetle-HumanScale/nld-50M-nld-eng).")
    p.add_argument("--dataset-name", default=None, help="HF dataset config name (optional).")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--pretokenized", action="store_true",
                   help="Dataset already has input_ids (skip tokenizer).")
    p.add_argument("--raw-text", dest="pretokenized", action="store_false",
                   help="Dataset is raw text (will tokenize on the fly).")
    p.set_defaults(pretokenized=True)
    p.add_argument("--submodule-template", default="layers.{i}",
                   help="Format template for submodule path (default: layers.{i} for PicoDecoderHF).")
    p.add_argument("--ctx-len", type=int, default=512)
    p.add_argument("--bs-lm", type=int, default=32)
    p.add_argument("--bs-sae", type=int, default=32_768)
    p.add_argument("--buffer-size", type=int, default=3_000_000)
    p.add_argument("--output-root", type=Path, default=Path("results"))
    p.add_argument("--overwrite", action="store_true")
    return p


def main():
    args = _build_parser().parse_args()
    train_sae_for_checkpoint(
        repo=args.repo,
        revision=args.revision,
        layer=args.layer,
        num_latents=args.num_latents,
        k=args.k,
        lr=args.lr,
        tokens=args.tokens,
        dataset_repo=args.dataset_repo,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        pretokenized=args.pretokenized,
        submodule_template=args.submodule_template,
        ctx_len=args.ctx_len,
        bs_lm=args.bs_lm,
        bs_sae=args.bs_sae,
        buffer_size=args.buffer_size,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
