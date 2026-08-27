"""extract_features.py — Run a trained SAE over Dutch + English text.

Produces the .npy files consumed by `representation_analysis.analyze_sae_features`:
  <features_dir>/<lang>.npy  shape=(n_tokens, num_latents) dtype=float16

Run as a module:
    python -m analyze.sae.extract_features --repo <repo> --revision <rev> --layer <L>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm

# Allow `python -m analyze.sae.extract_features` from repo root.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))         # analyze/
sys.path.insert(0, str(_THIS.parent.parent.parent))  # repo root

from analyze.sae.model import TopKSAE, TopKSAEConfig
from analyze.sae.nethook import Trace
from analyze.sae.paths import (
    feature_npy as resolve_feature_npy,
    features_dir as resolve_features_dir,
    sae_dir as resolve_sae_dir,
)


def _load_sae(sae_dir: Path, device) -> TopKSAE:
    cfg_path = sae_dir / "config.json"
    pt_path = sae_dir / "sae.pt"
    if not cfg_path.exists() or not pt_path.exists():
        raise FileNotFoundError(
            f"SAE artifacts missing in {sae_dir} (expected sae.pt + config.json)")
    with cfg_path.open() as f:
        cfg_dict = json.load(f)
    sae_cfg = TopKSAEConfig(
        d_in=cfg_dict["d_in"],
        num_latents=cfg_dict["num_latents"],
        k=cfg_dict["k"],
    )
    sae = TopKSAE(sae_cfg).to(device)
    state = torch.load(pt_path, map_location=device)
    sae.load_state_dict(state)
    sae.eval()
    return sae


def _iter_token_batches(
    sentences: list[str],
    tokenizer,
    *,
    batch_size: int,
    ctx_len: int,
    device,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (input_ids, attention_mask) batches up to ctx_len tokens each."""
    for start in range(0, len(sentences), batch_size):
        chunk = sentences[start : start + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=ctx_len,
        )
        yield enc["input_ids"].to(device), enc["attention_mask"].to(device)


@torch.no_grad()
def extract_features_for_checkpoint(
    repo: str,
    revision: str,
    layer: int,
    *,
    sae_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    langs: tuple[str, ...] = ("nld", "eng"),
    n_tokens_per_lang: int = 200_000,
    batch_size: int = 8,
    ctx_len: int = 512,
    submodule_template: str = "layers.{i}",
    num_latents: int | None = None,
    k: int | None = None,
    lr: float = 1e-3,
    output_root: Path | str = Path("results"),
) -> dict[str, Path]:
    """Run a trained SAE on FLORES Dutch + English, save .npy per language.

    Returns a dict {lang: path_to_npy}.
    """
    from ppl_utils import load_model_and_tokenizer, load_flores_sentences  # type: ignore

    # Resolve sae_dir if not given (uses the n/k/lr triple).
    if sae_dir is None:
        if num_latents is None or k is None:
            raise ValueError(
                "sae_dir not provided — must pass num_latents and k to resolve it.")
        sae_dir = resolve_sae_dir(repo, revision, layer, num_latents, k, lr, output_root)
    sae_dir = Path(sae_dir)

    if output_dir is None:
        output_dir = resolve_features_dir(repo, revision, layer, output_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = load_model_and_tokenizer(repo, revision=revision)
    if result is None:
        raise RuntimeError(f"Failed to load {repo}@{revision}")
    model, tokenizer = result
    model.eval()
    device = next(model.parameters()).device

    sae = _load_sae(sae_dir, device)
    submodule = submodule_template.format(i=layer)
    print(f"[sae-extract] {repo}@{revision} layer={layer} submodule={submodule!r}")

    out_paths: dict[str, Path] = {}
    for lang in langs:
        sentences = load_flores_sentences(lang)
        collected: list[np.ndarray] = []
        n_collected = 0

        pbar = tqdm(total=n_tokens_per_lang, desc=f"sae-extract[{lang}]", unit="tok")
        for input_ids, attention_mask in _iter_token_batches(
            sentences, tokenizer, batch_size=batch_size, ctx_len=ctx_len, device=device,
        ):
            if n_collected >= n_tokens_per_lang:
                break
            with Trace(model, submodule, stop=True) as trace:
                _ = model(input_ids)
            hidden = trace.output if torch.is_tensor(trace.output) else trace.output[0]
            # Flatten to (B*T, D), keep only valid (non-padding) positions.
            flat = hidden[attention_mask.bool()]
            # Match training-time normalization exactly.
            flat = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            feats = sae.encode(flat.float())  # (n_valid, num_latents)

            arr = feats.to(torch.float16).cpu().numpy()
            remaining = n_tokens_per_lang - n_collected
            arr = arr[:remaining]
            collected.append(arr)
            n_collected += arr.shape[0]
            pbar.update(arr.shape[0])
        pbar.close()

        out = np.concatenate(collected, axis=0) if collected else np.zeros(
            (0, sae.num_latents), dtype=np.float16
        )
        out_path = output_dir / f"{lang}.npy"
        np.save(out_path, out)
        print(f"[sae-extract] saved {out_path}  shape={out.shape}  dtype={out.dtype}")
        out_paths[lang] = out_path

    # Manifest.
    with (output_dir / "manifest.json").open("w") as f:
        json.dump(
            {
                "repo": repo,
                "revision": revision,
                "layer": layer,
                "submodule": submodule,
                "langs": list(langs),
                "n_tokens_per_lang": n_tokens_per_lang,
                "sae_dir": str(sae_dir),
                "num_latents": int(sae.num_latents),
                "k": int(sae.k),
            },
            f,
            indent=2,
        )
    return out_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract SAE feature activations for Dutch + English.")
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--sae-dir", type=Path, default=None,
                   help="Trained SAE dir (overrides n/k/lr resolution).")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--langs", nargs="+", default=["nld", "eng"])
    p.add_argument("--n-tokens", type=int, default=200_000, dest="n_tokens_per_lang")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--ctx-len", type=int, default=512)
    p.add_argument("--submodule-template", default="layers.{i}")
    p.add_argument("--num-latents", type=int, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output-root", type=Path, default=Path("results"))
    return p


def main():
    args = _build_parser().parse_args()
    extract_features_for_checkpoint(
        repo=args.repo,
        revision=args.revision,
        layer=args.layer,
        sae_dir=args.sae_dir,
        output_dir=args.output_dir,
        langs=tuple(args.langs),
        n_tokens_per_lang=args.n_tokens_per_lang,
        batch_size=args.batch_size,
        ctx_len=args.ctx_len,
        submodule_template=args.submodule_template,
        num_latents=args.num_latents,
        k=args.k,
        lr=args.lr,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
