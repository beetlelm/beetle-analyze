"""
analyze/sae/causal_ablation_bliss.py — §B causal validation.

Ablate the candidate L1-transfer features (zero them in the residual
stream after SAE-encoded → SAE-decoded reconstruction at the hooked layer)
and re-run BLiSS. The pre-registered prediction is that the false-friend
margin moves toward zero in b3/b5 ablated models, but does not in b1.

Outputs
-------
results/evals_paper/06_sae/bliss_ablation_results.csv
  Columns: model_id, curriculum, lang_pair, layer, n_ablated_features,
           bliss_margin_pre, bliss_margin_post, ci_lo, ci_hi, n_items
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from analyze._interp_helpers import RESULTS_PAPER, bootstrap_ci, write_csv_atomic

logger = logging.getLogger(__name__)


def _bliss_margin(model, tokenizer, items: pd.DataFrame, *, hook=None) -> tuple[float, list[float]]:
    """Mean and per-item false-friend margin under (optionally hooked) model.

    Items must have columns ``correct`` and ``false_friend``. The margin is
    defined as ``logP(correct) - logP(false_friend)``; positive = correct
    preferred. We support both per-token mean NLL and last-token NLL.
    """
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    margins: list[float] = []

    def _score(text: str) -> float:
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(input_ids=ids)
        logp = F.log_softmax(out.logits.float(), dim=-1)
        # Mean log-prob of the realised tokens (shifted by 1).
        token_lp = logp[:, :-1, :].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        return float(token_lp.mean().item())

    for _, r in items.iterrows():
        try:
            margins.append(_score(r["correct"]) - _score(r["false_friend"]))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ablation] item failed: %s", exc)

    return float(np.mean(margins)) if margins else float("nan"), margins


def _hook_ablate(model, sae, layer: int, feature_ids: list[int]):
    """Register a forward hook on ``hidden_states[layer]`` that:
       1. encodes activations to SAE features,
       2. zeros the listed features,
       3. decodes back into the residual stream.

    Returns the hook handle so the caller can remove it.
    """
    import torch

    feature_mask = np.ones(sae.n_features, dtype=np.float32)
    feature_mask[feature_ids] = 0.0
    feature_mask_t = None  # late-bind on the right device

    target_module = _find_layer_module(model, layer)

    def _forward_hook(module, inputs, output):
        nonlocal feature_mask_t
        h = output[0] if isinstance(output, tuple) else output
        feats = sae.encode(h)
        if feature_mask_t is None:
            feature_mask_t = torch.from_numpy(feature_mask).to(feats.device).to(feats.dtype)
        feats = feats * feature_mask_t
        recon = sae.decode(feats)
        new_h = h + (recon - sae.decode(sae.encode(h)))   # delta-only patch
        if isinstance(output, tuple):
            return (new_h,) + output[1:]
        return new_h

    return target_module.register_forward_hook(_forward_hook)


def _find_layer_module(model, layer_idx: int):
    """Best-effort lookup of the residual-stream module for ``layer_idx``."""
    for path in (
        ("model", "layers"), ("transformer", "h"), ("gpt_neox", "layers"),
    ):
        cur = model
        ok = True
        for attr in path:
            if not hasattr(cur, attr):
                ok = False; break
            cur = getattr(cur, attr)
        if ok and 0 <= layer_idx < len(cur):
            return cur[layer_idx]
    raise RuntimeError("could not locate transformer layer module for ablation")


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--curriculum", required=True)
    p.add_argument("--lang-pair", default="nld-eng")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--sae-dir", required=True)
    p.add_argument("--feature-table", required=True,
                    help="feature_labels.csv produced by label_l1_transfer.")
    p.add_argument("--n-ablate", type=int, default=10,
                    help="Top-N L1-transfer features to ablate.")
    p.add_argument("--bliss-csv", required=True,
                    help="CSV with columns ``correct`` and ``false_friend``.")
    p.add_argument("--out", default=str(RESULTS_PAPER / "06_sae"
                                         / "bliss_ablation_results.csv"))
    args = p.parse_args(list(argv) if argv is not None else None)

    from analyze.ppl_utils import load_model_and_tokenizer
    from analyze.sae.model import SAE

    pair = load_model_and_tokenizer(args.repo)
    if pair is None:
        raise SystemExit(f"could not load {args.repo}")
    model, tokenizer = pair
    model.eval()
    sae = SAE.load(Path(args.sae_dir) / "sae.pt")

    table = pd.read_csv(args.feature_table).sort_values(
        "l1_transfer_score", ascending=False
    )
    feat_ids = table.head(args.n_ablate)["feature_id"].astype(int).tolist()
    items = pd.read_csv(args.bliss_csv)

    pre_mean, pre_per_item = _bliss_margin(model, tokenizer, items)
    handle = _hook_ablate(model, sae, args.layer, feat_ids)
    try:
        post_mean, post_per_item = _bliss_margin(model, tokenizer, items)
    finally:
        handle.remove()

    deltas = [post - pre for pre, post in zip(pre_per_item, post_per_item)]
    ci_lo, ci_hi = bootstrap_ci(deltas)
    row = {
        "model_id": args.repo, "curriculum": args.curriculum,
        "lang_pair": args.lang_pair, "layer": args.layer,
        "n_ablated_features": len(feat_ids),
        "bliss_margin_pre": pre_mean, "bliss_margin_post": post_mean,
        "delta_mean": float(np.mean(deltas)) if deltas else float("nan"),
        "delta_ci_lo": ci_lo, "delta_ci_hi": ci_hi,
        "n_items": len(items),
    }
    out = Path(args.out)
    df = pd.DataFrame([row])
    if out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    write_csv_atomic(df, out)
    print(f"[causal_ablation_bliss] pre={pre_mean:.4f} post={post_mean:.4f} → {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
