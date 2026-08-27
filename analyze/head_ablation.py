"""
analyze/head_ablation.py — Per-head attention ablation for BeetleLM.

Implements the ablation experiment from Aoyama et al. §5: zero out a single
attention head, recompute model surprisal on a reading-time corpus, and
report the change in ∆LL (∆∆LL).  Repeated across heads × checkpoints.

Used by Paper-A RQ-A3 to test bilingual head specialisation: heads scored
as PS- or SAS-specialised under L1 vs L2 probes (from
`phase_transition.prefix_matching_score`) are clustered into
{L1-specialised, L2-specialised, shared, neither}, and the ablation impact
is reported per cluster × condition.

Approach: a forward pre-hook on each attention module zeros the head's
output projection contribution by setting its slice of the value
projection's output to zero before the softmax aggregation.  We avoid
patching the underlying CUDA kernels by intercepting ``forward`` on the
attention module, calling the original, and zeroing the head dimension on
the output tensor before the residual add.

CLI
---
    python -m analyze.head_ablation \
        --model       Beetle-2B/zho_L1-eng_L2_B5 \
        --corpus      celer \
        --languages   zho \
        --heads       all \
        --output      results/head_ablation/zho_eng_B5.csv
"""

from __future__ import annotations

import argparse
import gc
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "head_ablation"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Hook-based head ablation
# ---------------------------------------------------------------------------

class _HeadZeroHook:
    """Forward hook that zeros the contribution of head `head_idx` at layer `layer_idx`.

    Attaches to the attention module's *output*, where shape is typically
    (B, S, H * D_head) or a tuple whose first element matches.  We reshape
    to (B, S, H, D_head), zero the slice along the head axis, and reshape
    back.
    """
    def __init__(self, head_idx: int, n_heads: int):
        self.head_idx = head_idx
        self.n_heads = n_heads

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            primary = output[0]
            rest = output[1:]
        else:
            primary = output
            rest = ()
        if not torch.is_tensor(primary):
            return output
        B, S, HD = primary.shape
        if HD % self.n_heads != 0:
            return output
        D = HD // self.n_heads
        view = primary.view(B, S, self.n_heads, D).clone()
        view[:, :, self.head_idx, :] = 0.0
        new_primary = view.view(B, S, HD)
        if rest:
            return (new_primary, *rest)
        return new_primary


@contextmanager
def ablate_head(model: torch.nn.Module, layer_idx: int, head_idx: int) -> Iterator[None]:
    """Temporarily zero a single (layer, head)'s output contribution."""
    raw = getattr(model, "module", model)
    cfg = getattr(raw, "model_config", None)
    n_heads = int(getattr(cfg, "attention_n_heads", getattr(cfg, "n_heads", 0)))
    if n_heads <= 0:
        raise RuntimeError("[head_ablation] Could not infer n_heads from config.")

    # Pick the layer_idx-th attention module in iteration order.
    attn_modules = [m for m in raw.modules()
                    if m.__class__.__name__.lower().endswith("attention")]
    if layer_idx >= len(attn_modules):
        raise IndexError(f"layer_idx={layer_idx} but only {len(attn_modules)} layers.")

    handle = attn_modules[layer_idx].register_forward_hook(
        _HeadZeroHook(head_idx, n_heads)
    )
    try:
        yield
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# Surprisal recompute under ablation
# ---------------------------------------------------------------------------

def _surprisal_for_corpus(
    ilm_model,
    rt_df: pd.DataFrame,
) -> pd.Series:
    """Recompute surprisal for every row of `rt_df.FullTextMarked`."""
    from analyze.meco_surprisal import _get_surprisal
    return rt_df["FullTextMarked"].apply(lambda t: _get_surprisal(ilm_model, t))


def ablation_table(
    model_repo: str,
    revision: str,
    rt_df: pd.DataFrame,
    *,
    heads: Optional[list[tuple[int, int]]] = None,
    device: str = DEVICE,
) -> pd.DataFrame:
    """Compute ∆LL for the unablated baseline + every requested (layer, head).

    `heads` defaults to *every* (layer, head) inferred from model_config.
    Returns a DataFrame with columns:
        layer, head, delta_ll, delta_delta_ll  (vs. unablated)
        n              — sample size of the regression
    """
    from minicons import scorer
    from analyze.ppp_delta_ll import delta_ll

    ilm = scorer.IncrementalLMScorer(
        model_repo, device, tokenizer=model_repo,
        revision=revision, trust_remote_code=True,
    )
    raw = getattr(ilm.model, "module", ilm.model)
    cfg = raw.model_config
    n_layers = int(getattr(cfg, "n_layers", getattr(cfg, "num_layers", 0)))
    n_heads  = int(getattr(cfg, "attention_n_heads", getattr(cfg, "n_heads", 0)))
    if heads is None:
        heads = [(L, h) for L in range(n_layers) for h in range(n_heads)]

    # Baseline (no ablation)
    base_sur = _surprisal_for_corpus(ilm, rt_df)
    baseline_df = rt_df.copy()
    baseline_df["surprisal"] = base_sur
    base_stats = delta_ll(baseline_df, rt_df, surprisal_col="surprisal")
    base_delta = base_stats["delta_ll"]

    rows = [{
        "layer": -1, "head": -1,
        "delta_ll": base_delta,
        "delta_delta_ll": 0.0,
        "n": base_stats["n"],
    }]

    for (L, h) in tqdm(heads, desc=f"ablate[{model_repo.split('/')[-1]}]",
                       unit="head", leave=False):
        with ablate_head(ilm.model, L, h):
            sur = _surprisal_for_corpus(ilm, rt_df)
        df = rt_df.copy()
        df["surprisal"] = sur
        st = delta_ll(df, rt_df, surprisal_col="surprisal")
        rows.append({
            "layer": L, "head": h,
            "delta_ll": st["delta_ll"],
            "delta_delta_ll": st["delta_ll"] - base_delta,
            "n": st["n"],
        })

    del ilm, raw
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Per-head ablation ∆∆LL.")
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--corpus", required=True,
                   choices=["meco_l1", "meco_l2", "celer",
                            "dundee", "provo", "natural_stories"])
    p.add_argument("--language", default=None)
    p.add_argument("--heads", default="all",
                   help="'all' or comma-separated 'L,h' pairs e.g. '0,3;5,11'.")
    p.add_argument("--output", default=None)
    p.add_argument("--device", default=DEVICE)
    args = p.parse_args()

    from analyze.ppp_delta_ll import load_corpus
    rt_df = load_corpus(args.corpus, language=args.language)

    if args.heads == "all":
        heads = None
    else:
        heads = []
        for chunk in args.heads.split(";"):
            L, h = chunk.split(",")
            heads.append((int(L), int(h)))

    out = ablation_table(args.model, args.revision, rt_df,
                         heads=heads, device=args.device)
    out_path = Path(args.output) if args.output else (
        RESULTS_DIR / f"{args.model.replace('/', '__')}_{args.corpus}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[head_ablation] wrote → {out_path}")
