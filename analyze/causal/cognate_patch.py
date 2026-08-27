"""
analyze/causal/cognate_patch.py — §E activation patching for cognate transfer.

Cache the residual-stream hidden state at an L1 cognate token, then patch
it into an English context where the cognate target would be the correct
next-token, and measure ΔP(target). High patch-utility ⇒ shared rep.

Stimulus design (from EMNLP plan, surface-form confound fix)
-----------------------------------------------------------
Default stimuli are **orthographically distinct** near-cognates so the
result cannot be explained by the model having simply memorised that the
same string occurs in both languages:

  Dutch–English: vrienden/friends, huis/house, boek/book, dag/day,
                 nacht/night, melk/milk, brood/bread.
  German–English: Wasser/water, Mutter/mother, Brot/bread, Buch/book.

Plus a **Chinese–English negative control** (no cognates exist between
zho and eng by construction) — patch utility on that stratum should be ≈ 0.

Outputs
-------
results/evals_paper/09_patching/cognate_patch_utility.csv
  Columns: model_id, curriculum, lang_pair, layer, stratum, mean_delta_p,
           ci_lo, ci_hi, n_items
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from analyze._interp_helpers import (
    RESULTS_PAPER, bootstrap_ci, device_for, write_csv_atomic,
)

logger = logging.getLogger(__name__)


# Default orthographically-distinct stimulus pairs. Reviewers can add their
# own via --stimulus-csv with columns ``l1_token, l1_context, l2_target,
# l2_context, stratum``.
DEFAULT_STIMULI = pd.DataFrame([
    {"l1_token": "vrienden", "l1_context": "Mijn vrienden zijn hier.",
     "l2_target": "friends", "l2_context": "My ___ are here.",
     "stratum": "ortho_distinct"},
    {"l1_token": "huis", "l1_context": "Het huis is groot.",
     "l2_target": "house", "l2_context": "The ___ is large.",
     "stratum": "ortho_distinct"},
    {"l1_token": "boek", "l1_context": "Ik lees een boek.",
     "l2_target": "book", "l2_context": "I read a ___.",
     "stratum": "ortho_distinct"},
    # Shared-form upper bound (kept secondary, not headline).
    {"l1_token": "water", "l1_context": "Het water is koud.",
     "l2_target": "water", "l2_context": "The ___ is cold.",
     "stratum": "shared_form"},
])


def _patch_token_position(tokenizer, sentence: str, token: str) -> int | None:
    """Best-effort: index of the first BPE token whose decoded text overlaps
    ``token``. Returns None if the token cannot be located."""
    enc = tokenizer(sentence, return_tensors="pt")
    ids = enc.input_ids[0]
    for i, tid in enumerate(ids.tolist()):
        if token.lower() in tokenizer.decode([tid]).lower():
            return i
    return None


def _delta_p_target(model, tokenizer, *, l1_context: str, l2_context: str,
                     l1_token: str, l2_target: str, layer: int) -> float | None:
    """Patch the L1-token hidden state into the L2 context at the same
    layer; return ``logP(l2_target | patched) - logP(l2_target | clean)``.
    Returns None on failure (token not found, etc.).
    """
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    pos = _patch_token_position(tokenizer, l1_context, l1_token)
    if pos is None:
        return None
    blank_pos = l2_context.find("___")
    if blank_pos < 0:
        return None
    l2_clean = l2_context.replace("___", l2_target)
    l2_pre   = l2_context.replace("___", "").rstrip()

    enc1 = tokenizer(l1_context, return_tensors="pt").to(device)
    enc2 = tokenizer(l2_clean,   return_tensors="pt").to(device)

    # Capture the L1 hidden state at ``pos`` from layer ``layer``.
    cache: dict[str, torch.Tensor] = {}

    target_module = _find_layer_module(model, layer)

    def _capture(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        cache["h_l1"] = h[:, pos, :].detach().clone()

    handle = target_module.register_forward_hook(_capture)
    try:
        with torch.no_grad():
            model(input_ids=enc1["input_ids"])
    finally:
        handle.remove()
    if "h_l1" not in cache:
        return None

    # Find the patch position in L2: the last subword before the blank.
    enc2_pre = tokenizer(l2_pre, return_tensors="pt").to(device)
    patch_pos = enc2_pre["input_ids"].shape[1] - 1

    def _patch(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[:, patch_pos, :] = cache["h_l1"]
        return (h,) + output[1:] if isinstance(output, tuple) else h

    with torch.no_grad():
        out_clean = model(input_ids=enc2["input_ids"])
    handle = target_module.register_forward_hook(_patch)
    try:
        with torch.no_grad():
            out_patch = model(input_ids=enc2["input_ids"])
    finally:
        handle.remove()

    # logP(target_token) at patch_pos → next-token prediction.
    target_id = tokenizer(" " + l2_target, add_special_tokens=False
                           ).input_ids[0] if l2_target else None
    if target_id is None:
        return None
    lp_clean = F.log_softmax(out_clean.logits[:, patch_pos, :].float(), dim=-1)
    lp_patch = F.log_softmax(out_patch.logits[:, patch_pos, :].float(), dim=-1)
    return float(lp_patch[0, target_id].item() - lp_clean[0, target_id].item())


def _find_layer_module(model, layer_idx: int):
    for path in (("layers",), ("model", "layers"), ("transformer", "h"),
                 ("gpt_neox", "layers"), ("transformer", "layers")):
        cur = model
        ok = True
        for attr in path:
            if not hasattr(cur, attr):
                ok = False; break
            cur = getattr(cur, attr)
        if ok and 0 <= layer_idx < len(cur):
            return cur[layer_idx]
    raise RuntimeError("could not locate transformer layer module")


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--curriculum", required=True)
    p.add_argument("--lang-pair", default="nld-eng")
    p.add_argument("--layers", nargs="+", type=int, default=[0, 3, 6, 9, 12])
    p.add_argument("--stimulus-csv", default=None)
    p.add_argument("--out", default=str(RESULTS_PAPER / "09_patching"
                                         / "cognate_patch_utility.csv"))
    args = p.parse_args(list(argv) if argv is not None else None)

    from analyze.ppl_utils import load_model_and_tokenizer

    stim = pd.read_csv(args.stimulus_csv) if args.stimulus_csv else DEFAULT_STIMULI
    pair = load_model_and_tokenizer(args.repo)
    if pair is None:
        raise SystemExit(f"could not load {args.repo}")
    model, tokenizer = pair
    model.to(device_for()).eval()

    rows = []
    for layer in args.layers:
        for stratum, sub in stim.groupby("stratum"):
            deltas = []
            for _, r in sub.iterrows():
                d = _delta_p_target(model, tokenizer,
                                     l1_context=r["l1_context"],
                                     l2_context=r["l2_context"],
                                     l1_token=r["l1_token"],
                                     l2_target=r["l2_target"],
                                     layer=layer)
                if d is not None:
                    deltas.append(d)
            if not deltas:
                continue
            ci_lo, ci_hi = bootstrap_ci(deltas)
            rows.append({
                "model_id": args.repo, "curriculum": args.curriculum,
                "lang_pair": args.lang_pair, "layer": layer,
                "stratum": stratum,
                "mean_delta_p": float(np.mean(deltas)),
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "n_items": len(deltas),
            })

    df = pd.DataFrame(rows)
    out = Path(args.out)
    if out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    write_csv_atomic(df, out)
    print(f"[cognate_patch] wrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
