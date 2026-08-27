"""
analyze/lens/logit_lens_lang.py — §D logit-lens language-decision dynamics.

Across layers, track when the model "commits" to the input language by
projecting each layer's hidden state through the LM head and computing
top-k token competition between L1 and L2 vocab subsets:

    p_l1[layer]  = sum_{v ∈ V_L1} softmax(W_U @ h[layer])
    p_l2[layer]  = sum_{v ∈ V_L2} softmax(W_U @ h[layer])
    commit_layer = first layer where |p_l1 - p_l2| ≥ 0.5 (configurable)

V_L1 / V_L2 are derived heuristically from the tokenizer using a small
seed vocabulary per language; for the paper, swap in larger vocab sets.

Outputs
-------
results/evals_paper/08_lens/logit_lens_per_layer.csv
  Columns: model_id, curriculum, lang_pair, layer, sentence_id,
           input_lang, p_l1, p_l2, commit_layer
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from analyze._interp_helpers import (
    DEFAULT_BEETLE_REPOS, RESULTS_PAPER, device_for, write_csv_atomic,
)


# Seed words per language used to identify "language-coloured" vocab subsets.
SEED_WORDS = {
    "eng": ["the", "and", "of", "to", "in", "is", "you", "that", "with"],
    "nld": ["de", "het", "een", "van", "is", "en", "te", "dat", "met"],
    "deu": ["der", "die", "das", "und", "ist", "von", "in", "ein", "mit"],
    "zho": ["的", "是", "了", "我", "你", "他", "在", "和", "不"],
}


def _lang_vocab_indices(tokenizer, iso: str, top_n: int = 200) -> list[int]:
    """Heuristic: take token IDs whose decoded string overlaps a seed word
    or starts with one of its characters/substrings. Cheap and good enough
    for layer-wise commit detection."""
    seeds = SEED_WORDS.get(iso, [])
    ids = set()
    for w in seeds:
        for variant in (w, " " + w, w.upper(), w.capitalize()):
            try:
                tids = tokenizer.encode(variant, add_special_tokens=False)
            except Exception:
                continue
            ids.update(tids)
    # Pad with frequent tokens that contain any seed character/substring.
    vocab_size = getattr(tokenizer, "vocab_size", 32_000)
    extras: list[int] = []
    for tid in range(min(vocab_size, 50_000)):
        s = tokenizer.decode([tid]).strip().lower()
        if not s:
            continue
        if any(seed in s for seed in seeds):
            extras.append(tid)
        if len(extras) >= top_n:
            break
    ids.update(extras)
    return sorted(ids)


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repos", nargs="+",
                    default=list(DEFAULT_BEETLE_REPOS.values()))
    p.add_argument("--curricula", nargs="+",
                    default=list(DEFAULT_BEETLE_REPOS.keys()))
    p.add_argument("--l1", default="nld"); p.add_argument("--l2", default="eng")
    p.add_argument("--n-sentences", type=int, default=64)
    p.add_argument("--commit-threshold", type=float, default=0.5)
    p.add_argument("--out", default=str(RESULTS_PAPER / "08_lens"
                                         / "logit_lens_per_layer.csv"))
    args = p.parse_args(list(argv) if argv is not None else None)

    import torch
    import torch.nn.functional as F
    from analyze.ppl_utils import load_flores_sentences, load_model_and_tokenizer

    rows: list[dict] = []
    for repo, cur in zip(args.repos, args.curricula):
        pair = load_model_and_tokenizer(repo)
        if pair is None:
            continue
        model, tokenizer = pair
        device = device_for()
        model.to(device).eval()

        v_l1 = _lang_vocab_indices(tokenizer, args.l1)
        v_l2 = _lang_vocab_indices(tokenizer, args.l2)
        # LM head: ``model.lm_head.weight`` on standard causal LMs; custom
        # decoders (PicoDecoderHF, …) expose it via get_output_embeddings();
        # tied-weight models share with the input embedding.
        head = getattr(model, "lm_head", None)
        if head is None:
            try:
                head = model.get_output_embeddings()
            except Exception:
                head = None
        if head is None and hasattr(model, "get_input_embeddings"):
            try:
                head = model.get_input_embeddings()
            except Exception:
                head = None
        W_U = getattr(head, "weight", None)
        if W_U is None:
            continue  # (V, d)

        from analyze._interp_helpers import _find_layer_module_list
        layer_list = None
        for input_iso in (args.l1, args.l2):
            sents = load_flores_sentences(input_iso)[:args.n_sentences]
            for sid, s in enumerate(sents):
                ids = tokenizer(s, return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    out = model(input_ids=ids, output_hidden_states=True)
                hidden = getattr(out, "hidden_states", None)
                if hidden is None:
                    if layer_list is None:
                        layer_list = _find_layer_module_list(model)
                    if layer_list is None:
                        raise RuntimeError("no hidden_states and no layer ModuleList")
                    captured: list = []
                    handles = []
                    def _make_hook(idx):
                        def _hook(_m, _i, output):
                            h = output[0] if isinstance(output, tuple) else output
                            captured.append((idx, h.detach()))
                        return _hook
                    for idx, blk in enumerate(layer_list):
                        handles.append(blk.register_forward_hook(_make_hook(idx)))
                    with torch.no_grad():
                        _ = model(input_ids=ids)
                    for h_ in handles:
                        h_.remove()
                    captured.sort(key=lambda x: x[0])
                    hidden = [h for _, h in captured]
                commit_layer = -1
                for layer, h in enumerate(hidden):
                    last = h[:, -1, :].float()                # (1, d)
                    logits = last @ W_U.float().T              # (1, V)
                    probs = F.softmax(logits, dim=-1)[0]
                    p_l1 = float(probs[v_l1].sum().item())
                    p_l2 = float(probs[v_l2].sum().item())
                    if commit_layer < 0 and abs(p_l1 - p_l2) >= args.commit_threshold:
                        commit_layer = layer
                    rows.append({
                        "model_id": repo, "curriculum": cur,
                        "lang_pair": f"{args.l1}-{args.l2}", "layer": layer,
                        "sentence_id": sid, "input_lang": input_iso,
                        "p_l1": p_l1, "p_l2": p_l2,
                        "commit_layer": commit_layer,
                    })

    write_csv_atomic(pd.DataFrame(rows), args.out)
    print(f"[logit_lens_lang] wrote {len(rows)} rows → {args.out}")


if __name__ == "__main__":
    main()
