"""
analyze/sae/label_l1_transfer.py — §B feature-labelling pipeline.

For each SAE feature, compute mean activation on three probe sets:

  (a) L1 monolingual text         — FLORES devtest in L1.
  (b) L2 monolingual text         — FLORES devtest in L2.
  (c) BLiSS interlanguage probes  — false-friend / V2-leakage items.

A candidate "L1-transfer feature" is one with high activation on (c) and
low activation on (b). The threshold is tunable; we save the full per-
feature score table so downstream scripts can pick their own cutoff.

Outputs
-------
results/evals_paper/06_sae/<curriculum>__<langpair>__layer=<L>/feature_labels.csv
results/evals_paper/06_sae/l1_transfer_features.csv  (consolidated)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from analyze._interp_helpers import RESULTS_PAPER, device_for, write_csv_atomic

logger = logging.getLogger(__name__)


def _load_bliss_items(path: str | None) -> list[str]:
    """Load BLiSS probe items. Falls back to a tiny canonical list so the
    smoke path is self-contained."""
    if path:
        df = pd.read_csv(path)
        col = next((c for c in ["probe", "sentence", "item"] if c in df.columns),
                    None)
        if col:
            return df[col].dropna().astype(str).tolist()

    # Canonical fallback: V2 leakage + false-friend triples (Dutch-L1 → English).
    return [
        "Yesterday went I to the store",       # V2 leakage
        "Tomorrow will I see you",
        "She is becoming weird",                # raar/rare false-friend frame
        "He gave me a sleep present",
        "I will save you on the test",          # opslaan/save
    ]


def _feature_activations(model, tokenizer, sae, sentences: list[str], *,
                          layer: int, batch_size: int = 8) -> np.ndarray:
    """Mean SAE-feature activation per sentence; shape (n_sents, n_features)."""
    import torch

    device = next(model.parameters()).device
    out: list[np.ndarray] = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128,
                         return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device).float()
        with torch.no_grad():
            outputs = model(input_ids=ids, output_hidden_states=True)
        h = outputs.hidden_states[layer]            # (b, s, d)
        feats = sae.encode(h)                       # (b, s, n_features)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        sent_mean = (feats * mask.unsqueeze(-1)).sum(dim=1) / denom
        out.append(sent_mean.detach().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0,))


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--curriculum", required=True)
    p.add_argument("--lang-pair", default="nld-eng")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--l1", default="nld"); p.add_argument("--l2", default="eng")
    p.add_argument("--bliss-csv", default=None)
    p.add_argument("--sae-dir", required=True)
    p.add_argument("--n-mono", type=int, default=512)
    p.add_argument("--top-k-features", type=int, default=10)
    p.add_argument("--out-per-sae", default=None)
    p.add_argument("--out-consolidated",
                    default=str(RESULTS_PAPER / "06_sae"
                                / "l1_transfer_features.csv"))
    args = p.parse_args(list(argv) if argv is not None else None)

    from analyze.ppl_utils import load_flores_sentences, load_model_and_tokenizer
    from analyze.sae.model import SAE

    pair = load_model_and_tokenizer(args.repo)
    if pair is None:
        raise SystemExit(f"could not load {args.repo}")
    model, tokenizer = pair
    model.to(device_for()).eval()
    sae = SAE.load(Path(args.sae_dir) / "sae.pt")

    l1_sents = load_flores_sentences(args.l1)[:args.n_mono]
    l2_sents = load_flores_sentences(args.l2)[:args.n_mono]
    bliss = _load_bliss_items(args.bliss_csv)

    a_l1   = _feature_activations(model, tokenizer, sae, l1_sents, layer=args.layer)
    a_l2   = _feature_activations(model, tokenizer, sae, l2_sents, layer=args.layer)
    a_bliss = _feature_activations(model, tokenizer, sae, bliss, layer=args.layer)

    n_features = a_l1.shape[1] if a_l1.size else 0
    rows = []
    for f in range(n_features):
        score = float(a_bliss[:, f].mean() - a_l2[:, f].mean())
        rows.append({
            "model_id": args.repo, "curriculum": args.curriculum,
            "lang_pair": args.lang_pair, "layer": args.layer,
            "feature_id": f,
            "mean_act_l1":    float(a_l1[:, f].mean()),
            "mean_act_l2":    float(a_l2[:, f].mean()),
            "mean_act_bliss": float(a_bliss[:, f].mean()),
            "l1_transfer_score": score,
        })
    df = pd.DataFrame(rows).sort_values("l1_transfer_score", ascending=False)

    out_per = args.out_per_sae or str(Path(args.sae_dir) / "feature_labels.csv")
    write_csv_atomic(df, out_per)
    print(f"[label_l1_transfer] {n_features} features → {out_per}")

    # Append top-k to the consolidated table.
    top = df.head(args.top_k_features).copy()
    consolidated = Path(args.out_consolidated)
    if consolidated.exists():
        prior = pd.read_csv(consolidated)
        merged = pd.concat([prior, top], ignore_index=True)
        merged = merged.drop_duplicates(["model_id", "layer", "feature_id"],
                                         keep="last")
    else:
        merged = top
    write_csv_atomic(merged, consolidated)
    print(f"[label_l1_transfer] top-{args.top_k_features} appended → {consolidated}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
