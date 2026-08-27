"""
analyze/sae/chaining_analysis.py — §A↔§B chaining.

Layerwise activation spread of candidate L1-transfer features × per-layer
b1↔b5 CKA gap. The chaining figure shows that the layers where geometry
separates (§A) are the same layers where L1-transfer features fire (§B) —
turning the §A and §B results from a list into a chain.

Outputs
-------
results/evals_paper/16_chaining/l1_transfer_features_layerwise.csv
  Columns: model_id, curriculum, layer, feature_id, mean_act_bliss,
           cka_gap_b1_minus_curr
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from analyze._interp_helpers import RESULTS_PAPER, device_for, write_csv_atomic


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--curriculum", required=True)
    p.add_argument("--sae-dir", required=True)
    p.add_argument("--feature-table", required=True)
    p.add_argument("--cka-csv",
                    default=str(RESULTS_PAPER / "05_geometry"
                                / "cka_per_layer.csv"))
    p.add_argument("--bliss-csv", required=True)
    p.add_argument("--top-features", type=int, default=10)
    p.add_argument("--out", default=str(RESULTS_PAPER / "16_chaining"
                                         / "l1_transfer_features_layerwise.csv"))
    args = p.parse_args(list(argv) if argv is not None else None)

    from analyze.ppl_utils import load_model_and_tokenizer
    from analyze.sae.label_l1_transfer import _feature_activations
    from analyze.sae.model import SAE

    pair = load_model_and_tokenizer(args.repo)
    if pair is None:
        raise SystemExit(f"could not load {args.repo}")
    model, tokenizer = pair
    model.to(device_for()).eval()
    sae = SAE.load(Path(args.sae_dir) / "sae.pt")

    feat_ids = (pd.read_csv(args.feature_table)
                .sort_values("l1_transfer_score", ascending=False)
                .head(args.top_features)["feature_id"].astype(int).tolist())

    bliss = pd.read_csv(args.bliss_csv)
    items = bliss["correct"].astype(str).tolist() if "correct" in bliss.columns \
        else bliss.iloc[:, 0].astype(str).tolist()

    # Per-layer spread of the listed features over BLiSS items.
    n_layers = getattr(model.config, "num_hidden_layers", 12) + 1
    rows = []
    cka = pd.read_csv(args.cka_csv) if Path(args.cka_csv).exists() else pd.DataFrame()

    for layer in range(n_layers):
        try:
            acts = _feature_activations(model, tokenizer, sae, items, layer=layer)
        except Exception:
            continue
        for f in feat_ids:
            if f >= acts.shape[1]:
                continue
            mean_act = float(acts[:, f].mean())
            gap = float("nan")
            if not cka.empty:
                b1_row = cka[(cka["curriculum"] == "b1") & (cka["layer"] == layer)]
                cu_row = cka[(cka["curriculum"] == args.curriculum)
                              & (cka["layer"] == layer)]
                if not b1_row.empty and not cu_row.empty:
                    gap = float(b1_row["cka"].iloc[0] - cu_row["cka"].iloc[0])
            rows.append({
                "model_id": args.repo, "curriculum": args.curriculum,
                "layer": layer, "feature_id": f,
                "mean_act_bliss": mean_act,
                "cka_gap_b1_minus_curr": gap,
            })

    df = pd.DataFrame(rows)
    write_csv_atomic(df, args.out)
    print(f"[chaining_analysis] wrote {len(df)} rows → {args.out}")


if __name__ == "__main__":
    main()
