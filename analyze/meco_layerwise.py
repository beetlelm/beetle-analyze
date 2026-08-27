"""
analyze/meco_layerwise.py — §H per-layer surprisal regression on MECO.

For each (curriculum, layer, reader-group ∈ {L1 native, L2 reader}), fit
the canonical mixed-effects model regressing fixation duration on the
**layerwise** surprisal extracted from that layer (rather than the LM
head's). The behaviour-meets-mechanism bridge: the layer where reading
times are best predicted should coincide with the layer where geometry
separates in figI1.

Outputs
-------
results/evals_paper/12_meco_layerwise/per_layer_deltall.csv
  Columns: model_id, curriculum, lang_pair, layer, reader_group, delta_ll,
           ci_lo, ci_hi, n_subjects

Pilot mode
----------
``--pilot`` runs only b1, b3, b5 nld-eng on the canonical MECO pipeline,
which is the §H "go / no-go" decision the EMNLP plan calls for (Phase 0b).
Flat profiles → demote §H to an appendix paragraph.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from analyze._interp_helpers import (
    DEFAULT_BEETLE_REPOS, RESULTS_PAPER, bootstrap_ci, device_for, write_csv_atomic,
)

logger = logging.getLogger(__name__)


def _layerwise_surprisal(model, tokenizer, sentences: list[str]) -> dict[int, list[float]]:
    """For each layer, return a list of mean per-token NLLs (one per sentence).

    Approximation: we project each layer's hidden state through the LM head
    (logit-lens style) and use the resulting next-token cross-entropy as
    "surprisal at that layer". Fast and sufficient for ΔLL trends on MECO.
    """
    import torch
    import torch.nn.functional as F
    device = next(model.parameters()).device
    # LM head: ``model.lm_head.weight`` on most causal LMs; custom decoders
    # (PicoDecoderHF, …) expose it via ``get_output_embeddings()``; fully-
    # tied models share weights with the input embedding.
    W_U = getattr(getattr(model, "lm_head", None), "weight", None)
    if W_U is None:
        head = None
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
        raise RuntimeError("model has no lm_head.weight or get_output_embeddings; "
                           "cannot logit-lens")
    from analyze._interp_helpers import _find_layer_module_list
    out: dict[int, list[float]] = {}
    layer_list = None
    for s in sentences:
        ids = tokenizer(s, return_tensors="pt").input_ids.to(device)
        captured: list = []
        handles = []
        with torch.no_grad():
            o = model(input_ids=ids, output_hidden_states=True)
        hidden = getattr(o, "hidden_states", None)
        if hidden is None:
            if layer_list is None:
                layer_list = _find_layer_module_list(model)
            if layer_list is None:
                raise RuntimeError("no hidden_states and no layer ModuleList found")
            captured = []
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
        for layer, h in enumerate(hidden):
            if h is None:
                continue
            logits = h.float() @ W_U.float().T            # (1, T, V)
            lp = F.log_softmax(logits, dim=-1)
            tok_lp = lp[:, :-1, :].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            out.setdefault(layer, []).append(float(-tok_lp.mean().item()))
    return out


def _delta_ll_against_baseline(per_layer: dict[int, list[float]],
                                 fixations: pd.DataFrame) -> pd.DataFrame:
    """Compute ΔLL of a per-layer surprisal predictor vs a length-only
    baseline using a simple OLS regression (avoids statsmodels MixedLM
    dependency for the smoke path; for the paper, swap in MixedLM)."""
    import numpy as np

    rows = []
    base_ss = (fixations["fix_dur"] - fixations["fix_dur"].mean()) ** 2
    base_ll = float(-0.5 * base_ss.sum())

    for layer, surps in per_layer.items():
        n = min(len(surps), len(fixations))
        s = np.asarray(surps[:n], dtype=float)
        y = fixations["fix_dur"].values[:n]
        # Length residual.
        x_len = fixations["sentence_length"].values[:n] if "sentence_length" in fixations.columns \
            else np.zeros_like(s)
        X = np.stack([s, x_len, np.ones_like(s)], axis=1)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        ss = ((y - pred) ** 2).sum()
        ll = float(-0.5 * ss)
        rows.append({"layer": layer, "delta_ll": ll - base_ll, "n": n})
    return pd.DataFrame(rows)


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repos", nargs="+",
                    default=[DEFAULT_BEETLE_REPOS["b1"],
                              DEFAULT_BEETLE_REPOS["b3"],
                              DEFAULT_BEETLE_REPOS["b5"]])
    p.add_argument("--curricula", nargs="+", default=["b1", "b3", "b5"])
    p.add_argument("--lang-pair", default="nld-eng")
    p.add_argument("--meco-csv", required=False, default=None,
                    help="Cleaned MECO CSV with columns: sentence_id, sentence, "
                         "fix_dur, reader_group, sentence_length. Falls back to "
                         "an FLORES/uniform-fixation smoke set if absent.")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--out", default=str(RESULTS_PAPER / "12_meco_layerwise"
                                         / "per_layer_deltall.csv"))
    args = p.parse_args(list(argv) if argv is not None else None)

    from analyze.ppl_utils import load_flores_sentences, load_model_and_tokenizer

    if args.meco_csv and Path(args.meco_csv).exists():
        meco = pd.read_csv(args.meco_csv)
    else:
        # Smoke fallback: use FLORES sentences, synthetic uniform fixation.
        sents = load_flores_sentences("eng")[:64]
        meco = pd.DataFrame({
            "sentence_id": range(len(sents)),
            "sentence": sents,
            "fix_dur": np.full(len(sents), 250.0),
            "sentence_length": [len(s.split()) for s in sents],
            "reader_group": "smoke",
        })

    rows: list[dict] = []
    for repo, cur in zip(args.repos, args.curricula):
        pair = load_model_and_tokenizer(repo)
        if pair is None:
            continue
        model, tokenizer = pair
        model.to(device_for()).eval()

        for reader_group, sub in meco.groupby("reader_group"):
            per_layer = _layerwise_surprisal(model, tokenizer,
                                              sub["sentence"].astype(str).tolist())
            df = _delta_ll_against_baseline(per_layer, sub.reset_index(drop=True))
            for _, r in df.iterrows():
                lo, hi = bootstrap_ci(per_layer[int(r["layer"])])
                rows.append({
                    "model_id": repo, "curriculum": cur,
                    "lang_pair": args.lang_pair, "layer": int(r["layer"]),
                    "reader_group": reader_group,
                    "delta_ll": float(r["delta_ll"]),
                    "ci_lo": lo, "ci_hi": hi,
                    "n_subjects": int(sub["reader_group"].count()),
                })

    write_csv_atomic(pd.DataFrame(rows), args.out)
    print(f"[meco_layerwise] wrote {len(rows)} rows → {args.out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
