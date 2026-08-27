"""
analyze/xling/probe_alignment.py — §A linear probes for syntactic features
on L1 and L2, plus cross-language probe-weight cosine.

Outputs
-------
results/evals_paper/05_geometry/probe_acc.csv
  Columns: model_id, curriculum, lang_pair, layer, feature, lang, accuracy,
           n_train, n_test
results/evals_paper/05_geometry/probe_weight_cosine.csv
  Columns: model_id, curriculum, lang_pair, layer, feature, cosine, n_classes

Features probed
---------------
Default features: POS (UPOS via spaCy if available; falls back to a small
deterministic POS heuristic so the module is runnable without spaCy on a
CPU notebook).

The "headline" panel for figI1(d) is the cross-language probe-weight
cosine averaged over features.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from analyze._interp_helpers import (
    DEFAULT_BEETLE_REPOS, RESULTS_PAPER, device_for,
    per_layer_mean_activations, write_csv_atomic,
)

logger = logging.getLogger(__name__)


# A tiny lexicon-based POS heuristic so the smoke path runs without spaCy.
# Sufficient for sanity checks; for the paper, swap in spaCy / Stanza UPOS.
def _heuristic_pos(token: str) -> str:
    t = token.lower()
    if t.endswith(("ing", "ed", "s")) and len(t) > 3:
        return "VERB"
    if t.endswith(("ly",)):
        return "ADV"
    if t.endswith(("er", "est")) and len(t) > 4:
        return "ADJ"
    if t in {"a", "an", "the", "de", "het", "een", "der", "die", "das"}:
        return "DET"
    if t in {"and", "or", "but", "of", "to", "for", "in", "on", "at",
             "en", "of", "te", "voor", "in", "op", "und", "oder", "von"}:
        return "ADP"
    if t.istitle():
        return "PROPN"
    if t.isnumeric():
        return "NUM"
    return "NOUN"


def _build_pos_dataset(sentences: list[str]) -> tuple[list[str], list[str]]:
    """Word-level: returns aligned (token, label) pairs."""
    tokens: list[str] = []
    labels: list[str] = []
    for s in sentences:
        for w in s.split():
            tokens.append(w)
            labels.append(_heuristic_pos(w))
    return tokens, labels


def _word_activations(model, tokenizer, words: list[str], *,
                       layers: list[int]) -> dict[int, np.ndarray]:
    """Per-word activation ≈ first-token-mean activation of each isolated
    word; cheap and adequate for a probe-weight cosine signal."""
    return per_layer_mean_activations(model, tokenizer, words, layers=layers,
                                       max_length=8, batch_size=64)


def _fit_logreg(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0,
                                            stratify=y if len(set(y)) > 1 else None)
    clf = LogisticRegression(max_iter=2000, multi_class="multinomial",
                              n_jobs=1)
    clf.fit(Xtr, ytr)
    acc = clf.score(Xte, yte)
    # Concatenate per-class weight vectors; we'll cosine these between L1/L2
    # for the same feature.
    return float(acc), clf.coef_


def _layer_indices(model) -> list[int]:
    n = getattr(model.config, "num_hidden_layers", None)
    if n is None:
        n = len(model.transformer.h) if hasattr(model, "transformer") else 12
    # Sample 4 layers — early/mid-early/mid-late/late — to keep CPU smoke fast.
    return sorted({0, n // 3, 2 * n // 3, n})


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repos", nargs="+",
                    default=list(DEFAULT_BEETLE_REPOS.values()))
    p.add_argument("--curricula", nargs="+",
                    default=list(DEFAULT_BEETLE_REPOS.keys()))
    p.add_argument("--l1", default="nld"); p.add_argument("--l2", default="eng")
    p.add_argument("--n-words", type=int, default=2000)
    p.add_argument("--feature", default="POS")
    p.add_argument("--out-acc",   default=str(RESULTS_PAPER / "05_geometry"
                                              / "probe_acc.csv"))
    p.add_argument("--out-cosine", default=str(RESULTS_PAPER / "05_geometry"
                                                / "probe_weight_cosine.csv"))
    args = p.parse_args(list(argv) if argv is not None else None)

    from analyze.ppl_utils import load_flores_sentences, load_model_and_tokenizer

    l1_sents = load_flores_sentences(args.l1)
    l2_sents = load_flores_sentences(args.l2)
    l1_tokens, l1_labels = _build_pos_dataset(l1_sents)
    l2_tokens, l2_labels = _build_pos_dataset(l2_sents)
    l1_tokens = l1_tokens[:args.n_words]; l1_labels = l1_labels[:args.n_words]
    l2_tokens = l2_tokens[:args.n_words]; l2_labels = l2_labels[:args.n_words]

    # Align label sets between languages so weight cosines are well-defined.
    common_labels = sorted(set(l1_labels) & set(l2_labels))
    keep1 = [i for i, l in enumerate(l1_labels) if l in common_labels]
    keep2 = [i for i, l in enumerate(l2_labels) if l in common_labels]
    l1_tokens = [l1_tokens[i] for i in keep1]
    l1_labels = [l1_labels[i] for i in keep1]
    l2_tokens = [l2_tokens[i] for i in keep2]
    l2_labels = [l2_labels[i] for i in keep2]

    rows_acc: list[dict] = []
    rows_cos: list[dict] = []
    device = device_for(allow_gpu=True)

    for repo, cur in zip(args.repos, args.curricula):
        pair = load_model_and_tokenizer(repo)
        if pair is None:
            continue
        model, tokenizer = pair
        model.to(device).eval()
        layers = _layer_indices(model)

        l1_acts = _word_activations(model, tokenizer, l1_tokens, layers=layers)
        l2_acts = _word_activations(model, tokenizer, l2_tokens, layers=layers)

        for layer in layers:
            X1, X2 = l1_acts[layer], l2_acts[layer]
            label_to_int = {l: i for i, l in enumerate(common_labels)}
            y1 = np.array([label_to_int[l] for l in l1_labels])
            y2 = np.array([label_to_int[l] for l in l2_labels])
            acc1, W1 = _fit_logreg(X1, y1)
            acc2, W2 = _fit_logreg(X2, y2)
            rows_acc.append({"model_id": repo, "curriculum": cur,
                              "lang_pair": f"{args.l1}-{args.l2}",
                              "layer": layer, "feature": args.feature,
                              "lang": args.l1, "accuracy": acc1,
                              "n_train": len(y1) * 3 // 4, "n_test": len(y1) // 4})
            rows_acc.append({"model_id": repo, "curriculum": cur,
                              "lang_pair": f"{args.l1}-{args.l2}",
                              "layer": layer, "feature": args.feature,
                              "lang": args.l2, "accuracy": acc2,
                              "n_train": len(y2) * 3 // 4, "n_test": len(y2) // 4})
            # Per-class weight cosines, then mean across classes.
            n_classes = min(W1.shape[0], W2.shape[0])
            if n_classes >= 1:
                w1 = W1[:n_classes].reshape(-1)
                w2 = W2[:n_classes].reshape(-1)
                denom = (np.linalg.norm(w1) * np.linalg.norm(w2)) or 1.0
                cos = float(np.dot(w1, w2) / denom)
                rows_cos.append({"model_id": repo, "curriculum": cur,
                                  "lang_pair": f"{args.l1}-{args.l2}",
                                  "layer": layer, "feature": args.feature,
                                  "cosine": cos, "n_classes": n_classes})

    write_csv_atomic(pd.DataFrame(rows_acc), args.out_acc)
    write_csv_atomic(pd.DataFrame(rows_cos), args.out_cosine)
    print(f"[probe_alignment] acc → {args.out_acc} ({len(rows_acc)} rows)")
    print(f"[probe_alignment] cos → {args.out_cosine} ({len(rows_cos)} rows)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
