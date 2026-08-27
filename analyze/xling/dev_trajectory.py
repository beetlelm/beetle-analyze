"""
analyze/xling/dev_trajectory.py — §J developmental trajectory.

Runs §A's per-layer L1↔L2 CKA on a sequence of training checkpoints
(e.g., the b3 nld-eng FineWeb-2B trace, ~14 ckpts across the L1→L2
transition) and records ``cka(curriculum, layer, step)``.

Phase 0 dependency
------------------
This module assumes the b3 phase-scheduler bug is resolved per
``results/evals_b3_checkpoints/FINDING.md``. If the b3 trace is unusable,
fall back to the b5 nld-eng trace via ``--repo`` and reframe the resulting
panel as "L1 attrition trajectory at L2 onset" (see plan §J fallback).

Outputs
-------
results/evals_paper/14_dev_trajectory/per_layer_cka_by_step.csv
  Columns: model_id, curriculum, lang_pair, layer, step, cka, cka_lo,
           cka_hi, n_pairs
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from analyze._interp_helpers import (
    RESULTS_PAPER, device_for, per_layer_mean_activations, write_csv_atomic,
)
from analyze.xling.cka_full import linear_cka

logger = logging.getLogger(__name__)


def list_checkpoint_revisions(repo: str) -> list[str]:
    """List ``step-*`` HF revisions for ``repo``, sorted by step number.

    Returns ``["main"]`` if listing fails (offline / private), so the
    pipeline still runs against the final checkpoint.
    """
    try:
        from huggingface_hub import list_repo_refs
    except Exception:
        return ["main"]
    try:
        refs = list_repo_refs(repo)
        steps = []
        for b in getattr(refs, "branches", []) or []:
            name = b.name if hasattr(b, "name") else str(b)
            if name.startswith("step-"):
                try:
                    steps.append((int(name.split("-")[1]), name))
                except ValueError:
                    pass
        steps.sort()
        if steps:
            return [n for _, n in steps] + ["main"]
        return ["main"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("[dev_trajectory] could not list refs: %s", exc)
        return ["main"]


def cka_for_revision(repo: str, revision: str, *,
                      l1_sents: list[str], l2_sents: list[str]
                      ) -> list[dict]:
    from analyze.ppl_utils import load_model_and_tokenizer

    pair = load_model_and_tokenizer(repo, revision=revision)
    if pair is None:
        return []
    model, tokenizer = pair
    model.to(device_for(allow_gpu=True)).eval()
    l1_acts = per_layer_mean_activations(model, tokenizer, l1_sents)
    l2_acts = per_layer_mean_activations(model, tokenizer, l2_sents)

    rows = []
    for layer in sorted(l1_acts.keys() & l2_acts.keys()):
        X, Y = l1_acts[layer], l2_acts[layer]
        n = min(X.shape[0], Y.shape[0])
        rows.append({
            "model_id": repo, "revision": revision,
            "layer": layer, "cka": float(linear_cka(X[:n], Y[:n])),
            "n_pairs": n,
        })
    return rows


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="Beetle-FineWeb/beetle-bilingual-b3-nld-eng")
    p.add_argument("--curriculum", default="b3")
    p.add_argument("--l1", default="nld"); p.add_argument("--l2", default="eng")
    p.add_argument("--n-pairs", type=int, default=512)
    p.add_argument("--max-revisions", type=int, default=14,
                    help="Cap to N evenly-spaced ckpts (the EMNLP plan calls "
                         "for ~14 across the L1→L2 transition).")
    p.add_argument("--out", default=str(RESULTS_PAPER / "14_dev_trajectory"
                                         / "per_layer_cka_by_step.csv"))
    args = p.parse_args(list(argv) if argv is not None else None)

    from analyze.ppl_utils import load_flores_sentences

    l1 = load_flores_sentences(args.l1)[:args.n_pairs]
    l2 = load_flores_sentences(args.l2)[:args.n_pairs]

    revs = list_checkpoint_revisions(args.repo)
    if len(revs) > args.max_revisions:
        idx = np.linspace(0, len(revs) - 1, args.max_revisions).round().astype(int)
        revs = [revs[i] for i in idx]
    logger.info("[dev_trajectory] %d revisions: %s", len(revs), revs)

    all_rows: list[dict] = []
    for rev in revs:
        step = 10 ** 18 if rev == "main" else int(rev.split("-")[1])
        for r in cka_for_revision(args.repo, rev, l1_sents=l1, l2_sents=l2):
            r.update({"curriculum": args.curriculum,
                       "lang_pair": f"{args.l1}-{args.l2}", "step": step})
            all_rows.append(r)

    df = pd.DataFrame(all_rows)
    write_csv_atomic(df, args.out)
    print(f"[dev_trajectory] wrote {len(df)} rows → {args.out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
