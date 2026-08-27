#!/usr/bin/env python3
"""evals/eval_interlanguage.py

§4.2 L2 interlanguage error scoring.

For (correct, learner-error) sentence pairs, compute the surprisal margin
``margin = bpt(error) - bpt(correct)``.  Positive margin = model treats the
correct sentence as more probable, which is the desired monotone behaviour.

Three corpora:
  - **false_friend**: Dutch/English false-friend contexts shipped at
    ``data/cognates/false_friend_contexts.tsv`` — tightly-controlled L1-Dutch-
    influenced errors. Compares (intra_nl, intra_en) vs (cross_en_target,
    cross_nl_target). The bilingual nld-eng repos are predicted to *narrow*
    the surprisal margin on cross_*_target sentences relative to monolingual
    English.
  - **jfleg**: jhu-clsp/jfleg — sentence + list of native corrections (general
    L2 fluency errors, no L1 metadata). Margin per (model, item).
  - **lang8**: jvamvas/peer_lang8 — src/tgt pairs from Lang-8 (broad L2,
    multiple L1s, partial L1 metadata).

Output:
  results/<output-root>/<model>__rev=<rev>/<corpus>/items.csv
  results/<output-root>/<model>__rev=<rev>/<corpus>/summary.csv
  results/<output-root>/_aggregated.csv  (post-hoc by analyze.interlanguage_aggregate)
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evals.model_loader import (  # noqa: E402
    DEVICE, load_model, unload_model, setup_logging,
)
from evals.eval_bliss import score_sentences_bpt  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_false_friend(tsv_path: Path) -> list[dict]:
    """Pair each (word, intra_*) baseline with the matching cross_*_target as
    the learner-error counterpart. Treat intra as 'correct' and cross as
    'error' (the cross sentence puts the L1-meaning word into the L2 sentence
    or vice versa, which is the canonical false-friend interlanguage error).
    """
    rows: list[dict] = []
    with tsv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        per_word: dict[str, dict] = {}
        for r in reader:
            w = r["word"]; cond = r["condition"]; sent = r["sentence"]
            per_word.setdefault(w, {})[cond] = sent
    out = []
    for w, conds in per_word.items():
        if "intra_en" in conds and "cross_nl_target" in conds:
            out.append({
                "id": f"ff_{w}_en", "word": w, "lang": "en",
                "correct": conds["intra_en"],
                "error":   conds["cross_nl_target"],
                "direction": "nl_in_en",
            })
        if "intra_nl" in conds and "cross_en_target" in conds:
            out.append({
                "id": f"ff_{w}_nl", "word": w, "lang": "nl",
                "correct": conds["intra_nl"],
                "error":   conds["cross_en_target"],
                "direction": "en_in_nl",
            })
    return out


def load_jfleg(max_items: int = 0) -> list[dict]:
    from datasets import load_dataset
    items: list[dict] = []
    for split in ("validation", "test"):
        try:
            ds = load_dataset("jhu-clsp/jfleg", split=split)
        except Exception:
            continue
        for i, r in enumerate(ds):
            sent = (r.get("sentence") or "").strip()
            corrs = r.get("corrections") or []
            corr = next((c for c in corrs if c and c.strip() != sent), None)
            if not sent or not corr:
                continue
            if sent == corr:
                continue
            items.append({
                "id": f"jfleg_{split}_{i}",
                "correct": corr.strip(),
                "error":   sent,
                "split":   split,
            })
        if max_items > 0 and len(items) >= max_items:
            items = items[:max_items]
            break
    if max_items > 0:
        items = items[:max_items]
    return items


def load_lang8(max_items: int = 4000) -> list[dict]:
    from datasets import load_dataset
    items: list[dict] = []
    try:
        ds = load_dataset("jvamvas/peer_lang8", split="train", streaming=True)
    except Exception as e:
        logger.warning("peer_lang8 load failed: %s", e)
        return items
    for i, r in enumerate(ds):
        if i >= max_items * 3:  # over-pull then filter
            break
        src = r.get("src") or []
        tgt = r.get("tgt") or []
        if not src or not tgt:
            continue
        s_str = " ".join(src) if isinstance(src, list) else str(src)
        t_str = " ".join(tgt) if isinstance(tgt, list) else str(tgt)
        if s_str.strip() == t_str.strip():
            continue
        if not 4 <= len(s_str.split()) <= 80:
            continue
        items.append({
            "id": f"lang8_{i}",
            "correct": t_str.strip(),
            "error":   s_str.strip(),
        })
        if len(items) >= max_items:
            break
    return items


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def evaluate_corpus(repo: str, revision: str, corpus_name: str,
                    items: list[dict], output_root: Path,
                    batch_size: int) -> dict:
    pair = load_model(repo, DEVICE, revision=revision)
    if pair is None:
        logger.error("[%s @ %s] load failed", repo, revision)
        return {}
    model, tok = pair
    repo_safe = repo.replace("/", "_")
    rev_tag = "" if revision in ("main", "", None) else f"__rev={revision}"
    out_dir = output_root / f"{repo_safe}{rev_tag}" / corpus_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if not items:
        logger.warning("[%s] no items for corpus %s", repo, corpus_name)
        unload_model(model, tok)
        return {}
    correct = [it["correct"] for it in items]
    error   = [it["error"]   for it in items]
    t0 = time.time()
    bpt_corr = score_sentences_bpt(model, tok, correct, DEVICE, batch_size)
    bpt_err  = score_sentences_bpt(model, tok, error,   DEVICE, batch_size)
    logger.info("[%s @ %s] %s n=%d in %.1fs",
                repo, revision, corpus_name, len(items), time.time() - t0)

    # Per-item.
    items_csv = out_dir / "items.csv"
    keys = sorted({k for it in items for k in it.keys()
                   if k not in ("correct", "error")})
    cols = ["id"] + [k for k in keys if k != "id"] + [
        "bpt_correct", "bpt_error", "margin_bits"]
    with items_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        margins = []
        for it, bc, be in zip(items, bpt_corr, bpt_err):
            margin = float(be) - float(bc)
            row = {"id": it["id"]}
            for k in keys:
                if k == "id":
                    continue
                row[k] = it.get(k, "")
            row["bpt_correct"] = float(bc)
            row["bpt_error"]   = float(be)
            row["margin_bits"] = margin
            w.writerow(row)
            if math.isfinite(margin):
                margins.append(margin)
    summary = {
        "model": repo, "revision": revision, "corpus": corpus_name,
        "n_items": len(items),
        "n_finite_margins": len(margins),
        "mean_margin_bits": float(np.mean(margins)) if margins else float("nan"),
        "median_margin_bits": float(np.median(margins)) if margins else float("nan"),
        "frac_positive": float(np.mean(np.asarray(margins) > 0)) if margins else float("nan"),
    }
    sumcsv = out_dir / "summary.csv"
    with sumcsv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    logger.info("[%s @ %s] %s mean_margin=%.4f frac_pos=%.3f",
                repo, revision, corpus_name,
                summary["mean_margin_bits"], summary["frac_positive"])
    unload_model(model, tok)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--corpora", nargs="+",
                   default=["false_friend", "jfleg", "lang8"])
    p.add_argument("--false-friend-tsv", type=Path,
                   default=_REPO_ROOT / "data" / "cognates" / "false_friend_contexts.tsv")
    p.add_argument("--jfleg-max", type=int, default=0)
    p.add_argument("--lang8-max", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    p.add_argument("--world-size", type=int,
                   default=int(os.environ.get("WORLD_SIZE", "1")))
    args = p.parse_args(argv)

    args.output_root.mkdir(parents=True, exist_ok=True)
    setup_logging("interlanguage", args.output_root, args.rank)
    logger.info("eval_interlanguage rank %d/%d corpora=%s",
                args.rank, args.world_size, args.corpora)

    by_corpus: dict[str, list[dict]] = {}
    if "false_friend" in args.corpora:
        by_corpus["false_friend"] = load_false_friend(args.false_friend_tsv)
    if "jfleg" in args.corpora:
        by_corpus["jfleg"] = load_jfleg(max_items=args.jfleg_max)
    if "lang8" in args.corpora:
        by_corpus["lang8"] = load_lang8(max_items=args.lang8_max)
    for k, v in by_corpus.items():
        logger.info("loaded corpus %s: %d items", k, len(v))

    pairs = []
    for spec in args.models:
        if ":" in spec:
            repo, rev = spec.rsplit(":", 1)
        else:
            repo, rev = spec, "main"
        pairs.append((repo, rev))
    pairs = [p for i, p in enumerate(pairs) if i % args.world_size == args.rank]
    if not pairs:
        logger.info("rank %d: no pairs", args.rank)
        return 0

    for repo, rev in pairs:
        try:
            for cname, items in by_corpus.items():
                evaluate_corpus(repo, rev, cname, items,
                                args.output_root, args.batch_size)
        except Exception:
            logger.exception("[%s @ %s] failed", repo, rev)
    logger.info("eval_interlanguage done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
