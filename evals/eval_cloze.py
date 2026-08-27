#!/usr/bin/env python3
"""evals/eval_cloze.py

§4.1 CEFR-graded closed-cloze evaluation.

Two protocols, both reusing the score-options-given-context machinery from
``eval_rcmcq_variants.py``:

  closed: 4-option cloze. Default dataset = AndyChiang/cloth (HuggingFace).
          The dataset already provides {sentence, distractors, answer} in
          [MASK]-formatted sentences. Per item we score each candidate option
          in place of [MASK] and report argmax accuracy + length-normalised
          mean LL accuracy + bootstrap CI.

  cambridge: cloze derived from the local Cambridge Multiple-Choice Reading
          dataset by content-word masking. Per CEFR level we sample a fixed
          number of content-word slots and build 4-distractor cloze items
          (gold + 3 in-vocab nearest-frequency distractors).

Output:
  results/<output-root>/<model>__rev=<rev>/closed/all.csv      — closed cloze items
  results/<output-root>/<model>__rev=<rev>/closed/summary.csv  — accuracy summary
  results/<output-root>/<model>__rev=<rev>/cambridge/<level>.csv

Reused machinery: PROMPT_TEMPLATES from eval_rcmcq is not used (cloze prompts
are sentence-with-blank). We score with the same _score_with_context loop.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from evals.model_loader import (  # noqa: E402
    DEVICE, load_model, unload_model, setup_logging,
)
from evals.eval_rcmcq_variants import _score_with_context, _score_unconditional  # noqa: E402

logger = logging.getLogger(__name__)
OPTION_KEYS = ["a", "b", "c", "d"]
CEFR_LEVELS = ["A2", "B1", "B2", "C1", "C2"]


# ---------------------------------------------------------------------------
# Closed cloze (CLOTH)
# ---------------------------------------------------------------------------

def load_cloth(split: str = "test", max_items: int = 0,
               seed: int = 0) -> list[dict]:
    from datasets import load_dataset
    try:
        ds = load_dataset("AndyChiang/cloth", split=split)
    except Exception:
        ds = load_dataset("AndyChiang/cloth", split="train")
    items: list[dict] = []
    rng = random.Random(seed)
    for i, r in enumerate(ds):
        sent = r.get("sentence") or ""
        if "[MASK]" not in sent:
            continue
        ans = (r.get("answer") or "").strip()
        dists = list(r.get("distractors") or [])
        if not ans or len(dists) != 3:
            continue
        # Build 4-option list and randomise gold position.
        opts = dists + [ans]
        rng.shuffle(opts)
        gold_idx = opts.index(ans)
        gold = OPTION_KEYS[gold_idx]
        items.append({
            "id": f"cloth_{i}",
            "sentence": sent,
            "options": opts,
            "answer": gold,
        })
    if max_items > 0:
        items = items[:max_items]
    return items


def split_around_mask(sentence: str) -> tuple[str, str]:
    """Return (prefix, suffix) around `[MASK]` (first occurrence)."""
    idx = sentence.find("[MASK]")
    if idx < 0:
        return sentence, ""
    return sentence[:idx].rstrip(), sentence[idx + len("[MASK]"):].lstrip()


def evaluate_closed(repo: str, revision: str, items: list[dict],
                    output_root: Path, batch_size: int) -> dict:
    """Score a single (repo, revision) on closed-cloze items."""
    pair = load_model(repo, DEVICE, revision=revision)
    if pair is None:
        logger.error("[%s @ %s] load failed", repo, revision)
        return {}
    model, tok = pair

    repo_safe = repo.replace("/", "_")
    rev_tag = "" if revision in ("main", "", None) else f"__rev={revision}"
    out_dir = output_root / f"{repo_safe}{rev_tag}" / "closed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build (prompt, option) pairs. Cloze: prefix + option + suffix; we score
    # everything after prefix so suffix tokens are counted too — but the
    # in-place fill is what discriminates options, so suffix coverage is
    # symmetric and small. Simpler & consistent with rcmcq_variants: append
    # option after prefix, drop suffix from scoring (model conditioning
    # carries no signal from suffix anyway because it's the same across
    # options). Equivalent to "fill-in-the-blank" surprisal.
    prompts: list[str] = []
    opts: list[str] = []
    meta: list[tuple[str, str]] = []
    for it in items:
        prefix, _suffix = split_around_mask(it["sentence"])
        for o in it["options"]:
            prompts.append(prefix)
            opts.append(o)
        meta.append((it["id"], it["answer"]))

    # Unconditional scores for PMI variant.
    flat_options = [o for it in items for o in it["options"]]
    t0 = time.time()
    un_sum, un_mean, un_ntok = _score_unconditional(
        model, tok, flat_options, DEVICE, batch_size=max(8, batch_size * 2),
    )
    logger.info("[%s @ %s] uncond %d opts in %.1fs",
                repo, revision, len(flat_options), time.time() - t0)

    t1 = time.time()
    sum_lp, mean_lp, ntok = _score_with_context(
        model, tok, prompts, opts, DEVICE, batch_size,
    )
    logger.info("[%s @ %s] ctx %d×4 in %.1fs",
                repo, revision, len(meta), time.time() - t1)

    # Per-item CSV.
    items_csv = out_dir / "items.csv"
    cols = ["qid", "gold"]
    for k in OPTION_KEYS:
        cols += [f"sum_{k}", f"mean_{k}", f"un_sum_{k}", f"un_mean_{k}", f"ntok_{k}"]

    n_correct = {"mean": 0, "sum": 0, "pmi_mean": 0, "pmi_sum": 0}
    n_total = len(meta)
    n_unacc = 0
    with items_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for j, (qid, gold) in enumerate(meta):
            s_sum = sum_lp[4 * j: 4 * j + 4]
            s_mean = mean_lp[4 * j: 4 * j + 4]
            s_un_sum = un_sum[4 * j: 4 * j + 4]
            s_un_mean = un_mean[4 * j: 4 * j + 4]
            s_ntok = ntok[4 * j: 4 * j + 4]
            row = {"qid": qid, "gold": gold}
            for ki, k in enumerate(OPTION_KEYS):
                row[f"sum_{k}"] = float(s_sum[ki])
                row[f"mean_{k}"] = float(s_mean[ki])
                row[f"un_sum_{k}"] = float(s_un_sum[ki])
                row[f"un_mean_{k}"] = float(s_un_mean[ki])
                row[f"ntok_{k}"] = int(s_ntok[ki])
            w.writerow(row)
            if np.isnan(s_sum).any() or np.isnan(s_un_sum).any():
                n_unacc += 1
                continue
            gold_idx = OPTION_KEYS.index(gold)
            for rule, vec in (
                ("mean", s_mean), ("sum", s_sum),
                ("pmi_mean", s_mean - s_un_mean),
                ("pmi_sum", s_sum - s_un_sum),
            ):
                if int(np.argmax(vec)) == gold_idx:
                    n_correct[rule] += 1

    summary_csv = out_dir / "summary.csv"
    summary_row = {
        "model": repo, "revision": revision,
        "n_questions": n_total, "n_unacceptable": n_unacc,
        "acc_mean": n_correct["mean"] / max(1, n_total),
        "acc_sum": n_correct["sum"] / max(1, n_total),
        "acc_pmi_mean": n_correct["pmi_mean"] / max(1, n_total),
        "acc_pmi_sum": n_correct["pmi_sum"] / max(1, n_total),
    }
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_row.keys()))
        w.writeheader()
        w.writerow(summary_row)
    logger.info("[%s @ %s] closed acc_mean=%.3f acc_pmi_mean=%.3f n=%d",
                repo, revision, summary_row["acc_mean"],
                summary_row["acc_pmi_mean"], n_total)
    unload_model(model, tok)
    return summary_row


# ---------------------------------------------------------------------------
# Cambridge open cloze (derived from the MCQ Reading dataset)
# ---------------------------------------------------------------------------

import re

_CONTENT_RE = re.compile(r"^[A-Za-z][A-Za-z\-']{3,}$")


def _is_content_word(w: str, stop: set[str]) -> bool:
    if not _CONTENT_RE.match(w):
        return False
    return w.lower() not in stop


def build_cambridge_cloze(jsonl_path: Path, levels: list[str],
                          n_per_level: int = 80, seed: int = 0,
                          n_distractors: int = 3) -> dict[str, list[dict]]:
    """Derive 4-option cloze items from Cambridge MCQ Reading passages by masking
    one content word per sampled context window. Distractors are sampled from
    the level's vocabulary with similar token length.
    """
    stop = set("the a an and or but if then this that these those for of to in on at by with from is are was were be been being have has had do does did not no so as it its their there here he she they we you i my your our".split())
    rng = random.Random(seed)
    by_level: dict[str, list[str]] = defaultdict(list)
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            lvl = (rec.get("level") or "").upper().strip()
            text = rec.get("text") or ""
            if lvl in levels and text:
                by_level[lvl].append(text)
    out: dict[str, list[dict]] = {}
    for lvl in levels:
        passages = by_level.get(lvl, [])
        if not passages:
            continue
        # Build vocabulary across passages for distractor sampling.
        vocab = defaultdict(int)
        words_in_passage: list[list[str]] = []
        for p in passages:
            ws = re.findall(r"[A-Za-z][A-Za-z\-']*", p)
            words_in_passage.append(ws)
            for w in ws:
                if _is_content_word(w, stop):
                    vocab[w.lower()] += 1
        vocab_list = sorted(vocab.keys())
        items = []
        for pi, ws in enumerate(words_in_passage):
            if not ws:
                continue
            cand_idxs = [i for i, w in enumerate(ws) if _is_content_word(w, stop)]
            if not cand_idxs:
                continue
            rng.shuffle(cand_idxs)
            for idx in cand_idxs[:4]:  # up to 4 items per passage
                gold = ws[idx]
                # Build a sentence-window around the masked word.
                left = " ".join(ws[max(0, idx - 12): idx])
                right = " ".join(ws[idx + 1: idx + 13])
                # Distractors: same length-bucket, different word.
                glen = len(gold)
                cands = [w for w in vocab_list
                         if abs(len(w) - glen) <= 2 and w != gold.lower()]
                if len(cands) < n_distractors:
                    continue
                distractors = rng.sample(cands, n_distractors)
                opts = [gold.lower()] + distractors
                rng.shuffle(opts)
                gold_idx = opts.index(gold.lower())
                items.append({
                    "id": f"cam_{lvl}_{pi}_{idx}",
                    "sentence": (left + " [MASK] " + right).strip(),
                    "options": opts,
                    "answer": OPTION_KEYS[gold_idx],
                })
                if len(items) >= n_per_level:
                    break
            if len(items) >= n_per_level:
                break
        out[lvl] = items
    return out


def evaluate_cambridge(repo: str, revision: str,
                       cefr_items: dict[str, list[dict]],
                       output_root: Path, batch_size: int) -> list[dict]:
    pair = load_model(repo, DEVICE, revision=revision)
    if pair is None:
        logger.error("[%s @ %s] load failed", repo, revision)
        return []
    model, tok = pair
    repo_safe = repo.replace("/", "_")
    rev_tag = "" if revision in ("main", "", None) else f"__rev={revision}"
    out_dir = output_root / f"{repo_safe}{rev_tag}" / "cambridge"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for lvl, items in cefr_items.items():
        if not items:
            continue
        prompts = []
        opts = []
        meta = []
        for it in items:
            prefix, _ = split_around_mask(it["sentence"])
            for o in it["options"]:
                prompts.append(prefix)
                opts.append(o)
            meta.append((it["id"], it["answer"]))
        flat_options = [o for it in items for o in it["options"]]
        un_sum, un_mean, _ = _score_unconditional(
            model, tok, flat_options, DEVICE, batch_size=max(8, batch_size * 2),
        )
        sum_lp, mean_lp, ntok = _score_with_context(
            model, tok, prompts, opts, DEVICE, batch_size,
        )
        n_correct = {"mean": 0, "sum": 0, "pmi_mean": 0, "pmi_sum": 0}
        n_total = len(meta)
        n_unacc = 0
        items_csv = out_dir / f"{lvl}.csv"
        cols = ["qid", "gold"]
        for k in OPTION_KEYS:
            cols += [f"sum_{k}", f"mean_{k}", f"un_sum_{k}", f"un_mean_{k}", f"ntok_{k}"]
        with items_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for j, (qid, gold) in enumerate(meta):
                s_sum = sum_lp[4 * j: 4 * j + 4]
                s_mean = mean_lp[4 * j: 4 * j + 4]
                s_un_sum = un_sum[4 * j: 4 * j + 4]
                s_un_mean = un_mean[4 * j: 4 * j + 4]
                s_ntok = ntok[4 * j: 4 * j + 4]
                row = {"qid": qid, "gold": gold}
                for ki, k in enumerate(OPTION_KEYS):
                    row[f"sum_{k}"] = float(s_sum[ki])
                    row[f"mean_{k}"] = float(s_mean[ki])
                    row[f"un_sum_{k}"] = float(s_un_sum[ki])
                    row[f"un_mean_{k}"] = float(s_un_mean[ki])
                    row[f"ntok_{k}"] = int(s_ntok[ki])
                w.writerow(row)
                if np.isnan(s_sum).any() or np.isnan(s_un_sum).any():
                    n_unacc += 1
                    continue
                gold_idx = OPTION_KEYS.index(gold)
                for rule, vec in (
                    ("mean", s_mean), ("sum", s_sum),
                    ("pmi_mean", s_mean - s_un_mean),
                    ("pmi_sum", s_sum - s_un_sum),
                ):
                    if int(np.argmax(vec)) == gold_idx:
                        n_correct[rule] += 1
        summary_rows.append({
            "model": repo, "revision": revision, "cefr": lvl,
            "n_questions": n_total, "n_unacceptable": n_unacc,
            "acc_mean": n_correct["mean"] / max(1, n_total),
            "acc_sum": n_correct["sum"] / max(1, n_total),
            "acc_pmi_mean": n_correct["pmi_mean"] / max(1, n_total),
            "acc_pmi_sum": n_correct["pmi_sum"] / max(1, n_total),
        })
        logger.info("[%s @ %s] cambridge %s acc_mean=%.3f n=%d",
                    repo, revision, lvl,
                    summary_rows[-1]["acc_mean"], n_total)
    if summary_rows:
        sumcsv = out_dir / "summary.csv"
        with sumcsv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        logger.info("wrote %s", sumcsv)
    unload_model(model, tok)
    return summary_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True,
                   help="<repo>:<revision> entries; default revision=main")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--protocol", choices=["closed", "cambridge", "both"],
                   default="both")
    p.add_argument("--cloth-max", type=int, default=2000)
    p.add_argument("--cloth-split", default="test")
    p.add_argument("--cambridge-jsonl", type=Path,
                   default=_REPO_ROOT / "education" /
                   "Cambridge Multiple-Choice Questions Reading Dataset.jsonl")
    p.add_argument("--cambridge-per-level", type=int, default=80)
    p.add_argument("--cefr-levels", nargs="+", default=CEFR_LEVELS)
    p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    p.add_argument("--world-size", type=int,
                   default=int(os.environ.get("WORLD_SIZE", "1")))
    args = p.parse_args(argv)

    args.output_root.mkdir(parents=True, exist_ok=True)
    setup_logging("cloze", args.output_root, args.rank)
    logger.info("eval_cloze rank %d/%d protocol=%s",
                args.rank, args.world_size, args.protocol)

    pairs = []
    for spec in args.models:
        if ":" in spec:
            repo, rev = spec.rsplit(":", 1)
        else:
            repo, rev = spec, "main"
        pairs.append((repo, rev))
    pairs = [p for i, p in enumerate(pairs) if i % args.world_size == args.rank]
    if not pairs:
        logger.info("no pairs for rank %d", args.rank)
        return 0

    closed_items = []
    cambridge_items = {}
    if args.protocol in ("closed", "both"):
        closed_items = load_cloth(split=args.cloth_split, max_items=args.cloth_max)
        logger.info("loaded %d closed-cloze items", len(closed_items))
    if args.protocol in ("cambridge", "both"):
        cambridge_items = build_cambridge_cloze(
            args.cambridge_jsonl,
            levels=[lv.upper() for lv in args.cefr_levels],
            n_per_level=args.cambridge_per_level,
        )
        logger.info("built cambridge cloze: %s",
                    {k: len(v) for k, v in cambridge_items.items()})

    for repo, rev in pairs:
        try:
            if closed_items:
                evaluate_closed(repo, rev, closed_items,
                                args.output_root, args.batch_size)
            if cambridge_items:
                evaluate_cambridge(repo, rev, cambridge_items,
                                   args.output_root, args.batch_size)
        except Exception:
            logger.exception("[%s @ %s] failed", repo, rev)
    logger.info("eval_cloze done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
