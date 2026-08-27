#!/usr/bin/env python3
"""evals/eval_evp_trajectory.py

§4.3 EVP-style vocabulary acquisition trajectory.

For each (model repo, checkpoint) and each CEFR-tagged target word, compute
mean per-token surprisal (bits) of the target word in a small set of
controlled contexts. Aggregate across checkpoints to produce the per-curriculum
acquisition curve in plots/make_evp_trajectory.py.

Wordlist sources (combined; output records keep the source column):
  - data/evp/cefrj-vocabulary-profile-1.5.csv   (A1..B2)
  - data/evp/octanove-vocabulary-profile-c1c2-1.0.csv  (C1..C2)

Contexts (cheap, controlled):
  Each target word is scored in K templated 'I think the word X means …' style
  bridges so that surprisal is conditioned on a non-trivial left context.
  We use 3 templates per word; the per-word score is the mean across them.

Output (rank-disjoint):
  results/<output-root>/<model>__rev=<rev>/word_surprisal.csv
    columns: word, pos, cefr, source, mean_bits, n_contexts
  results/<output-root>/<model>__rev=<rev>/manifest.json    (rev metadata)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evals.model_loader import (  # noqa: E402
    DEVICE, load_model, unload_model, setup_logging, resolve_vocab_size,
)

logger = logging.getLogger(__name__)
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

CONTEXT_TEMPLATES = [
    "The teacher told us that the word {word}",
    "She wrote a sentence using the word {word}",
    "I learned today that {word}",
]


def load_wordlist(cefrj_csv: Path, octanove_csv: Path,
                  per_level: int = 0, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for src, path in [("cefrj", cefrj_csv), ("octanove", octanove_csv)]:
        if not path.exists():
            logger.warning("missing wordlist %s", path)
            continue
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                hw = (r.get("headword") or "").split("/")[0].strip()
                pos = (r.get("pos") or "").strip()
                cefr = (r.get("CEFR") or "").strip().upper()
                if not hw or cefr not in CEFR_LEVELS:
                    continue
                # Skip multi-word entries; keep single tokens with hyphens/apostrophes.
                if " " in hw or "(" in hw:
                    continue
                key = (hw.lower(), cefr)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"word": hw, "pos": pos, "cefr": cefr, "source": src})
    if per_level > 0:
        bucketed: dict[str, list[dict]] = {l: [] for l in CEFR_LEVELS}
        for r in out:
            bucketed[r["cefr"]].append(r)
        sub: list[dict] = []
        for l in CEFR_LEVELS:
            arr = bucketed[l]
            if len(arr) > per_level:
                idxs = rng.choice(len(arr), size=per_level, replace=False)
                arr = [arr[i] for i in idxs]
            sub.extend(arr)
        out = sub
    return out


@torch.no_grad()
def score_words_in_contexts(model, tok, words: list[str],
                            templates: list[str], device: str,
                            batch_size: int) -> np.ndarray:
    """Return mean surprisal (bits) of each target word, averaged across
    templates. Surprisal = sum of token log-probs over the target word
    tokens, conditional on the bridging template tokens.

    Returns array shape (n_words,) in bits.
    """
    LOG2 = math.log(2.0)
    vocab_size = resolve_vocab_size(model)

    # Build flat list of (template_idx, word_idx, prefix_text, full_text)
    flat: list[tuple[int, int, str, str]] = []
    for ti, t in enumerate(templates):
        for wi, w in enumerate(words):
            prefix = t.format(word="").rstrip()
            full = t.format(word=w)
            flat.append((ti, wi, prefix, full))

    n = len(flat)
    bits_per_target = np.zeros((len(templates), len(words)), dtype=np.float64)
    counts = np.zeros((len(templates), len(words)), dtype=np.int32)

    cur_bs = batch_size
    i = 0
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (
        tok.eos_token_id if tok.eos_token_id is not None else 0)
    while i < n:
        bs = min(cur_bs, n - i)
        sl = flat[i: i + bs]
        try:
            full_texts = [s[3] for s in sl]
            prefix_texts = [s[2] for s in sl]
            enc = tok(full_texts, padding=True, truncation=True, max_length=64,
                      return_tensors="pt", add_special_tokens=True)
            input_ids = enc["input_ids"].to(device).clamp(0, vocab_size - 1)
            attn = enc["attention_mask"].to(device)
            prefix_lens = [
                len(tok(p, add_special_tokens=True, truncation=True,
                        max_length=64)["input_ids"]) for p in prefix_texts
            ]
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(input_ids=input_ids, attention_mask=attn).logits
            logp = F.log_softmax(logits.float(), dim=-1)
            shift_lp = logp[:, :-1, :]
            shift_ids = input_ids[:, 1:]
            shift_mask = attn[:, 1:].clone().bool()
            tok_lp = shift_lp.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
            for b in range(bs):
                p_len = max(0, prefix_lens[b] - 1)
                shift_mask[b, :p_len] = False
            n_target = shift_mask.sum(dim=1).clamp_min(1)
            sum_lp = (tok_lp * shift_mask).sum(dim=1)
            mean_lp = (sum_lp / n_target).cpu().numpy()
            for k, (ti, wi, _pre, _full) in enumerate(sl):
                # negative log-prob in nats → bits
                bits_per_target[ti, wi] = -mean_lp[k] / LOG2
                counts[ti, wi] = 1
            i += bs
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and cur_bs > 1:
                cur_bs = max(1, cur_bs // 2)
                logger.warning("evp ctx OOM at bs=%d → %d", cur_bs * 2, cur_bs)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise

    # Mean across templates.
    bits_per_word = bits_per_target.mean(axis=0)
    return bits_per_word


def list_checkpoints(repo: str, max_n: int = 0) -> list[str]:
    """Enumerate step-N branches plus 'main', subsampled to max_n if set."""
    from huggingface_hub import list_repo_refs
    try:
        refs = list_repo_refs(repo)
    except Exception as e:
        logger.warning("list_repo_refs(%s) failed: %s", repo, e)
        return ["main"]
    revs: list[tuple[int, str]] = []
    for b in refs.branches:
        if b.name.startswith("step-"):
            try:
                step = int(b.name.split("-", 1)[1])
                revs.append((step, b.name))
            except ValueError:
                continue
    revs.sort()
    if max_n and len(revs) > max_n:
        idx = np.linspace(0, len(revs) - 1, max_n).round().astype(int).tolist()
        revs = [revs[i] for i in idx]
    out = [r[1] for r in revs]
    out.append("main")
    return out


def evaluate_repo(repo: str, words: list[dict], output_root: Path,
                  batch_size: int, revisions: list[str]) -> None:
    repo_safe = repo.replace("/", "_")
    word_strs = [w["word"] for w in words]
    rev_summary = []
    for rev in revisions:
        rev_tag = "" if rev in ("main", "", None) else f"__rev={rev}"
        out_dir = output_root / f"{repo_safe}{rev_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "word_surprisal.csv"
        if out_csv.exists() and out_csv.stat().st_size > 0:
            logger.info("[%s @ %s] skip (exists)", repo, rev)
            rev_summary.append({"revision": rev, "skipped": True})
            continue
        pair = load_model(repo, DEVICE, revision=rev)
        if pair is None:
            logger.warning("[%s @ %s] load failed", repo, rev)
            continue
        model, tok = pair
        t0 = time.time()
        bits = score_words_in_contexts(model, tok, word_strs,
                                       CONTEXT_TEMPLATES, DEVICE, batch_size)
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["word", "pos", "cefr", "source", "mean_bits"])
            for rec, b in zip(words, bits):
                w.writerow([rec["word"], rec["pos"], rec["cefr"], rec["source"],
                            f"{b:.6f}"])
        logger.info("[%s @ %s] %d words in %.1fs (mean=%.2f bits)",
                    repo, rev, len(words), time.time() - t0,
                    float(np.nanmean(bits)))
        rev_summary.append({"revision": rev, "n_words": len(words),
                            "mean_bits": float(np.nanmean(bits))})
        unload_model(model, tok)
    manifest_path = output_root / repo_safe / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "repo": repo, "revisions": rev_summary,
    }, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repos", nargs="+", required=True,
                   help="Beetle repos to trace (each gets its own checkpoint list)")
    p.add_argument("--max-checkpoints", type=int, default=14,
                   help="Subsample step-N branches (in addition to main)")
    p.add_argument("--cefrj-csv", type=Path,
                   default=_REPO_ROOT / "data" / "evp" /
                   "cefrj-vocabulary-profile-1.5.csv")
    p.add_argument("--octanove-csv", type=Path,
                   default=_REPO_ROOT / "data" / "evp" /
                   "octanove-vocabulary-profile-c1c2-1.0.csv")
    p.add_argument("--per-level", type=int, default=200)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    p.add_argument("--world-size", type=int,
                   default=int(os.environ.get("WORLD_SIZE", "1")))
    args = p.parse_args(argv)

    args.output_root.mkdir(parents=True, exist_ok=True)
    setup_logging("evp_trajectory", args.output_root, args.rank)
    logger.info("eval_evp_trajectory rank %d/%d", args.rank, args.world_size)

    words = load_wordlist(args.cefrj_csv, args.octanove_csv,
                          per_level=args.per_level)
    by_lvl: dict[str, int] = {}
    for w in words:
        by_lvl[w["cefr"]] = by_lvl.get(w["cefr"], 0) + 1
    logger.info("wordlist: %d total %s", len(words), sorted(by_lvl.items()))

    repos = [r for i, r in enumerate(args.repos)
             if i % args.world_size == args.rank]
    if not repos:
        logger.info("rank %d: no repos", args.rank)
        return 0

    for repo in repos:
        revs = list_checkpoints(repo, max_n=args.max_checkpoints)
        logger.info("[%s] %d revisions", repo, len(revs))
        try:
            evaluate_repo(repo, words, args.output_root,
                          args.batch_size, revs)
        except Exception:
            logger.exception("[%s] failed", repo)
    logger.info("eval_evp_trajectory done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
