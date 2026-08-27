#!/usr/bin/env python3
"""
evals/eval_race_student.py — RACE × checkpoint × student-persona sweep.

For each (checkpoint, student-persona) pair, score the four RACE answer
options under the model with multiple-choice log-likelihood (no
generation) and compute accuracy. Five personas span weak → excellent
student, with the *ideal* curve ``[0.00, 0.25, 0.50, 0.75, 1.00]``.

Per-checkpoint metrics:

  acc  : 5-tuple of accuracies, one per persona.
  corr : Pearson correlation of acc with the ideal curve.
  pen  : Σ_{i=1..4} max(0, acc[i] − acc[i+1])    (monotonicity violation)
  M    : corr − pen                              (overall stability score)

Differences from the source notebook
------------------------------------
* No generation → deterministic, ~50× faster, no temperature/top_p.
* Score per option = sum of token log-probs over (prompt + option),
  averaged over option-only tokens. The argmax option = predicted answer.
  Mirrors evals/eval_blimp.py:score_sentences but masks prompt tokens.
* Sharded by checkpoint across --rank/--world-size; RACE itself is
  identical for every rank, so no inter-rank comms.
* Ledger entry per (repo, revision) so re-runs are idempotent.

Usage
-----
  python evals/eval_race_student.py \\
      --repo Beetle-FineWeb2-24B/beetle-bilingual-l2-80-late-b5-fineweb-deu-eng \\
      --revisions main step-32 step-128 step-512 step-2048 \\
      --max-questions 64 --batch-size 8 \\
      --rank 0 --world-size 8 \\
      --output-root results
"""
from __future__ import annotations

import argparse
import ast
import csv
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "analyze"))

from evals.model_loader import (
    DEVICE, load_model, unload_model, setup_logging,
    resolve_vocab_size, log_gpu_mem,
)

logger = logging.getLogger(__name__)

# Ledger (best-effort)
try:
    from analyze.eval_ledger import is_done as _ledger_is_done
    from analyze.eval_ledger import record as _ledger_record
except Exception:
    def _ledger_is_done(*_a, **_kw) -> bool: return False
    def _ledger_record(*_a, **_kw) -> None: pass


# ---------------------------------------------------------------------------
# Personas + ideal curve  (verbatim from the source notebook)
# ---------------------------------------------------------------------------
PROMPTS: dict[int, str] = {
    1: "You are a very weak student. Answer the question.",
    2: "You are a weak student. Answer the question.",
    3: "You are an average student. Answer the question.",
    4: "You are a good student. Answer the question.",
    5: "You are an excellent student. Answer the question.",
}
IDEAL = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

# A B C D — RACE answer keys
_OPTION_KEYS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# RACE parsing
# ---------------------------------------------------------------------------

def _parse_item(ex: dict) -> list[tuple[str, str, list[str], str]]:
    """Yield (article, question, options, answer) tuples for one RACE row.

    The HF ``EleutherAI/race`` ``test`` split stores ``problems`` as either
    a list-of-dicts or a stringified Python literal. Tolerant to both.
    """
    article = ex["article"]
    probs = ex.get("problems", None)
    if probs is None:
        return []
    if isinstance(probs, str):
        try:
            probs = ast.literal_eval(probs)
        except Exception:
            return []
    if not isinstance(probs, list):
        return []
    out: list[tuple[str, str, list[str], str]] = []
    for p in probs:
        try:
            q    = p["question"]
            opts = p["options"]
            ans  = p["answer"]
            if not isinstance(opts, list) or len(opts) != 4:
                continue
            out.append((article, q, opts, ans))
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Multiple-choice loglik scoring
# ---------------------------------------------------------------------------

def _is_oom(exc: BaseException) -> bool:
    if torch.cuda.is_available():
        oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


@torch.no_grad()
def _score_option_logliks(
    model, tokenizer,
    prompts: list[str], options: list[str],
    device: str, batch_size: int,
) -> np.ndarray:
    """Per-pair mean-token log-likelihood of *option* given *prompt*.

    Returns a 1-D numpy array of shape (len(prompts),). The score is the
    mean log-prob of the option tokens (prompt tokens masked out), so
    options of unequal length stay comparable.
    """
    assert len(prompts) == len(options)
    vocab_size = resolve_vocab_size(model)
    n = len(prompts)
    out = np.full(n, fill_value=float("nan"), dtype=np.float64)
    cur_bs = batch_size
    i = 0
    while i < n:
        bs = min(cur_bs, n - i)
        sl = slice(i, i + bs)
        try:
            # Tokenize prompt-only and prompt+option separately so we can
            # locate the option-token span by length difference.
            enc_full = tokenizer(
                [p + " " + o for p, o in zip(prompts[sl], options[sl])],
                padding=True, truncation=True, max_length=1024,
                return_tensors="pt", add_special_tokens=True,
            )
            prompt_lens = [
                len(tokenizer(p, add_special_tokens=True)["input_ids"])
                for p in prompts[sl]
            ]

            input_ids      = enc_full["input_ids"].to(device).clamp(0, vocab_size - 1)
            attention_mask = enc_full["attention_mask"].to(device)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            log_probs = F.log_softmax(outputs.logits.float(), dim=-1)
            del outputs

            shift_lp   = log_probs[:, :-1, :]
            shift_ids  = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:].clone().bool()
            tok_lp     = shift_lp.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)

            # Mask out prompt tokens — only the option span counts.
            for b in range(bs):
                # prompt_lens[b] tokens are prompt; the rest is option.
                p_len = max(0, prompt_lens[b] - 1)  # -1: shift
                shift_mask[b, :p_len] = False

            n_opt = shift_mask.sum(dim=1).clamp_min(1)
            sum_lp = (tok_lp * shift_mask).sum(dim=1)
            avg_lp = (sum_lp / n_opt).cpu().numpy()
            out[i:i + bs] = avg_lp
        except BaseException as exc:
            if _is_oom(exc) and cur_bs > 1:
                new_bs = max(1, cur_bs // 2)
                logger.warning(
                    f"    RACE CUDA OOM at batch_size={cur_bs}; "
                    f"halving to {new_bs} and retrying"
                )
                cur_bs = new_bs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
        i += bs
    return out


def _evaluate_persona(
    model, tokenizer, items: list[tuple[str, str, list[str], str]],
    persona_prompt: str, device: str, batch_size: int,
) -> float:
    """Accuracy for one persona over *items*.

    Builds 4 (prompt, option) pairs per question, scores each, picks the
    argmax option. Correct = pred matches RACE answer key letter.
    """
    if not items:
        return 0.0
    flat_prompts: list[str] = []
    flat_options: list[str] = []
    answers: list[str] = []
    for article, q, opts, ans in items:
        base = (
            f"{persona_prompt}\n\n"
            f"{article}\n\nQuestion: {q}\n"
            f"A. {opts[0]}\nB. {opts[1]}\nC. {opts[2]}\nD. {opts[3]}\n"
            f"Answer:"
        )
        for opt_text in opts:
            flat_prompts.append(base)
            flat_options.append(" " + opt_text)
        answers.append(ans)

    scores = _score_option_logliks(
        model, tokenizer, flat_prompts, flat_options, device, batch_size,
    ).reshape(len(items), 4)
    preds = np.argmax(scores, axis=1)
    correct = sum(_OPTION_KEYS[p] == a for p, a in zip(preds, answers))
    return correct / len(items)


# ---------------------------------------------------------------------------
# Per-checkpoint M-score
# ---------------------------------------------------------------------------

def _M_score(acc: np.ndarray) -> tuple[float, float, float]:
    """Return (corr, penalty, M).

    corr     = Pearson(acc, IDEAL)  (0 if std(acc) == 0)
    penalty  = Σ max(0, acc[i] - acc[i+1])  — drops in accuracy as
               persona improves are penalised.
    M        = corr - penalty
    """
    if float(np.std(acc)) == 0.0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(acc, IDEAL)[0, 1])
    penalty = float(sum(max(0.0, acc[i] - acc[i + 1]) for i in range(4)))
    return corr, penalty, corr - penalty


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _step_int(rev: str) -> int | None:
    if rev == "main":
        return None
    if rev.startswith("step-") and rev[5:].isdigit():
        return int(rev[5:])
    if rev.isdigit():
        return int(rev)
    return None


def _output_csv(output_root: Path, repo: str) -> Path:
    slug = repo.replace("/", "__")
    return output_root / "evals" / "race" / f"race_{slug}.csv"


def _append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)


def evaluate(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    setup_logging("race", output_root, args.rank)

    # Shard checkpoints across ranks.
    revisions = list(args.revisions)
    my_revs = [r for i, r in enumerate(revisions)
               if i % args.world_size == args.rank]
    logger.info(f"[rank {args.rank}/{args.world_size}] {len(my_revs)} "
                f"revisions assigned for {args.repo}")

    # Load RACE once.
    from datasets import load_dataset
    ds = load_dataset(args.race_dataset, split=args.race_split)
    items: list[tuple[str, str, list[str], str]] = []
    for ex in ds:
        items.extend(_parse_item(ex))
        if len(items) >= args.max_questions:
            break
    items = items[:args.max_questions]
    logger.info(f"  RACE: {len(items)} questions loaded "
                f"(dataset={args.race_dataset}, split={args.race_split})")

    out_csv = _output_csv(output_root, args.repo)

    for rev in my_revs:
        if _ledger_is_done("race", args.repo, rev, "default", None,
                           output_root=str(output_root)):
            logger.info(f"  RACE SKIP {args.repo}@{rev} (ledger)")
            continue

        logger.info("=" * 60)
        logger.info(f"CHECKPOINT {args.repo}@{rev}")
        logger.info("=" * 60)
        t0 = time.time()
        result = load_model(args.repo, DEVICE, revision=rev)
        if result is None:
            logger.error(f"  load failed for {args.repo}@{rev}; skipping")
            _ledger_record("race", args.repo, rev, "default", None, "error",
                           output_root=str(output_root))
            continue
        model, tokenizer = result
        log_gpu_mem(logger, prefix=f"after-load {rev}")

        try:
            acc_list: list[float] = []
            for level in (1, 2, 3, 4, 5):
                acc = _evaluate_persona(
                    model, tokenizer, items,
                    PROMPTS[level], DEVICE, args.batch_size,
                )
                acc_list.append(acc)
                logger.info(f"  persona {level}: acc={acc:.3f}")
            acc_arr = np.array(acc_list)
            corr, pen, M = _M_score(acc_arr)
            logger.info(f"  corr={corr:.3f}  penalty={pen:.3f}  M={M:.3f}")
            row = {
                "repo": args.repo,
                "revision": rev,
                "step": _step_int(rev),
                "acc_1": round(acc_list[0], 4),
                "acc_2": round(acc_list[1], 4),
                "acc_3": round(acc_list[2], 4),
                "acc_4": round(acc_list[3], 4),
                "acc_5": round(acc_list[4], 4),
                "corr":  round(corr, 4),
                "penalty": round(pen, 4),
                "M":     round(M, 4),
                "n_items": len(items),
            }
            _append_row(out_csv, row)
            _ledger_record("race", args.repo, rev, "default", None, "ok",
                           output_root=str(output_root))
        except Exception:
            logger.error(f"  RACE ERROR {args.repo}@{rev}", exc_info=True)
            _ledger_record("race", args.repo, rev, "default", None, "error",
                           output_root=str(output_root))
        finally:
            unload_model(model, tokenizer)
            del model, tokenizer
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"  done in {(time.time()-t0)/60:.1f} min")

    logger.info(f"[rank {args.rank}] all assigned revisions complete → {out_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True)
    p.add_argument("--revisions", nargs="+", required=True,
                   help="Checkpoint revisions to evaluate (e.g. main step-512 step-2048).")
    p.add_argument("--max-questions", type=int, default=64,
                   help="Cap RACE questions per checkpoint (default 64).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--race-dataset", default="EleutherAI/race")
    p.add_argument("--race-split", default="test")
    p.add_argument("--output-root", default="results")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
