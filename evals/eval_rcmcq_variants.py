#!/usr/bin/env python3
"""
evals/eval_rcmcq_variants.py — RC-MCQ scorer with multiple normalization rules.

Mirrors evals/eval_rcmcq.py but computes, for each (prompt, cefr, question, option):

  sum_lp        sum of token log-probs over option tokens given context
  mean_lp       sum_lp / n_option_tokens   (this is what the baseline reports)
  uncond_sum    sum of token log-probs over option tokens with no context
  uncond_mean   uncond_sum / n_option_tokens
  pmi_sum       sum_lp - uncond_sum                         (PMI scoring)
  pmi_mean      mean_lp - uncond_mean                       (length-balanced PMI)

Unconditional scores depend only on the option text and the model, so they are
cached per-question and reused across all (prompt, cefr) cells.

Per-item CSV columns:
  qid, gold, sum_a..sum_d, mean_a..mean_d,
  un_sum_a..un_sum_d, un_mean_a..un_mean_d, ntok_a..ntok_d

Summary CSV columns:
  model, prompt, cefr, n_questions, n_unacceptable,
  acc_mean, acc_sum, acc_pmi_mean, acc_pmi_sum

Output:
  results/<output-root>/<model>[ __rev=...]/<prompt>/<cefr>.csv
  results/<output-root>/<model>[ __rev=...]/summary.csv
"""
from __future__ import annotations

import argparse
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

from evals.model_loader import (  # noqa: E402
    DEVICE, load_model, setup_logging, resolve_vocab_size, log_gpu_mem,
)
from evals.eval_rcmcq import (  # noqa: E402
    PROMPT_TEMPLATES, OPTION_KEYS, load_questions, render_prompt,
)

logger = logging.getLogger(__name__)


def _is_oom(exc: BaseException) -> bool:
    if torch.cuda.is_available():
        oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


@torch.no_grad()
def _score_with_context(
    model, tok, prompts, options, device, batch_size, max_length=1536,
):
    """Return arrays (sum_lp, mean_lp, n_tok), one entry per (prompt, option) pair.

    Scores option tokens conditional on the prompt; prompt tokens are masked out.
    """
    assert len(prompts) == len(options)
    vocab_size = resolve_vocab_size(model)
    n = len(prompts)
    sum_out = np.full(n, np.nan, dtype=np.float64)
    mean_out = np.full(n, np.nan, dtype=np.float64)
    ntok_out = np.zeros(n, dtype=np.int32)
    cur_bs = batch_size
    i = 0
    while i < n:
        bs = min(cur_bs, n - i)
        sl = slice(i, i + bs)
        try:
            enc = tok(
                [p + " " + o for p, o in zip(prompts[sl], options[sl])],
                padding=True, truncation=True, max_length=max_length,
                return_tensors="pt", add_special_tokens=True,
            )
            prompt_lens = [
                len(tok(p, add_special_tokens=True,
                        truncation=True, max_length=max_length)["input_ids"])
                for p in prompts[sl]
            ]
            input_ids = enc["input_ids"].to(device).clamp(0, vocab_size - 1)
            attn = enc["attention_mask"].to(device)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(input_ids=input_ids, attention_mask=attn).logits
            logp = F.log_softmax(logits.float(), dim=-1)
            shift_lp = logp[:, :-1, :]
            shift_ids = input_ids[:, 1:]
            shift_mask = attn[:, 1:].clone().bool()
            tok_lp = shift_lp.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
            for b in range(bs):
                p_len = max(0, prompt_lens[b] - 1)
                shift_mask[b, :p_len] = False
            n_opt = shift_mask.sum(dim=1).clamp_min(1)
            sum_lp = (tok_lp * shift_mask).sum(dim=1)
            mean_lp = sum_lp / n_opt
            sum_out[i:i + bs] = sum_lp.cpu().numpy()
            mean_out[i:i + bs] = mean_lp.cpu().numpy()
            ntok_out[i:i + bs] = n_opt.cpu().numpy()
            i += bs
        except BaseException as exc:
            if _is_oom(exc) and cur_bs > 1:
                cur_bs = max(1, cur_bs // 2)
                logger.warning("    rcmcq-ctx OOM at bs=%d → %d", cur_bs * 2, cur_bs)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
    return sum_out, mean_out, ntok_out


@torch.no_grad()
def _score_unconditional(model, tok, options, device, batch_size, max_length=256):
    """Return arrays (sum_lp, mean_lp, n_tok) for option tokens with no context.

    BOS (or eos as bos fallback) is prepended so the first option token still
    receives a non-trivial conditioning signal. Without this, the first token's
    score is meaningless and PMI is biased.
    """
    bos_id = tok.bos_token_id
    if bos_id is None:
        bos_id = tok.eos_token_id
    if bos_id is None:
        bos_id = 0
    vocab_size = resolve_vocab_size(model)

    n = len(options)
    sum_out = np.full(n, np.nan, dtype=np.float64)
    mean_out = np.full(n, np.nan, dtype=np.float64)
    ntok_out = np.zeros(n, dtype=np.int32)

    # Tokenize options once without special tokens, then prepend BOS manually.
    opt_ids = [
        tok(o, add_special_tokens=False, truncation=True,
            max_length=max_length - 1)["input_ids"]
        for o in options
    ]
    cur_bs = batch_size
    i = 0
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else bos_id
    while i < n:
        bs = min(cur_bs, n - i)
        try:
            batch_ids = []
            batch_masks = []
            n_opt_tokens = []
            max_len = 0
            for j in range(bs):
                ids = [bos_id] + opt_ids[i + j]
                batch_ids.append(ids)
                n_opt_tokens.append(len(opt_ids[i + j]))
                max_len = max(max_len, len(ids))
            for j in range(bs):
                pad_n = max_len - len(batch_ids[j])
                batch_masks.append([1] * len(batch_ids[j]) + [0] * pad_n)
                batch_ids[j] = batch_ids[j] + [pad_id] * pad_n
            input_ids = torch.tensor(batch_ids, dtype=torch.long, device=device).clamp(0, vocab_size - 1)
            attn = torch.tensor(batch_masks, dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(input_ids=input_ids, attention_mask=attn).logits
            logp = F.log_softmax(logits.float(), dim=-1)
            shift_lp = logp[:, :-1, :]
            shift_ids = input_ids[:, 1:]
            shift_mask = attn[:, 1:].clone().bool()
            tok_lp = shift_lp.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
            # Mask: token positions corresponding to option tokens are positions
            # 0..n_opt-1 in the shifted view (since BOS is the first token).
            for b in range(bs):
                shift_mask[b, n_opt_tokens[b]:] = False
            n_opt = shift_mask.sum(dim=1).clamp_min(1)
            sum_lp = (tok_lp * shift_mask).sum(dim=1)
            mean_lp = sum_lp / n_opt
            sum_out[i:i + bs] = sum_lp.cpu().numpy()
            mean_out[i:i + bs] = mean_lp.cpu().numpy()
            ntok_out[i:i + bs] = n_opt.cpu().numpy()
            i += bs
        except BaseException as exc:
            if _is_oom(exc) and cur_bs > 1:
                cur_bs = max(1, cur_bs // 2)
                logger.warning("    rcmcq-uncond OOM at bs=%d → %d", cur_bs * 2, cur_bs)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
    return sum_out, mean_out, ntok_out


def evaluate_model(repo, questions, cefr_levels, prompts, output_root,
                   batch_size, revision="main", adapter=None, output_tag=None):
    repo_safe = repo.replace("/", "_")
    if output_tag:
        dir_name = f"{repo_safe}__{output_tag}"
    elif revision and revision != "main":
        dir_name = f"{repo_safe}__rev={revision}"
    else:
        dir_name = repo_safe
    out_dir = output_root / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pair = load_model(repo, DEVICE, revision=revision)
    if pair is None:
        logger.error("[%s @ %s] load failed; skipping", repo, revision)
        return
    model, tok = pair

    if adapter:
        try:
            from peft import PeftModel
        except ImportError:
            logger.error("--adapter given but peft is not installed")
            return
        logger.info("Attaching LoRA adapter from %s", adapter)
        model = PeftModel.from_pretrained(model, adapter)
        model.eval()
    log_gpu_mem(logger, f"[{repo} @ {revision}] post-load")

    # ---- 1. Cache unconditional scores once per question/option ------------
    flat_options: list[str] = []
    for q in questions:
        flat_options.extend(q["options"])
    t0 = time.time()
    un_sum, un_mean, un_ntok = _score_unconditional(
        model, tok, flat_options, DEVICE, batch_size=max(8, batch_size * 2),
    )
    logger.info(
        "[%s] cached unconditional scores for %d options in %.1fs",
        repo, len(flat_options), time.time() - t0,
    )

    # ---- 2. Loop over (prompt, cefr) and score with context ----------------
    summary_rows: list[dict] = []
    for prompt_key in prompts:
        for cefr in cefr_levels:
            prompt_texts: list[str] = []
            option_texts: list[str] = []
            meta: list[tuple[str, str]] = []
            for q in questions:
                p = render_prompt(prompt_key, cefr, q["passage"], q["question"])
                for opt in q["options"]:
                    prompt_texts.append(p)
                    option_texts.append(opt)
                meta.append((q["id"], q["answer"]))
            if not prompt_texts:
                continue
            t1 = time.time()
            sum_lp, mean_lp, ntok = _score_with_context(
                model, tok, prompt_texts, option_texts, DEVICE, batch_size,
            )
            logger.info(
                "[%s] prompt=%s cefr=%s scored %d×4 ctx in %.1fs",
                repo, prompt_key, cefr, len(meta), time.time() - t1,
            )

            cell_dir = out_dir / prompt_key
            cell_dir.mkdir(parents=True, exist_ok=True)
            cell_csv = cell_dir / f"{cefr}.csv"
            cols = ["qid", "gold"]
            for k in OPTION_KEYS:
                cols += [f"sum_{k}", f"mean_{k}",
                         f"un_sum_{k}", f"un_mean_{k}", f"ntok_{k}"]

            n_unacc = 0
            n_correct = {"mean": 0, "sum": 0, "pmi_mean": 0, "pmi_sum": 0}
            with cell_csv.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                for j, (qid, gold) in enumerate(meta):
                    s_sum = sum_lp[4 * j: 4 * j + 4]
                    s_mean = mean_lp[4 * j: 4 * j + 4]
                    s_un_sum = un_sum[4 * j: 4 * j + 4]
                    s_un_mean = un_mean[4 * j: 4 * j + 4]
                    s_ntok = ntok[4 * j: 4 * j + 4]
                    if (np.isnan(s_sum).any() or np.isnan(s_un_sum).any()):
                        n_unacc += 1
                    row = {"qid": qid, "gold": gold}
                    for ki, k in enumerate(OPTION_KEYS):
                        row[f"sum_{k}"] = float(s_sum[ki])
                        row[f"mean_{k}"] = float(s_mean[ki])
                        row[f"un_sum_{k}"] = float(s_un_sum[ki])
                        row[f"un_mean_{k}"] = float(s_un_mean[ki])
                        row[f"ntok_{k}"] = int(s_ntok[ki])
                    w.writerow(row)
                    if np.isnan(s_sum).any() or np.isnan(s_un_sum).any():
                        continue
                    gold_idx = OPTION_KEYS.index(gold)
                    for rule, vec in (
                        ("mean", s_mean),
                        ("sum", s_sum),
                        ("pmi_mean", s_mean - s_un_mean),
                        ("pmi_sum", s_sum - s_un_sum),
                    ):
                        if int(np.argmax(vec)) == gold_idx:
                            n_correct[rule] += 1

            n_total = len(meta)
            summary_rows.append({
                "model": repo, "prompt": prompt_key, "cefr": cefr,
                "n_questions": n_total, "n_unacceptable": n_unacc,
                "acc_mean": n_correct["mean"] / max(1, n_total),
                "acc_sum": n_correct["sum"] / max(1, n_total),
                "acc_pmi_mean": n_correct["pmi_mean"] / max(1, n_total),
                "acc_pmi_sum": n_correct["pmi_sum"] / max(1, n_total),
            })

    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        if summary_rows:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    logger.info("[%s] wrote %s", repo, summary_csv)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--dataset", required=True)
    p.add_argument("--cefr-levels", nargs="+",
                   default=["A2", "B1", "B2", "C1", "C2"])
    p.add_argument("--prompts", nargs="+", default=list(PROMPT_TEMPLATES.keys()))
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    p.add_argument("--world-size", type=int,
                   default=int(os.environ.get("WORLD_SIZE", "1")))
    p.add_argument("--max-questions", type=int, default=0)
    p.add_argument("--adapter", default=None,
                   help="Path or HF repo of a PEFT/LoRA adapter to attach on top.")
    p.add_argument("--output-tag", default=None,
                   help="Override the per-model subdir name suffix; useful when "
                        "writing SFT'd adapter results so they don't collide with "
                        "the base-model rerun.")
    args = p.parse_args(argv)

    setup_logging("rcmcq_variants", args.output_root, args.rank)
    logger.info("eval_rcmcq_variants starting (rank %d / %d)",
                args.rank, args.world_size)

    raw = args.models
    models = [m.strip() for m in (raw.split(",") if "," in raw else raw.split())
              if m.strip()]
    models = [m for i, m in enumerate(models) if i % args.world_size == args.rank]
    if not models:
        logger.info("rank %d: no models assigned", args.rank)
        return 0

    qs = load_questions(Path(args.dataset))
    levels_set = {lv.upper() for lv in args.cefr_levels}
    qs = [q for q in qs if q["level"] in levels_set]
    if args.max_questions > 0:
        qs = qs[: args.max_questions]
    if not qs:
        logger.error("no questions left after filtering")
        return 2
    logger.info("loaded %d questions for levels %s",
                len(qs), sorted(levels_set))

    args.output_root.mkdir(parents=True, exist_ok=True)
    for repo in models:
        try:
            evaluate_model(
                repo, qs,
                cefr_levels=list(levels_set),
                prompts=args.prompts,
                output_root=args.output_root,
                batch_size=args.batch_size,
                revision=args.revision,
                adapter=args.adapter,
                output_tag=args.output_tag,
            )
        except Exception:
            logger.exception("[%s] failed", repo)
            continue
    logger.info("eval_rcmcq_variants finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
