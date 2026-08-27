#!/usr/bin/env python3
"""
evals/eval_rcmcq.py — Cambridge Reading Comprehension MCQ student-simulation
                      eval (Benedetto+24 protocol) on Beetle pretrained ckpts.

Adapts the Benedetto+24 ``llm-student-simulations`` protocol
(https://aclanthology.org/2024.findings-emnlp.663) to the local Cambridge MCQ
Reading Dataset shipped at::

    education/Cambridge Multiple-Choice Questions Reading Dataset.jsonl

Per (model × prompt × CEFR-persona × question) we compute the multiple-choice
log-likelihood of each option and take the argmax. Outputs:

  results/<output-root>/<model>/<prompt>/<cefr>.csv         — item-level
  results/<output-root>/<model>/summary.csv                 — accuracy and
      monotonicity penalty M = ρ_{L,I} − P, per (prompt, model).

The 6 LLM baseline rows (LLaMA-3-8B, Gemma-2B, Gemma-7B, Mistral-7B, GPT-3.5,
GPT-4o-mini) and their unacceptable-response counts in the paper come
DIRECTLY from Benedetto+24 Table 13 — they are NOT recomputed here. The plot
script ``plots/make_rcmcq_cefr.py`` hard-codes those numbers.

Usage:
  python evals/eval_rcmcq.py \\
      --models Beetle-HumanScale,Beetle-FineWeb-100M,Beetle-FineWeb-2B,Beetle-FineWeb2-24B \\
      --dataset 'education/Cambridge Multiple-Choice Questions Reading Dataset.jsonl' \\
      --cefr-levels A2 B1 B2 C1 C2 \\
      --prompts default cefr_persona stratified gradient \\
      --output-root results/evals_paper/03_rcmcq

Failures are recorded in the eval ledger under stage ``rcmcq``.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evals.model_loader import (  # noqa: E402
    DEVICE, load_model, unload_model, setup_logging,
    resolve_vocab_size, log_gpu_mem,
)

logger = logging.getLogger(__name__)

# Best-effort eval ledger
try:
    from analyze.eval_ledger import is_done as _ledger_is_done
    from analyze.eval_ledger import record as _ledger_record
except Exception:  # pragma: no cover
    def _ledger_is_done(*_a, **_kw) -> bool: return False
    def _ledger_record(*_a, **_kw) -> None: pass


# ---------------------------------------------------------------------------
# CEFR personas and prompt templates
# ---------------------------------------------------------------------------
CEFR_LEVELS_DEFAULT = ["A2", "B1", "B2", "C1", "C2"]

# Ideal accuracy curve: increases monotonically with simulated proficiency.
# An A2 simulated learner should answer C2 questions worse than a C2 learner.
_IDEAL_BY_N = {
    5: np.array([0.20, 0.40, 0.55, 0.70, 0.85]),
    6: np.array([0.10, 0.25, 0.45, 0.60, 0.80, 0.95]),
}

PROMPT_TEMPLATES: dict[str, str] = {
    # Variant 1 — minimal, baseline persona reference
    "default": (
        "You are a {cefr} English language learner. "
        "Read the passage and answer the question.\n\n"
        "Passage:\n{passage}\n\nQuestion: {question}\nAnswer:"
    ),
    # Variant 2 — explicit CEFR descriptor (Council of Europe wording paraphrase)
    "cefr_persona": (
        "You are a student of English at CEFR level {cefr}. "
        "Your reading comprehension and vocabulary are typical of a {cefr} learner. "
        "Read the following passage and choose the option that best answers the question.\n\n"
        "Passage:\n{passage}\n\nQuestion: {question}\nAnswer:"
    ),
    # Variant 3 — stratified, lists the level alongside the global CEFR scale
    "stratified": (
        "Imagine you are a learner whose English proficiency is at CEFR {cefr} "
        "(on the A1, A2, B1, B2, C1, C2 scale). "
        "Read and answer:\n\nPassage:\n{passage}\n\nQuestion: {question}\nAnswer:"
    ),
    # Variant 4 — gradient cue, paraphrase of Benedetto+24 Variant D
    "gradient": (
        "You are a {cefr} learner of English. {cefr_descriptor} "
        "Read the passage carefully and select the best answer.\n\n"
        "Passage:\n{passage}\n\nQuestion: {question}\nAnswer:"
    ),
}

CEFR_DESCRIPTORS = {
    "A1": "You can understand familiar everyday expressions.",
    "A2": "You can read short, simple texts on familiar topics.",
    "B1": "You can read straightforward factual texts on familiar subjects.",
    "B2": "You can read articles and reports on contemporary issues.",
    "C1": "You can read long, complex factual and literary texts.",
    "C2": "You can read with ease virtually all forms of written language.",
}

OPTION_KEYS = ["a", "b", "c", "d"]
OPTION_KEYS_UPPER = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------
def load_questions(jsonl_path: Path) -> list[dict]:
    """Return a flat list of question dicts:
        {id, level, passage, question, options:[4 strs], answer:'a'..'d'}.
    Tolerant to records that have a ``questions`` mapping/list."""
    out: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            level = (rec.get("level") or "").upper().strip()
            passage = rec.get("text") or ""
            qmap = rec.get("questions") or {}
            # ``questions`` may be a dict-of-dicts (id -> q) or a list-of-dicts.
            if isinstance(qmap, dict):
                items = list(qmap.items())
            elif isinstance(qmap, list):
                items = list(enumerate(qmap))
            else:
                continue
            for qkey, q in items:
                if not isinstance(q, dict):
                    continue
                # Two on-disk schemas are tolerated:
                # (A) flat: q["a"], q["b"], q["c"], q["d"] are option strings
                # (B) Cambridge MCQ: q["options"] = {"a": {"text": ...}, ...}
                raw_opts = q.get("options")
                if isinstance(raw_opts, dict):
                    opt_vals = []
                    for k in OPTION_KEYS:
                        ov = raw_opts.get(k)
                        if isinstance(ov, dict):
                            opt_vals.append(ov.get("text"))
                        else:
                            opt_vals.append(ov)
                else:
                    opt_vals = [q.get(k) for k in OPTION_KEYS]
                if any(o is None for o in opt_vals):
                    continue
                ans = (q.get("answer") or "").lower().strip()
                if ans not in OPTION_KEYS:
                    continue
                stem = q.get("text") or q.get("question") or q.get("stem") or ""
                if not stem:
                    continue
                out.append({
                    "id": f"{rec.get('id', '?')}::{q.get('id', q.get('q_id', qkey))}",
                    "level": level,
                    "passage": passage,
                    "question": stem,
                    "options": [str(o) for o in opt_vals],
                    "answer": ans,
                })
    return out


# ---------------------------------------------------------------------------
# Multiple-choice scoring (mirrors eval_race_student.py)
# ---------------------------------------------------------------------------
def _is_oom(exc: BaseException) -> bool:
    if torch.cuda.is_available():
        oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


@torch.no_grad()
def score_options(
    model, tokenizer,
    prompts: list[str], options: list[str],
    device: str, batch_size: int,
    max_length: int = 1536,
) -> np.ndarray:
    """Mean log-prob of option tokens given the prompt, padded as one batch."""
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
            enc = tokenizer(
                [p + " " + o for p, o in zip(prompts[sl], options[sl])],
                padding=True, truncation=True, max_length=max_length,
                return_tensors="pt", add_special_tokens=True,
            )
            prompt_lens = [
                len(tokenizer(p, add_special_tokens=True,
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
            avg_lp = (sum_lp / n_opt).cpu().numpy()
            out[i:i + bs] = avg_lp
            i += bs
        except BaseException as exc:
            if _is_oom(exc) and cur_bs > 1:
                cur_bs = max(1, cur_bs // 2)
                logger.warning("    rcmcq OOM at bs=%d → %d", cur_bs * 2, cur_bs)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
    return out


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------
def render_prompt(template_key: str, cefr: str, passage: str, question: str) -> str:
    tpl = PROMPT_TEMPLATES[template_key]
    return tpl.format(
        cefr=cefr,
        passage=passage,
        question=question,
        cefr_descriptor=CEFR_DESCRIPTORS.get(cefr, ""),
    )


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    repo: str,
    questions: list[dict],
    cefr_levels: list[str],
    prompts: list[str],
    output_root: Path,
    batch_size: int,
    revision: str = "main",
) -> None:
    repo_safe = repo.replace("/", "_")
    if revision and revision != "main":
        out_dir = output_root / f"{repo_safe}__rev={revision}"
    else:
        out_dir = output_root / repo_safe
    out_dir.mkdir(parents=True, exist_ok=True)

    if _ledger_is_done("rcmcq", repo, revision, "cambridge_mcq", "eng"):
        logger.info("[%s @ %s] ledger HIT — skipping", repo, revision)
        return

    model, tok = load_model(repo, DEVICE, revision=revision)
    log_gpu_mem(logger, f"[{repo} @ {revision}] post-load")

    summary_rows: list[dict] = []
    try:
        for prompt_key in prompts:
            for cefr in cefr_levels:
                cell_rows: list[dict] = []
                # Build 4 (prompt, option) pairs per question, score in one pass.
                prompt_texts: list[str] = []
                option_texts: list[str] = []
                meta: list[tuple[str, str, list[str]]] = []
                for q in questions:
                    p = render_prompt(prompt_key, cefr, q["passage"], q["question"])
                    for opt in q["options"]:
                        prompt_texts.append(p)
                        option_texts.append(opt)
                    meta.append((q["id"], q["answer"], q["options"]))
                if not prompt_texts:
                    continue

                t0 = time.time()
                scores = score_options(
                    model, tok, prompt_texts, option_texts,
                    DEVICE, batch_size,
                )
                dt = time.time() - t0
                logger.info(
                    "[%s] prompt=%s cefr=%s scored %d×4 options in %.1fs",
                    repo, prompt_key, cefr, len(meta), dt,
                )

                # Reduce: per question, argmax over its 4 options
                n_correct = 0
                n_unacc = 0
                for j, (qid, gold, _opts) in enumerate(meta):
                    s = scores[4 * j: 4 * j + 4]
                    if np.any(np.isnan(s)):
                        n_unacc += 1
                        cell_rows.append({
                            "qid": qid, "gold": gold, "pred": "?",
                            "correct": 0, "score_a": s[0], "score_b": s[1],
                            "score_c": s[2], "score_d": s[3],
                        })
                        continue
                    pred_idx = int(np.argmax(s))
                    pred = OPTION_KEYS[pred_idx]
                    correct = int(pred == gold)
                    n_correct += correct
                    cell_rows.append({
                        "qid": qid, "gold": gold, "pred": pred,
                        "correct": correct,
                        "score_a": s[0], "score_b": s[1],
                        "score_c": s[2], "score_d": s[3],
                    })

                cell_dir = out_dir / prompt_key
                cell_dir.mkdir(parents=True, exist_ok=True)
                cell_csv = cell_dir / f"{cefr}.csv"
                with cell_csv.open("w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(cell_rows[0].keys()))
                    w.writeheader()
                    w.writerows(cell_rows)

                acc = n_correct / max(1, len(meta))
                summary_rows.append({
                    "model": repo, "prompt": prompt_key, "cefr": cefr,
                    "n_questions": len(meta),
                    "n_correct": n_correct,
                    "n_unacceptable": n_unacc,
                    "accuracy": acc,
                })

        # Per-model summary + monotonicity score M = ρ_{L,I} − P
        for prompt_key in prompts:
            accs = [
                r["accuracy"] for r in summary_rows
                if r["prompt"] == prompt_key
            ]
            if len(accs) < 2:
                continue
            ideal = _IDEAL_BY_N.get(len(accs))
            if ideal is None:
                ideal = np.linspace(0.1, 0.9, len(accs))
            accs_arr = np.array(accs)
            # Pearson with ideal (rank correlation would require scipy)
            if accs_arr.std() == 0 or ideal.std() == 0:
                rho = 0.0
            else:
                rho = float(
                    ((accs_arr - accs_arr.mean()) * (ideal - ideal.mean())).mean()
                    / (accs_arr.std() * ideal.std())
                )
            pen = float(sum(
                max(0.0, accs_arr[i] - accs_arr[i + 1])
                for i in range(len(accs_arr) - 1)
            ))
            summary_rows.append({
                "model": repo, "prompt": prompt_key, "cefr": "_M",
                "n_questions": len(accs),
                "n_correct": -1, "n_unacceptable": -1,
                "accuracy": rho - pen,
            })

        summary_csv = out_dir / "summary.csv"
        with summary_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "model", "prompt", "cefr",
                "n_questions", "n_correct", "n_unacceptable", "accuracy",
            ])
            w.writeheader()
            w.writerows(summary_rows)
        logger.info("[%s] wrote %s", repo, summary_csv)

        _ledger_record("rcmcq", repo, revision, "cambridge_mcq", "eng", "ok")
    finally:
        unload_model(model, tok)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", required=True,
                   help="Comma- or space-separated list of HF repos.")
    p.add_argument("--revision", default="main",
                   help="HF branch/revision applied to all listed models. "
                        "For per-checkpoint sweeps the launcher should iterate "
                        "this argument and shard externally.")
    p.add_argument("--dataset", required=True,
                   help="Path to Cambridge MCQ JSONL.")
    p.add_argument("--cefr-levels", nargs="+", default=CEFR_LEVELS_DEFAULT)
    p.add_argument("--prompts", nargs="+",
                   default=list(PROMPT_TEMPLATES.keys()))
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-gpus", type=int, default=1,
                   help="Sharding hint; if >1, workers should set "
                        "CUDA_VISIBLE_DEVICES per rank externally.")
    p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    p.add_argument("--world-size", type=int,
                   default=int(os.environ.get("WORLD_SIZE", "1")))
    p.add_argument("--max-questions", type=int, default=0,
                   help="If >0, limit to first N questions per CEFR level "
                        "(smoke test).")
    args = p.parse_args(argv)

    setup_logging("rcmcq", args.output_root, args.rank)
    logger.info("eval_rcmcq starting (rank %d / %d)", args.rank, args.world_size)

    # Comma OR space separation
    raw_models = args.models
    if "," in raw_models:
        models = [m.strip() for m in raw_models.split(",") if m.strip()]
    else:
        models = [m for m in raw_models.split() if m]
    # Round-robin shard across world
    models = [m for i, m in enumerate(models) if i % args.world_size == args.rank]
    if not models:
        logger.info("rank %d: no models assigned", args.rank)
        return 0

    qs = load_questions(Path(args.dataset))
    logger.info("loaded %d questions from %s", len(qs), args.dataset)

    # Filter to requested CEFR levels
    levels_set = {lv.upper() for lv in args.cefr_levels}
    qs = [q for q in qs if q["level"] in levels_set]
    logger.info("after CEFR filter: %d questions across levels %s",
                len(qs), sorted(levels_set))
    if not qs:
        logger.error("no questions left after filtering")
        return 2

    if args.max_questions > 0:
        qs = qs[: args.max_questions]
        logger.info("smoke mode: capped to %d questions", len(qs))

    args.output_root.mkdir(parents=True, exist_ok=True)

    for repo in models:
        logger.info("=== model %s ===", repo)
        try:
            evaluate_model(
                repo, qs,
                cefr_levels=list(levels_set & set(CEFR_LEVELS_DEFAULT)) or args.cefr_levels,
                prompts=args.prompts,
                output_root=args.output_root,
                batch_size=args.batch_size,
                revision=args.revision,
            )
        except Exception:
            logger.exception("[%s] failed", repo)
            continue

    logger.info("eval_rcmcq finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
