#!/usr/bin/env python3
"""
evals/eval_universalcefr.py — UniversalCEFR proficiency-classification eval.

For each text in a UniversalCEFR corpus, the model is asked which CEFR level
(A1-C2) it belongs to. Two scoring modes are supported:

  generation  — apply the prompt, generate up to 10 tokens, parse for the first
                CEFR label found in the output (mirrors universalcefr-experiments/
                prompting/script.py). Best for instruction-tuned variants.

  surprisal   — for each candidate level L, compute mean per-token NLL of
                ``prompt + " " + L`` and pick argmin. Always produces a label;
                works on base LMs (default for un-tuned Beetle models).

Metrics per (model, corpus, language, gold_cefr) stratum (and aggregated):
  accuracy, macro precision/recall/F1, quadratic-weighted kappa (QWK),
  n_total, n_unknown, n_scored.

Prompts vendored from universalcefr-experiments/prompting/prompt_files into
``evals/prompts/universalcefr/lang_{read,write}/prompt_with_specs[_<lang>].txt``.
``<<TEXT>>`` is the placeholder substituted with each sample text.

Usage
-----
  python evals/eval_universalcefr.py \\
      --model Beetle-HumanScale/beetle-bilingual-...-deu-eng \\
      --corpora UniversalCEFR/elg-cefr-de \\
      --mode surprisal --task read --limit 50
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evals.model_loader import (
    DEVICE, load_model, unload_model, setup_logging, resolve_vocab_size,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CEFR_LABELS: List[str] = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_TO_ORDINAL: Dict[str, int] = {l: i for i, l in enumerate(CEFR_LABELS)}
UNKNOWN = "UNKNOWN"

PROMPT_DIR = _REPO_ROOT / "evals" / "prompts" / "universalcefr"

# Default corpora per ISO-639-1 language code. Extend by adding entries; the
# language field of each corpus is taken from the dataset itself, not this map.
DEFAULT_CORPORA: Dict[str, List[str]] = {
    "en": [
        "UniversalCEFR/cambridge-exams",
        "UniversalCEFR/elg-cefr-en",
        "UniversalCEFR/cefr-sp",
        "UniversalCEFR/icle500",
        "UniversalCEFR/cefr-asag",
    ],
    "de": [
        "UniversalCEFR/elg-cefr-de",
        "UniversalCEFR/merlin-de",
        "UniversalCEFR/deplain-web-doc",
        "UniversalCEFR/deplain-apa-doc",
    ],
    "nl": ["UniversalCEFR/elg-cefr-nl"],
    "fr": ["UniversalCEFR/kwiziq-fr"],
    "es": [
        "UniversalCEFR/hablacultura",
        "UniversalCEFR/kwiziq-es",
        "UniversalCEFR/caes",
    ],
    "it": ["UniversalCEFR/merlin-it"],
    "cs": ["UniversalCEFR/merlin-cs"],
    "et": ["UniversalCEFR/elle"],
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CEFRSample:
    text: str
    lang: str
    cefr: str
    fmt: str    # "sentence-level" | "paragraph-level" | "document-level"
    corpus: str  # e.g. "UniversalCEFR/elg-cefr-de"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _normalize_cefr(label: str) -> str:
    """Map a raw cefr_level to one of CEFR_LABELS, or UNKNOWN.

    UniversalCEFR sometimes uses "+", lowercase, or wrapped-in-whitespace
    forms (e.g. "a2+", "B1+"). We accept the leading 2-char canonical level
    and drop modifiers — script.py also skips ambiguous "+" entries; we
    instead retain the base level so coverage isn't gutted.
    """
    if not isinstance(label, str):
        return UNKNOWN
    s = label.strip().upper()
    if not s:
        return UNKNOWN
    # Take the first valid CEFR label that appears as a prefix
    for lvl in CEFR_LABELS:
        if s.startswith(lvl):
            return lvl
    return UNKNOWN


def load_corpus(corpus: str, split: str = "train", limit: Optional[int] = None) -> List[CEFRSample]:
    """Load a single UniversalCEFR HF dataset and yield CEFRSample records.

    Skips records with missing text or unrecognised CEFR level.
    """
    logger.info(f"Loading {corpus} (split={split}) …")
    try:
        ds = load_dataset(corpus, split=split)
    except Exception as exc:
        logger.error(f"  Could not load {corpus}: {exc}")
        return []

    samples: List[CEFRSample] = []
    skipped = 0
    for row in ds:
        text = row.get("text")
        cefr = _normalize_cefr(row.get("cefr_level") or row.get("cefr") or "")
        if not isinstance(text, str) or not text.strip() or cefr == UNKNOWN:
            skipped += 1
            continue
        samples.append(CEFRSample(
            text=text.strip(),
            lang=(row.get("lang") or "").strip().lower(),
            cefr=cefr,
            fmt=row.get("format") or "unknown",
            corpus=corpus,
        ))
        if limit is not None and len(samples) >= limit:
            break
    logger.info(f"  Loaded {len(samples):,} samples ({skipped:,} skipped)")
    return samples


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def load_prompt_template(task: str, lang: str, prompt_dir: Path = PROMPT_DIR) -> str:
    """Load a prompt template; fall back to the English file if lang-specific is absent.

    task ∈ {"read", "write"} → ``lang_read`` / ``lang_write`` directory.
    """
    sub = "lang_read" if task == "read" else "lang_write"
    base = prompt_dir / sub
    candidate = base / f"prompt_with_specs_{lang}.txt"
    if not candidate.exists():
        candidate = base / "prompt_with_specs_en.txt"
    if not candidate.exists():
        candidate = base / "prompt_with_specs.txt"
    if not candidate.exists():
        raise FileNotFoundError(f"No prompt template found in {base}")
    return candidate.read_text(encoding="utf-8")


def build_prompt(template: str, text: str) -> str:
    return template.replace("<<TEXT>>", text)


# ---------------------------------------------------------------------------
# Scoring — generation mode
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_generation(
    model, tokenizer, prompts: List[str], device: str,
    max_new_tokens: int = 10,
) -> List[str]:
    """Greedy-generate up to N tokens; return parsed CEFR label per prompt.

    Mirrors universalcefr-experiments/script.py — first label substring wins,
    otherwise UNKNOWN.
    """
    out: List[str] = []
    for p in prompts:
        enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = gen[0][enc["input_ids"].shape[-1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        out.append(_parse_cefr_from_output(decoded))
    return out


def _parse_cefr_from_output(decoded: str) -> str:
    s = decoded.upper()
    for lvl in CEFR_LABELS:
        if lvl in s:
            return lvl
    return UNKNOWN


# ---------------------------------------------------------------------------
# Scoring — surprisal mode
# ---------------------------------------------------------------------------

@torch.no_grad()
def _continuation_nll(
    model, tokenizer, prompt: str, continuation: str, device: str,
) -> float:
    """Mean per-token NLL (nats) of ``continuation`` given ``prompt``.

    Encodes prompt and prompt+continuation separately so we can mask the
    prompt tokens out of the loss. Lower = more probable.
    """
    vocab_size = resolve_vocab_size(model)
    full = prompt + " " + continuation

    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"]
    full_ids   = tokenizer(full,   add_special_tokens=True, return_tensors="pt")["input_ids"]

    n_prompt = prompt_ids.shape[1]
    n_total  = full_ids.shape[1]
    if n_total <= n_prompt:
        # Continuation produced no new tokens (e.g. tokenizer merged "C2" into
        # an existing prompt token). Treat as worst-case so it doesn't win.
        return float("inf")

    input_ids = full_ids.to(device).clamp(0, vocab_size - 1)
    labels = input_ids.clone()
    # Mask prompt positions; we only want NLL over the continuation tokens.
    labels[:, :n_prompt] = -100

    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
        outputs = model(input_ids=input_ids)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous().to(dtype=torch.long)
    shift_mask   = shift_labels != -100

    loss = F.cross_entropy(
        shift_logits.transpose(1, 2), shift_labels,
        ignore_index=-100, reduction="none",
    )
    n_tok = shift_mask.sum().clamp_min(1)
    mean_nll = (loss * shift_mask).sum() / n_tok
    return float(mean_nll.detach().cpu())


def predict_surprisal(
    model, tokenizer, prompts: List[str], device: str,
) -> List[str]:
    """Score each candidate CEFR level under each prompt; return argmin label."""
    out: List[str] = []
    for p in prompts:
        nlls = {lvl: _continuation_nll(model, tokenizer, p, lvl, device)
                for lvl in CEFR_LABELS}
        best = min(nlls.items(), key=lambda kv: kv[1])
        out.append(best[0])
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(gold: List[str], pred: List[str]) -> Dict[str, float]:
    """accuracy, macro P/R/F1, QWK over CEFR_LABELS. UNKNOWN excluded.

    Returns coverage counts so the caller can attach (n_total, n_unknown,
    n_scored) to the output row.
    """
    n_total = len(gold)
    valid = [(g, p) for g, p in zip(gold, pred) if p in CEFR_TO_ORDINAL and g in CEFR_TO_ORDINAL]
    n_scored = len(valid)
    n_unknown = n_total - n_scored

    if n_scored == 0:
        return {
            "n_total":    n_total,
            "n_scored":   0,
            "n_unknown":  n_unknown,
            "accuracy":   float("nan"),
            "macro_p":    float("nan"),
            "macro_r":    float("nan"),
            "macro_f1":   float("nan"),
            "qwk":        float("nan"),
        }

    g_arr = [g for g, _ in valid]
    p_arr = [p for _, p in valid]

    acc = accuracy_score(g_arr, p_arr)
    p_m = precision_score(g_arr, p_arr, average="macro", labels=CEFR_LABELS, zero_division=0)
    r_m = recall_score   (g_arr, p_arr, average="macro", labels=CEFR_LABELS, zero_division=0)
    f1_m = f1_score      (g_arr, p_arr, average="macro", labels=CEFR_LABELS, zero_division=0)

    g_ord = [CEFR_TO_ORDINAL[g] for g in g_arr]
    p_ord = [CEFR_TO_ORDINAL[p] for p in p_arr]
    try:
        qwk = cohen_kappa_score(g_ord, p_ord, labels=list(range(len(CEFR_LABELS))),
                                weights="quadratic")
    except Exception:
        qwk = float("nan")

    return {
        "n_total":    n_total,
        "n_scored":   n_scored,
        "n_unknown":  n_unknown,
        "accuracy":   round(acc, 6),
        "macro_p":    round(p_m, 6),
        "macro_r":    round(r_m, 6),
        "macro_f1":   round(f1_m, 6),
        "qwk":        round(float(qwk), 6),
    }


def confusion_dict(gold: List[str], pred: List[str]) -> Dict:
    """Confusion matrix as {gold_label: {pred_label: count}}."""
    labels = CEFR_LABELS + [UNKNOWN]
    cm = confusion_matrix(gold, pred, labels=labels)
    return {labels[i]: {labels[j]: int(cm[i][j]) for j in range(len(labels))}
            for i in range(len(labels))}


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def _append_row(output_file: Path, row: dict) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        output_file, mode="a", header=not output_file.exists(), index=False,
    )


# ---------------------------------------------------------------------------
# Per-model entry point (used by orchestrator and standalone CLI)
# ---------------------------------------------------------------------------

def run_universalcefr(
    model, tokenizer, repo: str, revision: str,
    corpora: List[str],
    output_dir: Path,
    mode: str = "surprisal",
    task: str = "read",
    split: str = "train",
    limit: Optional[int] = None,
    prompt_dir: Path = PROMPT_DIR,
    use_lang_prompts: bool = True,
    device: str = DEVICE,
    confusion_dir: Optional[Path] = None,
) -> None:
    """Run UniversalCEFR eval for one already-loaded model across N corpora.

    Output CSV layout (appended):
      model, branch, mode, task, corpus, lang, gold_cefr,
      n_total, n_scored, n_unknown, accuracy, macro_p, macro_r, macro_f1, qwk

    For each corpus, emits one row per gold_cefr stratum plus an "ALL" row.
    A global "(ALL,ALL,ALL)" row is emitted at the end if multiple corpora were run.
    """
    output_file = output_dir / "universalcefr_results.csv"
    confusion_dir = confusion_dir or (output_dir / "confusion")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ledger (best-effort; allows skipping completed (corpus, lang, mode) cells)
    try:
        from analyze.eval_ledger import is_done as _is_done
        from analyze.eval_ledger import record as _record
    except Exception:
        def _is_done(*_a, **_kw): return False
        def _record(*_a, **_kw): pass

    all_gold: List[str] = []
    all_pred: List[str] = []

    for corpus in corpora:
        # Per-corpus ledger key includes mode so generation/surprisal don't
        # collide if the same corpus is run twice with different modes.
        bench = f"universalcefr:{mode}:{task}"
        if _is_done("universalcefr", repo, revision, bench, corpus,
                    output_root=str(output_dir.parent.parent)):
            logger.info(f"  UniversalCEFR SKIP {corpus} (ledger)")
            continue

        samples = load_corpus(corpus, split=split, limit=limit)
        if not samples:
            _record("universalcefr", repo, revision, bench, corpus, "error",
                    output_root=str(output_dir.parent.parent))
            continue

        # Cache prompt templates per lang (most corpora are mono-lingual)
        templates: Dict[str, str] = {}
        prompts: List[str] = []
        for s in samples:
            lang_key = s.lang if (use_lang_prompts and s.lang) else "en"
            if lang_key not in templates:
                templates[lang_key] = load_prompt_template(task, lang_key, prompt_dir)
            prompts.append(build_prompt(templates[lang_key], s.text))

        t0 = time.time()
        logger.info(f"  UniversalCEFR {corpus}: {len(samples)} samples, mode={mode}, task={task}")
        try:
            if mode == "generation":
                preds = predict_generation(model, tokenizer, prompts, device)
            elif mode == "surprisal":
                preds = predict_surprisal(model, tokenizer, prompts, device)
            else:
                raise ValueError(f"Unknown mode '{mode}' (expected 'generation' or 'surprisal')")
        except Exception:
            logger.error(f"  UniversalCEFR ERROR {corpus}", exc_info=True)
            _record("universalcefr", repo, revision, bench, corpus, "error",
                    output_root=str(output_dir.parent.parent))
            continue
        elapsed = time.time() - t0

        gold = [s.cefr for s in samples]
        # The "language" we record is the first non-empty value seen in the corpus
        corpus_lang = next((s.lang for s in samples if s.lang), "")

        # Per-gold-level stratum rows
        by_gold: Dict[str, Tuple[List[str], List[str]]] = defaultdict(lambda: ([], []))
        for g, p in zip(gold, preds):
            gs, ps = by_gold[g]
            gs.append(g); ps.append(p)
        for g_lvl, (gs, ps) in sorted(by_gold.items()):
            m = compute_metrics(gs, ps)
            _append_row(output_file, {
                "model":     repo,
                "branch":    revision,
                "mode":      mode,
                "task":      task,
                "corpus":    corpus,
                "lang":      corpus_lang,
                "gold_cefr": g_lvl,
                **m,
            })

        # Per-corpus ALL row + confusion sidecar
        agg = compute_metrics(gold, preds)
        logger.info(
            f"    {corpus}: acc={agg['accuracy']:.3f} "
            f"f1={agg['macro_f1']:.3f} qwk={agg['qwk']:.3f} "
            f"({agg['n_scored']}/{agg['n_total']} scored, {elapsed:.1f}s)"
        )
        _append_row(output_file, {
            "model":     repo,
            "branch":    revision,
            "mode":      mode,
            "task":      task,
            "corpus":    corpus,
            "lang":      corpus_lang,
            "gold_cefr": "ALL",
            **agg,
        })

        # Save confusion matrix as JSON sidecar
        cm = confusion_dict(gold, preds)
        cm_dir = confusion_dir
        cm_dir.mkdir(parents=True, exist_ok=True)
        slug = repo.replace("/", "__")
        corpus_slug = corpus.replace("/", "__")
        cm_path = cm_dir / f"{slug}__{corpus_slug}__{mode}__{task}.json"
        cm_path.write_text(json.dumps(cm, indent=2), encoding="utf-8")

        all_gold.extend(gold)
        all_pred.extend(preds)
        _record("universalcefr", repo, revision, bench, corpus, "ok",
                output_root=str(output_dir.parent.parent))

    # Cross-corpus aggregate row
    if all_gold and len(corpora) > 1:
        agg = compute_metrics(all_gold, all_pred)
        _append_row(output_file, {
            "model":     repo,
            "branch":    revision,
            "mode":      mode,
            "task":      task,
            "corpus":    "ALL",
            "lang":      "ALL",
            "gold_cefr": "ALL",
            **agg,
        })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_corpora(args_corpora: Optional[List[str]],
                     args_languages: Optional[List[str]]) -> List[str]:
    """Resolve user-supplied corpora / languages into a flat HF-dataset list."""
    if args_corpora:
        return list(args_corpora)
    if args_languages:
        out: List[str] = []
        for lang in args_languages:
            out.extend(DEFAULT_CORPORA.get(lang, []))
        return out
    # Default: all language-matched corpora across en/de/nl
    return [c for lang in ("en", "de", "nl") for c in DEFAULT_CORPORA[lang]]


def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("universalcefr", output_dir.parent.parent, rank=0)

    corpora = _resolve_corpora(args.corpora, args.languages)
    logger.info(f"Resolved {len(corpora)} corpora: {corpora}")

    result = load_model(args.model, DEVICE, revision=args.revision)
    if result is None:
        logger.error(f"Could not load {args.model!r}@{args.revision!r} — aborting.")
        sys.exit(1)
    model, tokenizer = result

    # If the user supplied a LoRA adapter, attach it (no-op if peft missing)
    if args.adapter:
        try:
            from peft import PeftModel
            logger.info(f"Loading LoRA adapter from {args.adapter}")
            model = PeftModel.from_pretrained(model, args.adapter)
            model.eval()
        except Exception as exc:
            logger.error(f"Failed to attach adapter {args.adapter}: {exc}")
            sys.exit(2)

    run_universalcefr(
        model=model, tokenizer=tokenizer,
        repo=args.model, revision=args.revision,
        corpora=corpora,
        output_dir=output_dir,
        mode=args.mode, task=args.task,
        split=args.split, limit=args.limit,
        use_lang_prompts=not args.english_prompts,
    )

    unload_model(model, tokenizer)

    if (output_dir / "universalcefr_results.csv").exists():
        df = pd.read_csv(output_dir / "universalcefr_results.csv")
        # Show only this model's rows
        df = df[(df["model"] == args.model) & (df["branch"] == args.revision)]
        cols = ["mode", "task", "corpus", "lang", "gold_cefr",
                "n_scored", "accuracy", "macro_f1", "qwk"]
        present = [c for c in cols if c in df.columns]
        print("\n=== UniversalCEFR Results ===")
        print(df[present].to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UniversalCEFR proficiency-level classification eval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", required=True, metavar="ORG/REPO")
    p.add_argument("--revision", default="main")
    p.add_argument("--adapter", default=None,
                   help="Optional LoRA adapter dir to attach over the base model")
    p.add_argument("--corpora", nargs="+", default=None, metavar="HF/DATASET",
                   help="Explicit list of UniversalCEFR/* HF datasets")
    p.add_argument("--languages", nargs="+", default=None, metavar="ISO1",
                   help="Resolve via DEFAULT_CORPORA, e.g. --languages en de nl")
    p.add_argument("--mode", choices=["generation", "surprisal"], default="surprisal")
    p.add_argument("--task", choices=["read", "write"], default="read")
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=None,
                   help="Max samples per corpus (for smoke tests)")
    p.add_argument("--english-prompts", action="store_true",
                   help="Force the English prompt template for every sample "
                        "(default: per-sample language-specific prompts).")
    p.add_argument("--output", default="results/evals/universalcefr",
                   help="Output dir for universalcefr_results.csv + confusion/")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
