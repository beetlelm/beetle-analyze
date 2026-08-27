#!/usr/bin/env python3
"""
evals/eval_flores_ppl.py — FLORES-200 cross-lingual perplexity evaluation.

Scores all 1012 FLORES-200 devtest sentences per language under each model
and writes per-model-language CSVs plus an aggregated summary.

Language selection:
  Beetle models   — L1 + L2 auto-detected from repo name via model_naming.parse_repo()
  Baseline models — configurable list (default: eng nld deu zho)

Two model sources are supported:
  1. Beetle orgs — Beetle-HumanScale and Beetle-FineWeb
  2. Baselines   — EuroLLM, XGLM, BLOOM at multiple sizes

Completed (repo, lang) pairs are recorded in the eval ledger and skipped
on re-runs without re-scoring.

Usage
-----
  # Beetle-HumanScale models (L1+L2 auto-detected) + baselines on default langs
  python evals/eval_flores_ppl.py --models beetle-humanscale baselines

  # All Beetle orgs + baselines, restrict baselines to eng and nld
  python evals/eval_flores_ppl.py \\
      --models beetle-humanscale beetle-fineweb baselines \\
      --baseline-langs eng nld

  # 8-GPU fan-out, rank 3
  CUDA_VISIBLE_DEVICES=3 python evals/eval_flores_ppl.py \\
      --models beetle-humanscale baselines \\
      --rank 3 --world-size 8
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "analyze"))

from evals.baseline_models import resolve_models, BEETLE_ORGS, model_family_label
from evals.model_loader import (
    DEVICE, load_model, unload_model, setup_logging, log_gpu_mem,
    resolve_vocab_size,
)

# Import shared utilities from the main pipeline
from analyze.ppl_utils import (
    load_flores_sentences,
    nll_to_ppl,
    FLORES_LANG_MAP,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASELINE_LANGS = ["eng", "nld", "deu", "zho"]
EXPECTED_ROWS = 1012  # FLORES-200 devtest has 1012 sentences per language

# ---------------------------------------------------------------------------
# Ledger (best-effort — never crashes the eval loop)
# ---------------------------------------------------------------------------

try:
    from analyze.eval_ledger import is_done as _ledger_is_done
    from analyze.eval_ledger import record as _ledger_record
except Exception:
    def _ledger_is_done(*_a, **_kw) -> bool: return False   # type: ignore[misc]
    def _ledger_record(*_a, **_kw) -> None: pass             # type: ignore[misc]

# ---------------------------------------------------------------------------
# Language detection for Beetle models
# ---------------------------------------------------------------------------

def _detect_beetle_langs(repo: str) -> Optional[list[str]]:
    """Return [l1, l2] for a Beetle model repo, or None if unparseable."""
    try:
        from analyze.model_naming import parse_repo
        meta = parse_repo(repo)
        langs = [lg for lg in (meta.l1, meta.l2) if lg is not None]
        langs = [lg for lg in langs if lg in FLORES_LANG_MAP]
        return langs if langs else None
    except Exception:
        return None


def _is_beetle_repo(repo: str) -> bool:
    beetle_prefixes = tuple(v + "/" for v in BEETLE_ORGS.values())
    return repo.startswith(beetle_prefixes)


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

def _model_slug(repo: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", repo)


def _per_model_lang_path(
    output_dir: Path, repo: str, iso_code: str, revision: str = "main",
) -> Path:
    suffix = "" if revision == "main" else f"__{_model_slug(revision)}"
    return output_dir / f"ppl_{iso_code}_{_model_slug(repo)}{suffix}.csv"


# ---------------------------------------------------------------------------
# Batched scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_sentences_batched(
    sentences: list[str],
    model,
    tokenizer,
    goldfish_tokens: Optional[int],
    batch_size: int,
) -> list[float]:
    """Return mean per-token NLL for each sentence (lower = more likely).

    Uses padded batches with autocast for A100 throughput. Padding tokens
    are excluded from the NLL computation via the attention mask.
    """
    # Gemma 3 / PaliGemma and other multimodal configs hoist
    # ``max_position_embeddings`` onto ``config.text_config`` instead of the
    # top-level config, so check both before falling back.
    _cfg = model.config
    _text_cfg = getattr(_cfg, "text_config", None)
    max_len = min(
        goldfish_tokens or 1024,
        getattr(_cfg, "max_position_embeddings", None)
            or (getattr(_text_cfg, "max_position_embeddings", 1024) if _text_cfg else 1024),
    )
    device = next(model.parameters()).device
    vocab_size = resolve_vocab_size(model)
    nlls: list[float] = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device).clamp(0, vocab_size - 1)
        attention_mask = enc["attention_mask"].to(device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(input_ids=input_ids)
        log_probs = F.log_softmax(outputs.logits.float(), dim=-1)
        del outputs

        lp_shifted    = log_probs[:, :-1, :]
        ids_shifted   = input_ids[:, 1:]
        mask_shifted  = attention_mask[:, 1:].float()

        token_lp      = lp_shifted.gather(-1, ids_shifted.unsqueeze(-1)).squeeze(-1)
        per_token_nll = -(token_lp * mask_shifted)
        n_real        = mask_shifted.sum(dim=-1).clamp(min=1)
        batch_nlls    = (per_token_nll.sum(-1) / n_real).cpu().tolist()
        nlls.extend(batch_nlls)

    return nlls


# ---------------------------------------------------------------------------
# Per-model-language scoring
# ---------------------------------------------------------------------------

def score_model_language(
    repo: str,
    iso_code: str,
    sentences: list[str],
    goldfish_tokens: Optional[int],
    batch_size: int,
    revision: str = "main",
) -> Optional[pd.DataFrame]:
    """Score *sentences* under *repo*@*revision* and return a DataFrame, or None on failure."""
    result = load_model(repo, DEVICE, revision=revision)
    if result is None:
        return None
    model, tokenizer = result
    log_gpu_mem(logger, prefix=f"{repo.split('/')[-1]}@{revision}|{iso_code}")

    try:
        nlls = score_sentences_batched(sentences, model, tokenizer,
                                       goldfish_tokens, batch_size)
    finally:
        # Guarantee unload even if scoring raises mid-batch — otherwise
        # a long sweep accumulates GPU + host RSS until the node OOMs.
        unload_model(model, tokenizer)
        del model, tokenizer

    rows = [
        {
            "sentence_id": i,
            "sentence": sent,
            "repo": repo,
            "branch": revision,
            "revision": revision,
            "iso_code": iso_code,
            "nll": nll,
            "ppl": nll_to_ppl(nll),
        }
        for i, (sent, nll) in enumerate(zip(sentences, nlls))
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def _rebuild_summary(output_dir: Path, all_model_pairs: list) -> None:
    """Aggregate all per-(model, lang, revision) CSVs into ppl_summary.csv."""
    rows: list[dict] = []
    for repo, family in all_model_pairs:
        slug = _model_slug(repo)
        paths = list(output_dir.glob(f"ppl_*_{slug}.csv")) + \
                list(output_dir.glob(f"ppl_*_{slug}__*.csv"))
        for path in paths:
            try:
                df = pd.read_csv(path)
                if df.empty:
                    continue
                branch = df["branch"].iloc[0] if "branch" in df.columns else "main"
                rows.append({
                    "repo":          repo,
                    "model_family":  family,
                    "branch":        branch,
                    "revision":      branch,
                    "iso_code":      df["iso_code"].iloc[0],
                    "mean_nll":      df["nll"].mean(),
                    "mean_ppl":      df["ppl"].mean(),
                    "median_ppl":    df["ppl"].median(),
                    "std_ppl":       df["ppl"].std(),
                    "n_sentences":   len(df),
                })
            except Exception:
                pass

    if not rows:
        return
    summary = pd.DataFrame(rows)
    out_path = output_dir / "ppl_summary.csv"
    summary.to_csv(out_path, index=False)
    logger.info(f"Summary → {out_path} ({len(summary)} rows)")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging("ppl", output_dir, args.rank)

    baseline_langs: list[str] = args.baseline_langs or DEFAULT_BASELINE_LANGS
    unknown = [lg for lg in baseline_langs if lg not in FLORES_LANG_MAP]
    if unknown:
        logger.error(f"Unknown ISO codes (not in FLORES_LANG_MAP): {unknown}")
        sys.exit(1)

    # Resolve models and shard
    all_model_pairs = resolve_models(args.models, output_dir=output_dir.parent)
    my_models = [p for i, p in enumerate(all_model_pairs) if i % args.world_size == args.rank]
    logger.info(f"[rank {args.rank}/{args.world_size}] {len(my_models)} models to evaluate")

    # Pre-load FLORES sentences for baseline langs (shared across baseline models)
    flores_cache: dict[str, list[str]] = {}
    for iso in baseline_langs:
        try:
            flores_cache[iso] = load_flores_sentences(iso)
            logger.info(f"  Loaded FLORES/{iso}: {len(flores_cache[iso])} sentences")
        except ValueError as exc:
            logger.warning(f"Could not load FLORES for {iso}: {exc}")

    for repo, family in my_models:
        # Determine which languages to evaluate for this model
        if _is_beetle_repo(repo):
            langs = _detect_beetle_langs(repo)
            if not langs:
                logger.warning(f"Could not parse L1/L2 for {repo}, skipping")
                continue
        else:
            langs = [lg for lg in baseline_langs if lg in flores_cache]

        model_loaded = False
        model_obj = None
        tokenizer_obj = None

        for iso in langs:
            if _ledger_is_done("ppl", repo, args.revision, "flores", iso,
                               output_root=str(output_dir)):
                logger.info(f"SKIP {repo}@{args.revision} | {iso} — already in ledger")
                continue

            # Load sentences (from cache for baselines, fresh for Beetle langs not pre-loaded)
            if iso in flores_cache:
                sentences = flores_cache[iso]
            else:
                try:
                    sentences = load_flores_sentences(iso)
                    flores_cache[iso] = sentences
                except ValueError as exc:
                    logger.warning(f"SKIP {repo}|{iso}: {exc}")
                    continue

            logger.info(f"{repo}@{args.revision} | {iso}: scoring {len(sentences)} sentences "
                        f"(batch={args.batch_size}) …")

            df = score_model_language(repo, iso, sentences,
                                      args.goldfish, args.batch_size,
                                      revision=args.revision)
            if df is None:
                logger.warning(f"  Model load failed for {repo}@{args.revision}, skipping all langs")
                _ledger_record("ppl", repo, args.revision, "flores", iso, "error",
                               rank=args.rank, output_root=str(output_dir))
                break

            out_path = _per_model_lang_path(output_dir, repo, iso, args.revision)
            df.to_csv(out_path, index=False)
            logger.info(f"  Wrote {len(df)} rows → {out_path}  "
                        f"(mean_ppl={df['ppl'].mean():.1f})")
            _ledger_record("ppl", repo, args.revision, "flores", iso, "ok",
                           rank=args.rank, output_root=str(output_dir))

    # Rank 0: rebuild summary
    if args.rank == 0:
        _rebuild_summary(output_dir, all_model_pairs)

    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FLORES-200 cross-lingual perplexity evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--models", nargs="+", required=True,
        metavar="SPEC",
        help=(
            "Model specs: beetle-humanscale | beetle-fineweb | baselines | "
            "eurollm | xglm | bloom | <org/repo>"
        ),
    )
    p.add_argument(
        "--baseline-langs", nargs="+", default=None,
        metavar="ISO",
        help=(
            f"FLORES ISO-639-3 codes for baseline models "
            f"(default: {' '.join(DEFAULT_BASELINE_LANGS)}). "
            "Beetle models always use their L1+L2 automatically."
        ),
    )
    p.add_argument(
        "--output", default="results/evals/ppl",
        help="Output directory (default: results/evals/ppl)",
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help="Scoring batch size (default: 32, safe for A100 80 GB up to ~9B)",
    )
    p.add_argument(
        "--goldfish", type=int, default=512,
        help="Score only the first N tokens per sentence (default: 512). "
             "Set 0 to disable.",
    )
    p.add_argument(
        "--revision", default="main",
        help="HuggingFace branch / revision to evaluate (default: main). "
             "Recorded as the 'branch' column and namespaces the output CSVs.",
    )
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.goldfish == 0:
        args.goldfish = None
    evaluate(args)
