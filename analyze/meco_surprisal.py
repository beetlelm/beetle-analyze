"""
analyze/meco_surprisal.py — MECO Eye-Tracking Surprisal Evaluation.

Computes word-level surprisal for BeetleLM and baseline models against
MECO L2 reading-time stimuli. Surprisal values correlate with human
eye-tracking reading times (RT) — higher correlation = better cognitive fit.

Ported from beetlelm-rts-private/code/calculate_surprisal.py and extended
with checkpoint iteration and explicit memory management.

Key functions:
    score_checkpoint()       — Score stimuli at a single model checkpoint
    score_all_checkpoints()  — Score stimuli at every checkpoint of a model
    run_meco_eval()          — Evaluate a list of models, write results
"""

from __future__ import annotations

import gc
import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

tqdm.pandas()

try:
    from eval_ledger import record as _ledger_record
except Exception:  # pragma: no cover — ledger is best-effort
    def _ledger_record(*_a, **_kw) -> None:
        pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STIMULI = Path(__file__).resolve().parent.parent / "data" / "meco_l2_stims.tsv"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "meco"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------

def _cleanup_model():
    """Force cleanup of GPU memory between model loads."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Surprisal scoring
# ---------------------------------------------------------------------------

def _get_surprisal(ilm_model, marked_text: str) -> float:
    """
    Compute conditional surprisal for the target word in a marked sentence.

    The target word is delimited by asterisks in marked_text:
        "The cat *sat* on the mat"
    Returns -sum(log P(target | context)).

    Minicons dropped the ``reduction=`` kwarg around version 0.3.x. We try the
    legacy reducer path first; if the kwarg is unknown, fall back to the
    modern API: call ``conditional_score`` without reduction to get per-token
    log-probs and sum them.
    """
    parts = marked_text.split("*")
    if len(parts) < 3:
        return float("nan")

    context = parts[0].strip()
    target = parts[1].strip()

    # Legacy minicons path.
    try:
        return ilm_model.conditional_score(
            context, target,
            bos_token=False,
            reduction=lambda x: -x.sum(0).item(),
        )[0]
    except TypeError:
        # `reduction` not accepted — modern minicons (>= 0.3.x).
        pass
    except Exception:
        return float("nan")

    # Modern minicons path. conditional_score returns per-token logprobs
    # for each (context, target) pair; sum them and negate for surprisal.
    try:
        out = ilm_model.conditional_score(context, target, bos_token=False)
        # out is List[List[float]] or List[float]. Unwrap the single pair.
        first = out[0] if out else None
        if first is None:
            return float("nan")
        if hasattr(first, "__iter__") and not isinstance(first, (str, bytes)):
            # Per-token log-probs for the target
            return -float(sum(first))
        # Already reduced (float)
        return -float(first)
    except Exception:
        return float("nan")


# BeetleLM Pico model repos publish weights and tokenizer files on
# per-step branches (e.g. ``step-30000``), not on ``main``. The ``main``
# branch carries only ``config.json`` + ``pico_decoder.py`` — without weights
# or tokenizer.json — so loading at ``revision='main'`` fails twice:
#   1. AutoTokenizer can't resolve the custom ``PicoDecoderHFConfig``;
#   2. AutoModelForCausalLM finds no ``pytorch_model.bin``/safetensors.
# We therefore auto-resolve ``main`` to the highest-numbered ``step-*``
# branch when scoring a Pico model. As a defensive fallback if no step
# branches exist, we pre-load the matching multilingual 32k-vocab BPE
# (``BeetleLM/bpe_eng-nld-zho``, covers eng, nld, deu, zho).
_BEETLELM_PICO_TOKENIZER_FALLBACK = "BeetleLM/bpe_eng-nld-zho"


def _is_beetlelm_pico(model_path: str) -> bool:
    """Heuristic: does this repo look like a BeetleLM Pico model?"""
    name = model_path.lower()
    return name.startswith("beetlelm/beetlelm_") or "/beetlelm_" in name


def _resolve_pico_revision(model_path: str, requested: str) -> str:
    """Resolve ``main`` to the latest ``step-*`` branch for Pico repos.

    Returns ``requested`` unchanged if it is already a step branch or if
    branch enumeration fails (the caller will then surface the original
    error). Cached per repo to avoid repeated Hub round-trips inside a
    single Python process.
    """
    if requested != "main":
        return requested
    cache = _resolve_pico_revision._cache  # type: ignore[attr-defined]
    if model_path in cache:
        return cache[model_path]
    try:
        from huggingface_hub import list_repo_refs
        refs = list_repo_refs(model_path)
        steps = []
        for b in refs.branches:
            if b.name.startswith("step-"):
                try:
                    steps.append(int(b.name.split("-", 1)[1]))
                except (ValueError, IndexError):
                    continue
        if steps:
            resolved = f"step-{max(steps)}"
            print(f"[meco-surprisal] {model_path}: revision main → {resolved}")
            cache[model_path] = resolved
            return resolved
    except Exception as exc:  # pragma: no cover — defensive
        print(f"[meco-surprisal] could not enumerate refs for {model_path}: {exc}")
    cache[model_path] = requested
    return requested


_resolve_pico_revision._cache = {}  # type: ignore[attr-defined]


def _load_pico_tokenizer_fallback():
    """Load the shared BeetleLM Pico tokenizer fallback (multilingual BPE)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_BEETLELM_PICO_TOKENIZER_FALLBACK)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        elif tok.unk_token is not None:
            tok.pad_token = tok.unk_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
    return tok


def score_checkpoint(
    model_path: str,
    stimuli_path: str | Path = DEFAULT_STIMULI,
    device: str = DEVICE,
    revision: str = "main",
    rank: int = 0,
    world_size: int = 1,
) -> pd.DataFrame:
    """Score MECO stimuli under a single model checkpoint.

    When ``world_size > 1``, this rank scores only
    ``stimuli.iloc[rank::world_size]``. The returned DataFrame carries a
    ``row_id`` column so rank 0 can reassemble the full scored table
    after a barrier.

    Memory is explicitly released after scoring.
    """
    from minicons import scorer

    stimuli = pd.read_csv(stimuli_path, sep="\t").reset_index(drop=False)
    stimuli.rename(columns={"index": "row_id"}, inplace=True)
    if world_size > 1:
        stimuli = stimuli.iloc[rank::world_size].reset_index(drop=True)

    # Load model. For BeetleLM Pico repos, redirect a request for the empty
    # ``main`` branch to the latest ``step-*`` branch (where the actual
    # weights and tokenizer.json live).
    load_kw = {}
    is_pico = _is_beetlelm_pico(model_path)
    if is_pico:
        revision = _resolve_pico_revision(model_path, revision)
    if "beetlelm" in model_path.lower() or "beetle" in model_path.lower():
        load_kw = {"revision": revision, "trust_remote_code": True}

    if is_pico:
        # First try loading the tokenizer straight from the resolved revision
        # (step-* branches publish tokenizer.json). If that fails — e.g. the
        # branch lacks tokenizer files — fall back to the shared multilingual
        # BPE that all BeetleLM Pico models were trained on.
        from transformers import AutoTokenizer
        try:
            tokenizer_obj = AutoTokenizer.from_pretrained(
                model_path, revision=revision, trust_remote_code=True,
            )
        except Exception as exc:
            print(f"[meco-surprisal] {model_path}@{revision}: "
                  f"in-repo tokenizer unavailable ({type(exc).__name__}); "
                  f"using fallback {_BEETLELM_PICO_TOKENIZER_FALLBACK}")
            tokenizer_obj = _load_pico_tokenizer_fallback()
        ilm_model = scorer.IncrementalLMScorer(
            model_path, device,
            tokenizer=tokenizer_obj,
            **load_kw,
        )
    else:
        ilm_model = scorer.IncrementalLMScorer(
            model_path, device,
            tokenizer=model_path,
            **load_kw,
        )

    # Score this rank's marked sentences
    surprisals = stimuli["FullTextMarked"].progress_apply(
        lambda t: _get_surprisal(ilm_model, t)
    )

    # Cleanup
    del ilm_model
    _cleanup_model()

    result = stimuli.copy()
    result["surprisal"] = surprisals
    result["model"] = model_path
    result["revision"] = revision
    return result


def score_all_checkpoints(
    repo: str,
    stimuli_path: str | Path = DEFAULT_STIMULI,
    device: str = DEVICE,
) -> pd.DataFrame:
    """
    Score MECO stimuli at every step-N checkpoint of a HuggingFace model.

    Iterates over all step-N branches, loads one at a time, scores, and
    releases memory before loading the next.

    Returns a DataFrame with columns: [step, word_id, word, surprisal, model].
    """
    from huggingface_hub import list_repo_refs

    # Discover all step-N branches
    try:
        refs = list_repo_refs(repo)
        step_branches = sorted(
            [(int(m.group(1)), r.name)
             for r in refs.branches
             if (m := re.match(r"^step-(\d+)$", r.name))],
            key=lambda t: t[0],
        )
    except Exception as e:
        print(f"[meco] Could not list refs for {repo}: {e}")
        step_branches = []

    if not step_branches:
        print(f"[meco] No step-N branches found for {repo}, using main")
        step_branches = [(0, "main")]

    all_results = []
    for step_num, branch in tqdm(step_branches, desc=f"meco ckpts[{repo.split('/')[-1]}]",
                                 unit="ckpt", leave=False):
        print(f"[meco] {repo} @ {branch} (step {step_num})")
        df = score_checkpoint(repo, stimuli_path, device, revision=branch)
        df["step"] = step_num
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)


def run_meco_eval(
    models: list[str],
    stimuli_path: str | Path = DEFAULT_STIMULI,
    output_dir: str | Path = RESULTS_DIR,
    device: str = DEVICE,
    all_checkpoints: bool = False,
    rank: int = 0,
    world_size: int = 1,
    all_models: list[str] | None = None,
    output_root: "Path | str" = "results",
) -> Path:
    """
    Evaluate a list of models on MECO stimuli.

    Args:
        models:           List of HuggingFace model repo IDs (this rank's shard).
        stimuli_path:     Path to MECO stimuli TSV.
        output_dir:       Directory to write results.
        device:           "cuda" or "cpu".
        all_checkpoints:  If True, evaluate all checkpoints per model.
        rank, world_size: Multi-rank fan-out. Each rank writes its own
                          per-model CSV; rank 0 concatenates the full set.
        all_models:       Full model list used by rank 0 to assemble the
                          combined TSV after a barrier. Defaults to ``models``.

    Returns:
        Path to the combined results TSV (only rank 0 writes it).
    """
    from _rank_barrier import mark_done, reset, wait_for_all

    if rank == 0:
        reset("meco", world_size, output_root)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[meco] Evaluating {len(models)} model(s) on this rank "
          f"(all_checkpoints={all_checkpoints}, rank={rank}/{world_size})")
    shard_suffix = f"_rank{rank}of{world_size}" if world_size > 1 else ""
    for model in tqdm(models, desc="meco models", unit="model"):
        slug = model.split("/")[-1].replace("-", "__")
        print(f"\n[meco] Evaluating: {model}")

        if all_checkpoints:
            # Trajectory mode: do NOT shard stimuli within a checkpoint
            # (memory of checkpoints × stimuli can't be split per rank
            # cleanly here). Fall back to repo-level sharding only.
            df = score_all_checkpoints(model, stimuli_path, device)
            out_path = output_dir / f"checkpoint_trajectory_{slug}.csv"
        else:
            df = score_checkpoint(
                model, stimuli_path, device,
                rank=rank, world_size=world_size,
            )
            out_path = output_dir / f"surprisals_{slug}{shard_suffix}.csv"

        df.to_csv(out_path, index=False)
        print(f"[meco] Saved → {out_path}")
        # Ledger entries: per checkpoint when trajectory, else a single row.
        try:
            if all_checkpoints and "checkpoint" in df.columns:
                for ckpt in sorted(set(df["checkpoint"].astype(str).tolist())):
                    _ledger_record(
                        stage="meco", repo=model, checkpoint=ckpt,
                        benchmark="meco_l2", lang=None, status="ok",
                        rank=rank, output_root=output_root,
                    )
            else:
                _ledger_record(
                    stage="meco", repo=model, checkpoint="main",
                    benchmark="meco_l2", lang=None, status="ok",
                    rank=rank, output_root=output_root,
                )
        except Exception:  # pragma: no cover — ledger is best-effort
            pass

    # Signal completion; only rank 0 builds the combined TSV.
    mark_done("meco", rank, world_size, output_root)
    if rank != 0:
        return output_dir / "surprisals_all.tsv"
    if world_size > 1:
        wait_for_all("meco", world_size, output_root)

    # Rank 0 assembles the combined TSV by reading every per-model CSV on
    # disk. When within-stage sharding was used (world_size > 1 in
    # final-checkpoint mode), first reassemble per-model shard CSVs into
    # the canonical `surprisals_{slug}.csv`.
    summary_models = all_models if all_models is not None else models
    if world_size > 1 and not all_checkpoints:
        for model in summary_models:
            slug = model.split("/")[-1].replace("-", "__")
            canonical = output_dir / f"surprisals_{slug}.csv"
            shards = sorted(output_dir.glob(
                f"surprisals_{slug}_rank*of{world_size}.csv"
            ))
            if shards:
                merged = (pd.concat([pd.read_csv(p) for p in shards],
                                    ignore_index=True)
                          .sort_values("row_id")
                          .reset_index(drop=True))
                merged.to_csv(canonical, index=False)
                print(f"[meco] rank0 reassembled {len(shards)} shards → "
                      f"{canonical} ({len(merged)} rows)")
                for p in shards:
                    p.unlink(missing_ok=True)

    frames: list[pd.DataFrame] = []
    for model in summary_models:
        slug = model.split("/")[-1].replace("-", "__")
        if all_checkpoints:
            p = output_dir / f"checkpoint_trajectory_{slug}.csv"
        else:
            p = output_dir / f"surprisals_{slug}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))

    combined_path = output_dir / "surprisals_all.tsv"
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(combined_path, sep="\t", index=False)
        print(f"\n[meco] Combined results → {combined_path}")
    return combined_path


# ---------------------------------------------------------------------------
# Sweep evaluation: score all models in an HF org, rank by MECO correlation
# ---------------------------------------------------------------------------

def score_sweep_results(
    hf_org: str,
    stimuli_path: str | Path = DEFAULT_STIMULI,
    output_dir: str | Path = RESULTS_DIR,
    device: str = DEVICE,
    rt_column: str = "RT",
) -> Path:
    """
    Score all models in a HuggingFace org against MECO stimuli.

    For each model, scores at the final checkpoint only, then computes
    Spearman correlation between model surprisal and human reading times.
    Writes ranked results and a best_sweep_settings.json.

    Args:
        hf_org:        HuggingFace organization to list models from.
        stimuli_path:  Path to MECO stimuli TSV.
        output_dir:    Directory to write results.
        device:        "cuda" or "cpu".
        rt_column:     Column name for human reading times in stimuli TSV.

    Returns:
        Path to best_sweep_settings.json.
    """
    from huggingface_hub import HfApi, list_repo_refs

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    models = [m.id for m in api.list_models(author=hf_org) if "beetlelm" in m.id]
    print(f"[meco-sweep] Found {len(models)} models in {hf_org}")

    stimuli = pd.read_csv(stimuli_path, sep="\t")
    if rt_column not in stimuli.columns:
        raise ValueError(f"RT column '{rt_column}' not found in stimuli. "
                         f"Available: {list(stimuli.columns)}")

    results = []
    for model_repo in sorted(models):
        print(f"\n[meco-sweep] Scoring: {model_repo}")

        # Find the final (highest step) checkpoint
        try:
            refs = list_repo_refs(model_repo)
            step_branches = [
                (int(m.group(1)), r.name)
                for r in refs.branches
                if (m := re.match(r"^step-(\d+)$", r.name))
            ]
            if step_branches:
                final_step, final_branch = max(step_branches, key=lambda t: t[0])
            else:
                final_step, final_branch = 0, "main"
        except Exception as e:
            print(f"[meco-sweep]   Could not list refs: {e}")
            final_step, final_branch = 0, "main"

        # Score at final checkpoint
        try:
            df = score_checkpoint(model_repo, stimuli_path, device, revision=final_branch)
        except Exception as e:
            print(f"[meco-sweep]   Scoring failed: {e}")
            continue

        # Compute Spearman correlation with human RTs
        valid = df.dropna(subset=["surprisal"])
        if rt_column in valid.columns and len(valid) > 10:
            rho, pval = stats.spearmanr(valid["surprisal"], valid[rt_column])
        else:
            rho, pval = float("nan"), float("nan")

        # Extract sweep parameters from model name
        name = model_repo.split("/")[-1]
        params = _parse_sweep_params(name)

        results.append({
            "model": model_repo,
            "step": final_step,
            "branch": final_branch,
            "spearman_rho": rho,
            "spearman_pval": pval,
            "n_valid": len(valid),
            **params,
        })
        print(f"[meco-sweep]   rho={rho:.4f}, p={pval:.2e}, n={len(valid)}")

    # Write ranked results
    results_df = pd.DataFrame(results).sort_values("spearman_rho", ascending=False)
    results_path = output_dir / "sweep_ranked_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n[meco-sweep] Ranked results -> {results_path}")

    # Write best settings JSON
    best_settings = {}
    if len(results_df) > 0:
        best_row = results_df.iloc[0]
        best_settings = {
            "model": best_row["model"],
            "spearman_rho": float(best_row["spearman_rho"]),
            "ewc_lambda": best_row.get("ewc_lambda"),
            "maml_ratio": best_row.get("maml_ratio"),
            "lr_boost": best_row.get("lr_boost"),
            "lamol_lambda": best_row.get("lamol_lambda"),
        }
        # Clean NaN values
        best_settings = {k: v for k, v in best_settings.items()
                         if not (isinstance(v, float) and np.isnan(v))}

    best_path = output_dir / "best_sweep_settings.json"
    with open(best_path, "w") as f:
        json.dump(best_settings, f, indent=2)
    print(f"[meco-sweep] Best settings -> {best_path}")
    print(f"[meco-sweep] Best model: {best_settings.get('model', 'N/A')} "
          f"(rho={best_settings.get('spearman_rho', 'N/A')})")

    return best_path


def _parse_sweep_params(name: str) -> dict:
    """Extract sweep parameters from a model run name."""
    params = {
        "ewc_lambda": None,
        "maml_ratio": None,
        "lr_boost": None,
        "lamol_lambda": None,
    }

    m = re.search(r"ewc([\d.]+)", name)
    if m:
        params["ewc_lambda"] = float(m.group(1))

    m = re.search(r"maml([\d.]+)", name)
    if m:
        params["maml_ratio"] = float(m.group(1))

    m = re.search(r"lr([\d.]+)", name)
    if m:
        params["lr_boost"] = float(m.group(1))

    m = re.search(r"lamol([\d.]+)", name)
    if m:
        params["lamol_lambda"] = float(m.group(1))

    return params


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MECO surprisal evaluation")
    parser.add_argument("--models", nargs="+", required=True,
                        help="HuggingFace model repo IDs")
    parser.add_argument("--stimuli", default=str(DEFAULT_STIMULI),
                        help="Path to MECO stimuli TSV")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR),
                        help="Output directory")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="Evaluate all step-N checkpoints per model")
    parser.add_argument("--device", default=DEVICE,
                        help="Device (cuda or cpu)")
    parser.add_argument("--sweep-org", type=str, default=None,
                        help="HF org to sweep (scores all models, ranks by MECO correlation)")
    args = parser.parse_args()

    if args.sweep_org:
        score_sweep_results(
            hf_org=args.sweep_org,
            stimuli_path=args.stimuli,
            output_dir=args.output_dir,
            device=args.device,
        )
    else:
        run_meco_eval(
            models=args.models,
            stimuli_path=args.stimuli,
            output_dir=args.output_dir,
            device=args.device,
            all_checkpoints=args.all_checkpoints,
        )
