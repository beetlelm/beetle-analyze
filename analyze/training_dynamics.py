"""
training_dynamics.py — Training Dynamics & Scaling Analysis (Family D).

Implements:
  D1  Checkpoint Analysis: when cross-lingual alignment emerges
  D2  Scale Comparison: 100M vs 24B qualitative patterns
  D3  Data Ratio Ablation: Dutch/English ratio sweep (90/10 to 10/90)
  D4  Early-Imbalanced Control: temporal entrenchment vs data imbalance

Outputs
-------
results/dynamics/
  checkpoint_emergence.csv       — per-checkpoint alignment + CLI metrics
  scale_comparison.csv           — effect size comparison across scales
  ratio_ablation.csv             — CLI/cognate effects vs data ratio
  early_imbalanced_control.csv   — confound test results
"""

from __future__ import annotations

from analyze import _path  # noqa: F401

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from models import MODEL_GROUPS, get_bilingual_type, get_lang_pair
from ppl_utils import (
    DEVICE,
    clear_model_cache,
    load_flores_sentences,
    load_model_and_tokenizer,
    nll_to_ppl,
    sentence_log_likelihood,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("results/dynamics")


# ---------------------------------------------------------------------------
# D1: Checkpoint Analysis — When Does Bilingual Structure Emerge?
# ---------------------------------------------------------------------------

def compute_checkpoint_emergence(
    repo: str,
    lang_codes: list[str] = ("nld", "eng"),
    sample_size: int = 100,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Track when cross-lingual alignment and grammaticality emerge
    across training checkpoints.

    Measures at each checkpoint:
      - Per-language PPL (FLORES)
      - CLI proxy (PPL gap between languages)
      - Embedding norm statistics

    Args:
        repo:        Model repo with step-N branches.
        lang_codes:  Languages to evaluate.
        sample_size: Number of FLORES sentences per language.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = repo.replace("/", "__")
    out_path = OUTPUT_DIR / f"checkpoint_emergence_{slug}.csv"

    if out_path.exists() and not overwrite:
        print(f"[dynamics] Loading cached {out_path}")
        return pd.read_csv(out_path)

    from convergence import list_checkpoints, _load_checkpoint, _free, step_label

    sentences_by_lang = {}
    for code in lang_codes:
        try:
            all_sents = load_flores_sentences(code)
            step = max(1, len(all_sents) // sample_size)
            sentences_by_lang[code] = all_sents[::step][:sample_size]
        except (ValueError, Exception) as e:
            print(f"[dynamics] Could not load FLORES for {code}: {e}")

    if not sentences_by_lang:
        print(f"[dynamics] No FLORES data loaded — skipping {repo}")
        return pd.DataFrame()

    checkpoints = list_checkpoints(repo)
    print(f"[dynamics] {repo}: {len(checkpoints)} checkpoints x {len(sentences_by_lang)} langs")

    rows: list[dict] = []
    for revision in checkpoints:
        step = step_label(revision)
        result = _load_checkpoint(repo, revision)
        if result is None:
            continue
        model, tokenizer = result

        row = {
            "repo": repo,
            "bilingual_type": get_bilingual_type(repo),
            "revision": revision,
            "step": step,
        }

        for code, sentences in sentences_by_lang.items():
            nlls = [sentence_log_likelihood(s, model, tokenizer) for s in sentences]
            valid_nlls = [n for n in nlls if n < float("inf")]
            mean_nll = float(np.mean(valid_nlls)) if valid_nlls else float("inf")
            row[f"ppl_{code}"] = nll_to_ppl(mean_nll)
            row[f"nll_{code}"] = mean_nll

        # CLI proxy: PPL gap between languages
        ppl_vals = [row.get(f"ppl_{c}", float("inf")) for c in lang_codes]
        valid_ppls = [p for p in ppl_vals if p < float("inf")]
        if len(valid_ppls) >= 2:
            row["ppl_gap"] = max(valid_ppls) - min(valid_ppls)
            row["ppl_ratio"] = max(valid_ppls) / min(valid_ppls) if min(valid_ppls) > 0 else float("inf")

        rows.append(row)
        _free(model, tokenizer)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[dynamics] Saved {out_path} ({len(df)} checkpoints)")
    return df


# ---------------------------------------------------------------------------
# D2: Scale Comparison (100M vs 24B)
# ---------------------------------------------------------------------------

def compare_scales(
    results_100m: pd.DataFrame,
    results_24b: pd.DataFrame,
    metric_cols: list[str] | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Compare effect sizes between 100M and 24B scale models.

    Fits: effect_size ~ log(training_tokens) + condition

    Args:
        results_100m: Evaluation results for 100M-token models.
        results_24b:  Evaluation results for 24B-token models.
        metric_cols:  Columns to compare (default: all numeric columns).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "scale_comparison.csv"

    if out_path.exists() and not overwrite:
        print(f"[dynamics] Loading cached {out_path}")
        return pd.read_csv(out_path)

    results_100m = results_100m.copy()
    results_24b = results_24b.copy()
    results_100m["scale"] = "100M"
    results_24b["scale"] = "24B"

    combined = pd.concat([results_100m, results_24b], ignore_index=True)

    if metric_cols is None:
        metric_cols = [c for c in combined.select_dtypes(include=[np.number]).columns
                       if c not in ("step",)]

    rows: list[dict] = []
    for col in metric_cols:
        vals_100m = results_100m[col].dropna()
        vals_24b = results_24b[col].dropna()

        if len(vals_100m) < 2 or len(vals_24b) < 2:
            continue

        t_stat, t_pval = stats.ttest_ind(vals_100m, vals_24b, equal_var=False)

        rows.append({
            "metric": col,
            "mean_100m": vals_100m.mean(),
            "std_100m": vals_100m.std(),
            "mean_24b": vals_24b.mean(),
            "std_24b": vals_24b.std(),
            "difference": vals_24b.mean() - vals_100m.mean(),
            "t_statistic": t_stat,
            "p_value": t_pval,
            "n_100m": len(vals_100m),
            "n_24b": len(vals_24b),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[dynamics] Scale comparison saved {out_path}")
    return df


# ---------------------------------------------------------------------------
# D3: Data Ratio Ablation
# ---------------------------------------------------------------------------

def compute_ratio_ablation(
    ratio_models: dict[str, list[str]],
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Test how Dutch/English ratio affects CLI and cognate effects.

    Args:
        ratio_models: Dict mapping ratio label -> list of model repos.
                      e.g., {"90_10": [...], "75_25": [...], "50_50": [...]}
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "ratio_ablation.csv"

    if out_path.exists() and not overwrite:
        print(f"[dynamics] Loading cached {out_path}")
        return pd.read_csv(out_path)

    rows: list[dict] = []
    for ratio_label, repos in ratio_models.items():
        # Parse ratio
        parts = ratio_label.split("_")
        nl_pct = int(parts[0]) if len(parts) >= 2 else 50
        en_pct = int(parts[1]) if len(parts) >= 2 else 50

        for repo in repos:
            clear_model_cache()
            result = load_model_and_tokenizer(repo)
            if result is None:
                continue
            model, tokenizer = result

            # Measure PPL on both languages
            try:
                nl_sents = load_flores_sentences("nld")[:100]
                en_sents = load_flores_sentences("eng")[:100]
            except Exception:
                continue

            nl_nlls = [sentence_log_likelihood(s, model, tokenizer) for s in nl_sents]
            en_nlls = [sentence_log_likelihood(s, model, tokenizer) for s in en_sents]

            nl_ppl = nll_to_ppl(np.mean([n for n in nl_nlls if n < float("inf")]))
            en_ppl = nll_to_ppl(np.mean([n for n in en_nlls if n < float("inf")]))

            rows.append({
                "ratio": ratio_label,
                "nl_pct": nl_pct,
                "en_pct": en_pct,
                "repo": repo,
                "bilingual_type": get_bilingual_type(repo),
                "ppl_nld": nl_ppl,
                "ppl_eng": en_ppl,
                "ppl_gap": abs(nl_ppl - en_ppl),
                "ppl_ratio": nl_ppl / en_ppl if en_ppl > 0 else float("inf"),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[dynamics] Ratio ablation saved {out_path} ({len(df)} models)")
    return df


# ---------------------------------------------------------------------------
# D4: Early-Imbalanced Control
# ---------------------------------------------------------------------------

def compute_early_imbalanced_control(
    matched_models: dict[str, str],
    curriculum_models: dict[str, str],
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Determine whether AoE effects are driven by data imbalance alone vs
    temporal entrenchment.

    Compares:
      - Curriculum model (L2 introduced at step N with data ratio R)
      - Early-imbalanced control (both languages from step 0 at ratio R)

    If CLI is purely a data ratio effect, both should be equal.
    If temporal entrenchment matters, late introduction produces stronger CLI.

    Args:
        matched_models:    Dict mapping condition -> early-imbalanced control repo
        curriculum_models: Dict mapping condition -> curriculum model repo
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "early_imbalanced_control.csv"

    if out_path.exists() and not overwrite:
        print(f"[dynamics] Loading cached {out_path}")
        return pd.read_csv(out_path)

    rows: list[dict] = []

    try:
        nl_sents = load_flores_sentences("nld")[:100]
        en_sents = load_flores_sentences("eng")[:100]
    except Exception as e:
        print(f"[dynamics] Could not load FLORES: {e}")
        return pd.DataFrame()

    for condition in matched_models:
        for model_type, repo_map in [("early_imbalanced", matched_models),
                                      ("curriculum", curriculum_models)]:
            repo = repo_map.get(condition)
            if not repo:
                continue

            clear_model_cache()
            result = load_model_and_tokenizer(repo)
            if result is None:
                continue
            model, tokenizer = result

            nl_nlls = [sentence_log_likelihood(s, model, tokenizer) for s in nl_sents]
            en_nlls = [sentence_log_likelihood(s, model, tokenizer) for s in en_sents]

            nl_ppl = nll_to_ppl(np.mean([n for n in nl_nlls if n < float("inf")]))
            en_ppl = nll_to_ppl(np.mean([n for n in en_nlls if n < float("inf")]))

            rows.append({
                "condition": condition,
                "model_type": model_type,
                "repo": repo,
                "ppl_nld": nl_ppl,
                "ppl_eng": en_ppl,
                "ppl_gap": abs(nl_ppl - en_ppl),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[dynamics] Early-imbalanced control saved {out_path}")
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_training_dynamics(
    repos: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run all training dynamics analyses (D1-D4)."""
    if repos is None:
        repos = MODEL_GROUPS.get("nld", [])[:5]  # Limit for compute

    results = {}

    # D1: Checkpoint emergence
    print("\n" + "=" * 60 + "\n D1: CHECKPOINT EMERGENCE\n" + "=" * 60)
    for repo in repos[:3]:
        r = compute_checkpoint_emergence(repo, overwrite=overwrite)
        results[f"emergence_{repo}"] = r

    # D2: Scale comparison (requires results from both scales)
    # This is a post-hoc analysis using pre-computed results
    print("\n" + "=" * 60 + "\n D2: SCALE COMPARISON\n" + "=" * 60)
    print("[dynamics] D2 requires pre-computed results at both 100M and 24B scales.")
    print("[dynamics] Run A1/B1/C1 at both scales first, then call compare_scales().")

    # D3: Ratio ablation
    print("\n" + "=" * 60 + "\n D3: RATIO ABLATION\n" + "=" * 60)
    print("[dynamics] D3 requires ratio-specific models.")
    print("[dynamics] Provide ratio_models dict to compute_ratio_ablation().")

    # D4: Early-imbalanced control
    print("\n" + "=" * 60 + "\n D4: EARLY-IMBALANCED CONTROL\n" + "=" * 60)
    print("[dynamics] D4 requires matched early-imbalanced and curriculum models.")
    print("[dynamics] Provide matched_models and curriculum_models dicts.")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training dynamics & scaling analysis")
    parser.add_argument("--repos", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_training_dynamics(repos=args.repos, overwrite=args.overwrite)
