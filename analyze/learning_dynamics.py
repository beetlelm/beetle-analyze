"""
learning_dynamics.py — Cross-Family Learning Dynamics Analysis.

Covers:
  - Loss curves for a single language pair across all conditions
  - Consistency of patterns across language pairs and directions
  - Catastrophic forgetting detection with trajectory analysis
  - Transfer learning evidence (L2 BLiMP/MultiBLiMP vs L1 distance)
  - Comparison with popular multilingual LMs and B-GPT models

Evaluation uses FLORES-200 for PPL and existing benchmark results
(MultiBLiMP, BLiMP-NL, ZhoBLiMP, XCOMPS, XNLI) for downstream tasks.

Outputs
-------
results/dynamics/
  loss_curves.csv                 — per-checkpoint PPL per language per condition
  cross_pair_consistency.csv      — pattern correlation across language pairs
  forgetting_trajectories.csv     — CF trajectory across checkpoints
  transfer_evidence.csv           — L2 accuracy vs L1 typological distance
  baseline_comparison.csv         — BeetleLM vs multilingual LMs vs B-GPT
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
    FLORES_LANG_MAP,
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

# B-GPT models from https://huggingface.co/collections/catherinearnett/b-gpt
BGPT_MODELS = [
    "catherinearnett/b-gpt-nld-eng",
    "catherinearnett/b-gpt-deu-eng",
    "catherinearnett/b-gpt-fra-eng",
    "catherinearnett/b-gpt-zho-eng",
    "catherinearnett/b-gpt-spa-eng",
    "catherinearnett/b-gpt-jpn-eng",
    "catherinearnett/b-gpt-ita-eng",
    "catherinearnett/b-gpt-rus-eng",
]

# Popular multilingual LMs for comparison
MULTILINGUAL_BASELINES = [
    "ai-forever/mGPT",
    "facebook/xglm-564M",
    "bigscience/bloom-560m",
]

# Typological distances to English (rough WALS/Ethnologue based)
# Higher = more distant
TYPOLOGICAL_DISTANCE_TO_ENG = {
    "nld": 0.15,   # Dutch (close Germanic)
    "deu": 0.20,   # German (Germanic)
    "fra": 0.35,   # French (Romance)
    "spa": 0.35,   # Spanish (Romance)
    "ita": 0.35,   # Italian (Romance)
    "cat": 0.35,   # Catalan (Romance)
    "por": 0.35,   # Portuguese (Romance)
    "swe": 0.20,   # Swedish (Germanic)
    "dan": 0.20,   # Danish (Germanic)
    "pol": 0.50,   # Polish (Slavic)
    "rus": 0.55,   # Russian (Slavic)
    "bul": 0.55,   # Bulgarian (Slavic)
    "ell": 0.60,   # Greek (Hellenic)
    "tur": 0.70,   # Turkish (Turkic)
    "fas": 0.65,   # Persian (Indo-Iranian)
    "hin": 0.60,   # Hindi (Indo-Aryan)
    "ind": 0.75,   # Indonesian (Austronesian)
    "tgl": 0.75,   # Tagalog (Austronesian)
    "zho": 0.85,   # Chinese (Sino-Tibetan)
    "jpn": 0.90,   # Japanese (Japonic)
    "tam": 0.80,   # Tamil (Dravidian)
    "arb": 0.70,   # Arabic (Semitic)
    "eus": 0.95,   # Basque (isolate)
}


# ---------------------------------------------------------------------------
# Loss Curves
# ---------------------------------------------------------------------------

def compute_loss_curves(
    repos: list[str],
    lang_codes: list[str] = ("nld", "eng"),
    sample_size: int = 100,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Compute per-checkpoint PPL for each language across conditions.

    Produces loss curves suitable for:
      - Comparing conditions within a language pair
      - Detecting catastrophic forgetting (PPL increase on L1)
      - Identifying transfer learning (PPL decrease on L2)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "loss_curves.csv"

    if out_path.exists() and not overwrite:
        print(f"[learning] Loading cached {out_path}")
        return pd.read_csv(out_path)

    from convergence import list_checkpoints, _load_checkpoint, _free, step_label

    # Load FLORES sentences
    sentences_by_lang = {}
    for code in lang_codes:
        try:
            all_sents = load_flores_sentences(code)
            step = max(1, len(all_sents) // sample_size)
            sentences_by_lang[code] = all_sents[::step][:sample_size]
        except Exception as e:
            print(f"[learning] Could not load FLORES for {code}: {e}")

    rows: list[dict] = []
    for repo in repos:
        checkpoints = list_checkpoints(repo)
        bil_type = get_bilingual_type(repo)
        lang_pair = get_lang_pair(repo)

        print(f"[learning] {repo}: {len(checkpoints)} checkpoints")

        for revision in checkpoints:
            step = step_label(revision)
            result = _load_checkpoint(repo, revision)
            if result is None:
                continue
            model, tokenizer = result

            for code, sentences in sentences_by_lang.items():
                nlls = [sentence_log_likelihood(s, model, tokenizer) for s in sentences]
                valid_nlls = [n for n in nlls if n < float("inf")]
                mean_nll = float(np.mean(valid_nlls)) if valid_nlls else float("inf")

                rows.append({
                    "repo": repo,
                    "bilingual_type": bil_type,
                    "lang_pair": lang_pair,
                    "revision": revision,
                    "step": step,
                    "eval_lang": code,
                    "mean_nll": mean_nll,
                    "mean_ppl": nll_to_ppl(mean_nll),
                    "std_nll": float(np.std(valid_nlls)) if valid_nlls else float("nan"),
                    "median_ppl": nll_to_ppl(float(np.median(valid_nlls))) if valid_nlls else float("inf"),
                })

            _free(model, tokenizer)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[learning] Loss curves saved {out_path} ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Cross-Pair Consistency
# ---------------------------------------------------------------------------

def compute_cross_pair_consistency(
    loss_curves: pd.DataFrame,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Test consistency of learning dynamics patterns across language pairs.

    For each pair of language pairs with the same bilingual type,
    computes Spearman correlation of their loss curves.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "cross_pair_consistency.csv"

    if out_path.exists() and not overwrite:
        print(f"[learning] Loading cached {out_path}")
        return pd.read_csv(out_path)

    rows: list[dict] = []

    for bil_type in loss_curves["bilingual_type"].unique():
        type_data = loss_curves[loss_curves["bilingual_type"] == bil_type]
        pairs = type_data["lang_pair"].unique()

        for i, p1 in enumerate(pairs):
            for p2 in pairs[i + 1:]:
                d1 = type_data[type_data["lang_pair"] == p1].sort_values("step")
                d2 = type_data[type_data["lang_pair"] == p2].sort_values("step")

                # Align by step number
                common_steps = set(d1["step"]) & set(d2["step"])
                if len(common_steps) < 3:
                    continue

                d1_aligned = d1[d1["step"].isin(common_steps)].sort_values("step")
                d2_aligned = d2[d2["step"].isin(common_steps)].sort_values("step")

                rho, pval = stats.spearmanr(d1_aligned["mean_ppl"], d2_aligned["mean_ppl"])

                rows.append({
                    "bilingual_type": bil_type,
                    "pair_1": p1,
                    "pair_2": p2,
                    "spearman_rho": rho,
                    "spearman_pval": pval,
                    "n_common_steps": len(common_steps),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[learning] Cross-pair consistency saved {out_path}")
    return df


# ---------------------------------------------------------------------------
# Catastrophic Forgetting Detection
# ---------------------------------------------------------------------------

def detect_catastrophic_forgetting(
    loss_curves: pd.DataFrame,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Detect catastrophic forgetting from loss curve trajectories.

    CF indicators:
      - PPL increase on L1 over training (positive slope in late training)
      - CF_score = PPL_bilingual - PPL_monolingual (from forgetting.py)
      - CF_log = log(PPL_bilingual / PPL_monolingual)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "forgetting_trajectories.csv"

    if out_path.exists() and not overwrite:
        print(f"[learning] Loading cached {out_path}")
        return pd.read_csv(out_path)

    rows: list[dict] = []

    for (repo, lang), group in loss_curves.groupby(["repo", "eval_lang"]):
        group = group.sort_values("step")
        if len(group) < 3:
            continue

        steps = group["step"].values.astype(float)
        ppls = group["mean_ppl"].values

        # Filter out inf values
        valid = ppls < float("inf")
        if valid.sum() < 3:
            continue

        steps_valid = steps[valid]
        ppls_valid = ppls[valid]

        # Overall trend
        slope, intercept, r, p, se = stats.linregress(steps_valid, ppls_valid)

        # Late-training trend (last 50% of checkpoints)
        mid = len(steps_valid) // 2
        if mid >= 2:
            late_slope, _, _, late_p, _ = stats.linregress(
                steps_valid[mid:], ppls_valid[mid:]
            )
        else:
            late_slope, late_p = float("nan"), float("nan")

        # Detect forgetting: PPL increase in late training
        ppl_min = ppls_valid.min()
        ppl_final = ppls_valid[-1]
        forgetting_delta = ppl_final - ppl_min

        rows.append({
            "repo": repo,
            "bilingual_type": get_bilingual_type(repo),
            "eval_lang": lang,
            "overall_slope": slope,
            "overall_r": r,
            "overall_p": p,
            "late_slope": late_slope,
            "late_p": late_p,
            "ppl_min": ppl_min,
            "ppl_final": ppl_final,
            "forgetting_delta": forgetting_delta,
            "forgetting_detected": forgetting_delta > 0 and late_slope > 0,
            "n_checkpoints": len(steps_valid),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[learning] Forgetting trajectories saved {out_path}")
    return df


# ---------------------------------------------------------------------------
# Transfer Learning Evidence
# ---------------------------------------------------------------------------

def compute_transfer_evidence(
    benchmark_results: pd.DataFrame,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Test whether L2 BLiMP/MultiBLiMP accuracy correlates with L1 distance.

    Hypothesis: More similar L1s produce higher L2 scores (positive transfer).

    Args:
        benchmark_results: DataFrame with columns: model, lang, accuracy
                           (from MultiBLiMP or BLiMP evaluation)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "transfer_evidence.csv"

    if out_path.exists() and not overwrite:
        print(f"[learning] Loading cached {out_path}")
        return pd.read_csv(out_path)

    # Extract L1 language from model name
    benchmark_results = benchmark_results.copy()
    benchmark_results["l1_lang"] = benchmark_results["model"].apply(
        lambda m: get_lang_pair(m).split("-")[0].split("_")[0]
    )
    benchmark_results["typo_distance"] = benchmark_results["l1_lang"].map(
        TYPOLOGICAL_DISTANCE_TO_ENG
    )

    valid = benchmark_results.dropna(subset=["typo_distance", "accuracy"])
    if len(valid) < 3:
        print("[learning] Not enough data for transfer analysis")
        return pd.DataFrame()

    # Correlation: distance vs accuracy
    rho, pval = stats.spearmanr(valid["typo_distance"], valid["accuracy"])

    # Aggregate per L1 language
    agg = valid.groupby(["l1_lang", "typo_distance"]).agg(
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        n_models=("model", "nunique"),
    ).reset_index()

    agg["spearman_rho"] = rho
    agg["spearman_pval"] = pval

    agg.to_csv(out_path, index=False)
    print(f"[learning] Transfer evidence: rho={rho:.3f}, p={pval:.3e}")
    print(f"[learning] Saved {out_path}")
    return agg


# ---------------------------------------------------------------------------
# Baseline Model Comparison
# ---------------------------------------------------------------------------

def evaluate_baseline_models(
    baseline_repos: list[str] | None = None,
    lang_codes: list[str] = ("nld", "eng", "deu", "fra", "zho"),
    sample_size: int = 200,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Evaluate multilingual LMs and B-GPT on FLORES-200 for comparison.

    Baselines include B-GPT models and popular multilingual LMs
    (mGPT, XGLM, BLOOM).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "baseline_comparison.csv"

    if out_path.exists() and not overwrite:
        print(f"[learning] Loading cached {out_path}")
        return pd.read_csv(out_path)

    if baseline_repos is None:
        baseline_repos = BGPT_MODELS + MULTILINGUAL_BASELINES

    rows: list[dict] = []

    for repo in baseline_repos:
        clear_model_cache()
        result = load_model_and_tokenizer(repo)
        if result is None:
            print(f"[learning] SKIPPED {repo} — failed to load")
            continue
        model, tokenizer = result

        for code in lang_codes:
            if code not in FLORES_LANG_MAP:
                continue
            try:
                sentences = load_flores_sentences(code)[:sample_size]
            except Exception as e:
                print(f"[learning] Could not load FLORES {code}: {e}")
                continue

            nlls = [sentence_log_likelihood(s, model, tokenizer) for s in sentences]
            valid_nlls = [n for n in nlls if n < float("inf")]
            mean_nll = float(np.mean(valid_nlls)) if valid_nlls else float("inf")

            rows.append({
                "repo": repo,
                "model_type": _classify_baseline(repo),
                "eval_lang": code,
                "mean_nll": mean_nll,
                "mean_ppl": nll_to_ppl(mean_nll),
                "std_nll": float(np.std(valid_nlls)) if valid_nlls else float("nan"),
                "n_sentences": len(valid_nlls),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[learning] Baseline comparison saved {out_path} ({len(df)} rows)")
    return df


def _classify_baseline(repo: str) -> str:
    """Classify a baseline model by type."""
    lower = repo.lower()
    if "b-gpt" in lower:
        return "bgpt"
    elif "mgpt" in lower:
        return "mgpt"
    elif "xglm" in lower:
        return "xglm"
    elif "bloom" in lower:
        return "bloom"
    elif "beetlelm" in lower:
        return "beetlelm"
    else:
        return "other"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_learning_dynamics(
    repos: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run all learning dynamics analyses."""
    if repos is None:
        repos = MODEL_GROUPS.get("nld", [])[:5]

    results = {}

    # Loss curves
    print("\n" + "=" * 60 + "\n LOSS CURVES\n" + "=" * 60)
    loss_df = compute_loss_curves(repos, overwrite=overwrite)
    results["loss_curves"] = loss_df

    # Cross-pair consistency
    if not loss_df.empty:
        print("\n" + "=" * 60 + "\n CROSS-PAIR CONSISTENCY\n" + "=" * 60)
        results["consistency"] = compute_cross_pair_consistency(loss_df, overwrite=overwrite)

        print("\n" + "=" * 60 + "\n FORGETTING DETECTION\n" + "=" * 60)
        results["forgetting"] = detect_catastrophic_forgetting(loss_df, overwrite=overwrite)

    # Transfer evidence (requires benchmark results)
    print("\n" + "=" * 60 + "\n TRANSFER EVIDENCE\n" + "=" * 60)
    for bm_file in ["results/multiblimp_results.csv", "results/blimp_eng_results.csv"]:
        if Path(bm_file).exists():
            bm_df = pd.read_csv(bm_file)
            if "accuracy" in bm_df.columns or "acc" in bm_df.columns:
                if "accuracy" not in bm_df.columns:
                    bm_df["accuracy"] = bm_df["acc"]
                results["transfer"] = compute_transfer_evidence(bm_df, overwrite=overwrite)
                break
    else:
        print("[learning] SKIPPED transfer analysis — no benchmark results found")

    # Baseline comparison
    print("\n" + "=" * 60 + "\n BASELINE COMPARISON\n" + "=" * 60)
    results["baselines"] = evaluate_baseline_models(overwrite=overwrite)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Learning dynamics analysis")
    parser.add_argument("--repos", nargs="+", default=None)
    parser.add_argument("--baselines-only", action="store_true",
                        help="Only evaluate baseline models")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.baselines_only:
        evaluate_baseline_models(overwrite=args.overwrite)
    else:
        run_learning_dynamics(repos=args.repos, overwrite=args.overwrite)
