"""
cli_eval.py — Cross-Lingual Interference (CLI) Evaluation (Family B).

Implements:
  B1  BLiMP Grammaticality Accuracy as CLI Metric
  B2  Dutch-Specific Grammaticality Probe (V2, separable verbs, de/het)
  B3  Age-of-Exposure x CLI Interaction
  B4  FCE Learner Preference Test
  B5  L1 Translation Priming Experiment

CLI index = (bilingual_accuracy - monolingual_accuracy) / monolingual_accuracy
Positive = facilitation; Negative = interference.

Outputs
-------
results/cli/
  cli_index.csv             — CLI per model, per BLiMP phenomenon
  dutch_probes.csv           — Dutch grammaticality accuracy per model
  aoe_cli_curve.csv          — CLI vs L2 introduction step
  learner_preference.csv     — delta surprisal for learner errors
  translation_priming.csv    — priming effect per phenomenon
"""

from __future__ import annotations

from analyze import _path  # noqa: F401

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy import stats

from models import MODEL_GROUPS, get_bilingual_type
from ppl_utils import (
    DEVICE,
    clear_model_cache,
    load_model_and_tokenizer,
    sentence_log_likelihood,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("results/cli")

DUTCH_PROBES_FILE = (
    Path(__file__).resolve().parent.parent / "data" / "cli" / "dutch_grammaticality_probes.tsv"
)


# ---------------------------------------------------------------------------
# B1: BLiMP Grammaticality as CLI Metric
# ---------------------------------------------------------------------------

def compute_cli_index(
    bilingual_results: pd.DataFrame,
    mono_results: pd.DataFrame,
    group_col: str = "phenomenon",
) -> pd.DataFrame:
    """
    Compute CLI index from BLiMP evaluation results.

    CLI = (bilingual_accuracy - monolingual_accuracy) / monolingual_accuracy

    Args:
        bilingual_results: DataFrame with columns: model, phenomenon, accuracy
        mono_results:      DataFrame with columns: model, phenomenon, accuracy
                           (single monolingual English baseline)
        group_col:         Column to group by (default: BLiMP phenomenon)

    Returns:
        DataFrame with CLI index per bilingual model per phenomenon.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get mono baseline accuracy per phenomenon
    mono_acc = mono_results.groupby(group_col)["accuracy"].mean().to_dict()

    rows: list[dict] = []
    for (model, phenom), group in bilingual_results.groupby(["model", group_col]):
        bi_acc = group["accuracy"].mean()
        m_acc = mono_acc.get(phenom, 0.5)

        if m_acc == 0:
            cli = float("nan")
        else:
            cli = (bi_acc - m_acc) / m_acc

        rows.append({
            "model": model,
            "bilingual_type": get_bilingual_type(model),
            "phenomenon": phenom,
            "bilingual_accuracy": bi_acc,
            "monolingual_accuracy": m_acc,
            "cli_index": cli,
            "cli_direction": "facilitation" if cli > 0 else "interference",
        })

    return pd.DataFrame(rows)


def run_blimp_cli(
    blimp_results_path: Path | str = "results/blimp_eng_results.csv",
    mono_model: str = "BeetleLM/beetlelm_eng_mono",
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Compute CLI index from existing BLiMP results.

    Expects a CSV with columns: model, phenomenon, accuracy (or correct/total).
    """
    out_path = OUTPUT_DIR / "cli_index.csv"
    if out_path.exists() and not overwrite:
        print(f"[cli] Loading cached {out_path}")
        return pd.read_csv(out_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    blimp_path = Path(blimp_results_path)
    if not blimp_path.exists():
        print(f"[cli] BLiMP results not found at {blimp_path}")
        return pd.DataFrame()

    all_results = pd.read_csv(blimp_path)

    # Handle different column naming conventions
    if "accuracy" not in all_results.columns:
        if "correct" in all_results.columns and "total" in all_results.columns:
            all_results["accuracy"] = all_results["correct"] / all_results["total"]
        elif "acc" in all_results.columns:
            all_results["accuracy"] = all_results["acc"]
        else:
            print(f"[cli] Cannot find accuracy column in {blimp_path}")
            return pd.DataFrame()

    if "phenomenon" not in all_results.columns and "paradigm" in all_results.columns:
        all_results["phenomenon"] = all_results["paradigm"]

    # Split mono vs bilingual
    mono_mask = all_results["model"].str.contains(mono_model, na=False)
    mono_results = all_results[mono_mask]
    bilingual_results = all_results[~mono_mask]

    if mono_results.empty:
        print(f"[cli] No mono baseline found for {mono_model}")
        return pd.DataFrame()

    cli_df = compute_cli_index(bilingual_results, mono_results)
    cli_df.to_csv(out_path, index=False)
    print(f"[cli] Saved {out_path} ({len(cli_df)} rows)")
    return cli_df


# ---------------------------------------------------------------------------
# B2: Dutch-Specific Grammaticality Probe
# ---------------------------------------------------------------------------

def evaluate_dutch_probes(
    repos: list[str],
    probes_path: Path = DUTCH_PROBES_FILE,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Evaluate models on Dutch grammaticality minimal pairs.

    Tests Dutch-specific phenomena:
      - V2 word order (Dutch obligatory, unlike English)
      - Separable verb prefixes
      - De/het gender agreement

    Probes TSV expected columns: phenomenon, acceptable, unacceptable

    Accuracy = P(acceptable) > P(unacceptable) per pair.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "dutch_probes.csv"

    if out_path.exists() and not overwrite:
        print(f"[cli] Loading cached {out_path}")
        return pd.read_csv(out_path)

    if not probes_path.exists():
        print(f"[cli] Dutch probes not found at {probes_path}")
        return pd.DataFrame()

    probes = pd.read_csv(probes_path, sep="\t")

    rows: list[dict] = []
    for repo in repos:
        clear_model_cache()
        result = load_model_and_tokenizer(repo)
        if result is None:
            continue
        model, tokenizer = result
        bil_type = get_bilingual_type(repo)

        for phenom, phenom_df in probes.groupby("phenomenon"):
            correct = 0
            total = 0
            for _, pair in phenom_df.iterrows():
                nll_good = sentence_log_likelihood(pair["acceptable"], model, tokenizer)
                nll_bad = sentence_log_likelihood(pair["unacceptable"], model, tokenizer)
                if nll_good < nll_bad:  # lower NLL = higher probability
                    correct += 1
                total += 1

            rows.append({
                "model": repo,
                "bilingual_type": bil_type,
                "phenomenon": phenom,
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total > 0 else float("nan"),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[cli] Dutch probes saved {out_path} ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# B3: Age-of-Exposure x CLI Interaction
# ---------------------------------------------------------------------------

def compute_aoe_cli_curve(
    cli_df: pd.DataFrame,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Plot CLI score vs L2 introduction step.

    Fits: CLI ~ log(introduction_step) + scale + (1|seed)

    Maps bilingual types to approximate L2 introduction steps:
      simultaneous/balanced -> 0K
      part_time -> 16K
      late -> 32K
      heritage -> 48K
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "aoe_cli_curve.csv"

    if out_path.exists() and not overwrite:
        print(f"[cli] Loading cached {out_path}")
        return pd.read_csv(out_path)

    step_map = {
        "simultaneous": 0,
        "balanced": 0,
        "part_time": 16000,
        "late": 32000,
        "heritage": 48000,
    }

    cli_df = cli_df.copy()
    cli_df["l2_intro_step"] = cli_df["bilingual_type"].map(step_map)

    valid = cli_df.dropna(subset=["l2_intro_step", "cli_index"])
    if valid.empty:
        print("[cli] No valid AoE-CLI data")
        return pd.DataFrame()

    # Aggregate CLI per (bilingual_type, phenomenon)
    agg = valid.groupby(["bilingual_type", "l2_intro_step"]).agg(
        mean_cli=("cli_index", "mean"),
        std_cli=("cli_index", "std"),
        n_models=("model", "nunique"),
    ).reset_index()

    # Fit log-linear relationship
    x = agg["l2_intro_step"].values.astype(float)
    y = agg["mean_cli"].values
    # Add 1 to avoid log(0)
    x_log = np.log1p(x)

    if len(x_log) >= 3:
        slope, intercept, r, p, se = stats.linregress(x_log, y)
        agg["log_fit_slope"] = slope
        agg["log_fit_intercept"] = intercept
        agg["log_fit_r"] = r
        agg["log_fit_p"] = p
    else:
        agg["log_fit_slope"] = float("nan")

    agg.to_csv(out_path, index=False)
    print(f"[cli] AoE-CLI curve saved {out_path}")
    return agg


# ---------------------------------------------------------------------------
# B4: FCE Learner Preference Test
# ---------------------------------------------------------------------------

def compute_learner_preference(
    repos: list[str],
    error_correct_pairs: pd.DataFrame,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Test whether bilingual models prefer Dutch-learner English constructions.

    delta_S = surprisal(correct) - surprisal(error)
    Positive delta_S = model prefers the error (lower surprisal for error).

    Args:
        repos:               Models to evaluate.
        error_correct_pairs: DataFrame with columns: error_sentence, correct_sentence,
                             error_type
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "learner_preference.csv"

    if out_path.exists() and not overwrite:
        print(f"[cli] Loading cached {out_path}")
        return pd.read_csv(out_path)

    rows: list[dict] = []
    for repo in repos:
        clear_model_cache()
        result = load_model_and_tokenizer(repo)
        if result is None:
            continue
        model, tokenizer = result
        bil_type = get_bilingual_type(repo)

        for _, pair in error_correct_pairs.iterrows():
            nll_error = sentence_log_likelihood(pair["error_sentence"], model, tokenizer)
            nll_correct = sentence_log_likelihood(pair["correct_sentence"], model, tokenizer)

            rows.append({
                "model": repo,
                "bilingual_type": bil_type,
                "error_type": pair.get("error_type", "unknown"),
                "surprisal_error": nll_error,
                "surprisal_correct": nll_correct,
                "delta_s": nll_correct - nll_error,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[cli] Learner preference saved {out_path} ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# B5: L1 Translation Priming
# ---------------------------------------------------------------------------

def compute_translation_priming(
    repos: list[str],
    blimp_pairs: pd.DataFrame,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Test whether Dutch prime changes English grammaticality judgment.

    Condition 1: P(English sentence)
    Condition 2: P(English sentence | Dutch prime)
    Priming effect = log_prob(cond2) - log_prob(cond1)

    blimp_pairs expected columns: english_sentence, dutch_prime, phenomenon
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "translation_priming.csv"

    if out_path.exists() and not overwrite:
        print(f"[cli] Loading cached {out_path}")
        return pd.read_csv(out_path)

    rows: list[dict] = []
    for repo in repos:
        clear_model_cache()
        result = load_model_and_tokenizer(repo)
        if result is None:
            continue
        model, tokenizer = result
        bil_type = get_bilingual_type(repo)

        for _, pair in blimp_pairs.iterrows():
            eng_sent = pair["english_sentence"]
            dutch_prime = pair.get("dutch_prime", "")

            # Condition 1: English only
            nll_eng = sentence_log_likelihood(eng_sent, model, tokenizer)

            # Condition 2: Dutch prime + English
            if dutch_prime:
                primed = f"{dutch_prime} {eng_sent}"
                nll_primed = sentence_log_likelihood(primed, model, tokenizer)
            else:
                nll_primed = float("nan")

            rows.append({
                "model": repo,
                "bilingual_type": bil_type,
                "phenomenon": pair.get("phenomenon", "unknown"),
                "nll_english_only": nll_eng,
                "nll_primed": nll_primed,
                "priming_effect": nll_eng - nll_primed,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[cli] Translation priming saved {out_path} ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_cli_evaluation(
    repos: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Run all CLI evaluation analyses (B1-B5).

    B1 uses existing BLiMP results (no model loading needed).
    B2-B5 require model loading and stimulus files.
    """
    if repos is None:
        repos = MODEL_GROUPS.get("nld", []) + MODEL_GROUPS.get("eng", [])
        repos = list(dict.fromkeys(repos))

    results = {}

    # B1: BLiMP CLI index
    print("\n" + "=" * 60 + "\n B1: BLIMP CLI INDEX\n" + "=" * 60)
    cli_df = run_blimp_cli(overwrite=overwrite)
    results["cli_index"] = cli_df

    # B2: Dutch grammaticality probes
    print("\n" + "=" * 60 + "\n B2: DUTCH GRAMMATICALITY PROBES\n" + "=" * 60)
    results["dutch_probes"] = evaluate_dutch_probes(repos, overwrite=overwrite)

    # B3: AoE x CLI interaction (requires B1)
    if not cli_df.empty:
        print("\n" + "=" * 60 + "\n B3: AGE-OF-EXPOSURE x CLI\n" + "=" * 60)
        results["aoe_cli"] = compute_aoe_cli_curve(cli_df, overwrite=overwrite)

    # B4 and B5 require external data files (FCE corpus, BLiMP translations)
    # that must be provided separately.  Stub orchestration:
    fce_path = Path(__file__).resolve().parent.parent / "data" / "cli" / "fce_learner_errors.tsv"
    if fce_path.exists():
        print("\n" + "=" * 60 + "\n B4: FCE LEARNER PREFERENCE\n" + "=" * 60)
        fce_data = pd.read_csv(fce_path, sep="\t")
        results["learner_preference"] = compute_learner_preference(
            repos, fce_data, overwrite=overwrite
        )
    else:
        print(f"[cli] SKIPPED B4 — {fce_path} not found")

    priming_path = Path(__file__).resolve().parent.parent / "data" / "cli" / "blimp_dutch_primes.tsv"
    if priming_path.exists():
        print("\n" + "=" * 60 + "\n B5: TRANSLATION PRIMING\n" + "=" * 60)
        priming_data = pd.read_csv(priming_path, sep="\t")
        results["translation_priming"] = compute_translation_priming(
            repos, priming_data, overwrite=overwrite
        )
    else:
        print(f"[cli] SKIPPED B5 — {priming_path} not found")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Lingual Interference evaluation")
    parser.add_argument("--repos", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_cli_evaluation(repos=args.repos, overwrite=args.overwrite)
