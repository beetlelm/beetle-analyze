"""
cognate_surprisal.py — Cognate & False-Friend Surprisal Analysis.

Implements experiment families:
  A1  Cognate Surprisal Analysis
  A4  False Friend Interference (surprisal x context language)
  A5  Frequency Dissociation Analysis
  E1  Cross-Lingual Surprisal Asymmetry (novel)

All experiments compare surprisal across vocabulary sharing conditions
(Full Overlap, Minimal Overlap, Friends-Only, False-Friends-Only) and
model scales (100M, 24B).

Outputs
-------
results/cognates/
  cognate_surprisal.csv             — per-word surprisal: cognates vs controls
  false_friend_interference.csv     — surprisal x context language
  frequency_dissociation.csv        — regression with frequency controls
  surprisal_asymmetry.csv           — L1->L2 vs L2->L1 facilitation
"""

from __future__ import annotations

from analyze import _path  # noqa: F401

import gc
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy import stats

from models import MODEL_GROUPS, VOCAB_CONDITIONS, get_bilingual_type
from ppl_utils import (
    DEVICE,
    LOAD_FAILURES,
    clear_model_cache,
    load_model_and_tokenizer,
    sentence_log_likelihood,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("results/cognates")

COGNATE_FILE = Path(__file__).resolve().parent.parent / "data" / "cognates" / "dutch_english_cognates.tsv"
FALSE_FRIEND_FILE = Path(__file__).resolve().parent.parent / "data" / "cognates" / "dutch_english_false_friends.tsv"
FALSE_FRIEND_CONTEXTS_FILE = Path(__file__).resolve().parent.parent / "data" / "cognates" / "false_friend_contexts.tsv"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cognate_list(path: Path = COGNATE_FILE) -> pd.DataFrame:
    """
    Load Dutch-English cognate list.

    Expected columns: word, dutch_form, english_form, dutch_freq, english_freq,
                      control_nl, control_en
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Cognate list not found at {path}. "
            "Provide a TSV with columns: word, dutch_form, english_form, "
            "dutch_freq, english_freq, control_nl, control_en"
        )
    return pd.read_csv(path, sep="\t")


def load_false_friend_list(path: Path = FALSE_FRIEND_FILE) -> pd.DataFrame:
    """
    Load Dutch-English false friend (interlingual homograph) list.

    Expected columns: word, dutch_meaning, english_meaning, dutch_freq,
                      english_freq, control_nl, control_en
    """
    if not path.exists():
        raise FileNotFoundError(
            f"False friend list not found at {path}. "
            "Provide a TSV with columns: word, dutch_meaning, english_meaning, "
            "dutch_freq, english_freq, control_nl, control_en"
        )
    return pd.read_csv(path, sep="\t")


def load_false_friend_contexts(path: Path = FALSE_FRIEND_CONTEXTS_FILE) -> pd.DataFrame:
    """
    Load false friend context sentences for A4.

    Expected columns: word, condition (intra_nl, intra_en, cross_nl_target,
                      cross_en_target), sentence
    """
    if not path.exists():
        raise FileNotFoundError(
            f"False friend contexts not found at {path}. "
            "Provide a TSV with columns: word, condition, sentence"
        )
    return pd.read_csv(path, sep="\t")


# ---------------------------------------------------------------------------
# A1: Cognate Surprisal Analysis
# ---------------------------------------------------------------------------

def compute_cognate_surprisal(
    repos: list[str],
    cognates: pd.DataFrame,
    context_sentences: dict[str, list[str]],
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Compute surprisal for cognates vs frequency-matched controls.

    For each model, computes surprisal(w | context) for cognate words and
    their frequency-matched controls.  The cognate facilitation effect is
    the difference: surprisal(control) - surprisal(cognate).

    Args:
        repos:             Model repo IDs to evaluate.
        cognates:          DataFrame from load_cognate_list().
        context_sentences: Dict mapping word -> list of context sentences
                           where the word appears (marked with *word*).
        overwrite:         Re-compute even if cached.

    Returns:
        DataFrame with columns: repo, bilingual_type, word, word_type
        (cognate/control), surprisal, context_lang.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "cognate_surprisal.csv"

    if out_path.exists() and not overwrite:
        print(f"[cognate] Loading cached {out_path}")
        return pd.read_csv(out_path)

    rows: list[dict] = []

    for repo in repos:
        clear_model_cache()
        result = load_model_and_tokenizer(repo)
        if result is None:
            continue
        model, tokenizer = result
        bil_type = get_bilingual_type(repo)

        for _, row in cognates.iterrows():
            for word_type, word_col, lang in [
                ("cognate", "english_form", "eng"),
                ("cognate", "dutch_form", "nld"),
                ("control", "control_en", "eng"),
                ("control", "control_nl", "nld"),
            ]:
                word = row.get(word_col)
                if pd.isna(word) or not word:
                    continue

                sentences = context_sentences.get(word, [])
                for sent in sentences:
                    nll = sentence_log_likelihood(sent, model, tokenizer)
                    rows.append({
                        "repo": repo,
                        "bilingual_type": bil_type,
                        "word": word,
                        "word_type": word_type,
                        "context_lang": lang,
                        "surprisal": nll,
                        "cognate_pair": row.get("word", ""),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[cognate] Saved {out_path} ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# A4: False Friend Interference
# ---------------------------------------------------------------------------

def compute_false_friend_interference(
    repos: list[str],
    contexts: pd.DataFrame,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Test whether false friends show higher surprisal in cross-lingual context.

    For each false friend, measures surprisal in:
      - Intra-lingual Dutch context + Dutch target
      - Intra-lingual English context + English target
      - Cross-lingual context + false-friend target (confusion context)
      - Disambiguation context

    Key prediction: Full Overlap models show more confusion (higher surprisal)
    in cross-lingual context; Minimal Overlap shows less.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "false_friend_interference.csv"

    if out_path.exists() and not overwrite:
        print(f"[false_friend] Loading cached {out_path}")
        return pd.read_csv(out_path)

    rows: list[dict] = []

    for repo in repos:
        clear_model_cache()
        result = load_model_and_tokenizer(repo)
        if result is None:
            continue
        model, tokenizer = result
        bil_type = get_bilingual_type(repo)

        for _, ctx_row in contexts.iterrows():
            nll = sentence_log_likelihood(ctx_row["sentence"], model, tokenizer)
            rows.append({
                "repo": repo,
                "bilingual_type": bil_type,
                "word": ctx_row["word"],
                "condition": ctx_row["condition"],
                "surprisal": nll,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[false_friend] Saved {out_path} ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# A5: Frequency Dissociation Analysis
# ---------------------------------------------------------------------------

def compute_frequency_dissociation(
    cognate_df: pd.DataFrame,
    word_freqs: pd.DataFrame,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Disentangle cross-language frequency from embedding overlap effects.

    Fits regression: surprisal ~ word_type + log_L1_freq + log_L2_freq
                     + vocab_condition + interactions

    Tests whether vocab_condition effect survives frequency control.

    Args:
        cognate_df:  Output from compute_cognate_surprisal().
        word_freqs:  DataFrame with columns: word, l1_freq, l2_freq.
        overwrite:   Re-compute even if cached.

    Returns:
        DataFrame with regression coefficients per vocab condition.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "frequency_dissociation.csv"

    if out_path.exists() and not overwrite:
        print(f"[freq_dissoc] Loading cached {out_path}")
        return pd.read_csv(out_path)

    merged = cognate_df.merge(word_freqs, on="word", how="left")
    merged["log_l1_freq"] = np.log1p(merged.get("l1_freq", 0))
    merged["log_l2_freq"] = np.log1p(merged.get("l2_freq", 0))

    rows: list[dict] = []
    for bil_type in merged["bilingual_type"].unique():
        subset = merged[merged["bilingual_type"] == bil_type].dropna(
            subset=["surprisal", "log_l1_freq", "log_l2_freq"]
        )
        if len(subset) < 10:
            continue

        # Simple OLS: surprisal ~ word_type_numeric + log_l1_freq + log_l2_freq
        subset = subset.copy()
        subset["word_type_num"] = (subset["word_type"] == "cognate").astype(float)

        X_cols = ["word_type_num", "log_l1_freq", "log_l2_freq"]
        X = subset[X_cols].values
        X = np.column_stack([np.ones(len(X)), X])  # intercept
        y = subset["surprisal"].values

        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            rows.append({
                "bilingual_type": bil_type,
                "intercept": beta[0],
                "beta_word_type": beta[1],
                "beta_log_l1_freq": beta[2],
                "beta_log_l2_freq": beta[3],
                "n_observations": len(subset),
            })
        except np.linalg.LinAlgError:
            continue

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[freq_dissoc] Saved {out_path}")
    return df


# ---------------------------------------------------------------------------
# E1: Cross-Lingual Surprisal Asymmetry
# ---------------------------------------------------------------------------

def compute_surprisal_asymmetry(
    cognate_df: pd.DataFrame,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Test whether cognate facilitation is symmetric or asymmetric.

    Computes facilitation (control_surprisal - cognate_surprisal) separately
    for Dutch and English contexts, then measures asymmetry.

    Prediction: 75% Dutch training -> stronger facilitation in Dutch
    (L1 dominance asymmetry).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "surprisal_asymmetry.csv"

    if out_path.exists() and not overwrite:
        print(f"[asymmetry] Loading cached {out_path}")
        return pd.read_csv(out_path)

    rows: list[dict] = []

    for (repo, bil_type), group in cognate_df.groupby(["repo", "bilingual_type"]):
        for lang in ["eng", "nld"]:
            lang_data = group[group["context_lang"] == lang]
            cognate_surp = lang_data[lang_data["word_type"] == "cognate"]["surprisal"].mean()
            control_surp = lang_data[lang_data["word_type"] == "control"]["surprisal"].mean()

            if np.isnan(cognate_surp) or np.isnan(control_surp):
                continue

            facilitation = control_surp - cognate_surp
            rows.append({
                "repo": repo,
                "bilingual_type": bil_type,
                "context_lang": lang,
                "mean_cognate_surprisal": cognate_surp,
                "mean_control_surprisal": control_surp,
                "facilitation": facilitation,
            })

    df = pd.DataFrame(rows)

    # Compute asymmetry per model
    if not df.empty:
        asym_rows = []
        for repo, rdf in df.groupby("repo"):
            eng_fac = rdf[rdf["context_lang"] == "eng"]["facilitation"].values
            nld_fac = rdf[rdf["context_lang"] == "nld"]["facilitation"].values
            if len(eng_fac) > 0 and len(nld_fac) > 0:
                asym_rows.append({
                    "repo": repo,
                    "bilingual_type": rdf["bilingual_type"].iloc[0],
                    "facilitation_eng": float(eng_fac[0]),
                    "facilitation_nld": float(nld_fac[0]),
                    "asymmetry": float(abs(nld_fac[0] - eng_fac[0])),
                    "dominant_lang": "nld" if nld_fac[0] > eng_fac[0] else "eng",
                })
        if asym_rows:
            df = pd.DataFrame(asym_rows)

    df.to_csv(out_path, index=False)
    print(f"[asymmetry] Saved {out_path}")
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_cognate_analysis(
    repos: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Run all cognate/false-friend analyses (A1, A4, A5, E1).

    Args:
        repos:     Model repos to evaluate.  Defaults to all Dutch-English models.
        overwrite: Re-compute even if cached results exist.

    Returns:
        Dict of result DataFrames keyed by analysis name.
    """
    if repos is None:
        repos = MODEL_GROUPS.get("nld", []) + MODEL_GROUPS.get("eng", [])
        repos = list(dict.fromkeys(repos))

    results = {}

    # Load stimuli
    try:
        cognates = load_cognate_list()
    except FileNotFoundError as e:
        print(f"[cognate] SKIPPED A1/A5/E1 — {e}")
        cognates = None

    try:
        ff_contexts = load_false_friend_contexts()
    except FileNotFoundError as e:
        print(f"[cognate] SKIPPED A4 — {e}")
        ff_contexts = None

    # A1: Cognate surprisal
    if cognates is not None:
        # Build context sentences from cognate list
        context_sentences: dict[str, list[str]] = {}
        for _, row in cognates.iterrows():
            for col in ["english_form", "dutch_form", "control_en", "control_nl"]:
                word = row.get(col)
                if pd.notna(word) and word:
                    # Context sentences should be provided in the data file
                    # or generated externally; here we use a placeholder
                    context_sentences.setdefault(word, [])

        print("\n" + "=" * 60 + "\n A1: COGNATE SURPRISAL\n" + "=" * 60)
        results["cognate_surprisal"] = compute_cognate_surprisal(
            repos, cognates, context_sentences, overwrite=overwrite,
        )

    # A4: False friend interference
    if ff_contexts is not None:
        print("\n" + "=" * 60 + "\n A4: FALSE FRIEND INTERFERENCE\n" + "=" * 60)
        results["false_friend_interference"] = compute_false_friend_interference(
            repos, ff_contexts, overwrite=overwrite,
        )

    # A5: Frequency dissociation (requires A1 output)
    if "cognate_surprisal" in results and cognates is not None:
        print("\n" + "=" * 60 + "\n A5: FREQUENCY DISSOCIATION\n" + "=" * 60)
        word_freqs = cognates[["word", "dutch_freq", "english_freq"]].rename(
            columns={"dutch_freq": "l1_freq", "english_freq": "l2_freq"}
        )
        results["frequency_dissociation"] = compute_frequency_dissociation(
            results["cognate_surprisal"], word_freqs, overwrite=overwrite,
        )

    # E1: Surprisal asymmetry (requires A1 output)
    if "cognate_surprisal" in results:
        print("\n" + "=" * 60 + "\n E1: SURPRISAL ASYMMETRY\n" + "=" * 60)
        results["surprisal_asymmetry"] = compute_surprisal_asymmetry(
            results["cognate_surprisal"], overwrite=overwrite,
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cognate & false-friend surprisal analysis")
    parser.add_argument("--repos", nargs="+", default=None,
                        help="Model repos to evaluate")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_cognate_analysis(repos=args.repos, overwrite=args.overwrite)
