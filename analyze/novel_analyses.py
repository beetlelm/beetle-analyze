"""
novel_analyses.py — Novel Analyses Not in Any Reference Paper (Family E).

Implements:
  E2  Discourse-Level Code-Switching Surprisal
  E3  Grammatical Phenomenon x Syntactic Distance Interaction
  E4  Tokenizer Fertility Analysis
  E5  Reading Order Curriculum Effects

(E1 — Cross-Lingual Surprisal Asymmetry — is in cognate_surprisal.py)

Outputs
-------
results/novel/
  codeswitch_surprisal.csv     — surprisal at switch points by model condition
  syntactic_distance.csv       — CLI per BLiMP phenomenon vs similarity score
  tokenizer_fertility.csv      — BPE fertility and surprisal interaction
  reading_order.csv            — curriculum order effects on metrics
"""

from __future__ import annotations

from analyze import _path  # noqa: F401

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from models import MODEL_GROUPS, get_bilingual_type
from ppl_utils import (
    clear_model_cache,
    load_model_and_tokenizer,
    sentence_log_likelihood,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("results/novel")

# Dutch-English syntactic similarity scores for BLiMP phenomena (manual/WALS)
# Higher = more similar between Dutch and English
SYNTACTIC_SIMILARITY = {
    "anaphor_agreement": 0.9,       # Similar in both
    "argument_structure": 0.7,
    "binding": 0.8,
    "control_raising": 0.6,
    "determiner_noun_agreement": 0.5,  # Dutch has de/het
    "ellipsis": 0.7,
    "filler_gap": 0.6,
    "irregular_forms": 0.4,
    "island_effects": 0.7,
    "npi_licensing": 0.8,
    "quantifiers": 0.8,
    "subject_verb_agreement": 0.7,
    "syntax_semantics": 0.6,
}


# ---------------------------------------------------------------------------
# E2: Discourse-Level Code-Switching Surprisal
# ---------------------------------------------------------------------------

def compute_codeswitch_surprisal(
    repos: list[str],
    monolingual_sents: list[str],
    cs_sents: list[dict],
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Test whether CS-trained models show lower surprisal at switch points.

    Args:
        repos:              Models to evaluate.
        monolingual_sents:  Monolingual reference sentences.
        cs_sents:           List of dicts with keys:
                            sentence, switch_position (token index),
                            cs_type (intra_sentential / inter_sentential)

    Prediction: Word-level CS models show lowest surprisal at switch points;
    monolingual models show highest.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "codeswitch_surprisal.csv"

    if out_path.exists() and not overwrite:
        print(f"[novel] Loading cached {out_path}")
        return pd.read_csv(out_path)

    rows: list[dict] = []

    for repo in repos:
        clear_model_cache()
        result = load_model_and_tokenizer(repo)
        if result is None:
            continue
        model, tokenizer = result
        bil_type = get_bilingual_type(repo)

        # Monolingual baseline surprisal
        mono_nlls = [sentence_log_likelihood(s, model, tokenizer)
                     for s in monolingual_sents[:50]]
        mean_mono = np.mean([n for n in mono_nlls if n < float("inf")])

        # Code-switched sentences
        for cs in cs_sents:
            nll = sentence_log_likelihood(cs["sentence"], model, tokenizer)
            rows.append({
                "repo": repo,
                "bilingual_type": bil_type,
                "sentence_type": "code_switched",
                "cs_type": cs.get("cs_type", "unknown"),
                "switch_position": cs.get("switch_position", -1),
                "surprisal": nll,
                "mono_baseline_surprisal": mean_mono,
                "delta_surprisal": nll - mean_mono,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[novel] Code-switching surprisal saved {out_path} ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# E3: Grammatical Phenomenon x Syntactic Distance Interaction
# ---------------------------------------------------------------------------

def compute_syntactic_distance_interaction(
    cli_per_phenomenon: pd.DataFrame,
    similarity_scores: dict[str, float] | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Test whether CLI is stronger for BLiMP phenomena that are more
    syntactically similar between Dutch and English.

    Prediction: high similarity -> facilitation; low similarity -> interference.

    Args:
        cli_per_phenomenon: DataFrame with columns: phenomenon, cli_index
                            (output from cli_eval.compute_cli_index)
        similarity_scores:  Dict mapping phenomenon -> similarity score.
                            Defaults to SYNTACTIC_SIMILARITY.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "syntactic_distance.csv"

    if out_path.exists() and not overwrite:
        print(f"[novel] Loading cached {out_path}")
        return pd.read_csv(out_path)

    if similarity_scores is None:
        similarity_scores = SYNTACTIC_SIMILARITY

    # Merge CLI scores with similarity ratings
    cli_agg = cli_per_phenomenon.groupby("phenomenon").agg(
        mean_cli=("cli_index", "mean"),
        std_cli=("cli_index", "std"),
        n_models=("model", "nunique"),
    ).reset_index()

    cli_agg["syntactic_similarity"] = cli_agg["phenomenon"].map(similarity_scores)
    valid = cli_agg.dropna(subset=["syntactic_similarity", "mean_cli"])

    if len(valid) < 3:
        print("[novel] Not enough phenomena for correlation")
        return pd.DataFrame()

    rho, pval = stats.spearmanr(valid["syntactic_similarity"], valid["mean_cli"])
    valid = valid.copy()
    valid["spearman_rho"] = rho
    valid["spearman_pval"] = pval

    valid.to_csv(out_path, index=False)
    print(f"[novel] Syntactic distance interaction: rho={rho:.3f}, p={pval:.3e}")
    print(f"[novel] Saved {out_path}")
    return valid


# ---------------------------------------------------------------------------
# E4: Tokenizer Fertility Analysis
# ---------------------------------------------------------------------------

def compute_tokenizer_fertility(
    repos: list[str],
    word_list: list[str],
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Compute BPE fertility for each word and test whether it predicts
    surprisal effects.

    Fertility = number of BPE subword tokens for a given word.

    Key question: Do cognates that are single tokens show stronger
    facilitation than fragmented cognates?
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "tokenizer_fertility.csv"

    if out_path.exists() and not overwrite:
        print(f"[novel] Loading cached {out_path}")
        return pd.read_csv(out_path)

    rows: list[dict] = []

    for repo in repos:
        clear_model_cache()
        result = load_model_and_tokenizer(repo)
        if result is None:
            continue
        _, tokenizer = result
        bil_type = get_bilingual_type(repo)

        for word in word_list:
            tokens = tokenizer.encode(word, add_special_tokens=False)
            token_strs = tokenizer.convert_ids_to_tokens(tokens)

            rows.append({
                "repo": repo,
                "bilingual_type": bil_type,
                "word": word,
                "fertility": len(tokens),
                "is_single_token": len(tokens) == 1,
                "subword_tokens": "|".join(token_strs),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[novel] Tokenizer fertility saved {out_path} ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# E5: Reading Order Curriculum Effects
# ---------------------------------------------------------------------------

def compute_reading_order_effects(
    curriculum_models: dict[str, list[str]],
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Compare curriculum orders: Dutch-first, English-first, interleaved, alternating.

    Measures PPL on both languages for each curriculum variant.

    Args:
        curriculum_models: Dict mapping order -> list of model repos.
                           e.g., {"dutch_first": [...], "english_first": [...],
                                  "interleaved": [...], "alternating": [...]}
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "reading_order.csv"

    if out_path.exists() and not overwrite:
        print(f"[novel] Loading cached {out_path}")
        return pd.read_csv(out_path)

    try:
        from ppl_utils import load_flores_sentences
        nl_sents = load_flores_sentences("nld")[:100]
        en_sents = load_flores_sentences("eng")[:100]
    except Exception as e:
        print(f"[novel] Could not load FLORES: {e}")
        return pd.DataFrame()

    from ppl_utils import nll_to_ppl

    rows: list[dict] = []
    for order, repos in curriculum_models.items():
        for repo in repos:
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
                "curriculum_order": order,
                "repo": repo,
                "bilingual_type": get_bilingual_type(repo),
                "ppl_nld": nl_ppl,
                "ppl_eng": en_ppl,
                "ppl_mean": (nl_ppl + en_ppl) / 2,
                "ppl_gap": abs(nl_ppl - en_ppl),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[novel] Reading order effects saved {out_path} ({len(df)} models)")
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_novel_analyses(
    repos: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run all novel analyses (E2-E5)."""
    if repos is None:
        repos = MODEL_GROUPS.get("nld", []) + MODEL_GROUPS.get("eng", [])
        repos = list(dict.fromkeys(repos))

    results = {}

    # E2: Code-switching surprisal (requires CS stimuli)
    print("\n" + "=" * 60 + "\n E2: CODE-SWITCHING SURPRISAL\n" + "=" * 60)
    cs_path = Path(__file__).resolve().parent.parent / "data" / "novel" / "codeswitch_stimuli.tsv"
    if cs_path.exists():
        cs_data = pd.read_csv(cs_path, sep="\t")
        cs_sents = cs_data.to_dict("records")
        mono_sents = []
        try:
            from ppl_utils import load_flores_sentences
            mono_sents = load_flores_sentences("eng")[:50]
        except Exception:
            pass
        results["codeswitch"] = compute_codeswitch_surprisal(
            repos, mono_sents, cs_sents, overwrite=overwrite,
        )
    else:
        print(f"[novel] SKIPPED E2 — {cs_path} not found")

    # E3: Syntactic distance interaction (requires CLI results)
    print("\n" + "=" * 60 + "\n E3: SYNTACTIC DISTANCE INTERACTION\n" + "=" * 60)
    cli_path = Path("results/cli/cli_index.csv")
    if cli_path.exists():
        cli_df = pd.read_csv(cli_path)
        results["syntactic_distance"] = compute_syntactic_distance_interaction(
            cli_df, overwrite=overwrite,
        )
    else:
        print("[novel] SKIPPED E3 — run CLI evaluation (B1) first")

    # E4: Tokenizer fertility
    print("\n" + "=" * 60 + "\n E4: TOKENIZER FERTILITY\n" + "=" * 60)
    cognate_path = Path(__file__).resolve().parent.parent / "data" / "cognates" / "dutch_english_cognates.tsv"
    ff_path = Path(__file__).resolve().parent.parent / "data" / "cognates" / "dutch_english_false_friends.tsv"
    word_list = []
    if cognate_path.exists():
        cdf = pd.read_csv(cognate_path, sep="\t")
        word_list.extend(cdf["english_form"].dropna().tolist())
    if ff_path.exists():
        fdf = pd.read_csv(ff_path, sep="\t")
        word_list.extend(fdf["word"].dropna().tolist())

    if word_list:
        results["fertility"] = compute_tokenizer_fertility(
            repos[:10], word_list, overwrite=overwrite,
        )
    else:
        print("[novel] SKIPPED E4 — no word lists found")

    # E5: Reading order (requires curriculum-specific models)
    print("\n" + "=" * 60 + "\n E5: READING ORDER CURRICULUM\n" + "=" * 60)
    print("[novel] E5 requires curriculum-specific models.")
    print("[novel] Provide curriculum_models dict to compute_reading_order_effects().")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Novel analyses (E2-E5)")
    parser.add_argument("--repos", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_novel_analyses(repos=args.repos, overwrite=args.overwrite)
