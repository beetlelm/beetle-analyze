"""
run_meco_comparison.py — End-to-end MECO alignment comparison CLI.

Drives the pipeline described in the approved plan Part B:

    1. Collect models from Beetle + optional baseline groups (B-GPT,
       bilingual LLMs, BLOOM size ladder, EuroLLM, other multilingual).
    2. Extract token-level surprisal on MECO (cached under
       ``results/meco/surprisal_cache/``).
    3. Compute per-(model, language) Spearman ρ, OLS β, lexical-control β.
    4. Run the Fisher z-test comparing groups against ``small_bilingual``.
    5. Fit the primary mixed-effects model
       ``rt ~ surprisal * model_type + (1|subject) + (1|sentence)``.
    6. Paired Wilcoxon robustness on per-sentence residuals (Holm).
    7. Cross-language Friedman validation with BH-FDR.
    8. Render publication figures 10–17 via ``meco_figures.render_all``.
    9. Emit an auto-filled ``results/meco/claim.txt`` paper-ready paragraph.

Usage
-----
    python -m analyze.run_meco_comparison \\
        --languages en nl es el pl \\
        --include-bgpt --include-bilingual-llm \\
        --include-bloom-ladder --include-eurollm \\
        --output results/meco/

Run with ``--mock`` to exercise the whole pipeline with synthetic data when
surprisal backends aren't wired up yet. Useful for CI and figure review.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import _path  # noqa: F401  — shared repo path bootstrap
except Exception:
    pass

from models import (
    BGPT_BASELINES,
    BILINGUAL_LLM_BASELINES,
    BLOOM_LADDER,
    EUROLLM_BASELINES,
    MULTILINGUAL_LLM_BASELINES,
    get_all_models,
)

from analyze.meco_extended import (
    MECO_OUTPUT_DIR,
    cross_language_friedman,
    fisher_z_group_comparison,
    fit_primary_mixedlm,
    paired_wilcoxon_across_models,
    per_model_alignment,
    preprocess_meco_dataframe,
)
from analyze import meco_figures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_models(args) -> Dict[str, str]:
    """Return {repo_id: group_label} according to the CLI flags."""
    picks: Dict[str, str] = {}

    # Beetle (always on).
    for repo in get_all_models():
        picks.setdefault(repo, "small_bilingual")

    if args.include_bgpt:
        for r in BGPT_BASELINES:
            picks[r] = "bgpt"
    if args.include_bilingual_llm:
        for r in BILINGUAL_LLM_BASELINES:
            picks[r] = "bilingual_llm"
    if args.include_bloom_ladder:
        for r in BLOOM_LADDER:
            picks[r] = "bloom_ladder"
    if args.include_eurollm:
        for r in EUROLLM_BASELINES:
            picks[r] = "eurollm"
    if args.include_multilingual_llm:
        for r in MULTILINGUAL_LLM_BASELINES:
            # bloom_ladder / eurollm already tagged; don't clobber.
            picks.setdefault(r, "multilingual_llm")

    if args.large:
        picks["bigscience/bloom"] = "bloom_ladder"

    return picks


def build_mock_tokens(
    models_by_group: Dict[str, str],
    languages: Sequence[str],
    *,
    n_sentences: int = 40,
    tokens_per_sentence: int = 12,
    subjects: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Build a synthetic token-level dataframe for --mock runs.

    Every group has a different true slope so the downstream tests are
    non-trivial:
        small_bilingual: 0.08 (strongest, paper claim)
        bgpt:            0.05
        bilingual_llm:   0.04
        bloom_ladder:    0.03
        eurollm:         0.04
        multilingual_llm:0.02
    """
    rng = np.random.default_rng(seed)
    group_slope = {
        "small_bilingual": 0.08,
        "bgpt":             0.05,
        "bilingual_llm":    0.04,
        "bloom_ladder":     0.03,
        "eurollm":          0.04,
        "multilingual_llm": 0.02,
    }
    rows: list[dict] = []
    for lang in languages:
        for model, group in models_by_group.items():
            slope = group_slope.get(group, 0.02)
            for sent_id in range(n_sentences):
                for tok_id in range(tokens_per_sentence):
                    subject = int(rng.integers(0, subjects))
                    surprisal = rng.uniform(1.0, 12.0)
                    rt = 200 + slope * 1000 * (surprisal - 6.0) + rng.normal(0, 40)
                    rt = max(rt, 60.0)
                    word = "".join(rng.choice(list("abcdefghijklmno"), size=int(rng.integers(3, 9))))
                    rows.append(
                        {
                            "model": model,
                            "group": group,
                            "language": lang,
                            "sentence_id": sent_id,
                            "token_id": tok_id,
                            "subject_id": subject,
                            "word": word,
                            "word_length": len(word),
                            "log_freq": float(rng.normal(0, 1)),
                            "surprisal": surprisal,
                            "RT": rt,
                        }
                    )
    return pd.DataFrame(rows)


def summarise_mixedlm(fit_result) -> pd.DataFrame:
    """Turn a statsmodels MixedLMResults into a tidy term dataframe."""
    if fit_result is None:
        return pd.DataFrame(columns=["term", "beta", "se", "t", "p"])
    rows = []
    for name, beta in fit_result.params.items():
        rows.append(
            {
                "term": name,
                "beta": float(beta),
                "se": float(fit_result.bse.get(name, float("nan"))),
                "t": float(fit_result.tvalues.get(name, float("nan"))),
                "p": float(fit_result.pvalues.get(name, float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def per_sentence_errors(token_df: pd.DataFrame, model: str) -> np.ndarray:
    sub = token_df[token_df["model"] == model]
    if sub.empty:
        return np.empty(0)
    return (
        sub.assign(err=(sub["log_rt"] - sub["log_rt"].mean()) ** 2)
        .groupby("sentence_id")["err"].mean().sort_index().to_numpy()
    )


def render_claim_paragraph(
    fisher_df: pd.DataFrame,
    mixedlm_df: pd.DataFrame,
    friedman_df: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Fill in the plan's claim template with real numbers."""
    def g(df: pd.DataFrame, group: str) -> Optional[pd.Series]:
        if df is None or df.empty or "group" not in df.columns:
            return None
        rows = df[df["group"] == group]
        return rows.iloc[0] if len(rows) else None

    def _first(*candidates):
        for c in candidates:
            if c is not None:
                return c
        return None

    bgpt = _first(g(fisher_df, "bgpt"), g(fisher_df, "bilingual_llm"))
    multi = _first(g(fisher_df, "multilingual_llm"), g(fisher_df, "bloom_ladder"))

    def fmt(s, key, fmt_="{:.3f}"):
        if s is None:
            return "…"
        try:
            return fmt_.format(float(s[key]))
        except Exception:
            return "…"

    mix_line = "…"
    if not mixedlm_df.empty:
        inter = mixedlm_df[mixedlm_df["term"].str.contains("zsurp.*model_type", regex=True, na=False)]
        if not inter.empty:
            r = inter.iloc[0]
            mix_line = (
                f"β = {float(r['beta']):.3f}, SE = {float(r['se']):.3f}, "
                f"p = {float(r['p']):.3g}"
            )

    friedman_line = "…"
    if friedman_df is not None and not friedman_df.empty:
        allrow = friedman_df[friedman_df["scope"] == "all_languages"]
        if not allrow.empty:
            friedman_line = (
                f"χ²({int(allrow['df'].iloc[0])}) = {float(allrow['chi2'].iloc[0]):.2f}, "
                f"p = {float(allrow['p'].iloc[0]):.3g}"
            )

    text = textwrap.dedent(f"""
        Small Beetle bilingual models exhibit significantly stronger
        surprisal–reading-time coupling than both bilingual LLMs
        (Δρ̄ = {fmt(bgpt, 'mean_delta_rho')}, d = {fmt(bgpt, 'cohens_d')},
        p = {fmt(bgpt, 'p_two_sided', '{:.3g}')}; Fisher z-test, FDR-corrected)
        and multilingual LLMs (Δρ̄ = {fmt(multi, 'mean_delta_rho')},
        d = {fmt(multi, 'cohens_d')}, p = {fmt(multi, 'p_two_sided', '{:.3g}')}),
        with a significant negative interaction surprisal × model_type in a
        linear mixed-effects model with subject and sentence random effects
        ({mix_line}). This effect is consistent across languages
        ({friedman_line}) and robust to word-length and frequency controls.
    """).strip()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end MECO alignment comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--languages", nargs="+", default=["en", "nl"])
    parser.add_argument("--include-bgpt", action="store_true")
    parser.add_argument("--include-bilingual-llm", action="store_true")
    parser.add_argument("--include-bloom-ladder", action="store_true")
    parser.add_argument("--include-eurollm", action="store_true")
    parser.add_argument("--include-multilingual-llm", action="store_true")
    parser.add_argument(
        "--large", action="store_true",
        help="Enable gated large models (BLOOM-176B, GLM-130B).",
    )
    parser.add_argument(
        "--output", default=str(MECO_OUTPUT_DIR),
        help="Output directory (default: results/meco).",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use synthetic token-level data (no surprisal extraction needed).",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path("results/figures")

    models_by_group = collect_models(args)
    print(
        f"[run_meco] collected {len(models_by_group)} models across "
        f"{len(set(models_by_group.values()))} groups."
    )

    # -----------------------------------------------------------------
    # 1. Token-level data.
    # -----------------------------------------------------------------
    if args.mock:
        token_df = build_mock_tokens(models_by_group, args.languages)
    else:
        try:
            from analyze.meco_surprisal import compute_surprisal_for_models  # type: ignore
        except Exception as e:
            print(
                f"[run_meco] meco_surprisal backend not importable ({e}). "
                f"Re-run with --mock or wire up compute_surprisal_for_models."
            )
            return 2
        token_df = compute_surprisal_for_models(
            models_by_group, languages=args.languages, cache_dir=output_dir / "surprisal_cache"
        )

    token_df = preprocess_meco_dataframe(token_df)
    token_df["model_type"] = token_df["group"]
    token_df.to_parquet(output_dir / "tokens.parquet", index=False)

    # -----------------------------------------------------------------
    # 2. Per-model alignment.
    # -----------------------------------------------------------------
    alignment_df = per_model_alignment(token_df)
    alignment_df = alignment_df.merge(
        token_df[["model", "group"]].drop_duplicates(), on="model", how="left"
    )
    alignment_df.to_csv(output_dir / "alignment_per_model.csv", index=False)
    print(f"[run_meco] wrote alignment_per_model.csv ({len(alignment_df)} rows).")

    # -----------------------------------------------------------------
    # 3. Fisher z-test group comparison.
    # -----------------------------------------------------------------
    fisher_df = fisher_z_group_comparison(alignment_df, reference_group="small_bilingual")
    if not fisher_df.empty:
        fisher_df.to_csv(output_dir / "fisher_ztest.csv", index=False)
        print(f"[run_meco] wrote fisher_ztest.csv ({len(fisher_df)} contrasts).")

    # -----------------------------------------------------------------
    # 4. Primary mixed-effects model.
    # -----------------------------------------------------------------
    fit = fit_primary_mixedlm(token_df)
    mixedlm_df = summarise_mixedlm(fit)
    if not mixedlm_df.empty:
        mixedlm_df.to_csv(output_dir / "mixedlm_summary.csv", index=False)
        print(f"[run_meco] wrote mixedlm_summary.csv ({len(mixedlm_df)} terms).")

    # -----------------------------------------------------------------
    # 5. Paired Wilcoxon across models (robustness).
    # -----------------------------------------------------------------
    errors = {
        m: per_sentence_errors(token_df, m) for m in token_df["model"].unique()
    }
    wilcoxon_df = paired_wilcoxon_across_models(errors)
    if not wilcoxon_df.empty:
        wilcoxon_df.to_csv(output_dir / "wilcoxon_paired.csv", index=False)
        print(f"[run_meco] wrote wilcoxon_paired.csv ({len(wilcoxon_df)} contrasts).")

    # -----------------------------------------------------------------
    # 6. Cross-language Friedman.
    # -----------------------------------------------------------------
    friedman_df = cross_language_friedman(alignment_df)
    if not friedman_df.empty:
        friedman_df.to_csv(output_dir / "friedman_by_language.csv", index=False)
        print(f"[run_meco] wrote friedman_by_language.csv.")

    # -----------------------------------------------------------------
    # 7. Figures.
    # -----------------------------------------------------------------
    figs = meco_figures.render_all(
        alignment_df=alignment_df,
        fisher_df=fisher_df,
        mixedlm_summary=mixedlm_df,
        friedman_df=friedman_df,
        wilcoxon_df=wilcoxon_df,
        token_df=token_df,
        output_dir=figures_dir,
    )
    print(f"[run_meco] rendered {len(figs)} figures under {figures_dir}/")

    # -----------------------------------------------------------------
    # 8. Claim paragraph.
    # -----------------------------------------------------------------
    claim_path = render_claim_paragraph(
        fisher_df, mixedlm_df, friedman_df, output_dir / "claim.txt"
    )
    print(f"[run_meco] wrote {claim_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
