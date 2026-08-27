"""
forgetting.py — Catastrophic Forgetting analysis for BeetleLM models.

Strategy
--------
Catastrophic forgetting is measured as the *degradation in L1 perplexity*
when moving from the monolingual baseline to a bilingual model.

  CF_score(model, lang) = mean_PPL(bilingual, lang) - mean_PPL(mono, lang)

A positive score means the bilingual model is *worse* on the L1 than the
mono baseline → evidence of catastrophic forgetting.
We also compute a log-ratio version:
  CF_log_ratio = log(mean_PPL_bilingual / mean_PPL_mono)

Focus languages: German (deu), Dutch (nld), Chinese (zho).

Outputs
-------
results/forgetting/
  forgetting_raw_<iso>.csv      — per-sentence delta NLL for every bilingual model
  forgetting_summary.csv        — mean CF_score + CF_log_ratio per model
  forgetting_by_type.csv        — mean CF_score aggregated by bilingual type
"""

from __future__ import annotations
from analyze import _path  # noqa: F401  — adds repo root to sys.path

from pathlib import Path

import numpy as np
import pandas as pd

from models import DUTCH_MODELS, GERMAN_MODELS, CHINESE_MODELS, get_bilingual_type
from ppl_utils import (
    GOLDFISH_TOKENS,
    LOAD_FAILURES,
    get_best_revision,
    load_flores_sentences,
    nll_to_ppl,
    score_sentences,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("results/forgetting")

FORGETTING_LANGS: dict[str, dict] = {
    "deu": {
        "mono":       "BeetleLM/beetlelm_deu_mono",
        "candidates": [m for m in GERMAN_MODELS if "mono" not in m],
    },
    "nld": {
        "mono":       "BeetleLM/beetlelm_nld_mono",
        "candidates": [m for m in DUTCH_MODELS if "mono" not in m],
    },
    "zho": {
        "mono":       "BeetleLM/beetlelm_zho_mono",
        "candidates": [m for m in CHINESE_MODELS if "mono" not in m],
    },
}


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_forgetting(
    iso_code: str,
    goldfish_tokens: int | None = GOLDFISH_TOKENS,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    For a single focus language, compute per-sentence forgetting scores.

    Returns a DataFrame with columns:
      sentence_id, sentence, repo, revision, bilingual_type,
      nll_mono, nll_bilingual, delta_nll, delta_ppl, cf_log_ratio

    Models that fail to load are skipped and recorded in LOAD_FAILURES.
    If the mono baseline fails to load, returns an empty DataFrame.
    """
    if iso_code not in FORGETTING_LANGS:
        print(
            f"[forgetting] No mono baseline configured for {iso_code!r} "
            f"(supported: {sorted(FORGETTING_LANGS)}) — skipping."
        )
        return pd.DataFrame()

    config     = FORGETTING_LANGS[iso_code]
    mono_repo  = config["mono"]
    candidates = config["candidates"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUTPUT_DIR / f"forgetting_raw_{iso_code}.csv"

    if cache_path.exists() and not overwrite:
        print(f"[forgetting] Loading cached {cache_path}")
        return pd.read_csv(cache_path)

    sentences = load_flores_sentences(iso_code)

    mono_rev  = get_best_revision(mono_repo)
    print(f"[forgetting] Scoring mono baseline: {mono_repo} @ {mono_rev}")
    mono_indexed = score_sentences(sentences, mono_repo, goldfish_tokens)

    if mono_indexed is None:
        print(f"[forgetting] SKIPPED {iso_code} — mono baseline {mono_repo} failed to load.")
        return pd.DataFrame()
    # score_sentences now returns [(sentence_id, nll), ...]; restore the flat
    # list aligned to the original `sentences` order.
    mono_nlls = [nll for _, nll in sorted(mono_indexed, key=lambda x: x[0])]

    rows: list[dict] = []
    for repo in candidates:
        revision = get_best_revision(repo)
        print(f"[forgetting] Scoring {repo} @ {revision} …")
        bi_indexed = score_sentences(sentences, repo, goldfish_tokens)

        if bi_indexed is None:
            print(f"[forgetting] SKIPPED {repo} — model failed to load.")
            continue
        bi_nlls = [nll for _, nll in sorted(bi_indexed, key=lambda x: x[0])]

        bil_type = get_bilingual_type(repo)
        for i, (sent, m_nll, b_nll) in enumerate(zip(sentences, mono_nlls, bi_nlls)):
            m_ppl = nll_to_ppl(m_nll)
            b_ppl = nll_to_ppl(b_nll)
            rows.append({
                "iso_code":       iso_code,
                "sentence_id":    i,
                "sentence":       sent,
                "repo":           repo,
                "revision":       revision,
                "bilingual_type": bil_type,
                "nll_mono":       m_nll,
                "nll_bilingual":  b_nll,
                "ppl_mono":       m_ppl,
                "ppl_bilingual":  b_ppl,
                "delta_nll":      b_nll - m_nll,
                "delta_ppl":      b_ppl - m_ppl,
                "cf_log_ratio":   (
                    np.log(b_ppl / m_ppl)
                    if (m_ppl > 0 and b_ppl < float("inf"))
                    else float("nan")
                ),
            })

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    print(f"  → saved {cache_path}")
    return df


def run_forgetting_analysis(
    lang_codes: list[str] | None = None,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run forgetting analysis for all focus languages.

    Returns (raw_df, summary_df, by_type_df).
    """
    if lang_codes is None:
        lang_codes = list(FORGETTING_LANGS.keys())

    frames = [compute_forgetting(c, overwrite=overwrite) for c in lang_codes]
    frames = [f for f in frames if not f.empty]

    if not frames:
        print("[forgetting] No data — all models may have failed to load.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    raw_df = pd.concat(frames, ignore_index=True)

    summary_df = (
        raw_df
        .groupby(["iso_code", "repo", "revision", "bilingual_type"])
        .agg(
            mean_delta_nll    = ("delta_nll",    "mean"),
            mean_delta_ppl    = ("delta_ppl",    "mean"),
            mean_cf_log_ratio = ("cf_log_ratio", "mean"),
            pct_worse         = ("delta_nll",    lambda x: (x > 0).mean() * 100),
            n_sentences       = ("delta_nll",    "count"),
        )
        .reset_index()
        .sort_values(["iso_code", "mean_delta_ppl"])
    )

    by_type_df = (
        raw_df
        .groupby(["iso_code", "bilingual_type"])
        .agg(
            mean_delta_ppl    = ("delta_ppl",    "mean"),
            mean_cf_log_ratio = ("cf_log_ratio", "mean"),
            pct_worse         = ("delta_nll",    lambda x: (x > 0).mean() * 100),
            n_models          = ("repo",         "nunique"),
        )
        .reset_index()
        .sort_values(["iso_code", "mean_delta_ppl"])
    )

    summary_path = OUTPUT_DIR / "forgetting_summary.csv"
    by_type_path = OUTPUT_DIR / "forgetting_by_type.csv"
    summary_df.to_csv(summary_path, index=False)
    by_type_df.to_csv(by_type_path, index=False)

    print(f"\n[forgetting] Summary  → {summary_path}")
    print(f"[forgetting] By-type  → {by_type_path}")
    print("\n--- Top 5 most forgetful models per language ---")
    print(
        summary_df
        .groupby("iso_code")
        .apply(lambda g: g.nlargest(5, "mean_delta_ppl"))
        [["iso_code", "repo", "revision", "bilingual_type", "mean_delta_ppl", "pct_worse"]]
        .to_string()
    )

    if LOAD_FAILURES:
        print(f"\n[forgetting] {'='*56}")
        print(f"[forgetting] SKIPPED {len(LOAD_FAILURES)} model(s) due to load errors:")
        for repo, reason in LOAD_FAILURES.items():
            print(f"  ✗  {repo}")
            print(f"     {reason}")
        print(f"[forgetting] {'='*56}")

    return raw_df, summary_df, by_type_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run catastrophic forgetting analysis.")
    parser.add_argument("--langs", nargs="+", choices=["deu", "nld", "zho"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_forgetting_analysis(lang_codes=args.langs, overwrite=args.overwrite)