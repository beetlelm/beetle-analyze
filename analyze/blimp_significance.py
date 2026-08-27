"""
blimp_significance.py — Directional, non-parametric, and mixed-effects
statistical tests for BLiMP / BLiMP-NL / ZhoBLiMP (and any future
per-phenomenon benchmark CSV).

Reads the existing results/{blimp_eng,blimp_nl,zhoblimp}_results.csv files
(produced by eval_alta_blimp.py and cli_eval.py) and produces, per task:

  results/blimp_significance/{task}_paired.csv             — Δx̄, SD, Cohen's d,
                                                              paired-t p (two-sided
                                                              primary + one-sided
                                                              exploratory),
                                                              paired-permutation p,
                                                              Bonferroni and BH-FDR.
  results/blimp_significance/{task}_friedman.json          — χ², df, p.
  results/blimp_significance/{task}_wilcoxon_pairwise.csv  — pair, W, p_raw, p_holm.
  results/blimp_significance/{task}_mixedlm_summary.csv    — β, SE, t, p.

All tests keep two-sided as the PRIMARY test; one-sided p-values are
reported alongside and explicitly labelled as exploratory per the approved
plan.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy import stats as scipy_stats

try:
    from statsmodels.stats.multitest import multipletests
except ImportError as e:
    raise SystemExit(
        "statsmodels is required for blimp_significance — install via "
        "`pip install statsmodels`."
    ) from e


logger = logging.getLogger("blimp_significance")
if not logger.handlers:
    import sys
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# Effect size helpers
# ─────────────────────────────────────────────────────────────────────────────

def cohens_d_pooled(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d (independent samples)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return 0.0 if pooled == 0 else float((a.mean() - b.mean()) / pooled)


def cohens_dz(diffs: np.ndarray) -> float:
    """Paired Cohen's d_z."""
    diffs = np.asarray(diffs, dtype=float)
    if diffs.size < 2:
        return 0.0
    sd = diffs.std(ddof=1)
    return 0.0 if sd == 0 else float(diffs.mean() / sd)


# ─────────────────────────────────────────────────────────────────────────────
# Core statistics
# ─────────────────────────────────────────────────────────────────────────────

def score_differences(
    bil: pd.DataFrame,
    mono: pd.DataFrame,
    *,
    phenomenon_col: str = "phenomenon",
    language_col: Optional[str] = "eval_language",
    score_col: str = "accuracy",
) -> pd.DataFrame:
    """
    Compute per-language Δx̄, SD of Δ, and Cohen's d (pooled) between
    bilingual and monolingual accuracies, matched by ``phenomenon_col``.

    When ``language_col`` is ``None`` the comparison is pooled across
    languages (single row).
    """
    if language_col is None or language_col not in bil.columns:
        bil = bil.assign(_lang="all")
        mono = mono.assign(_lang="all")
        language_col = "_lang"

    rows: list[dict] = []
    for lang, bsub in bil.groupby(language_col):
        msub = mono[mono[language_col] == lang]
        if msub.empty:
            continue
        merged = bsub.merge(
            msub, on=phenomenon_col, suffixes=("_bil", "_mono"),
        )
        if merged.empty:
            continue
        diffs = merged[f"{score_col}_bil"].to_numpy() - merged[f"{score_col}_mono"].to_numpy()
        rows.append(
            {
                "language": lang,
                "n_phenomena": int(len(diffs)),
                "mean_delta": float(diffs.mean()),
                "std_delta": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
                "cohens_d": cohens_d_pooled(
                    merged[f"{score_col}_bil"].to_numpy(),
                    merged[f"{score_col}_mono"].to_numpy(),
                ),
                "cohens_dz": cohens_dz(diffs),
            }
        )
    return pd.DataFrame(rows)


def paired_ttests(
    diffs: np.ndarray,
    *,
    alternative: str = "two-sided",
) -> float:
    """Paired t-test on a vector of per-phenomenon differences."""
    diffs = np.asarray(diffs, dtype=float)
    if diffs.size < 2:
        return float("nan")
    _, p = scipy_stats.ttest_1samp(diffs, popmean=0.0, alternative=alternative)
    return float(p)


def paired_permutation(
    a: Sequence[float],
    b: Sequence[float],
    *,
    alternative: str = "two-sided",
    n_resamples: int = 9999,
    seed: int = 0,
) -> float:
    """Distribution-free paired permutation (sign-flip) test."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    diffs = a - b
    observed = diffs.mean()
    n = diffs.size

    if n <= 20:
        # Exact: enumerate 2^n sign patterns.
        extreme = 0
        total = 2 ** n
        for mask in range(total):
            signs = np.array([1 if (mask >> k) & 1 else -1 for k in range(n)])
            m = (signs * diffs).mean()
            if alternative == "two-sided" and abs(m) >= abs(observed):
                extreme += 1
            elif alternative == "greater" and m >= observed:
                extreme += 1
            elif alternative == "less" and m <= observed:
                extreme += 1
        return extreme / total

    # Monte-Carlo.
    count = 0
    for _ in range(n_resamples):
        signs = rng.choice([-1, 1], size=n)
        m = (signs * diffs).mean()
        if alternative == "two-sided":
            if abs(m) >= abs(observed):
                count += 1
        elif alternative == "greater":
            if m >= observed:
                count += 1
        elif alternative == "less":
            if m <= observed:
                count += 1
    return (count + 1) / (n_resamples + 1)


def correct_pvalues(pvals: Sequence[float]) -> Dict[str, np.ndarray]:
    """Return both Bonferroni and BH-FDR corrected arrays."""
    pvals = np.asarray(list(pvals), dtype=float)
    _, p_bonf, _, _ = multipletests(pvals, method="bonferroni")
    _, p_fdr,  _, _ = multipletests(pvals, method="fdr_bh")
    return {"bonferroni": p_bonf, "fdr_bh": p_fdr}


def friedman_across_models(
    accuracy_matrix: pd.DataFrame,
) -> Dict[str, float]:
    """
    Friedman χ² across k models on n phenomena. ``accuracy_matrix`` has one
    row per phenomenon and one column per model.
    """
    mat = accuracy_matrix.dropna(how="any")
    if mat.shape[0] < 3 or mat.shape[1] < 3:
        return {"chi2": float("nan"), "df": float("nan"), "p": float("nan"),
                "n_phenomena": int(mat.shape[0]), "n_models": int(mat.shape[1])}
    stat, p = scipy_stats.friedmanchisquare(*[mat[c].to_numpy() for c in mat.columns])
    return {
        "chi2": float(stat),
        "df": int(mat.shape[1] - 1),
        "p": float(p),
        "n_phenomena": int(mat.shape[0]),
        "n_models": int(mat.shape[1]),
    }


def pairwise_wilcoxon_holm(
    accuracy_matrix: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank with Holm-Bonferroni correction across
    all model columns of ``accuracy_matrix`` (phenomena are the paired
    observations).
    """
    mat = accuracy_matrix.dropna(how="any")
    models = list(mat.columns)
    rows: list[dict] = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            va = mat[a].to_numpy(); vb = mat[b].to_numpy()
            try:
                W, p = scipy_stats.wilcoxon(va, vb, alternative="two-sided")
            except ValueError:
                # All differences zero: skip.
                continue
            rows.append(
                {
                    "model_a": a, "model_b": b,
                    "W": float(W), "p_raw": float(p),
                    "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        _, p_adj, _, _ = multipletests(df["p_raw"].to_numpy(), alpha=alpha, method="holm")
        df["p_holm"] = p_adj
    return df


def fit_blimp_mixedlm(
    df: pd.DataFrame,
    *,
    accuracy_col: str = "accuracy",
    training_type_col: str = "training_type",
    english_status_col: str = "english_status",
    phenomenon_col: str = "phenomenon",
    model_col: Optional[str] = None,
):
    """
    Primary mixed-effects model:
        accuracy ~ training_type * english_status + (1 | phenomenon)

    If ``model_col`` is supplied, a sensitivity-analysis version with
    ``(1 | model)`` plus a phenomenon variance component is attempted.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        logger.warning("statsmodels not available; skipping mixedlm.")
        return None, None

    needed = {accuracy_col, training_type_col, english_status_col, phenomenon_col}
    missing = needed - set(df.columns)
    if missing:
        logger.warning("fit_blimp_mixedlm: missing columns %s", missing)
        return None, None
    data = df.dropna(subset=list(needed))
    if len(data) < 20:
        return None, None

    formula = (
        f"{accuracy_col} ~ C({training_type_col}) * C({english_status_col})"
    )
    try:
        primary = smf.mixedlm(formula, data=data, groups=data[phenomenon_col]).fit(
            reml=True, method="lbfgs"
        )
    except Exception as e:
        logger.warning("Primary mixedlm failed: %s", e)
        primary = None

    sensitivity = None
    if model_col and model_col in data.columns:
        try:
            sensitivity = smf.mixedlm(
                formula,
                data=data,
                groups=data[model_col],
                vc_formula={phenomenon_col: f"0 + C({phenomenon_col})"},
            ).fit(reml=True, method="lbfgs")
        except Exception as e:
            logger.warning("Sensitivity mixedlm failed: %s", e)

    return primary, sensitivity


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def analyse_task(
    task_name: str,
    bilingual_csv: Path,
    monolingual_csv: Path,
    *,
    output_dir: Path,
    score_col: str = "accuracy",
    phenomenon_col: str = "phenomenon",
    language_col: Optional[str] = "eval_language",
    model_col: str = "model",
) -> Dict[str, Path]:
    """Run the full BLiMP significance pipeline for a single task."""
    out_paths: Dict[str, Path] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    bil = pd.read_csv(bilingual_csv)
    mono = pd.read_csv(monolingual_csv)

    # 1. Directional paired summary per language.
    diffs_df = score_differences(
        bil, mono,
        phenomenon_col=phenomenon_col,
        language_col=language_col,
        score_col=score_col,
    )
    if not diffs_df.empty:
        # Attach paired-t (two-sided primary + one-sided exploratory) and
        # permutation p-values, then correct.
        p_two, p_up, p_down, p_perm = [], [], [], []
        for _, row in diffs_df.iterrows():
            lang = row["language"]
            if language_col and language_col in bil.columns:
                bsub = bil[bil[language_col] == lang]
                msub = mono[mono[language_col] == lang]
            else:
                bsub, msub = bil, mono
            merged = bsub.merge(msub, on=phenomenon_col, suffixes=("_b", "_m"))
            a = merged[f"{score_col}_b"].to_numpy()
            b = merged[f"{score_col}_m"].to_numpy()
            d = a - b
            p_two.append(paired_ttests(d, alternative="two-sided"))
            p_up.append(paired_ttests(d, alternative="greater"))
            p_down.append(paired_ttests(d, alternative="less"))
            p_perm.append(paired_permutation(a, b, alternative="two-sided"))
        diffs_df["p_two_sided"] = p_two
        diffs_df["p_one_sided_greater"] = p_up   # exploratory
        diffs_df["p_one_sided_less"] = p_down    # exploratory
        diffs_df["p_permutation"] = p_perm
        corr = correct_pvalues([p for p in p_two if not pd.isna(p)] or [1.0])
        # Align back (only over rows with valid p_two).
        valid = ~diffs_df["p_two_sided"].isna()
        diffs_df.loc[valid, "p_bonferroni"] = corr["bonferroni"]
        diffs_df.loc[valid, "p_fdr_bh"] = corr["fdr_bh"]
    out_paths["paired"] = output_dir / f"{task_name}_paired.csv"
    diffs_df.to_csv(out_paths["paired"], index=False)

    # 2. Friedman across models (phenomena as blocks).
    if model_col in bil.columns:
        pivot = bil.pivot_table(
            index=phenomenon_col, columns=model_col, values=score_col, aggfunc="mean"
        )
        friedman = friedman_across_models(pivot)
        out_paths["friedman"] = output_dir / f"{task_name}_friedman.json"
        out_paths["friedman"].write_text(json.dumps(friedman, indent=2))

        # 3. Pairwise Wilcoxon + Holm.
        wilcox = pairwise_wilcoxon_holm(pivot)
        out_paths["wilcoxon"] = output_dir / f"{task_name}_wilcoxon_pairwise.csv"
        wilcox.to_csv(out_paths["wilcoxon"], index=False)

    # 4. Mixed-effects.
    mixed_df = bil.copy()
    # Heuristic derivation of factors from the canonical column naming used
    # in blimp_eng_results.csv / blimp_nl_results.csv:
    if "bilingual_type" in mixed_df.columns:
        mixed_df["training_type"] = mixed_df["bilingual_type"].apply(
            lambda s: "simultaneous" if "simultaneous" in str(s)
            else ("sequential" if "sequential" in str(s) else "other")
        )
    if "lang_pair" in mixed_df.columns:
        mixed_df["english_status"] = mixed_df["lang_pair"].apply(
            lambda s: "L1" if str(s).startswith("en") else "L2"
        )

    primary, sensitivity = fit_blimp_mixedlm(
        mixed_df,
        accuracy_col=score_col,
        phenomenon_col=phenomenon_col,
        model_col=model_col,
    )

    def summarise(fit) -> pd.DataFrame:
        if fit is None:
            return pd.DataFrame(columns=["term", "beta", "se", "t", "p"])
        rows = [
            {
                "term": name,
                "beta": float(beta),
                "se": float(fit.bse.get(name, float("nan"))),
                "t": float(fit.tvalues.get(name, float("nan"))),
                "p": float(fit.pvalues.get(name, float("nan"))),
            }
            for name, beta in fit.params.items()
        ]
        return pd.DataFrame(rows)

    primary_df = summarise(primary)
    out_paths["mixedlm"] = output_dir / f"{task_name}_mixedlm_summary.csv"
    primary_df.to_csv(out_paths["mixedlm"], index=False)

    if sensitivity is not None:
        sens_df = summarise(sensitivity)
        out_paths["mixedlm_sens"] = output_dir / f"{task_name}_mixedlm_sensitivity.csv"
        sens_df.to_csv(out_paths["mixedlm_sens"], index=False)

    return out_paths
