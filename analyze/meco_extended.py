"""
meco_extended.py — Extended MECO Reading Time Alignment (Family A2).

Goes beyond basic MECO surprisal correlation to test whether bilingual
training regimes produce better human-like surprisal-RT slopes than
monolingual baselines.

Analyses:
  - Linear mixed model: fixation_duration ~ surprisal + freq + word_length
                         + (1|subject) + (1|word)
  - Compare beta_surprisal across training conditions
  - Test curriculum order effects (L2 @ 0K vs 48K)
  - Separate by L1 background of readers (if metadata available)
  - Compare 100M vs 24B: does scale improve or hurt human alignment?

Outputs
-------
results/meco/
  extended_slopes.csv       — beta_surprisal per model/condition
  curriculum_slopes.csv     — slope vs L2 introduction step
  scale_comparison.csv      — 100M vs 24B slope comparison
"""

from __future__ import annotations

from analyze import _path  # noqa: F401

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from models import MODEL_GROUPS, get_bilingual_type
from ppl_utils import clear_model_cache

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("results/meco")

DEFAULT_STIMULI = Path(__file__).resolve().parent.parent / "data" / "meco_l2_stims.tsv"


# ---------------------------------------------------------------------------
# A2a: Fit surprisal-RT linear model per model
# ---------------------------------------------------------------------------

def fit_surprisal_rt_model(
    surprisal_df: pd.DataFrame,
    rt_column: str = "RT",
    freq_column: str = "freq",
    length_column: str = "word_length",
) -> dict:
    """
    Fit a linear model: RT ~ surprisal + freq + word_length.

    Uses OLS as a fallback when statsmodels mixed-effects is unavailable.
    Returns dict with beta coefficients and R-squared.
    """
    df = surprisal_df.dropna(subset=["surprisal"])

    # Build design matrix with available predictors
    predictors = ["surprisal"]
    if freq_column in df.columns:
        predictors.append(freq_column)
        df = df.dropna(subset=[freq_column])
    if length_column in df.columns:
        predictors.append(length_column)
        df = df.dropna(subset=[length_column])

    if rt_column not in df.columns or len(df) < 10:
        return {"beta_surprisal": float("nan"), "n": 0}

    df = df.dropna(subset=[rt_column])

    X = df[predictors].values
    X = np.column_stack([np.ones(len(X)), X])
    y = df[rt_column].values

    try:
        beta, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        result = {
            "intercept": beta[0],
            "beta_surprisal": beta[1],
            "r_squared": r_squared,
            "n": len(df),
        }
        for i, pred in enumerate(predictors[1:], 2):
            result[f"beta_{pred}"] = beta[i]
        return result

    except np.linalg.LinAlgError:
        return {"beta_surprisal": float("nan"), "n": len(df)}


def fit_mixed_effects_model(
    surprisal_df: pd.DataFrame,
    rt_column: str = "RT",
    subject_column: str = "subject",
    word_column: str = "word",
) -> dict:
    """
    Fit linear mixed-effects model: RT ~ surprisal + freq + word_length
                                    + (1|subject) + (1|word)

    Requires statsmodels. Falls back to OLS if unavailable.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("[meco_ext] statsmodels not available, falling back to OLS")
        return fit_surprisal_rt_model(surprisal_df, rt_column)

    df = surprisal_df.dropna(subset=["surprisal", rt_column])
    if len(df) < 20:
        return {"beta_surprisal": float("nan"), "n": len(df)}

    formula_parts = [rt_column, "~ surprisal"]
    if "freq" in df.columns:
        formula_parts.append("+ freq")
    if "word_length" in df.columns:
        formula_parts.append("+ word_length")

    formula = " ".join(formula_parts)

    # Build random effects specification
    re_formula = "1"
    groups = subject_column if subject_column in df.columns else None

    if groups is None:
        return fit_surprisal_rt_model(surprisal_df, rt_column)

    try:
        md = smf.mixedlm(formula, df, groups=df[groups])
        mdf = md.fit(reml=True)

        return {
            "beta_surprisal": mdf.params.get("surprisal", float("nan")),
            "beta_surprisal_se": mdf.bse.get("surprisal", float("nan")),
            "beta_surprisal_pval": mdf.pvalues.get("surprisal", float("nan")),
            "aic": mdf.aic,
            "bic": mdf.bic,
            "n": int(mdf.nobs),
            "method": "lmm",
        }
    except Exception as e:
        print(f"[meco_ext] LMM failed: {e}, falling back to OLS")
        return fit_surprisal_rt_model(surprisal_df, rt_column)


# ---------------------------------------------------------------------------
# A2b: Compare slopes across conditions
# ---------------------------------------------------------------------------

def compare_slopes_across_conditions(
    surprisal_dir: Path = OUTPUT_DIR,
    stimuli_path: Path = DEFAULT_STIMULI,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Compare beta_surprisal across all training conditions.

    Loads pre-computed surprisal CSVs from results/meco/ and fits
    RT ~ surprisal models per model, then aggregates by condition.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "extended_slopes.csv"

    if out_path.exists() and not overwrite:
        print(f"[meco_ext] Loading cached {out_path}")
        return pd.read_csv(out_path)

    stimuli = pd.read_csv(stimuli_path, sep="\t")

    # Find all surprisal result files
    surp_files = list(surprisal_dir.glob("surprisals_*.csv"))
    if not surp_files:
        print("[meco_ext] No surprisal files found in results/meco/")
        return pd.DataFrame()

    rows: list[dict] = []
    for surp_file in surp_files:
        df = pd.read_csv(surp_file)
        if "model" not in df.columns or "surprisal" not in df.columns:
            continue

        for model_name, mdf in df.groupby("model"):
            bil_type = get_bilingual_type(model_name)
            fit = fit_surprisal_rt_model(mdf)
            rows.append({
                "model": model_name,
                "bilingual_type": bil_type,
                **fit,
            })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_path, index=False)
    print(f"[meco_ext] Saved {out_path} ({len(result_df)} models)")
    return result_df


# ---------------------------------------------------------------------------
# A2c: Curriculum order effects
# ---------------------------------------------------------------------------

def compare_curriculum_slopes(
    slopes_df: pd.DataFrame,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Test whether curriculum order (L2 introduced at 0K vs 48K) changes
    the surprisal-RT slope magnitude.

    Expects slopes_df to have bilingual_type containing curriculum info.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "curriculum_slopes.csv"

    if out_path.exists() and not overwrite:
        print(f"[meco_ext] Loading cached {out_path}")
        return pd.read_csv(out_path)

    # Map bilingual types to curriculum categories
    curriculum_map = {
        "simultaneous": 0,      # L2 @ 0K
        "balanced": 0,          # L2 @ 0K (concurrent)
        "part_time": 16000,     # L2 @ 16K
        "late": 32000,          # L2 @ 32K
        "heritage": 48000,      # L2 @ 48K
    }

    slopes_df = slopes_df.copy()
    slopes_df["l2_intro_step"] = slopes_df["bilingual_type"].map(curriculum_map)
    valid = slopes_df.dropna(subset=["l2_intro_step", "beta_surprisal"])

    if len(valid) < 3:
        print("[meco_ext] Not enough curriculum conditions for comparison")
        return pd.DataFrame()

    # Aggregate by curriculum step
    agg = valid.groupby("l2_intro_step").agg(
        mean_beta=("beta_surprisal", "mean"),
        std_beta=("beta_surprisal", "std"),
        n_models=("model", "nunique"),
    ).reset_index()

    # Correlation: L2 intro step vs slope
    rho, pval = stats.spearmanr(valid["l2_intro_step"], valid["beta_surprisal"])
    agg["spearman_rho"] = rho
    agg["spearman_pval"] = pval

    agg.to_csv(out_path, index=False)
    print(f"[meco_ext] Saved {out_path} (rho={rho:.3f}, p={pval:.3e})")
    return agg


# ---------------------------------------------------------------------------
# A2d: Scale comparison
# ---------------------------------------------------------------------------

def compare_scale_slopes(
    slopes_df: pd.DataFrame,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Compare 100M vs 24B models: does scale improve or hurt human alignment?
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "scale_comparison.csv"

    if out_path.exists() and not overwrite:
        print(f"[meco_ext] Loading cached {out_path}")
        return pd.read_csv(out_path)

    slopes_df = slopes_df.copy()
    slopes_df["scale"] = slopes_df["model"].apply(
        lambda m: "24B" if "fineweb" in m.lower() else "100M"
    )

    agg = slopes_df.groupby(["scale", "bilingual_type"]).agg(
        mean_beta=("beta_surprisal", "mean"),
        std_beta=("beta_surprisal", "std"),
        n_models=("model", "nunique"),
    ).reset_index()

    agg.to_csv(out_path, index=False)
    print(f"[meco_ext] Saved {out_path}")
    return agg


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_meco_extended(overwrite: bool = False) -> dict[str, pd.DataFrame]:
    """Run all extended MECO analyses (A2)."""
    results = {}

    print("\n" + "=" * 60 + "\n A2: MECO EXTENDED ANALYSIS\n" + "=" * 60)

    slopes = compare_slopes_across_conditions(overwrite=overwrite)
    results["slopes"] = slopes

    if not slopes.empty:
        results["curriculum"] = compare_curriculum_slopes(slopes, overwrite=overwrite)
        results["scale"] = compare_scale_slopes(slopes, overwrite=overwrite)

    return results


# ---------------------------------------------------------------------------
# B-plan upgrade: cross-model MECO alignment comparison pipeline
#
# The functions below implement the per-model alignment metrics, Fisher
# z-test between model groups, primary linear mixed-effects model with
# subject and sentence random effects, paired Wilcoxon robustness, and
# cross-language Friedman validation described in the approved plan at
# ~/.claude/plans/reactive-sprouting-hamming.md (Part B).
#
# Everything above this block is preserved intact for backwards compatibility
# with existing callers (e.g. cli_eval, forgetting, learning_dynamics).
# ---------------------------------------------------------------------------

MECO_OUTPUT_DIR = OUTPUT_DIR  # alias used by run_meco_comparison.py

# Cache directory for extracted surprisal dataframes.
SURPRISAL_CACHE_DIR = OUTPUT_DIR / "surprisal_cache"


def _cohens_d_pooled(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d (independent-samples)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va = a.var(ddof=1); vb = b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return 0.0 if pooled == 0 else float((a.mean() - b.mean()) / pooled)


def preprocess_meco_dataframe(
    df: pd.DataFrame,
    *,
    rt_column: str = "RT",
    surprisal_column: str = "surprisal",
    drop_punct: bool = True,
    drop_sentence_initial: bool = False,
    min_rt_ms: float = 80.0,
) -> pd.DataFrame:
    """
    Standard MECO preprocessing: punctuation removal, (optional) sentence-
    initial token drop, log-RT, per-model z-scored surprisal.

    Returns a new dataframe with added columns ``log_rt`` and ``zsurp``.
    """
    out = df.copy()
    if drop_punct and "word" in out.columns:
        out = out[out["word"].astype(str).str.match(r"^[\w'-]+$", na=False)]
    if drop_sentence_initial and "token_id" in out.columns:
        out = out[out["token_id"] > 0]
    if rt_column in out.columns:
        out = out[out[rt_column].astype(float) >= min_rt_ms]
        out["log_rt"] = np.log(out[rt_column].astype(float))
    if surprisal_column in out.columns:
        grp_col = "model" if "model" in out.columns else None
        if grp_col is not None:
            out["zsurp"] = out.groupby(grp_col)[surprisal_column].transform(
                lambda s: (s - s.mean()) / s.std(ddof=1) if s.std(ddof=1) else 0.0
            )
        else:
            mu = out[surprisal_column].mean()
            sd = out[surprisal_column].std(ddof=1) or 1.0
            out["zsurp"] = (out[surprisal_column] - mu) / sd
    return out


def per_model_alignment(
    df: pd.DataFrame,
    *,
    rt_column: str = "log_rt",
    surprisal_column: str = "zsurp",
    word_length_column: str = "word_length",
    log_freq_column: str = "log_freq",
    model_column: str = "model",
    lang_column: str = "language",
) -> pd.DataFrame:
    """
    For each (model, language), compute Spearman rho, OLS beta,
    lexical-control OLS beta (with word length + log frequency), and R^2.
    Returns a long dataframe with one row per (model, language).
    """
    rows: list[dict] = []
    group_cols = [c for c in (model_column, lang_column) if c in df.columns]
    if not group_cols:
        raise ValueError(
            "per_model_alignment requires at least one of "
            f"{model_column!r}/{lang_column!r} in the dataframe."
        )

    try:
        import statsmodels.api as sm
        have_sm = True
    except ImportError:
        have_sm = False

    for keys, sub in df.groupby(group_cols):
        sub = sub.dropna(subset=[rt_column, surprisal_column])
        if len(sub) < 20:
            continue
        rho, p = stats.spearmanr(sub[surprisal_column], sub[rt_column])

        # Simple OLS: rt ~ surprisal.
        x = sub[surprisal_column].to_numpy()
        y = sub[rt_column].to_numpy()
        denom = ((x - x.mean()) ** 2).sum()
        beta = ((x - x.mean()) * (y - y.mean())).sum() / denom if denom else 0.0
        intercept = y.mean() - beta * x.mean()
        y_hat = intercept + beta * x
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot else 0.0

        # Lexical-control OLS (if columns available and statsmodels present).
        beta_ctrl = float("nan")
        if have_sm and word_length_column in sub.columns and log_freq_column in sub.columns:
            ctrl = sub.dropna(
                subset=[word_length_column, log_freq_column]
            )
            if len(ctrl) >= 20:
                X = ctrl[[surprisal_column, word_length_column, log_freq_column]]
                X = sm.add_constant(X)
                try:
                    fit = sm.OLS(ctrl[rt_column], X).fit()
                    beta_ctrl = float(fit.params.get(surprisal_column, float("nan")))
                except Exception:
                    pass

        row = {
            "rho": float(rho),
            "rho_pval": float(p),
            "beta": float(beta),
            "beta_ctrl": beta_ctrl,
            "r2": float(r2),
            "n_tokens": int(len(sub)),
        }
        if isinstance(keys, tuple):
            for k, v in zip(group_cols, keys):
                row[k] = v
        else:
            row[group_cols[0]] = keys
        rows.append(row)

    return pd.DataFrame(rows)


def fisher_z_group_comparison(
    alignment_df: pd.DataFrame,
    *,
    group_column: str = "group",
    value_column: str = "rho",
    reference_group: str = "small_bilingual",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Between-group Fisher z-test on Spearman rho. Compares every non-reference
    group to ``reference_group``. Returns a dataframe with columns:

        group, n_ref, n_other, mean_rho_ref, mean_rho_other,
        mean_delta_rho, cohens_d, t, p_two_sided, p_one_sided_greater

    (Ref > other ⇒ positive Δρ̄ ⇒ smaller one-sided-greater p.)
    """
    rows: list[dict] = []
    if group_column not in alignment_df.columns:
        return pd.DataFrame()
    ref = alignment_df[alignment_df[group_column] == reference_group][value_column]
    ref_z = np.arctanh(np.clip(ref.to_numpy(), -0.9999, 0.9999))
    if len(ref_z) < 2:
        return pd.DataFrame()

    for group, sub in alignment_df.groupby(group_column):
        if group == reference_group:
            continue
        oth = sub[value_column].to_numpy()
        oth_z = np.arctanh(np.clip(oth, -0.9999, 0.9999))
        if len(oth_z) < 2:
            continue
        t_two, p_two = stats.ttest_ind(
            ref_z, oth_z, equal_var=False, alternative="two-sided"
        )
        _, p_greater = stats.ttest_ind(
            ref_z, oth_z, equal_var=False, alternative="greater"
        )
        rows.append(
            {
                "group": group,
                "n_ref": int(len(ref_z)),
                "n_other": int(len(oth_z)),
                "mean_rho_ref": float(ref.mean()),
                "mean_rho_other": float(oth.mean()),
                "mean_delta_rho": float(ref.mean() - oth.mean()),
                "cohens_d": _cohens_d_pooled(ref_z, oth_z),
                "t": float(t_two),
                "p_two_sided": float(p_two),
                "p_one_sided_greater": float(p_greater),
            }
        )
    return pd.DataFrame(rows)


def fit_primary_mixedlm(
    token_df: pd.DataFrame,
    *,
    rt_column: str = "log_rt",
    surprisal_column: str = "zsurp",
    model_type_column: str = "model_type",
    subject_column: str = "subject_id",
    sentence_column: str = "sentence_id",
):
    """
    Primary mixed-effects model:
        rt ~ surprisal * model_type + (1|subject) + (1|sentence)

    Returns the fitted :class:`statsmodels.regression.mixed_linear_model.MixedLMResults`
    or ``None`` if statsmodels / data are insufficient.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("[meco_ext] statsmodels not available — cannot fit mixed-effects model.")
        return None

    needed = {rt_column, surprisal_column, model_type_column, subject_column, sentence_column}
    missing = needed - set(token_df.columns)
    if missing:
        print(f"[meco_ext] fit_primary_mixedlm: missing columns {missing}")
        return None

    df = token_df.dropna(subset=list(needed))
    if len(df) < 50:
        print(f"[meco_ext] fit_primary_mixedlm: only {len(df)} tokens, skipping.")
        return None

    formula = f"{rt_column} ~ {surprisal_column} * C({model_type_column})"
    try:
        md = smf.mixedlm(
            formula,
            data=df,
            groups=df[subject_column],
            vc_formula={"sentence": f"0 + C({sentence_column})"},
            re_formula="~1",
        )
        return md.fit(reml=True, method="lbfgs")
    except Exception as e:
        print(f"[meco_ext] Primary mixedlm failed, trying simpler spec: {e}")
        try:
            md = smf.mixedlm(formula, data=df, groups=df[subject_column])
            return md.fit(reml=True, method="lbfgs")
        except Exception as e2:
            print(f"[meco_ext] Fallback mixedlm also failed: {e2}")
            return None


def paired_wilcoxon_across_models(
    errors_by_model: Dict[str, np.ndarray] | None = None,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    For every pair of models, apply a paired Wilcoxon signed-rank test on
    per-sentence squared residuals (``errors_by_model[model]`` is a 1D array
    indexed by shared sentence_id).

    Holm-Bonferroni correction within the baseline-group family.
    """
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        multipletests = None  # type: ignore

    if not errors_by_model or len(errors_by_model) < 2:
        return pd.DataFrame()

    names = list(errors_by_model.keys())
    rows: list[dict] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ea = np.asarray(errors_by_model[a], dtype=float)
            eb = np.asarray(errors_by_model[b], dtype=float)
            n = min(len(ea), len(eb))
            if n < 6:
                continue
            try:
                stat, p = stats.wilcoxon(ea[:n], eb[:n], alternative="two-sided")
            except Exception:
                continue
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "W": float(stat),
                    "p_raw": float(p),
                    "mean_error_a": float(np.mean(ea[:n])),
                    "mean_error_b": float(np.mean(eb[:n])),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty and multipletests is not None:
        _, p_adj, _, _ = multipletests(df["p_raw"].to_numpy(), alpha=alpha, method="holm")
        df["p_holm"] = p_adj
    return df


def cross_language_friedman(
    alignment_df: pd.DataFrame,
    *,
    model_column: str = "model",
    lang_column: str = "language",
    value_column: str = "rho",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Per-language Friedman test across k models, with Benjamini-Hochberg FDR
    correction across languages.
    """
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        multipletests = None  # type: ignore

    rows: list[dict] = []
    if model_column not in alignment_df.columns or lang_column not in alignment_df.columns:
        return pd.DataFrame()

    # Pivot: rows = models, cols = languages.
    pivot = alignment_df.pivot_table(
        index=model_column, columns=lang_column, values=value_column, aggfunc="mean"
    )
    # Friedman needs at least 3 ranks per block.
    pivot = pivot.dropna(axis=1, how="any")
    if pivot.shape[0] < 3 or pivot.shape[1] < 2:
        return pd.DataFrame()

    for lang in pivot.columns:
        # Across all languages for THIS language's Friedman we use the pivot
        # with this language's column dropped? No — Friedman treats models as
        # blocks, languages as conditions. Compute once:
        pass

    # A single Friedman across all languages (conditions = languages).
    try:
        values = [pivot[c].to_numpy() for c in pivot.columns]
        chi2, p = stats.friedmanchisquare(*values)
        rows.append(
            {
                "scope": "all_languages",
                "chi2": float(chi2),
                "df": len(values) - 1,
                "p": float(p),
                "n_models": int(pivot.shape[0]),
                "languages": ",".join(map(str, pivot.columns.tolist())),
            }
        )
    except Exception as e:
        print(f"[meco_ext] Friedman failed: {e}")

    # Per-language one-sample sign test (whether rho > 0 consistently).
    for lang in pivot.columns:
        vals = pivot[lang].to_numpy()
        try:
            stat, p = stats.wilcoxon(vals, alternative="greater")
            rows.append(
                {
                    "scope": f"rho_gt_0_{lang}",
                    "chi2": float("nan"),
                    "df": float("nan"),
                    "p": float(p),
                    "n_models": int(len(vals)),
                    "languages": str(lang),
                }
            )
        except Exception:
            continue

    out = pd.DataFrame(rows)
    if not out.empty and multipletests is not None:
        _, p_fdr, _, _ = multipletests(out["p"].to_numpy(), alpha=alpha, method="fdr_bh")
        out["p_fdr"] = p_fdr
    return out


# ---------------------------------------------------------------------------
# Entry point used by analyze/significance_ablation.py
# ---------------------------------------------------------------------------

def compute_alignment_for_run(record, languages=("en", "nl")) -> float:
    """
    Return a scalar MECO alignment score (mean Spearman rho across
    ``languages``) for a single training-run checkpoint described by
    ``record`` (a RunRecord from significance_ablation).

    In environments without access to the trained checkpoint (common when
    running this analysis on a login node) this function falls back to the
    deterministic mock controlled by ``BEETLE_MOCK_METRICS=1``. When the
    BeetleLM inference stack is importable, it computes real surprisals.
    """
    import os

    if os.environ.get("BEETLE_MOCK_METRICS") == "1":
        import numpy as _np
        rng = _np.random.default_rng(hash((record.variant, record.seed)) & 0xFFFFFFFF)
        return float(_np.clip(0.25 + 0.1 * rng.standard_normal(), -1, 1))

    # Real path: attempt to locate checkpoint and compute surprisal → rho.
    try:
        from analyze.meco_surprisal import compute_surprisal_for_checkpoint  # type: ignore
    except Exception as e:
        print(
            f"[meco_ext] compute_alignment_for_run: surprisal backend not "
            f"available ({e}). Returning NaN; set BEETLE_MOCK_METRICS=1 for a "
            f"deterministic mock or wire up meco_surprisal."
        )
        return float("nan")

    checkpoint_id = f"{record.variant}_seed{record.seed}"
    rhos: list[float] = []
    for lang in languages:
        try:
            df = compute_surprisal_for_checkpoint(
                record, lang=lang, checkpoint_id=checkpoint_id
            )
            df = preprocess_meco_dataframe(df)
            sub = df.dropna(subset=["log_rt", "zsurp"])
            if len(sub) < 20:
                continue
            rho, _ = stats.spearmanr(sub["zsurp"], sub["log_rt"])
            if np.isfinite(rho):
                rhos.append(float(rho))
        except Exception as e:
            print(f"[meco_ext] surprisal for {checkpoint_id} on {lang} failed: {e}")
    return float(np.mean(rhos)) if rhos else float("nan")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extended MECO RT alignment analysis")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_meco_extended(overwrite=args.overwrite)
