"""
meco_figures.py — Publication-ready MECO alignment figures.

Produces figures 10–17 of the approved plan
(``~/.claude/plans/reactive-sprouting-hamming.md`` Part B.8). Every figure is:

* rendered at DPI 300 with a serif font family,
* saved under ``results/figures/`` with the ``NN_meco_*.png`` convention
  continuing from ``05_phenomenon_breakdown.png`` in ``plot_results.py``,
* coloured by baseline group (small_bilingual / bgpt / bilingual_llm /
  bloom_ladder / eurollm / multilingual_llm) via ``GROUP_PALETTE``.

The functions take pandas dataframes (typically produced by
``meco_extended.per_model_alignment`` or ``fisher_z_group_comparison``) so
they are decoupled from the heavy surprisal-extraction step and can be
re-run after the fact with cached inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


# ---------------------------------------------------------------------------
# Style — delegated to analyze/plot_style.py
# ---------------------------------------------------------------------------

from plot_style import (
    apply_acl_style, savefig_pub, BASELINE_PALETTE,
    WIDTH_SINGLE, WIDTH_DOUBLE, DPI,
)

apply_acl_style()

FIGURE_DIR = Path("results/figures")

# Public alias retained for backwards compatibility with any caller that
# imports GROUP_PALETTE directly.
GROUP_PALETTE: Dict[str, str] = BASELINE_PALETTE

DEFAULT_GROUP_ORDER = [
    "small_bilingual", "bgpt", "bilingual_llm",
    "bloom_ladder", "eurollm", "multilingual_llm",
]


def _set_style():
    """Back-compat no-op; the global style is applied at import time."""
    apply_acl_style()


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save(fig, out: Path) -> Path:
    """Save a figure as both PDF (vector) and PNG (300 DPI preview)."""
    paths = savefig_pub(fig, Path(out).with_suffix(""))
    return paths["pdf"] if isinstance(paths, dict) else paths[0]


# ---------------------------------------------------------------------------
# Figure 10: ρ bar plot, grouped + faceted by language
# ---------------------------------------------------------------------------

def fig_10_correlation_bar(
    alignment_df: pd.DataFrame,
    *,
    output_dir: Path = FIGURE_DIR,
    group_column: str = "group",
    language_column: str = "language",
    value_column: str = "rho",
) -> Path:
    _set_style()
    _ensure_dir(output_dir)
    df = alignment_df.copy()
    df[group_column] = pd.Categorical(
        df[group_column], categories=DEFAULT_GROUP_ORDER, ordered=True
    )
    g = sns.catplot(
        data=df,
        x=group_column,
        y=value_column,
        col=language_column,
        kind="bar",
        palette=GROUP_PALETTE,
        errorbar=("ci", 95),
        height=3.2,
        aspect=0.9,
        sharey=True,
    )
    g.set_axis_labels("Model group", r"Spearman $\rho$ (surprisal $\sim$ log RT)")
    for ax in g.axes.flat:
        ax.axhline(0, color="grey", lw=0.6, linestyle="--")
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30)
            lbl.set_ha("right")
    out = output_dir / "10_meco_correlation_bar.png"
    _save(g.figure, out)
    plt.close(g.figure)
    return out


# ---------------------------------------------------------------------------
# Figure 11: β ± SE per model
# ---------------------------------------------------------------------------

def fig_11_regression_slope(
    alignment_df: pd.DataFrame,
    *,
    output_dir: Path = FIGURE_DIR,
    beta_column: str = "beta",
    se_column: Optional[str] = None,
    model_column: str = "model",
    group_column: str = "group",
) -> Path:
    _set_style()
    _ensure_dir(output_dir)
    df = alignment_df.sort_values([group_column, beta_column])
    fig, ax = plt.subplots(figsize=(6, max(4, 0.22 * len(df))))
    colors = [GROUP_PALETTE.get(g, "#7f7f7f") for g in df[group_column]]
    y = np.arange(len(df))
    se = df[se_column].to_numpy() if se_column and se_column in df.columns else None
    ax.errorbar(
        df[beta_column].to_numpy(), y,
        xerr=se, fmt="o", color="black",
        ecolor="grey", capsize=2, lw=0.7,
    )
    ax.scatter(df[beta_column].to_numpy(), y, c=colors, s=30, zorder=3)
    ax.axvline(0, color="grey", lw=0.6, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(df[model_column].to_list(), fontsize=7)
    ax.set_xlabel(r"$\beta_{\mathrm{surprisal}}$ (log RT per z-surprisal)")
    ax.set_title("Per-model surprisal → log-RT slope")
    handles = [mpl.patches.Patch(color=GROUP_PALETTE[g], label=g)
               for g in DEFAULT_GROUP_ORDER if g in df[group_column].unique()]
    ax.legend(handles=handles, loc="best", fontsize=8, frameon=False)
    out = output_dir / "11_meco_regression_slope.png"
    _save(fig, out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 12 (KEY): scatter + regression line per group, faceted by language
# ---------------------------------------------------------------------------

def fig_12_scatter_surprisal_rt(
    token_df: pd.DataFrame,
    *,
    output_dir: Path = FIGURE_DIR,
    group_column: str = "group",
    language_column: str = "language",
    surprisal_column: str = "zsurp",
    rt_column: str = "log_rt",
    max_points_per_panel: int = 20_000,
    seed: int = 0,
) -> Path:
    _set_style()
    _ensure_dir(output_dir)
    rng = np.random.default_rng(seed)
    frames = []
    for (lang, grp), sub in token_df.groupby([language_column, group_column]):
        if len(sub) > max_points_per_panel:
            idx = rng.choice(len(sub), max_points_per_panel, replace=False)
            sub = sub.iloc[idx]
        frames.append(sub)
    plot_df = pd.concat(frames, ignore_index=True)
    plot_df[group_column] = pd.Categorical(
        plot_df[group_column], categories=DEFAULT_GROUP_ORDER, ordered=True
    )
    g = sns.lmplot(
        data=plot_df,
        x=surprisal_column,
        y=rt_column,
        hue=group_column,
        col=language_column,
        palette=GROUP_PALETTE,
        scatter_kws={"alpha": 0.06, "s": 4, "rasterized": True},
        line_kws={"lw": 1.4},
        height=3.4,
        aspect=0.95,
        ci=95,
        n_boot=200,
    )
    g.set_axis_labels("z-scored surprisal", "log(RT)")
    out = output_dir / "12_meco_scatter_surprisal_rt.png"
    _save(g.figure, out)
    plt.close(g.figure)
    return out


# ---------------------------------------------------------------------------
# Figure 13: Δρ forest (Beetle − baseline) with 95% CI
# ---------------------------------------------------------------------------

def fig_13_forest_deltarho(
    fisher_df: pd.DataFrame,
    *,
    output_dir: Path = FIGURE_DIR,
) -> Path:
    _set_style()
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(5.5, max(2.2, 0.4 * len(fisher_df))))
    y = np.arange(len(fisher_df))
    delta = fisher_df["mean_delta_rho"].to_numpy()
    # Approximate 95% CI via Fisher z ± 1.96/sqrt(n-3) — rough but informative.
    n_ref = fisher_df["n_ref"].to_numpy()
    ci = 1.96 / np.sqrt(np.maximum(n_ref - 3, 1))
    colors = [GROUP_PALETTE.get(g, "#7f7f7f") for g in fisher_df["group"]]
    ax.errorbar(delta, y, xerr=ci, fmt="o", color="black", ecolor="grey", capsize=2)
    ax.scatter(delta, y, c=colors, s=50, zorder=3)
    ax.axvline(0, color="grey", lw=0.6, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(fisher_df["group"].to_list())
    ax.set_xlabel(r"$\Delta \rho$ (Beetle $-$ baseline group)")
    ax.set_title("Group-level MECO alignment advantage")
    out = output_dir / "13_meco_forest_deltarho.png"
    _save(fig, out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 14: mixed-effects β ± SE
# ---------------------------------------------------------------------------

def fig_14_mixedlm_effects(
    mixedlm_summary: pd.DataFrame,
    *,
    output_dir: Path = FIGURE_DIR,
    term_column: str = "term",
    beta_column: str = "beta",
    se_column: str = "se",
) -> Path:
    _set_style()
    _ensure_dir(output_dir)
    df = mixedlm_summary.copy().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6, max(2.2, 0.35 * len(df))))
    y = np.arange(len(df))
    ax.errorbar(
        df[beta_column].to_numpy(), y,
        xerr=df[se_column].to_numpy(),
        fmt="o", color="black", ecolor="grey", capsize=2,
    )
    ax.axvline(0, color="grey", lw=0.6, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(df[term_column].to_list(), fontsize=8)
    ax.set_xlabel(r"$\beta$ (log-RT units)")
    ax.set_title("Mixed-effects: rt $\\sim$ surprisal $\\times$ model_type")
    out = output_dir / "14_meco_mixedlm_effects.png"
    _save(fig, out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 15: k × language heatmap of ρ, annotated with per-language Friedman p
# ---------------------------------------------------------------------------

def fig_15_friedman_heatmap(
    alignment_df: pd.DataFrame,
    friedman_df: Optional[pd.DataFrame] = None,
    *,
    output_dir: Path = FIGURE_DIR,
    model_column: str = "model",
    language_column: str = "language",
    value_column: str = "rho",
) -> Path:
    _set_style()
    _ensure_dir(output_dir)
    pivot = alignment_df.pivot_table(
        index=model_column, columns=language_column, values=value_column
    )
    fig, ax = plt.subplots(
        figsize=(1.6 + 0.9 * pivot.shape[1], 0.25 * pivot.shape[0] + 2)
    )
    sns.heatmap(pivot, ax=ax, cmap="RdBu_r", center=0, annot=False, cbar_kws={"label": r"$\rho$"})
    if friedman_df is not None and not friedman_df.empty:
        all_row = friedman_df[friedman_df["scope"] == "all_languages"]
        if not all_row.empty:
            chi2 = float(all_row["chi2"].iloc[0])
            p = float(all_row["p"].iloc[0])
            ax.set_title(
                rf"Per-model $\rho$ by language — Friedman $\chi^2={chi2:.2f}$, p={p:.2e}"
            )
    ax.set_xlabel("Language")
    ax.set_ylabel("Model")
    out = output_dir / "15_meco_friedman_heatmap.png"
    _save(fig, out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 16: BLOOM size ladder with Beetle + EuroLLM reference bands
# ---------------------------------------------------------------------------

_BLOOM_PARAM_COUNT = {
    "bigscience/bloom-560m": 0.56e9,
    "bigscience/bloom-1b1":  1.1e9,
    "bigscience/bloom-1b7":  1.7e9,
    "bigscience/bloom-3b":   3e9,
    "bigscience/bloom-7b1":  7.1e9,
    "bigscience/bloom":      176e9,
}


def fig_16_bloom_scale_curve(
    alignment_df: pd.DataFrame,
    *,
    output_dir: Path = FIGURE_DIR,
    model_column: str = "model",
    group_column: str = "group",
    language_column: str = "language",
    value_column: str = "rho",
) -> Path:
    _set_style()
    _ensure_dir(output_dir)
    langs = sorted(alignment_df[language_column].dropna().unique())
    fig, axes = plt.subplots(
        1, max(1, len(langs)), figsize=(3.6 * max(1, len(langs)), 3.6), sharey=True,
    )
    if len(langs) == 1:
        axes = [axes]

    for ax, lang in zip(axes, langs):
        sub = alignment_df[alignment_df[language_column] == lang]
        bloom = sub[sub[group_column] == "bloom_ladder"].copy()
        bloom["params"] = bloom[model_column].map(_BLOOM_PARAM_COUNT)
        bloom = bloom.dropna(subset=["params"]).sort_values("params")
        if not bloom.empty:
            ax.plot(
                bloom["params"].to_numpy(), bloom[value_column].to_numpy(),
                "o-", color=GROUP_PALETTE["bloom_ladder"], label="BLOOM ladder",
            )
            ax.set_xscale("log")

        # Reference bands.
        for grp, label in [("small_bilingual", "Beetle"), ("eurollm", "EuroLLM")]:
            ref = sub[sub[group_column] == grp][value_column].to_numpy()
            if len(ref):
                ax.axhspan(
                    ref.mean() - ref.std(ddof=1 if len(ref) > 1 else 0),
                    ref.mean() + ref.std(ddof=1 if len(ref) > 1 else 0),
                    color=GROUP_PALETTE[grp], alpha=0.15,
                )
                ax.axhline(ref.mean(), color=GROUP_PALETTE[grp], lw=1.2, label=label)

        ax.set_title(f"{lang}")
        ax.set_xlabel("Parameters (log scale)")
        ax.axhline(0, color="grey", lw=0.5, ls=":")
        ax.legend(fontsize=8, frameon=False)

    axes[0].set_ylabel(r"Spearman $\rho$ on MECO")
    out = output_dir / "16_meco_bloom_scale_curve.png"
    _save(fig, out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 17: Beetle vs EuroLLM paired bar per language
# ---------------------------------------------------------------------------

def fig_17_eurollm_vs_beetle(
    alignment_df: pd.DataFrame,
    wilcoxon_df: Optional[pd.DataFrame] = None,
    *,
    output_dir: Path = FIGURE_DIR,
    language_column: str = "language",
    group_column: str = "group",
    value_column: str = "rho",
) -> Path:
    _set_style()
    _ensure_dir(output_dir)
    sub = alignment_df[
        alignment_df[group_column].isin(["small_bilingual", "eurollm"])
    ].copy()
    sub[group_column] = pd.Categorical(
        sub[group_column], categories=["small_bilingual", "eurollm"], ordered=True
    )
    g = sns.catplot(
        data=sub, x=language_column, y=value_column, hue=group_column,
        kind="bar", palette=GROUP_PALETTE,
        errorbar=("ci", 95), height=3.6, aspect=1.1,
    )
    g.set_axis_labels("Language", r"Spearman $\rho$")
    # Annotate with Holm-corrected Wilcoxon stars if the dataframe has a
    # "language" column. Per-language significance is optional; when the
    # caller's wilcoxon_df is keyed by model pairs only (no language), we
    # silently skip the annotations rather than raising.
    if (
        wilcoxon_df is not None
        and not wilcoxon_df.empty
        and "p_holm" in wilcoxon_df.columns
        and "language" in wilcoxon_df.columns
    ):
        ax = g.axes.flat[0]
        for i, lang in enumerate(sorted(sub[language_column].unique())):
            row = wilcoxon_df[wilcoxon_df["language"] == lang]
            if row.empty:
                continue
            p = float(row["p_holm"].iloc[0])
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ax.text(i, ax.get_ylim()[1] * 0.95, stars, ha="center", fontsize=10)
    out = output_dir / "17_meco_eurollm_vs_beetle.png"
    _save(g.figure, out)
    plt.close(g.figure)
    return out


# ---------------------------------------------------------------------------
# Convenience orchestrator
# ---------------------------------------------------------------------------

def render_all(
    alignment_df: pd.DataFrame,
    *,
    fisher_df: Optional[pd.DataFrame] = None,
    mixedlm_summary: Optional[pd.DataFrame] = None,
    friedman_df: Optional[pd.DataFrame] = None,
    wilcoxon_df: Optional[pd.DataFrame] = None,
    token_df: Optional[pd.DataFrame] = None,
    output_dir: Path = FIGURE_DIR,
) -> Dict[str, Path]:
    """Render all figures for which the requisite dataframe is available."""
    out: Dict[str, Path] = {}
    out["10"] = fig_10_correlation_bar(alignment_df, output_dir=output_dir)
    out["11"] = fig_11_regression_slope(alignment_df, output_dir=output_dir)
    if token_df is not None:
        out["12"] = fig_12_scatter_surprisal_rt(token_df, output_dir=output_dir)
    if fisher_df is not None and not fisher_df.empty:
        out["13"] = fig_13_forest_deltarho(fisher_df, output_dir=output_dir)
    if mixedlm_summary is not None and not mixedlm_summary.empty:
        out["14"] = fig_14_mixedlm_effects(mixedlm_summary, output_dir=output_dir)
    out["15"] = fig_15_friedman_heatmap(
        alignment_df, friedman_df, output_dir=output_dir
    )
    out["16"] = fig_16_bloom_scale_curve(alignment_df, output_dir=output_dir)
    out["17"] = fig_17_eurollm_vs_beetle(
        alignment_df, wilcoxon_df, output_dir=output_dir
    )
    return out
