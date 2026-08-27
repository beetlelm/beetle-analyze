"""
visualise.py — All plots for the BeetleLM analysis pipeline.

Produces publication-quality figures saved to results/figures/.

Plots
-----
1.  ppl_heatmap()          PPL per language × condition-type heatmap
2.  forgetting_barplot()   CF score (delta PPL) per model, grouped by type
3.  forgetting_scatter()   mono PPL vs bilingual PPL scatter (per sentence)
4.  vocab_overlap_plot()   shared vocab % + mean cosine dist, by condition
5.  probe_pca_plot()       PCA of probe-word embeddings per language
6.  cka_heatmap()          pairwise CKA similarity matrix
7.  rt_ranking_plot()      Pearson r ranking, coloured by condition type
8.  condition_summary()    One-pane overview: PPL / CF / RT by condition
"""

from __future__ import annotations
from analyze import _path  # noqa: F401  — adds repo root to sys.path

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

from plot_style import (
    apply_acl_style, savefig_pub,
    color_for_condition, marker_for_cl, linestyle_for_family,
    WIDTH_SINGLE, WIDTH_DOUBLE,
)

apply_acl_style()

FIGURE_DIR = Path("results/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Back-compat alias so existing call-sites keep working.
# New code should call color_for_condition() directly.
class _ConditionColours:
    def __getitem__(self, key):
        return color_for_condition(key)
    def get(self, key, default=None):
        c = color_for_condition(key)
        return c if c else (default or "#aaa")

CONDITION_COLOURS = _ConditionColours()

LANG_LABELS = {
    "nld": "Dutch",
    "deu": "German",
    "zho": "Chinese",
    "fra": "French",
    "fas": "Persian",
    "bul": "Bulgarian",
    "eng": "English",
}


def _condition_legend(ax, types: list[str]) -> None:
    handles = [
        mpatches.Patch(color=color_for_condition(t), label=t.replace("_", "-"))
        for t in types
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8)


# Module-level hook set by analyze/run.py::_run_figures to tag the current
# figure variant ("final" or "trajectory"). When set, _save_pub appends it
# to the output stem so trajectory and final variants coexist on disk.
CURRENT_FIGURE_VARIANT: str | None = None


def _variant_suffix(name: str) -> str:
    """Append the current figure variant suffix to *name* if one is active."""
    if CURRENT_FIGURE_VARIANT and not name.endswith(
        ("_final", "_trajectory")
    ):
        return f"{name}_{CURRENT_FIGURE_VARIANT}"
    return name


def _save_pub(fig, name: str) -> Path:
    """Save as PDF+PNG via plot_style.savefig_pub. Returns the PDF path.

    When ``CURRENT_FIGURE_VARIANT`` is set (via ``analyze.run._run_figures``),
    the output stem gains a ``_final`` or ``_trajectory`` suffix so both
    variants can coexist on disk.
    """
    name = _variant_suffix(name)
    paths = savefig_pub(fig, FIGURE_DIR / Path(name).stem)
    # Back-compat: savefig_pub now returns a dict {format: path}. Old
    # callers expected (pdf, png); return just the PDF path here.
    return paths["pdf"] if isinstance(paths, dict) else paths[0]


def _collapse_to_final(df):
    """Filter a PPL-summary DataFrame to one row per (repo, iso_code).

    Delegates to :func:`ppl_utils.final_revision_row_mask`, which picks
    ``main`` (or the max ``step-N``) per repo. Safe no-op when the
    DataFrame already has only one row per (repo, iso_code).
    """
    try:
        from ppl_utils import final_revision_row_mask
    except Exception:
        return df
    if "revision" not in df.columns or "repo" not in df.columns:
        return df
    mask = final_revision_row_mask(df)
    return df[mask] if mask.any() else df


def _also_png(pdf_path: Path, fig) -> None:
    """Write a matching PNG sibling at publication DPI.

    Back-compat shim for ``fig.savefig(out); _also_png(out, fig)`` call-sites
    in this module. Uses the same rcParams DPI as savefig_pub so every PDF
    gets an equivalent-quality PNG preview without per-call duplication.
    """
    png = Path(pdf_path).with_suffix(".png")
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.02)


# ---------------------------------------------------------------------------
# 1. PPL heatmap
# ---------------------------------------------------------------------------

def ppl_heatmap(summary_path: str | Path = "results/ppl/ppl_summary.csv") -> Path:
    """
    Heatmap: rows = bilingual_type, columns = iso_code, values = mean PPL.

    Final-checkpoint snapshot: when ``ppl_summary.csv`` carries multiple
    revisions per repo (CKPT_SCOPE=all|both), collapses to the final
    revision per repo via ``ppl_utils.final_revision_row_mask``.
    """
    df = pd.read_csv(summary_path)
    df = _collapse_to_final(df)
    sys.path.insert(0, ".")
    from models import get_bilingual_type

    df["bilingual_type"] = df["repo"].apply(get_bilingual_type)

    pivot = (
        df
        .groupby(["bilingual_type", "iso_code"])["mean_ppl"]
        .mean()
        .unstack(fill_value=np.nan)
    )
    lang_order = [c for c in ["eng","nld","deu","zho","fra","fas","bul"] if c in pivot.columns]
    pivot = pivot[lang_order]
    type_order = ["mono","balanced","simultaneous","sequential","part_time","late","heritage"]
    known = [t for t in type_order if t in pivot.index]
    extras = [t for t in pivot.index if t not in type_order]
    pivot = pivot.reindex(known + extras)

    if pivot.empty or pivot.shape[1] == 0:
        print(f"[viz] ppl_heatmap: no rows after filtering — skipping")
        return FIGURE_DIR / "ppl_heatmap.pdf"

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".0f",
        cmap="YlOrRd", linewidths=0.5, linecolor="#ddd",
        cbar_kws={"label": "Mean PPL (↓ better)"},
    )
    ax.set_xticklabels([LANG_LABELS.get(c, c) for c in pivot.columns], rotation=30, ha="right")
    ax.set_yticklabels([t.replace("_", "-") for t in pivot.index], rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Mean FLORES Perplexity by Condition Type & Language", fontweight="bold", pad=12)
    plt.tight_layout()

    out = FIGURE_DIR / "ppl_heatmap.pdf"
    fig.savefig(out); _also_png(out, fig)
    plt.close(fig)
    print(f"[viz] saved {out}")
    return out


# ---------------------------------------------------------------------------
# 2. Catastrophic forgetting bar plot
# ---------------------------------------------------------------------------

def forgetting_barplot(
    summary_path: str | Path = "results/forgetting/forgetting_summary.csv",
) -> Path:
    """
    Grouped bar chart: delta PPL per model, faceted by language.
    Red = forgetting (delta > 0), blue = improvement (delta < 0).
    """
    df = pd.read_csv(summary_path)
    langs = df["iso_code"].unique()

    fig, axes = plt.subplots(1, len(langs), figsize=(5 * len(langs), 5), sharey=False)
    if len(langs) == 1:
        axes = [axes]

    for ax, lang in zip(axes, langs):
        sub = df[df["iso_code"] == lang].sort_values("mean_delta_ppl")
        colours = [CONDITION_COLOURS.get(t, "#aaa") for t in sub["bilingual_type"]]
        bars = ax.barh(range(len(sub)), sub["mean_delta_ppl"], color=colours, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(
            [r.split("beetlelm_")[-1] for r in sub["repo"]],
            fontsize=7
        )
        ax.set_title(LANG_LABELS.get(lang, lang), fontweight="bold")
        ax.set_xlabel("Δ PPL vs mono (↑ = forgetting)")

    # Shared legend
    _condition_legend(axes[-1], list(CONDITION_COLOURS.keys()))
    fig.suptitle("Catastrophic Forgetting: PPL Degradation from Mono Baseline", fontweight="bold")
    plt.tight_layout()

    out = FIGURE_DIR / "forgetting_barplot.pdf"
    fig.savefig(out); _also_png(out, fig)
    plt.close(fig)
    print(f"[viz] saved {out}")
    return out


# ---------------------------------------------------------------------------
# 3. Forgetting scatter (per-sentence)
# ---------------------------------------------------------------------------

def forgetting_scatter(
    raw_dir: str | Path = "results/forgetting",
    lang: str = "deu",
) -> Path:
    """
    Scatter plot: mono PPL (x) vs bilingual PPL (y) per sentence.
    Points above y=x show forgetting.
    """
    raw_path = Path(raw_dir) / f"forgetting_raw_{lang}.csv"
    df = pd.read_csv(raw_path)

    fig, ax = plt.subplots(figsize=(6, 6))
    for bil_type, grp in df.groupby("bilingual_type"):
        ax.scatter(
            grp["ppl_mono"], grp["ppl_bilingual"],
            alpha=0.15, s=8,
            color=CONDITION_COLOURS.get(bil_type, "#aaa"),
            label=bil_type.replace("_", "-"),
        )

    lim = max(df["ppl_mono"].quantile(0.95), df["ppl_bilingual"].quantile(0.95))
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="y=x (no change)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Mono PPL")
    ax.set_ylabel("Bilingual PPL")
    ax.set_title(
        f"Per-Sentence PPL: Mono vs Bilingual — {LANG_LABELS.get(lang, lang)}",
        fontweight="bold"
    )
    ax.legend(frameon=False, fontsize=8, markerscale=2)
    plt.tight_layout()

    out = FIGURE_DIR / f"forgetting_scatter_{lang}.pdf"
    fig.savefig(out); _also_png(out, fig)
    plt.close(fig)
    print(f"[viz] saved {out}")
    return out


# ---------------------------------------------------------------------------
# 4. Vocabulary overlap plot
# ---------------------------------------------------------------------------

def vocab_overlap_plot(
    overlap_dir: str | Path = "results/embeddings",
    lang: str = "deu",
) -> Path:
    df = pd.read_csv(Path(overlap_dir) / f"vocab_overlap_{lang}.csv")
    df_sorted = df.sort_values("mean_cosine_dist", ascending=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    colours = [CONDITION_COLOURS.get(t, "#aaa") for t in df_sorted["bilingual_type"]]
    x = range(len(df_sorted))

    ax1.bar(x, df_sorted["shared_vocab_pct"], color=colours, edgecolor="white")
    ax1.set_ylabel("Shared vocab with mono (%)")
    ax1.set_title(
        f"Vocabulary Overlap & Embedding Drift — {LANG_LABELS.get(lang, lang)}",
        fontweight="bold"
    )

    ax2.bar(x, df_sorted["mean_cosine_dist"], color=colours, edgecolor="white")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(
        [r.split("beetlelm_")[-1] for r in df_sorted["repo"]],
        rotation=45, ha="right", fontsize=7
    )
    ax2.set_ylabel("Mean cosine distance from mono")

    _condition_legend(ax1, df_sorted["bilingual_type"].unique().tolist())
    plt.tight_layout()

    out = FIGURE_DIR / f"vocab_overlap_{lang}.pdf"
    fig.savefig(out); _also_png(out, fig)
    plt.close(fig)
    print(f"[viz] saved {out}")
    return out


# ---------------------------------------------------------------------------
# 5. Probe-word PCA plot
# ---------------------------------------------------------------------------

def probe_pca_plot(
    pca_dir: str | Path = "results/embeddings",
    lang: str = "deu",
) -> Path:
    """
    2D scatter of probe-word embeddings for each condition.
    Each word gets a distinct marker; conditions are coloured.
    """
    df = pd.read_csv(Path(pca_dir) / f"probe_pca_{lang}.csv")
    words = df["word"].unique()
    word_markers = ["o","s","^","D","v","P","X","*","h","8"]
    marker_map = {w: word_markers[i % len(word_markers)] for i, w in enumerate(words)}

    fig, ax = plt.subplots(figsize=(8, 7))

    for bil_type, grp in df.groupby("bilingual_type"):
        for word, wgrp in grp.groupby("word"):
            ax.scatter(
                wgrp["pc1"], wgrp["pc2"],
                color=CONDITION_COLOURS.get(bil_type, "#aaa"),
                marker=marker_map[word],
                alpha=0.75, s=60,
            )

    # Annotate mono positions
    mono_rows = df[df["bilingual_type"] == "mono"]
    for _, r in mono_rows.iterrows():
        ax.annotate(
            r["word"], (r["pc1"], r["pc2"]),
            fontsize=7, ha="center", va="bottom",
            color="#2c2c2c",
        )

    # Build dual legend: conditions + word shapes
    cond_handles = [
        Line2D([0],[0], marker="o", color=CONDITION_COLOURS.get(t,"#aaa"),
               label=t.replace("_","-"), linestyle="None", markersize=7)
        for t in df["bilingual_type"].unique()
    ]
    word_handles = [
        Line2D([0],[0], marker=marker_map[w], color="grey",
               label=w, linestyle="None", markersize=7)
        for w in words[:6]
    ]
    leg1 = ax.legend(handles=cond_handles, frameon=False, fontsize=8,
                     loc="upper left", title="Condition")
    ax.add_artist(leg1)
    ax.legend(handles=word_handles, frameon=False, fontsize=8,
              loc="lower right", title="Word")

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(
        f"Probe-Word Embedding PCA — {LANG_LABELS.get(lang, lang)} (mono-anchored)",
        fontweight="bold"
    )
    plt.tight_layout()

    out = FIGURE_DIR / f"probe_pca_{lang}.pdf"
    fig.savefig(out); _also_png(out, fig)
    plt.close(fig)
    print(f"[viz] saved {out}")
    return out


# ---------------------------------------------------------------------------
# 6. CKA heatmap
# ---------------------------------------------------------------------------

def cka_heatmap(
    cka_dir: str | Path = "results/embeddings",
    lang: str = "deu",
) -> Path:
    cka_df = pd.read_csv(Path(cka_dir) / f"cka_matrix_{lang}.csv", index_col=0)
    labels = [r.split("beetlelm_")[-1] for r in cka_df.index]

    fig, ax = plt.subplots(figsize=(max(8, len(cka_df) * 0.35),) * 2)
    sns.heatmap(
        cka_df.values, ax=ax,
        xticklabels=labels, yticklabels=labels,
        cmap="viridis", vmin=0, vmax=1,
        annot=len(cka_df) <= 15, fmt=".2f",
        linewidths=0.3, linecolor="#eee",
        cbar_kws={"label": "Linear CKA (↑ more similar)"},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    ax.set_title(
        f"Pairwise Representational Similarity (CKA) — {LANG_LABELS.get(lang, lang)}",
        fontweight="bold", pad=12
    )
    plt.tight_layout()

    out = FIGURE_DIR / f"cka_heatmap_{lang}.pdf"
    fig.savefig(out); _also_png(out, fig)
    plt.close(fig)
    print(f"[viz] saved {out}")
    return out


# ---------------------------------------------------------------------------
# 7. Reading time Pearson r ranking
# ---------------------------------------------------------------------------

def rt_ranking_plot(
    rt_dir: str | Path = "results/reading_time",
    corpus: str = "provo",
    lang: str = "eng",
) -> Path:
    df = pd.read_csv(Path(rt_dir) / f"rt_summary_{corpus}_{lang}.csv")

    fig, ax = plt.subplots(figsize=(7, max(4, len(df) * 0.35)))
    colours = [CONDITION_COLOURS.get(t, "#aaa") for t in df["bilingual_type"]]
    ax.barh(range(len(df)), df["pearson_r"], color=colours, edgecolor="white")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([r.split("beetlelm_")[-1] for r in df["repo"]], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Pearson r (surprisal ~ reading time)")
    ax.set_title(
        f"Reading Time Prediction — {corpus.upper()} / {LANG_LABELS.get(lang, lang)}",
        fontweight="bold"
    )
    _condition_legend(ax, df["bilingual_type"].unique().tolist())
    plt.tight_layout()

    out = FIGURE_DIR / f"rt_ranking_{corpus}_{lang}.pdf"
    fig.savefig(out); _also_png(out, fig)
    plt.close(fig)
    print(f"[viz] saved {out}")
    return out


# ---------------------------------------------------------------------------
# 8. Condition summary overview (PPL / CF / RT in one figure)
# ---------------------------------------------------------------------------

def condition_summary(
    ppl_path:       str | Path = "results/ppl/ppl_summary.csv",
    forgetting_path:str | Path = "results/forgetting/forgetting_summary.csv",
    rt_path:        str | Path | None = None,
    lang: str = "deu",
) -> Path:
    """
    Three-panel summary for a single language:
      Left:  Mean PPL by condition type
      Centre: Mean CF score (delta PPL) by condition type
      Right:  Mean Pearson r by condition type (if rt_path provided)
    """
    sys.path.insert(0, ".")
    from models import get_bilingual_type

    ppl_df = pd.read_csv(ppl_path)
    ppl_df = _collapse_to_final(ppl_df)
    ppl_df = ppl_df[ppl_df["iso_code"] == lang].copy()
    ppl_df["bilingual_type"] = ppl_df["repo"].apply(get_bilingual_type)
    ppl_agg = ppl_df.groupby("bilingual_type")["mean_ppl"].mean()

    cf_df = pd.read_csv(forgetting_path)
    cf_df  = cf_df[cf_df["iso_code"] == lang]
    cf_agg = cf_df.groupby("bilingual_type")["mean_delta_ppl"].mean()

    n_panels = 3 if rt_path else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    def _bar(ax, series, title, xlabel):
        types = series.index.tolist()
        colours = [CONDITION_COLOURS.get(t, "#aaa") for t in types]
        ax.bar(range(len(series)), series.values, color=colours, edgecolor="white")
        ax.set_xticks(range(len(series)))
        ax.set_xticklabels([t.replace("_", "-") for t in types], rotation=30, ha="right")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(xlabel)

    _bar(axes[0], ppl_agg,   "Mean PPL",              "Perplexity (↓ better)")
    _bar(axes[1], cf_agg,    "Catastrophic Forgetting","Δ PPL from mono (↑ = worse)")

    if rt_path:
        rt_df = pd.read_csv(rt_path)
        rt_agg = rt_df.groupby("bilingual_type")["pearson_r"].mean()
        _bar(axes[2], rt_agg, "RT Fit", "Pearson r (↑ better)")

    fig.suptitle(
        f"Condition Summary — {LANG_LABELS.get(lang, lang)}",
        fontweight="bold", fontsize=14, y=1.02
    )
    plt.tight_layout()

    out = FIGURE_DIR / f"condition_summary_{lang}.pdf"
    fig.savefig(out); _also_png(out, fig)
    plt.close(fig)
    print(f"[viz] saved {out}")
    return out


# ===========================================================================
# CONVERGENCE PLOTS  (checkpoint-trajectory visualisations)
# ===========================================================================

def ppl_convergence_curves(
    traj_dir: str | Path = "results/convergence",
    repos: list[str] | None = None,
    overlay_langs: bool = True,
) -> Path:
    """
    Line plot: PPL vs training step for each model, one panel per language.

    If overlay_langs=True, L1 and L2 curves for each model are shown in the
    same panel with a solid/dashed line style — so L1 forgetting inflection
    points are visible against the L2 learning curve.

    repos: filter to a subset of models (default: all found in CSV).
    """
    combined_path = Path(traj_dir) / "all_ppl_trajectories.csv"
    df = pd.read_csv(combined_path)
    if repos:
        df = df[df["repo"].isin(repos)]

    # Add bilingual_type for colouring
    sys.path.insert(0, ".")
    from models import get_bilingual_type
    df["bilingual_type"] = df["repo"].apply(get_bilingual_type)
    df["short_name"] = df["repo"].apply(lambda r: r.split("beetlelm_")[-1])

    langs = sorted(df["lang"].unique())
    fig, axes = plt.subplots(1, len(langs), figsize=(6 * len(langs), 5), sharey=False)
    if len(langs) == 1:
        axes = [axes]

    for ax, lang in zip(axes, langs):
        sub = df[df["lang"] == lang].sort_values("step")
        for name, grp in sub.groupby("short_name"):
            bil_type = grp["bilingual_type"].iloc[0]
            colour   = CONDITION_COLOURS.get(bil_type, "#aaa")
            # Dashed if this lang is the L2 (i.e. it doesn't appear in the
            # model name before the hyphen)
            linestyle = "--" if lang not in name.split("-")[0].split("_")[0] else "-"
            ax.plot(
                grp["step"], grp["mean_ppl"],
                color=colour, linewidth=1.5, linestyle=linestyle,
                alpha=0.75, label=f"{name} ({bil_type})",
            )

        ax.set_xlabel("Training step")
        ax.set_ylabel("FLORES PPL (↓ better)")
        ax.set_title(f"{LANG_LABELS.get(lang, lang)}", fontweight="bold")
        ax.set_yscale("log")

    # Shared legend
    _condition_legend(axes[-1], [t for t in CONDITION_COLOURS if t != "unknown"])
    # Add linestyle legend
    solid_line  = Line2D([0],[0], color="grey", linewidth=1.5, linestyle="-",  label="L1")
    dashed_line = Line2D([0],[0], color="grey", linewidth=1.5, linestyle="--", label="L2")
    axes[0].legend(handles=[solid_line, dashed_line], frameon=False, fontsize=9,
                   loc="upper right", title="Role")

    fig.suptitle("PPL Convergence Curves by Training Step", fontweight="bold", fontsize=14)
    plt.tight_layout()

    out = FIGURE_DIR / "ppl_convergence_curves.pdf"
    fig.savefig(out); _also_png(out, fig)
    plt.close(fig)
    print(f"[viz] saved {out}")
    return out


def ppl_convergence_overlay(
    traj_dir: str | Path = "results/convergence",
    focus_repos: list[str] | None = None,
    lang: str = "nld",
) -> Path:
    """
    Overlaid convergence curves for a SINGLE language — one line per model,
    coloured by condition type.

    Best used for the key comparison (e.g. Dutch L1 PPL across all Dutch models)
    to show which conditions converge fastest / best without forgetting.
    """
    combined_path = Path(traj_dir) / "all_ppl_trajectories.csv"
    df = pd.read_csv(combined_path)
    df = df[df["lang"] == lang]
    if focus_repos:
        df = df[df["repo"].isin(focus_repos)]

    sys.path.insert(0, ".")
    from models import get_bilingual_type
    df["bilingual_type"] = df["repo"].apply(get_bilingual_type)
    df["short_name"] = df["repo"].apply(lambda r: r.split("beetlelm_")[-1])

    fig, ax = plt.subplots(figsize=(9, 5))

    for name, grp in df.sort_values("step").groupby("short_name"):
        bil_type = grp["bilingual_type"].iloc[0]
        ax.plot(
            grp["step"], grp["mean_ppl"],
            color=CONDITION_COLOURS.get(bil_type, "#aaa"),
            linewidth=2, alpha=0.85,
            label=f"{name}",
        )
        # Mark the final point
        last = grp.sort_values("step").iloc[-1]
        ax.scatter(last["step"], last["mean_ppl"],
                   color=CONDITION_COLOURS.get(bil_type, "#aaa"), s=30, zorder=5)

    ax.set_xlabel("Training step")
    ax.set_ylabel("FLORES PPL (log scale, ↓ better)")
    ax.set_yscale("log")
    ax.set_title(
        f"Convergence Overlay — {LANG_LABELS.get(lang, lang)} PPL across conditions",
        fontweight="bold"
    )
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
    plt.tight_layout()

    out = FIGURE_DIR / f"ppl_convergence_overlay_{lang}.pdf"
    fig.savefig(out); _also_png(out, fig)
    plt.close(fig)
    print(f"[viz] saved {out}")
    return out


def drift_trajectory_plot(
    traj_dir: str | Path = "results/convergence",
    repo: str | None = None,
    word_subset: list[str] | None = None,
) -> Path:
    """
    Line plot: per-word cosine distance from step-0 across training steps.

    Shows how dramatically individual word representations move in embedding
    space during the bilingual training cycle.
    One line per probe word; shading = IQR across words.
    """
    traj_dir = Path(traj_dir)

    if repo:
        slug = repo.replace("/", "__")
        paths = [traj_dir / f"drift_traj_{slug}.csv"]
    else:
        paths = list(traj_dir.glob("drift_traj_*.csv"))

    if not paths:
        raise FileNotFoundError(f"No drift trajectory CSVs found in {traj_dir}")

    frames = [pd.read_csv(p) for p in paths if p.exists()]
    df = pd.concat(frames, ignore_index=True)

    if word_subset:
        df = df[df["word"].isin(word_subset)]

    repos_in_df = df["repo"].unique()
    fig, axes = plt.subplots(1, len(repos_in_df),
                              figsize=(7 * len(repos_in_df), 5), sharey=True)
    if len(repos_in_df) == 1:
        axes = [axes]

    for ax, r in zip(axes, repos_in_df):
        sub = df[df["repo"] == r].sort_values("step")
        short = r.split("beetlelm_")[-1]

        # Shaded band: median ± IQR across all words
        step_agg = sub.groupby("step")["cosine_dist_step0"].agg(
            ["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
        )
        step_agg.columns = ["median", "q25", "q75"]
        ax.fill_between(step_agg.index, step_agg["q25"], step_agg["q75"],
                         alpha=0.15, color="#4e79a7", label="IQR")
        ax.plot(step_agg.index, step_agg["median"],
                color="#4e79a7", linewidth=2, label="Median")

        # Individual word lines (thin, coloured by word rank)
        words = sub["word"].unique()
        cmap  = plt.cm.tab10
        for i, word in enumerate(words[:8]):  # cap at 8 for readability
            wgrp = sub[sub["word"] == word]
            ax.plot(wgrp["step"], wgrp["cosine_dist_step0"],
                    linewidth=0.8, alpha=0.6, color=cmap(i), label=word)

        ax.set_xlabel("Training step")
        ax.set_ylabel("Cosine distance from step-0")
        ax.set_title(short, fontweight="bold")
        ax.legend(fontsize=7, frameon=False, ncol=2)

    fig.suptitle("Embedding Drift: Word Representation Movement Across Training",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()

    slug_str = repo.replace("/", "__") if repo else "all"
    out = FIGURE_DIR / f"drift_trajectory_{slug_str}.pdf"
    fig.savefig(out); _also_png(out, fig)
    plt.close(fig)
    print(f"[viz] saved {out}")
    return out


def cka_convergence_plot(
    traj_dir: str | Path = "results/convergence",
    repos: list[str] | None = None,
) -> Path:
    """
    CKA between consecutive checkpoints vs training step.

    CKA ≈ 1.0 → representations stable (converged)
    Sudden drops → representational phase transitions

    Overlay multiple models to compare convergence speed by condition.
    """
    traj_dir = Path(traj_dir)

    if repos:
        paths = [traj_dir / f"cka_traj_{r.replace('/', '__')}.csv" for r in repos]
    else:
        paths = list(traj_dir.glob("cka_traj_*.csv"))

    frames = [pd.read_csv(p) for p in paths if p.exists()]
    if not frames:
        raise FileNotFoundError(f"No CKA trajectory CSVs found in {traj_dir}")

    df = pd.concat(frames, ignore_index=True)

    sys.path.insert(0, ".")
    from models import get_bilingual_type
    df["bilingual_type"] = df["repo"].apply(get_bilingual_type)
    df["short_name"] = df["repo"].apply(lambda r: r.split("beetlelm_")[-1])

    # Use step_b as the x-axis (the "arrival" checkpoint)
    fig, (ax_cka, ax_change) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for name, grp in df.sort_values("step_b").groupby("short_name"):
        bil_type = grp["bilingual_type"].iloc[0]
        colour   = CONDITION_COLOURS.get(bil_type, "#aaa")
        ax_cka.plot(grp["step_b"], grp["cka"],
                    color=colour, linewidth=1.5, alpha=0.8, label=name)
        ax_change.plot(grp["step_b"], grp["change"],
                       color=colour, linewidth=1.5, alpha=0.8)

    ax_cka.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax_cka.set_ylabel("CKA (consecutive steps, ↑ = stable)")
    ax_cka.set_ylim(0, 1.05)
    ax_cka.legend(fontsize=7, frameon=False, ncol=3, loc="lower right")
    ax_cka.set_title("Representational Stability Across Training (CKA)", fontweight="bold")

    ax_change.axhline(0.0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax_change.set_ylabel("1 − CKA (representational change, ↓ = stable)")
    ax_change.set_xlabel("Training step")

    plt.tight_layout()

    out = FIGURE_DIR / "cka_convergence.pdf"
    fig.savefig(out); _also_png(out, fig)
    plt.close(fig)
    print(f"[viz] saved {out}")
    return out


def forgetting_inflection_plot(
    traj_dir: str | Path = "results/convergence",
    lang: str = "nld",
) -> Path:
    """
    For each bilingual model with Dutch/German/Chinese as L1, plot L1 PPL and
    L2 PPL on the same axes across training steps.

    Highlights the 'forgetting inflection point' — the step where L1 PPL
    starts rising even as L2 PPL is still falling.
    """
    combined_path = Path(traj_dir) / "all_ppl_trajectories.csv"
    df = pd.read_csv(combined_path)

    # Filter to models that have BOTH the focus lang and at least one other
    repos_with_lang = df[df["lang"] == lang]["repo"].unique()
    df_focus = df[df["repo"].isin(repos_with_lang)]

    repos_bilingual = [r for r in repos_with_lang if "mono" not in r]
    if not repos_bilingual:
        print(f"[viz] No bilingual models found for {lang}")
        return FIGURE_DIR / f"forgetting_inflection_{lang}.pdf"

    ncols = min(3, len(repos_bilingual))
    nrows = (len(repos_bilingual) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, repo in enumerate(repos_bilingual):
        ax  = axes[idx // ncols][idx % ncols]
        sub = df_focus[df_focus["repo"] == repo].sort_values("step")

        l1_df = sub[sub["lang"] == lang]
        l2_langs = [l for l in sub["lang"].unique() if l != lang]

        ax.plot(l1_df["step"], l1_df["mean_ppl"],
                color="#e15759", linewidth=2, label=f"L1 ({lang.upper()})")

        for l2 in l2_langs:
            l2_df = sub[sub["lang"] == l2]
            ax.plot(l2_df["step"], l2_df["mean_ppl"],
                    color="#4e79a7", linewidth=2, linestyle="--",
                    label=f"L2 ({l2.upper()})")

        # Find and annotate the forgetting inflection point:
        # First step where L1 PPL exceeds the previous step's L1 PPL
        # after initially decreasing
        if len(l1_df) > 2:
            l1_ppl = l1_df["mean_ppl"].values
            l1_step= l1_df["step"].values
            for k in range(1, len(l1_ppl) - 1):
                if l1_ppl[k] > l1_ppl[k-1] and l1_ppl[k-1] < l1_ppl[0]:
                    ax.axvline(l1_step[k], color="#f28e2b", linewidth=1.2,
                               linestyle=":", alpha=0.9)
                    ax.annotate("↑L1\nforgetting", (l1_step[k], l1_ppl[k]),
                                fontsize=7, color="#f28e2b", ha="left")
                    break

        ax.set_yscale("log")
        ax.set_xlabel("Step")
        ax.set_ylabel("PPL")
        ax.set_title(repo.split("beetlelm_")[-1], fontweight="bold", fontsize=9)
        ax.legend(fontsize=7, frameon=False)

    # Hide unused panels
    for idx in range(len(repos_bilingual), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        f"L1/L2 PPL Trajectories with Forgetting Inflection Points — {LANG_LABELS.get(lang, lang)}",
        fontweight="bold", fontsize=13
    )
    plt.tight_layout()

    out = FIGURE_DIR / f"forgetting_inflection_{lang}.pdf"
    fig.savefig(out); _also_png(out, fig)
    plt.close(fig)
    print(f"[viz] saved {out}")
    return out


# ---------------------------------------------------------------------------
# MECO Eye-Tracking Plots
# ---------------------------------------------------------------------------

def meco_checkpoint_trajectory_plot(
    traj_dir: str | Path = "results/meco",
) -> Path:
    """
    Plot surprisal vs training step for each model/condition.

    Reads checkpoint_trajectory_*.csv files from the MECO results directory.
    """
    traj_dir = Path(traj_dir)
    traj_files = sorted(traj_dir.glob("checkpoint_trajectory_*.csv"))
    if not traj_files:
        print("[viz] No MECO checkpoint trajectory files found")
        return FIGURE_DIR / "meco_checkpoint_trajectories.pdf"

    fig, ax = plt.subplots(figsize=(12, 6))

    for f in traj_files:
        df = pd.read_csv(f)
        model_slug = f.stem.replace("checkpoint_trajectory_", "")

        # Mean surprisal per step
        step_means = df.groupby("step")["surprisal"].mean()

        # Determine condition for coloring
        condition = "unknown"
        for cond in CONDITION_COLOURS:
            if cond in model_slug:
                condition = cond
                break

        ax.plot(step_means.index, step_means.values,
                color=CONDITION_COLOURS.get(condition, "#aaaaaa"),
                linewidth=1.5, alpha=0.8, label=model_slug[:30])

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Surprisal")
    ax.set_title("MECO Surprisal Across Training Checkpoints", fontweight="bold")
    ax.legend(fontsize=6, ncol=2, frameon=False, loc="upper right")
    plt.tight_layout()

    out = FIGURE_DIR / "meco_checkpoint_trajectories.pdf"
    fig.savefig(out); _also_png(out, fig)
    plt.close(fig)
    print(f"[viz] saved {out}")
    return out


def meco_correlation_heatmap(
    results_path: str | Path = "results/meco/surprisals_all.tsv",
) -> Path:
    """
    Heatmap of Pearson r (model surprisal vs human reading time) by model.

    Requires a combined surprisal TSV with model columns and RT data.
    """
    results_path = Path(results_path)
    if not results_path.exists():
        print(f"[viz] {results_path} not found — run meco_surprisal.py first")
        return FIGURE_DIR / "meco_correlation_heatmap.pdf"

    df = pd.read_csv(results_path, sep="\t")

    # Find model columns (those with 'surprisal' or model name patterns)
    model_cols = [c for c in df.columns if c not in
                  {"FullTextMarked", "FullText", "step", "word_id", "word",
                   "model", "revision", "surprisal"}
                  and df[c].dtype in ("float64", "float32", "int64")]

    if not model_cols:
        print("[viz] No model columns found in combined results")
        return FIGURE_DIR / "meco_correlation_heatmap.pdf"

    # Compute correlations (placeholder — needs RT data merge)
    fig, ax = plt.subplots(figsize=(10, 6))
    corr_matrix = df[model_cols].corr()
    sns.heatmap(corr_matrix, ax=ax, cmap="RdBu_r", center=0,
                annot=True, fmt=".2f", square=True)
    ax.set_title("Model Surprisal Inter-Correlation (MECO)", fontweight="bold")
    # tight_layout incompatible with seaborn's colorbar on newer matplotlib;
    # savefig(..., bbox_inches="tight") below handles margins instead.

    out = FIGURE_DIR / "meco_correlation_heatmap.pdf"
    fig.savefig(out); _also_png(out, fig)
    plt.close(fig)
    print(f"[viz] saved {out}")
    return out


def meco_condition_comparison(
    results_dir: str | Path = "results/meco",
) -> Path:
    """
    Bar chart comparing mean surprisal across training conditions.
    """
    results_dir = Path(results_dir)
    surp_files = sorted(results_dir.glob("surprisals_*.csv"))
    if not surp_files:
        print("[viz] No surprisal files found")
        return FIGURE_DIR / "meco_condition_comparison.pdf"

    records = []
    for f in surp_files:
        if f.name == "surprisals_all.tsv":
            continue
        df = pd.read_csv(f)
        slug = f.stem.replace("surprisals_", "")

        condition = "unknown"
        for cond in CONDITION_COLOURS:
            if cond in slug:
                condition = cond
                break

        records.append({
            "model": slug[:25],
            "condition": condition,
            "mean_surprisal": df["surprisal"].mean(),
        })

    if not records:
        print("[viz] No surprisal data to plot")
        return FIGURE_DIR / "meco_condition_comparison.pdf"

    plot_df = pd.DataFrame(records).sort_values("mean_surprisal")

    fig, ax = plt.subplots(figsize=(10, max(4, len(records) * 0.4)))
    colors = [CONDITION_COLOURS.get(r["condition"], "#aaaaaa") for _, r in plot_df.iterrows()]
    ax.barh(plot_df["model"], plot_df["mean_surprisal"], color=colors, edgecolor="white")
    ax.set_xlabel("Mean Surprisal (lower = better language model)")
    ax.set_title("MECO Surprisal by Model/Condition", fontweight="bold")

    _condition_legend(ax, list(plot_df["condition"].unique()))
    plt.tight_layout()

    out = FIGURE_DIR / "meco_condition_comparison.pdf"
    fig.savefig(out); _also_png(out, fig)
    plt.close(fig)
    print(f"[viz] saved {out}")
    return out


# ---------------------------------------------------------------------------
# BLiMP significance figures (06–09) — additive.
#
# Produced by analyze/blimp_significance.py. Consume the {task}_paired.csv,
# {task}_wilcoxon_pairwise.csv, and {task}_mixedlm_summary.csv files under
# results/blimp_significance/. Saved under results/figures/ with the
# "0N_blimp_..." prefix continuing the existing numbering scheme.
# ---------------------------------------------------------------------------

def blimp_fig_06_effects(paired_csv: Path, task: str,
                         output_dir: Path = FIGURE_DIR) -> Path:
    """Per-language forest plot of Δx̄ with Cohen's d annotations and
    two-sided FDR-corrected significance stars (primary). One-sided p is
    included as an exploratory annotation only."""
    df = pd.read_csv(paired_csv)
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.45 * len(df))))
    y = np.arange(len(df))
    err = df["std_delta"].to_numpy() / np.sqrt(np.maximum(df["n_phenomena"].to_numpy(), 1))
    ax.errorbar(df["mean_delta"].to_numpy(), y, xerr=1.96 * err, fmt="o",
                color="black", ecolor="grey", capsize=2)
    ax.axvline(0, color="grey", lw=0.6, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(df["language"].astype(str).to_list())
    ax.set_xlabel(r"$\Delta \bar{x}$ (bilingual $-$ monolingual)")
    ax.set_title(f"BLiMP {task}: per-language bilingual advantage")
    # Annotate Cohen's d + two-sided FDR stars (primary).
    for yi, row in enumerate(df.itertuples()):
        stars = ""
        p_fdr = getattr(row, "p_fdr_bh", float("nan"))
        if pd.notna(p_fdr):
            if p_fdr < 0.001: stars = "***"
            elif p_fdr < 0.01: stars = "**"
            elif p_fdr < 0.05: stars = "*"
        ax.text(
            ax.get_xlim()[1] * 0.95, yi,
            f"d={getattr(row, 'cohens_d', float('nan')):.2f} {stars}",
            fontsize=8, va="center", ha="right",
        )
    out = output_dir / f"06_blimp_effects_{task}.png"
    fig.savefig(out); _also_png(out, fig)
    plt.close(fig)
    print(f"[viz] saved {out}")
    return out


def blimp_fig_07_wilcoxon_heatmap(wilcoxon_csv: Path, task: str,
                                  output_dir: Path = FIGURE_DIR) -> Path:
    """k×k Holm-corrected Wilcoxon pairwise heatmap (−log10(p))."""
    df = pd.read_csv(wilcoxon_csv)
    if df.empty or "p_holm" not in df.columns:
        print(f"[viz] skipping 07_blimp_wilcoxon for {task}: empty/no p_holm")
        return output_dir / f"07_blimp_wilcoxon_heatmap_{task}.png"
    models = sorted(set(df["model_a"]).union(df["model_b"]))
    mat = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)
    for _, r in df.iterrows():
        val = -np.log10(max(float(r["p_holm"]), 1e-12))
        mat.loc[r["model_a"], r["model_b"]] = val
        mat.loc[r["model_b"], r["model_a"]] = val
    np.fill_diagonal(mat.values, 0.0)
    fig, ax = plt.subplots(figsize=(1.6 + 0.35 * len(models), 1.4 + 0.35 * len(models)))
    sns.heatmap(mat, cmap="viridis", cbar_kws={"label": r"$-\log_{10}(p_{Holm})$"},
                xticklabels=True, yticklabels=True, ax=ax, square=True)
    ax.set_title(f"BLiMP {task}: pairwise Wilcoxon (Holm)")
    ax.tick_params(axis="x", labelrotation=45)
    out = output_dir / f"07_blimp_wilcoxon_heatmap_{task}.png"
    fig.savefig(out); _also_png(out, fig)
    plt.close(fig)
    print(f"[viz] saved {out}")
    return out


def blimp_fig_08_mixedlm_effects(mixedlm_csv: Path, task: str,
                                 output_dir: Path = FIGURE_DIR) -> Path:
    """β ± SE coefficient plot for the BLiMP mixed-effects fit."""
    df = pd.read_csv(mixedlm_csv)
    if df.empty:
        print(f"[viz] skipping 08_blimp_mixedlm for {task}: empty summary")
        return output_dir / f"08_blimp_mixedlm_effects_{task}.png"
    df = df.sort_values("beta")
    fig, ax = plt.subplots(figsize=(6, max(2, 0.35 * len(df))))
    y = np.arange(len(df))
    ax.errorbar(df["beta"].to_numpy(), y,
                xerr=df["se"].to_numpy(),
                fmt="o", color="black", ecolor="grey", capsize=2)
    ax.axvline(0, color="grey", lw=0.6, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(df["term"].to_list(), fontsize=8)
    ax.set_xlabel(r"$\beta$ (accuracy units)")
    ax.set_title(f"BLiMP {task}: mixed-effects terms (phenomenon RE)")
    out = output_dir / f"08_blimp_mixedlm_effects_{task}.png"
    fig.savefig(out); _also_png(out, fig)
    plt.close(fig)
    print(f"[viz] saved {out}")
    return out


def blimp_fig_09_phenomenon_by_model(
    bilingual_csv: Path, monolingual_csv: Path, task: str,
    *, output_dir: Path = FIGURE_DIR,
    phenomenon_col: str = "phenomenon",
    score_col: str = "accuracy",
    bilingual_type_col: str = "bilingual_type",
) -> Path:
    """Faceted per-phenomenon accuracy (bilingual vs monolingual)."""
    bil = pd.read_csv(bilingual_csv); mono = pd.read_csv(monolingual_csv)
    mono = mono.assign(**{bilingual_type_col: "monolingual"})
    both = pd.concat([bil, mono], ignore_index=True)
    g = sns.catplot(
        data=both, x=phenomenon_col, y=score_col,
        hue=bilingual_type_col, kind="bar",
        height=4.5, aspect=2.2, errorbar=None,
    )
    g.set_xticklabels(rotation=85, fontsize=7)
    g.set_axis_labels("Phenomenon", "Accuracy")
    g.figure.suptitle(f"BLiMP {task}: per-phenomenon accuracy", y=1.02)
    out = output_dir / f"09_blimp_phenomenon_by_model_{task}.png"
    g.figure.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(g.figure)
    print(f"[viz] saved {out}")
    return out


def render_blimp_figures(
    task: str,
    bilingual_csv: Path,
    monolingual_csv: Path,
    *,
    significance_dir: Path = Path("results/blimp_significance"),
    output_dir: Path = FIGURE_DIR,
) -> dict[str, Path]:
    """Render all four BLiMP figures for one task if the source CSVs exist."""
    out: dict[str, Path] = {}
    paired = significance_dir / f"{task}_paired.csv"
    wilc   = significance_dir / f"{task}_wilcoxon_pairwise.csv"
    mixed  = significance_dir / f"{task}_mixedlm_summary.csv"
    if paired.exists():
        out["06"] = blimp_fig_06_effects(paired, task, output_dir=output_dir)
    if wilc.exists():
        out["07"] = blimp_fig_07_wilcoxon_heatmap(wilc, task, output_dir=output_dir)
    if mixed.exists():
        out["08"] = blimp_fig_08_mixedlm_effects(mixed, task, output_dir=output_dir)
    if Path(bilingual_csv).exists() and Path(monolingual_csv).exists():
        out["09"] = blimp_fig_09_phenomenon_by_model(
            Path(bilingual_csv), Path(monolingual_csv), task, output_dir=output_dir,
        )
    return out


# ===========================================================================
# COMPARISON FIGURES (paper-aligned: per-model / per-language / factor pairs)
# ===========================================================================

def _load_ppl_summary(path: str | Path = "results/ppl/ppl_summary.csv") -> pd.DataFrame:
    """Load ``ppl_summary.csv`` and collapse to final revision per repo.

    All downstream comparison figures (per-model, per-language factor,
    per-condition family etc.) are final-snapshot by nature and assume one
    row per (repo, iso_code). When ``CKPT_SCOPE=all|both`` produces a
    multi-revision summary, this helper collapses it back to the final
    revision via ``ppl_utils.final_revision_row_mask``.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    return _collapse_to_final(pd.read_csv(path))


def figure_per_model(
    repo: str,
    lang: str,
    *,
    out_dir: Path = FIGURE_DIR / "per_model",
) -> Path:
    """Single-model summary page: PPL trajectory if available, otherwise final PPL bar."""
    out_dir.mkdir(parents=True, exist_ok=True)
    from models import get_bilingual_type
    cond = get_bilingual_type(repo)

    summary = _load_ppl_summary()
    row = summary[(summary["repo"] == repo) & (summary["iso_code"] == lang)]
    fig, ax = fig_single(height=2.2)
    if row.empty:
        ax.text(0.5, 0.5, f"No PPL for\n{repo}\n({lang})", ha="center",
                va="center", transform=ax.transAxes, fontsize=7)
    else:
        ax.bar([cond], [row["mean_ppl"].iloc[0]],
               color=color_for_condition(cond), width=0.6)
        ax.set_ylabel(f"mean PPL ({lang})")
    ax.set_title(Path(repo).name, fontsize=8)
    stem = out_dir / f"{repo.replace('/', '__')}__{lang}"
    return _save_pub(fig, stem.name)


def figure_per_language(
    lang: str,
    *,
    family: str | None = None,
    condition: str | None = None,
    out_dir: Path = FIGURE_DIR / "per_language",
) -> Path:
    """Bar plot of mean PPL per condition for one language, optionally filtered by family."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, ".")
    from analyze.model_naming import parse_repo

    summary = _load_ppl_summary()
    summary = summary[summary["iso_code"] == lang].copy()

    def _keep(repo: str) -> bool:
        meta = parse_repo(repo)
        if family and meta.family != family:
            return False
        if condition and meta.condition != condition:
            return False
        return True

    summary = summary[summary["repo"].map(_keep)]
    if summary.empty:
        raise FileNotFoundError(
            f"No PPL rows for lang={lang} family={family} cond={condition}"
        )
    summary["condition"] = summary["repo"].map(lambda r: parse_repo(r).condition)
    grouped = summary.groupby("condition")["mean_ppl"].mean().sort_index()

    fig, ax = fig_single(height=2.3)
    colors = [color_for_condition(c) for c in grouped.index]
    ax.bar(grouped.index, grouped.values, color=colors, width=0.6)
    ax.set_ylabel(f"mean PPL ({lang})")
    ax.set_title(
        f"{lang.upper()} PPL by condition"
        + (f" — {family}" if family else "")
    )
    suffix = "_".join(filter(None, [family, condition, lang]))
    return _save_pub(fig, f"per_language_{suffix}")


def figure_compare(
    repos: list[str],
    axis: str,
    lang: str,
    *,
    out_dir: Path = FIGURE_DIR / "compare",
) -> Path:
    """Paired comparison figure for one of the six paper factors.

    ``axis`` ∈ {"onset","ratio","transition","burst","episodic","cl"}.
    Aggregates across seeds if present.
    """
    from analyze.model_naming import parse_repo, grouping_key

    summary = _load_ppl_summary()
    rows = summary[(summary["iso_code"] == lang) & summary["repo"].isin(repos)]
    if rows.empty:
        raise FileNotFoundError(
            f"No PPL data for axis={axis} lang={lang} (n={len(repos)})"
        )

    # Group by seed-stripped label, compute mean ± SE.
    keys = [grouping_key(parse_repo(r)) for r in rows["repo"]]
    rows = rows.assign(key=keys)
    agg = rows.groupby("key")["mean_ppl"].agg(["mean", "sem"]).reset_index()
    agg = agg.sort_values("mean")

    fig, ax = fig_single(height=2.5)
    xs = range(len(agg))
    ax.bar(xs, agg["mean"], yerr=agg["sem"].fillna(0.0),
           color=[color_for_condition(
               parse_repo(next(r for r in repos if grouping_key(parse_repo(r)) == k)).condition
           ) for k in agg["key"]],
           capsize=2, width=0.6)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(agg["key"], rotation=30, ha="right", fontsize=6)
    ax.set_ylabel(f"mean PPL ({lang})")
    ax.set_title(f"Factor: {axis} — {lang.upper()}")
    return _save_pub(fig, f"compare_{axis}_{lang}")


def figure_family_comparison(
    lang: str,
    condition: str,
    *,
    out_dir: Path = FIGURE_DIR / "family",
) -> Path:
    """Humanscale vs FineWeb vs BeetleLM-legacy on one panel for a given condition."""
    from analyze.model_naming import parse_repo

    summary = _load_ppl_summary()
    summary = summary[summary["iso_code"] == lang].copy()
    summary["family"]    = summary["repo"].map(lambda r: parse_repo(r).family)
    summary["condition"] = summary["repo"].map(lambda r: parse_repo(r).condition)
    summary = summary[summary["condition"] == condition]
    if summary.empty:
        raise FileNotFoundError(
            f"No data for lang={lang} cond={condition} across families"
        )

    agg = summary.groupby("family")["mean_ppl"].agg(["mean", "sem"]).reset_index()
    fig, ax = fig_single(height=2.2)
    xs = range(len(agg))
    ax.bar(xs, agg["mean"], yerr=agg["sem"].fillna(0.0),
           color=[color_for_condition(condition)] * len(agg),
           edgecolor="black", linewidth=0.8, width=0.6, capsize=2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(agg["family"], rotation=0)
    ax.set_ylabel(f"mean PPL ({lang})")
    ax.set_title(f"{condition} — family comparison ({lang.upper()})")
    return _save_pub(fig, f"family_compare_{condition}_{lang}")


if __name__ == "__main__":
    # Generate all available plots for all languages
    for lang in ["deu", "nld", "zho"]:
        try: ppl_heatmap()
        except Exception as e: print(f"[viz] ppl_heatmap failed: {e}")

        try: forgetting_barplot()
        except Exception as e: print(f"[viz] forgetting_barplot failed: {e}")

        for lang in ["deu", "nld", "zho"]:
            for fn in [forgetting_scatter, vocab_overlap_plot, probe_pca_plot, cka_heatmap,
                       ppl_convergence_overlay, drift_trajectory_plot, forgetting_inflection_plot]:
                try: fn(lang=lang)
                except Exception as e: print(f"[viz] {fn.__name__}({lang}) failed: {e}")

    # Convergence overlays
    for fn in [ppl_convergence_curves, cka_convergence_plot]:
        try: fn()
        except Exception as e: print(f"[viz] {fn.__name__} failed: {e}")
