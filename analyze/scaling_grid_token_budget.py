"""scaling_grid_token_budget.py — Holistic 2x2 token-budget scaling figure.

Investigates the effect of FineWeb training token budget (100M / 2B / 24B) on
four orthogonal evaluation axes for the nld-eng Beetle bilingual LMs:

    [BLiSS RP@0]     [FLORES BPT (Dutch)]
    [CEFR gradient]  [BLiMP-NL accuracy]

The four panels share a log-scaled x-axis in tokens, so reviewers can see at a
glance that the four eval suites live on *different* scaling regimes:
pedagogical metrics are non-monotonic, structural alignment is curriculum-
conditional, syntax saturates early, translation scales cleanly.

Inputs (read-only):
  - paper/emnlp/main_body/data/F7_bliss_curriculum_x_scale.csv  (BLiSS RP@0)
  - paper/emnlp/main_body/data/F10_flores_scale_curve.csv       (FLORES BPT)
  - plots/headline_summary_table.csv                            (CEFR-grad)
  - results/evals/blimp/blimp_results.csv                       (BLiMP-NL)

Outputs:
  - results/figures/main/fig_token_budget_scaling.{pdf,png}
  - results/figures/appendix/fig_token_budget_scaling.{pdf,png}

Missing cells (e.g. BLiSS @ 2B, CEFR-grad @ 24B) are rendered as a hollow
marker with an inline "pending" annotation so the figure stays informative
while the underlying eval is dispatched.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze.plot_style import (
    PALETTE_ACL,
    apply_acl_style,
    fig_double,
    savefig_pub,
    style_axes,
)

_log = logging.getLogger("scaling_grid_token_budget")
ROOT = Path(__file__).resolve().parents[1]

# Token-budget axis (in B tokens, log-spaced labels).
TOKEN_BUDGETS = [0.1, 2.0, 24.0]
TOKEN_LABELS  = ["100M", "2B", "24B"]

# Curricula on the x-axis. All five core curricula + monolingual baseline.
CURRICULA = ["B1", "B2", "B3", "B4", "B5", "mono"]

# Per-budget bar colours (cool→warm, perceptually ordered with token budget).
BUDGET_COLOURS = {
    0.1:  "#a6cee3",   # light blue   — 100M
    2.0:  "#1f78b4",   # mid blue     — 2B
    24.0: "#08306b",   # dark navy    — 24B
}
BUDGET_LABELS = {0.1: "100M", 2.0: "2B", 24.0: "24B"}


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders — each returns a curriculum × tokens_b → value (+ optional CI)
# ─────────────────────────────────────────────────────────────────────────────

def _load_bliss() -> pd.DataFrame:
    """BLiSS RP@0 (Dutch L1) for nld-eng. 2B column is currently empty."""
    df = pd.read_csv(ROOT / "paper/emnlp/main_body/data/F7_bliss_curriculum_x_scale.csv")
    df = df[(df["pair"] == "nld-eng") & (df["target_lang"] == "Dutch")].copy()
    # Map scale → tokens_b. F7 has only HumanScale (100M) and FineWeb (24B).
    scale_to_b = {"HumanScale": 0.1, "FineWeb": 24.0}
    df["tokens_b"] = df["scale"].map(scale_to_b)
    df = df.rename(columns={"condition": "curriculum", "rp0": "y"})
    return df[["curriculum", "tokens_b", "y", "ci_lo", "ci_hi"]]


def _load_flores() -> pd.DataFrame:
    """FLORES BPT (Dutch direction). All three scales present for B1-B5."""
    df = pd.read_csv(ROOT / "paper/emnlp/main_body/data/F10_flores_scale_curve.csv")
    df = df.rename(columns={"scale_b": "tokens_b", "nld_bpt": "y"})
    return df[["curriculum", "tokens_b", "y"]]


def _load_cefr_grad() -> pd.DataFrame:
    """CEFR-stratified BPT gradient (C2 − A1, bits). Higher = better.

    HumanScale → 0.1B, FineWeb-2B → 2B. FineWeb-24B not yet evaluated
    (gated on the 500M / 1B scale-up run).
    """
    df = pd.read_csv(ROOT / "plots/headline_summary_table.csv")
    df = df[df["lang_pair"] == "nld-eng"].copy()
    family_to_b = {"HumanScale": 0.1, "FineWeb-2B": 2.0, "FineWeb-24B": 24.0}
    df["tokens_b"] = df["family"].map(family_to_b)
    df["curriculum"] = df["curriculum"].str.upper().replace({"MONO": "mono"})
    df = df.rename(columns={"cefr_gradient_bits": "y"})
    df = df.dropna(subset=["tokens_b", "y"])
    return df[["curriculum", "tokens_b", "y"]]


def _load_blimp_nl() -> pd.DataFrame:
    """BLiMP-NL accuracy. Parses curriculum + scale from the model slug."""
    # Use the dedicated blimp_nl CSV (uniform schema). The combined
    # blimp_results.csv has heterogeneous column counts that silently
    # drop the FineWeb rows under on_bad_lines="skip".
    df = pd.read_csv(ROOT / "results/blimp/blimp_nl_results.csv")
    df = df[df["benchmark"] == "blimp_nl"].copy()
    # Restrict to nld-eng bilingual + monolingual-eng baselines.
    df = df[
        df["model"].str.contains("nld-eng", regex=False)
        | df["model"].str.contains("monolingual-humanscale-eng", regex=False)
    ].copy()
    # Drop variants/postfix tags so we only see clean curriculum cells.
    df = df[~df["model"].str.contains("postfix|alibi|ewc|lamol|safe|pre-pretrain|moe|sas|tilt|lr-boost|gni|align|lang-id|layer-growth|episodic")]

    family_to_b = {
        "Beetle-FineWeb-100M": 0.1,
        "Beetle-FineWeb-2B":   2.0,
        "Beetle-FineWeb":      24.0,
        "Beetle-HumanScale":   0.1,    # for the mono baseline only
    }

    def parse(model: str):
        org, repo = model.split("/", 1)
        tokens_b = family_to_b.get(org, np.nan)
        if "monolingual-humanscale-eng" in repo:
            curr = "mono"
        elif "balanced-b1" in repo:
            curr = "B1"
        elif "simultaneous-b2" in repo:
            curr = "B2"
        elif "sequential-33-67-b3" in repo:
            curr = "B3"
        elif "classroom-20-b4" in repo:
            curr = "B4"
        elif "late-b5" in repo:
            curr = "B5"
        else:
            curr = None
        return curr, tokens_b

    df[["curriculum", "tokens_b"]] = df["model"].apply(
        lambda m: pd.Series(parse(m))
    )
    df = df.dropna(subset=["curriculum", "tokens_b"])
    df = df.rename(columns={"accuracy": "y"})
    # Average duplicate rows (eval suite was re-run with iso_code added).
    df = df.groupby(["curriculum", "tokens_b"], as_index=False)["y"].mean()
    return df[["curriculum", "tokens_b", "y"]]


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _draw_panel(ax, df: pd.DataFrame, *, ylabel: str, title: str,
                higher_is_better: bool = True, ci: bool = False,
                ylim: tuple[float, float] | None = None,
                baseline: float | None = None,
                include_mono: bool = True) -> None:
    """Draw one panel as grouped bars: curricula on x, 3 bars per group.

    Bars are drawn relative to ``baseline`` (default = ylim lower bound) so
    small differences are visible without zero-padding the y-axis.
    """
    bar_w = 0.26
    offsets = {0.1: -bar_w, 2.0: 0.0, 24.0: bar_w}
    curricula = CURRICULA if include_mono else [c for c in CURRICULA if c != "mono"]

    if ylim is not None:
        ax.set_ylim(*ylim)
    base = baseline if baseline is not None else (
        ylim[0] if ylim is not None else 0.0
    )

    x_centres = np.arange(len(curricula))
    for curr_idx, curr in enumerate(curricula):
        for tb in TOKEN_BUDGETS:
            row = df[(df["curriculum"] == curr) & (df["tokens_b"] == tb)]
            x = x_centres[curr_idx] + offsets[tb]
            if row.empty:
                continue   # leave a gap; reader sees the missing column.
            y = float(row["y"].iloc[0])
            err = None
            if ci and {"ci_lo", "ci_hi"}.issubset(row.columns):
                lo, hi = float(row["ci_lo"].iloc[0]), float(row["ci_hi"].iloc[0])
                err = [[max(0, y - lo)], [max(0, hi - y)]]
            ax.bar(
                x, y - base, width=bar_w * 0.9, bottom=base,
                color=BUDGET_COLOURS[tb], edgecolor="white",
                linewidth=0.4, zorder=3,
                yerr=err, error_kw={"linewidth": 0.6, "ecolor": "#222",
                                    "capsize": 1.5, "capthick": 0.6},
            )

    ax.set_xticks(x_centres)
    ax.set_xticklabels(curricula)
    ax.set_xlim(-0.6, len(curricula) - 0.4)
    style_axes(ax, ylabel=ylabel, title=title, grid=True)
    if not higher_is_better:
        ax.invert_yaxis()


def make_figure(out_main: Path, out_appendix: Path) -> None:
    apply_acl_style()

    bliss = _load_bliss()
    flores = _load_flores()
    cefr  = _load_cefr_grad()
    blimp = _load_blimp_nl()

    fig, axes = fig_double(height=4.6, nrows=2, ncols=2)
    (ax_bliss, ax_flores), (ax_cefr, ax_blimp) = axes

    _draw_panel(
        ax_bliss, bliss,
        ylabel="BLiSS RP@0 (Dutch L2)",
        title="(a) Cross-lingual L2 alignment",
        ci=True, ylim=(50, 65), include_mono=False,
    )
    _draw_panel(
        ax_flores, flores,
        ylabel="FLORES BPT (Dutch, ↓)",
        title="(b) Translation quality",
        higher_is_better=False, ylim=(8.0, 5.5), baseline=8.0,
        include_mono=False,
    )
    _draw_panel(
        ax_cefr, cefr,
        ylabel="CEFR gradient (bits, C2−A1)",
        title="(c) Pedagogical stratification",
        ylim=(-0.6, 3.5), baseline=0.0,
    )
    ax_cefr.axhline(0, color="#444", linewidth=0.7, linestyle="-", zorder=1)
    _draw_panel(
        ax_blimp, blimp,
        ylabel="BLiMP-NL accuracy",
        title="(d) Dutch syntactic competence",
        ylim=(0.45, 0.85),
    )

    # Shared x-label.
    for ax in (ax_cefr, ax_blimp):
        ax.set_xlabel("Curriculum")

    # Pending-eval annotations for the two known gaps (placed above panel).
    ax_bliss.text(
        0.99, 0.02, "2B column: eval pending",
        transform=ax_bliss.transAxes, ha="right", va="bottom",
        fontsize=6, style="italic", color="#666",
    )
    ax_cefr.text(
        0.99, 0.02, "24B column: gated on 500M run",
        transform=ax_cefr.transAxes, ha="right", va="bottom",
        fontsize=6, style="italic", color="#666",
    )

    # Shared legend (token budget = colour).
    handles = [
        mlines.Line2D([], [], color=BUDGET_COLOURS[tb], marker="s",
                      markersize=7, linestyle="None",
                      label=f"FineWeb-{BUDGET_LABELS[tb]} tokens")
        for tb in TOKEN_BUDGETS
    ]
    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(handles), frameon=False, fontsize=7,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    for out in (out_main, out_appendix):
        out.parent.mkdir(parents=True, exist_ok=True)
        savefig_pub(fig, out)
    plt.close(fig)
    _log.info("wrote %s and %s", out_main, out_appendix)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_main = ROOT / "results/figures/main/fig_token_budget_scaling"
    out_appendix = ROOT / "results/figures/appendix/fig_token_budget_scaling"
    make_figure(out_main, out_appendix)


if __name__ == "__main__":
    main()
