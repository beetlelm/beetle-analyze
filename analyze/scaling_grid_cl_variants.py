"""scaling_grid_cl_variants.py — CL-variant scaling figure.

Companion to ``scaling_grid_token_budget.py``. Shows how continual-learning
variants of the B3 sequential curriculum scale across FineWeb token budgets
on the two BLiMP suites (English and Dutch).

X-axis ticks: B3 (control), B3-ewc, B3-lamol, B3-pre-pretrain, B3-alibi
Token budgets (bar colour): FW-100M / FW-2B  (24B not yet evaluated for CL
variants — B3 control has FW-24B BLiMP data but the variants do not).

Inputs (read-only):
  - results/evals/blimp/blimp_results.csv   (benchmark = blimp_eng / blimp_nl)

Outputs:
  - results/figures/main/fig_cl_variants_scaling.{pdf,png}
  - results/figures/appendix/fig_cl_variants_scaling.{pdf,png}
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze.plot_style import (
    apply_acl_style,
    fig_double,
    savefig_pub,
    style_axes,
)

_log = logging.getLogger("scaling_grid_cl_variants")
ROOT = Path(__file__).resolve().parents[1]

# Variants to plot (in display order).
VARIANTS = ["B3", "B3-ewc", "B3-lamol", "B3-pre-pretrain", "B3-alibi"]

TOKEN_BUDGETS = [0.1, 2.0, 24.0]
BUDGET_COLOURS = {0.1: "#a6cee3", 2.0: "#1f78b4", 24.0: "#08306b"}
BUDGET_LABELS  = {0.1: "100M",   2.0: "2B",      24.0: "24B"}


def _parse_variant(model: str) -> str | None:
    """Map a B3 model slug to its variant tag."""
    if "sequential-33-67-b3" not in model and "sequential-0-100-b3" not in model:
        return None
    if "nld-eng" not in model:
        return None
    # Order matters — most specific suffix first.
    suffixes = [
        ("ewc", "B3-ewc"),
        ("lamol", "B3-lamol"),
        ("pre-pretrain", "B3-pre-pretrain"),
        ("alibi", "B3-alibi"),
    ]
    for needle, label in suffixes:
        if needle in model:
            return label
    # Plain B3 control (no postfix). Reject other postfixes we're not plotting.
    rejected = ("safe", "moe", "tilt", "sas", "lr", "gni", "lang-id",
                "layer-growth", "align", "episodic")
    if any(r in model for r in rejected):
        return None
    return "B3"


def _load_blimp(benchmark: str) -> pd.DataFrame:
    # Use the dedicated per-suite CSVs (uniform schema). The combined
    # blimp_results.csv has heterogeneous column counts that silently
    # drop the FineWeb rows under on_bad_lines="skip".
    fname = {"blimp_eng": "blimp_eng_results.csv",
             "blimp_nl":  "blimp_nl_results.csv"}[benchmark]
    df = pd.read_csv(ROOT / "results/blimp" / fname)
    df = df[df["benchmark"] == benchmark].copy()
    family_to_b = {
        "Beetle-FineWeb-100M":   0.1,
        "Beetle-FineWeb-2B":     2.0,
        "Beetle-FineWeb":        24.0,
        "Beetle-nld-eng-Variants": 2.0,   # the postfix-tagged 2B variants
    }
    def parse(model: str):
        org = model.split("/", 1)[0]
        return _parse_variant(model), family_to_b.get(org, np.nan)
    df[["variant", "tokens_b"]] = df["model"].apply(
        lambda m: pd.Series(parse(m))
    )
    df = df.dropna(subset=["variant", "tokens_b"])
    df = df.rename(columns={"accuracy": "y"})
    df = df.groupby(["variant", "tokens_b"], as_index=False)["y"].mean()
    return df[["variant", "tokens_b", "y"]]


def _draw_panel(ax, df: pd.DataFrame, *, ylabel: str, title: str,
                ylim: tuple[float, float]) -> None:
    bar_w = 0.26
    offsets = {0.1: -bar_w, 2.0: 0.0, 24.0: bar_w}
    ax.set_ylim(*ylim)
    base = ylim[0]

    x_centres = np.arange(len(VARIANTS))
    for v_idx, variant in enumerate(VARIANTS):
        for tb in TOKEN_BUDGETS:
            row = df[(df["variant"] == variant) & (df["tokens_b"] == tb)]
            x = x_centres[v_idx] + offsets[tb]
            if row.empty:
                continue
            y = float(row["y"].iloc[0])
            ax.bar(
                x, y - base, width=bar_w * 0.9, bottom=base,
                color=BUDGET_COLOURS[tb], edgecolor="white",
                linewidth=0.4, zorder=3,
            )

    ax.set_xticks(x_centres)
    ax.set_xticklabels(VARIANTS, rotation=20, ha="right")
    ax.set_xlim(-0.6, len(VARIANTS) - 0.4)
    style_axes(ax, ylabel=ylabel, title=title, grid=True)


def make_figure(out_main: Path, out_appendix: Path) -> None:
    apply_acl_style()

    blimp_eng = _load_blimp("blimp_eng")
    blimp_nl  = _load_blimp("blimp_nl")

    fig, (ax_eng, ax_nl) = fig_double(height=2.6, ncols=2, nrows=1)

    _draw_panel(
        ax_eng, blimp_eng,
        ylabel="BLiMP-eng accuracy",
        title="(a) English syntactic competence",
        ylim=(0.50, 0.72),
    )
    _draw_panel(
        ax_nl, blimp_nl,
        ylabel="BLiMP-NL accuracy",
        title="(b) Dutch syntactic competence",
        ylim=(0.55, 0.82),
    )

    for ax in (ax_eng, ax_nl):
        ax.set_xlabel("B3 sequential + CL variant")

    handles = [
        mlines.Line2D([], [], color=BUDGET_COLOURS[tb], marker="s",
                      markersize=7, linestyle="None",
                      label=f"FineWeb-{BUDGET_LABELS[tb]} tokens")
        for tb in TOKEN_BUDGETS
    ]
    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=len(handles), frameon=False, fontsize=7,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for out in (out_main, out_appendix):
        out.parent.mkdir(parents=True, exist_ok=True)
        savefig_pub(fig, out)
    plt.close(fig)
    _log.info("wrote %s and %s", out_main, out_appendix)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_main = ROOT / "results/figures/main/fig_cl_variants_scaling"
    out_appendix = ROOT / "results/figures/appendix/fig_cl_variants_scaling"
    make_figure(out_main, out_appendix)


if __name__ == "__main__":
    main()
