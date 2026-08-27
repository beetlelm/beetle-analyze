"""
variant_comparison_100m.py — EMNLP-style variant-comparison figure for the
100M nld-eng B3 sweep.

One panel per downstream metric (BLiMP-eng, BLiMP-nl, BLiSS-RP@0,
BLiSS-RP@τ, BLiSS-NGS, BLiSS-CPS), bars for {baseline, ewc, lamol,
pre-pretrain}. Highlights per-variant Δ vs baseline so the reader can
see at a glance which training strategy moved the needle.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze.plot_style import (
    apply_acl_style,
    fig_one_half,
    fig_double,
    style_axes,
    despine,
    savefig_pub,
)

RESULT = Path("/root/beetle-analyze/results/100m-variants")
FIG_DIR = RESULT / "figs"

# `safe` is a real manipulation variant (the eval ledger confirms a
# distinct HF repo `…-nld-eng-safe`), not a baseline alias. The previous
# version of this script relabelled `safe → baseline` here, which (combined
# with a separate cache-namespacing bug in the FLORES driver) was the source
# of the byte-identical pre-pretrain ≡ baseline result. Keep variants distinct.
VARIANT_ORDER = ["baseline", "safe", "ewc", "lamol", "maml",
                 "pre-pretrain", "alibi", "layer-growth"]
VARIANT_COLORS = {
    "baseline":     "#000000",
    "safe":         "#999999",
    "ewc":          "#EE6677",
    "lamol":        "#4477AA",
    "maml":         "#228833",
    "pre-pretrain": "#CCBB44",
    "alibi":        "#AA3377",
    "layer-growth": "#66CCEE",
}


def _load_blimp() -> pd.DataFrame:
    df = pd.read_csv(RESULT / "blimp/blimp_results.csv")
    df["variant"] = df["model"].str.extract(r"-100m-nld-eng-([\w-]+?)$")[0]
    return df.pivot_table(
        index="variant", columns="benchmark", values="accuracy", aggfunc="first"
    )[["blimp_eng", "blimp_nl"]] * 100


def _load_bliss() -> pd.DataFrame:
    df = pd.read_csv(RESULT / "bliss/bliss_results.csv")
    df["variant"] = df["model"].str.extract(r"-100m-nld-eng-([\w-]+?)$")[0]

    def wavg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            col: (g[col] * g["n_items"]).sum() / g["n_items"].sum()
            for col in ("RP_at_0", "RP_at_tau", "NGS", "CPS")
        })

    return df.groupby("variant").apply(wavg)


def _load_meco() -> pd.DataFrame:
    df = pd.read_csv(RESULT / "meco/all_variants.csv")
    return df.groupby("variant")["surprisal"].agg(["mean", "median", "std"])


def _load_flores() -> pd.DataFrame:
    # Prefer the v2 (correctly namespaced) summary if present.
    v2 = RESULT / "flores/ppl_summary_v2.csv"
    if v2.exists():
        df = pd.read_csv(v2)
        # Recover variant label from the repo column.
        stem = "beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-nld-eng"
        def _v(r: str) -> str:
            name = r.rsplit("/", 1)[-1]
            if name == stem:
                return "baseline"
            if name.startswith(stem + "-"):
                return name[len(stem) + 1:]
            return name
        df["variant"] = df["repo"].apply(_v)
        # Resolve final step.
        def _step(rev):
            if not isinstance(rev, str) or rev == "main":
                return None
            if rev.startswith("step-"):
                try: return int(rev.split("-", 1)[1])
                except ValueError: return None
            try: return int(rev)
            except ValueError: return None
        df["step"] = df["revision"].apply(_step)
        df = df.dropna(subset=["step"])
        df = df[df["step"] == df["step"].max()].copy()
        df["ppl"] = df["mean_nll"].apply(math.exp)
        df = df.rename(columns={"iso_code": "iso"})
        return df.pivot(index="variant", columns="iso", values="ppl")[["eng", "nld"]]

    df = pd.read_csv(RESULT / "flores/ppl_summary.csv")
    df = df[df["step"] == 9096].copy()
    df["ppl"] = df["mean_nll"].apply(math.exp)
    return df.pivot(index="variant", columns="iso", values="ppl")[["eng", "nld"]]


def plot_metrics_grid() -> Path:
    apply_acl_style()
    blimp = _load_blimp()
    bliss = _load_bliss()
    meco  = _load_meco()
    flores = _load_flores()

    metrics = [
        ("BLiMP eng %",  blimp["blimp_eng"], "↑"),
        ("BLiMP nld %",  blimp["blimp_nl"],  "↑"),
        ("FLORES ppl eng", flores["eng"],  "↓"),
        ("FLORES ppl nld", flores["nld"],  "↓"),
        ("BLiSS RP@0",   bliss["RP_at_0"],   "↑"),
        ("BLiSS RP@τ",   bliss["RP_at_tau"], "↑"),
        ("BLiSS NGS",    bliss["NGS"],       "↑"),
        ("BLiSS CPS",    bliss["CPS"],       "↑"),
    ]

    # 8 metrics × 8 variants → keep 2×4 grid; bars per variant inside each panel.
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.0), constrained_layout=True)
    for ax, (title, series, direction) in zip(axes.flat, metrics):
        present = [v for v in VARIANT_ORDER if v in series.index]
        vals = [series.loc[v] for v in present]
        colors = [VARIANT_COLORS[v] for v in present]
        bars = ax.bar(range(len(present)), vals, color=colors, edgecolor="black",
                      linewidth=0.4, width=0.7)

        # baseline reference line
        if "baseline" in series.index:
            ax.axhline(series.loc["baseline"], color="0.4", linestyle="--",
                       linewidth=0.8, zorder=0)

        # value labels
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6)

        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, rotation=30, ha="right", fontsize=7)
        style_axes(ax, ylabel=None, title=f"{title} ({direction})")
        ax.tick_params(axis="y", labelsize=6)
        despine(ax)

        # tighten y to (min*0.99, max*1.01) for clarity on small differences
        if direction == "↑":
            ax.set_ylim(min(vals) * 0.97, max(vals) * 1.03)
        else:
            ax.set_ylim(min(vals) * 0.95, max(vals) * 1.05)

    fig.suptitle("100M nld-eng B3 — variant comparison (final ckpt step-9096)",
                 fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_variant_comparison_100m"
    savefig_pub(fig, out, formats=("pdf", "png"))
    plt.close(fig)
    return out.with_suffix(".pdf")


def emit_summary_table() -> Path:
    """Markdown table summarising all metrics."""
    blimp = _load_blimp()
    bliss = _load_bliss()
    meco  = _load_meco()
    flores = _load_flores()

    rows = []
    for v in VARIANT_ORDER:
        row = {"variant": v}
        if v in blimp.index:
            row["BLiMP-eng %"] = round(blimp.loc[v, "blimp_eng"], 3)
            row["BLiMP-nl %"]  = round(blimp.loc[v, "blimp_nl"], 3)
        if v in flores.index:
            row["FLORES eng PPL"] = round(flores.loc[v, "eng"], 2)
            row["FLORES nld PPL"] = round(flores.loc[v, "nld"], 2)
        if v in bliss.index:
            for c in ("RP_at_0", "RP_at_tau", "NGS", "CPS"):
                row[f"BLiSS {c}"] = round(bliss.loc[v, c], 3)
        if v in meco.index:
            row["MECO surprisal"] = round(meco.loc[v, "mean"], 3)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("variant")
    out = RESULT / "variant_metrics_summary.csv"
    df.to_csv(out)
    md = RESULT / "variant_metrics_summary.md"
    md.write_text("# 100M nld-eng B3 — variant metrics summary\n\n"
                  + df.to_markdown() + "\n")
    return md


def main() -> None:
    p = plot_metrics_grid()
    md = emit_summary_table()
    print(f"figure → {p}")
    print(f"summary → {md}")


if __name__ == "__main__":
    main()
