"""
flores_grid_100m.py — main grid figure for the 100M nld-eng B3 manipulation
sweep.

Layout: 2 rows × N cols (default N = 8 variants):
    row 0 = English NLL trajectory
    row 1 = Dutch   NLL trajectory
    cols  = baseline, safe, ewc, lamol, maml, pre-pretrain, alibi, layer-growth

In every panel we draw the *baseline* trajectory in light grey as a fixed
reference, then the variant-of-this-column in its accent colour on top. This
makes the per-variant deviation from baseline read at a glance — the answer
to "did this manipulation move the needle?".

A dotted vertical line marks the L1→L2 schedule switch (step 5000 for B3).

Source CSV: results/100m-variants/flores/ppl_summary_v2.csv (produced by
scripts/eval_flores_100m_variants.py). Falls back to the legacy
ppl_summary.csv with a warning printed.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analyze.plot_style import (  # noqa: E402
    apply_acl_style,
    style_axes,
    despine,
    savefig_pub,
)

RESULT_DIR = REPO_ROOT / "results" / "100m-variants"
FIG_DIR = RESULT_DIR / "figs"

VARIANT_ORDER = [
    "baseline", "safe", "ewc", "lamol",
    "maml", "pre-pretrain", "alibi", "layer-growth",
]

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

# B3 sequential 33-67: switch happens around step 5000 (canonical).
SCHEDULE_BOUNDARY = 5000

STEM = "beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-nld-eng"


def _variant_from_repo(repo: str) -> str:
    """Recover the variant label from a HF repo path produced by
    eval_flores_100m_variants.py."""
    name = repo.rsplit("/", 1)[-1]
    if name == STEM:
        return "baseline"
    if name.startswith(STEM + "-"):
        return name[len(STEM) + 1:]
    return name  # unknown — keep as-is so anomalies are visible


def _step_from_revision(rev: str) -> int | None:
    if not isinstance(rev, str):
        return None
    if rev == "main":
        return None  # exclude from trajectory
    if rev.startswith("step-"):
        try:
            return int(rev.split("-", 1)[1])
        except ValueError:
            return None
    try:
        return int(rev)
    except ValueError:
        return None


def _load() -> pd.DataFrame:
    """Return tidy DataFrame [variant, step, iso, mean_nll]."""
    new_path = RESULT_DIR / "flores" / "ppl_summary_v2.csv"
    if new_path.exists():
        df = pd.read_csv(new_path)
        df["variant"] = df["repo"].apply(_variant_from_repo)
        df["step"] = df["revision"].apply(_step_from_revision)
        df = df.dropna(subset=["step"]).copy()
        df["step"] = df["step"].astype(int)
        df = df.rename(columns={"iso_code": "iso"})
        return df[["variant", "step", "iso", "mean_nll"]]

    legacy = RESULT_DIR / "flores" / "ppl_summary.csv"
    print(f"[grid] WARNING: {new_path} not found, falling back to {legacy}. "
          "This file is the buggy first-round output where pre-pretrain ≡ "
          "baseline — the resulting figure will not show the manipulation "
          "effect. Re-run scripts/eval_flores_100m_variants.sh.",
          file=sys.stderr)
    df = pd.read_csv(legacy)
    return df[["variant", "step", "iso", "mean_nll"]]


def plot_grid(*, log_y: bool = True) -> Path:
    apply_acl_style()
    df = _load()

    present = [v for v in VARIANT_ORDER if v in set(df["variant"])]
    if "baseline" not in present:
        print("[grid] WARNING: no `baseline` rows — reference grey trace will "
              "be omitted.", file=sys.stderr)
    n = len(present)

    # Width scales with column count; one panel per (lang, variant).
    fig, axes = plt.subplots(
        2, n,
        figsize=(1.45 * n + 0.7, 4.0),
        sharex=True, sharey="row",
        constrained_layout=True,
    )
    if n == 1:
        axes = axes.reshape(2, 1)

    base_eng = (df[(df["variant"] == "baseline") & (df["iso"] == "eng")]
                .sort_values("step")) if "baseline" in present else None
    base_nld = (df[(df["variant"] == "baseline") & (df["iso"] == "nld")]
                .sort_values("step")) if "baseline" in present else None

    for j, variant in enumerate(present):
        for i, (iso, base) in enumerate([("eng", base_eng), ("nld", base_nld)]):
            ax = axes[i, j]
            # Reference: baseline in grey under everything.
            if base is not None and not base.empty and variant != "baseline":
                ax.plot(base["step"].clip(lower=1), base["mean_nll"],
                        color="0.75", linewidth=1.0, zorder=1,
                        label="baseline" if (i == 0 and j == 0) else None)
            sub = (df[(df["variant"] == variant) & (df["iso"] == iso)]
                   .sort_values("step"))
            if not sub.empty:
                ax.plot(sub["step"].clip(lower=1), sub["mean_nll"],
                        color=VARIANT_COLORS.get(variant, "C0"),
                        linewidth=1.5, marker="o", markersize=3, zorder=2,
                        label=variant if (i == 0 and j == 0) else None)
            ax.axvline(SCHEDULE_BOUNDARY, color="0.55",
                       linestyle=":", linewidth=0.7, zorder=0)
            ax.set_xscale("log")
            if log_y:
                ax.set_yscale("log")
            despine(ax)
            ax.tick_params(axis="both", labelsize=6)
            if i == 0:
                ax.set_title(variant, fontsize=7,
                             color=VARIANT_COLORS.get(variant, "k"))
            if j == 0:
                style_axes(ax,
                           ylabel=f"{iso.upper()} mean NLL"
                                  + (" (log)" if log_y else ""),
                           xlabel="step (log)" if i == 1 else None)
            else:
                style_axes(ax, ylabel=None,
                           xlabel="step (log)" if i == 1 else None)

    fig.suptitle(
        "FLORES NLL trajectories  ·  nld-eng B3 100M  ·  variant grid "
        f"(switch @ step {SCHEDULE_BOUNDARY})",
        fontsize=9,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_flores_grid_100m"
    savefig_pub(fig, out, formats=("pdf", "png"))
    plt.close(fig)
    return out.with_suffix(".pdf")


def emit_final_table() -> Path:
    """Persist a concise per-variant table at step-9096 (eng / nld PPL +
    Δ vs baseline). Lands at results/100m-variants/flores/final_step_table.md."""
    df = _load()
    final = df[df["step"] == df["step"].max()].copy()
    final["ppl"] = final["mean_nll"].apply(math.exp)
    pivot = final.pivot_table(index="variant", columns="iso", values="ppl")
    if "baseline" in pivot.index:
        for col in pivot.columns:
            pivot[f"Δ {col}"] = pivot[col] - pivot.loc["baseline", col]

    pivot = pivot.reindex([v for v in VARIANT_ORDER if v in pivot.index])
    out = RESULT_DIR / "flores" / "final_step_table.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"# FLORES PPL @ final ckpt (step-{int(df['step'].max())})\n\n"
        + pivot.round(2).to_markdown() + "\n"
    )
    return out


def main() -> None:
    pdf = plot_grid()
    md = emit_final_table()
    print(f"grid figure → {pdf}")
    print(f"final table → {md}")


if __name__ == "__main__":
    main()
