"""
learning_dynamics_100m.py — EMNLP-ready figures for the 100M nld-eng B3 sweep.

Two figures, both styled by `analyze.plot_style`:

  1. fig_loss_curve_100m.pdf
     train_loss vs step, one line per variant. Source = SWEEP.log "Step N |
     loss=…" lines (parsed from the per-job log file). x-axis log-scale to
     match the dyadic checkpoint schedule.

  2. fig_flores_ppl_100m.pdf
     mean FLORES PPL on eng and nld vs step, per variant. Source =
     /root/beetle-analyze/results/100m-variants/flores/ppl_summary.csv,
     produced by script/eval_100m_variants/eval_flores_per_ckpt.py.

Both figures land in /root/beetle-analyze/results/100m-variants/figs/ as
PDF + PNG so they can be added to the EMNLP draft straight from the
100m-variants branch.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from analyze.plot_style import (
    apply_acl_style,
    fig_one_half,
    style_axes,
    savefig_pub,
    despine,
    compact_legend,
    CL_VARIANT_MARKERS,
)

# ── Paths ──────────────────────────────────────────────────────────────────
LOG_DIR = Path("/root/beetlelm/logs/nld_eng_b3_100m_safe")
FLORES_SUMMARY = Path(
    "/root/beetle-analyze/results/100m-variants/flores/ppl_summary.csv"
)
FIG_DIR = Path("/root/beetle-analyze/results/100m-variants/figs")

# ── Variants & per-variant style (extends plot_style.CL_VARIANT_MARKERS) ──
# log-file basename ↔ short variant label used in plots.
LOG_FILES: dict[str, str] = {
    "baseline":     "fineweb_nld_eng_b3_100m.log",
    "ewc":          "fineweb_nld_eng_b3_100m_ewc.log",
    "lamol":        "fineweb_nld_eng_b3_100m_lamol.log",
    "maml":         "fineweb_nld_eng_b3_100m_maml.log",
    "pre-pretrain": "fineweb_nld_eng_b3_100m_pre_pretrain.log",
    "alibi":        "fineweb_nld_eng_b3_100m_alibi.log",
    "layer-growth": "fineweb_nld_eng_b3_100m_layer_growth.log",
}

# Re-use existing markers; add the three new variants.
VARIANT_MARKERS: dict[str, str] = {
    "baseline":     CL_VARIANT_MARKERS.get("none", "o"),
    "ewc":          CL_VARIANT_MARKERS["ewc"],
    "lamol":        CL_VARIANT_MARKERS["lamol"],
    "maml":         CL_VARIANT_MARKERS["maml"],
    "pre-pretrain": "P",   # plus-filled
    "alibi":        "X",   # x-filled
    "layer-growth": "h",   # hex
}

# Paul Tol bright + dark2 mix; high contrast & colour-blind safe.
VARIANT_COLORS: dict[str, str] = {
    "baseline":     "#000000",
    "ewc":          "#EE6677",   # red
    "lamol":        "#4477AA",   # blue
    "maml":         "#228833",   # green
    "pre-pretrain": "#CCBB44",   # yellow
    "alibi":        "#AA3377",   # purple
    "layer-growth": "#66CCEE",   # cyan
}

VARIANT_ORDER = list(LOG_FILES.keys())


# ── Helpers ────────────────────────────────────────────────────────────────
_STEP_LOSS = re.compile(
    r"Step\s+(\d+)\s*\|\s*loss=([0-9.]+)"
)
_PHASE = re.compile(r"phase=(\w+)")


def _parse_log(path: Path) -> pd.DataFrame:
    """Pull (step, loss, phase) tuples out of a single training log."""
    if not path.exists():
        return pd.DataFrame(columns=["step", "loss", "phase"])
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        m = _STEP_LOSS.search(line)
        if not m:
            continue
        step, loss = int(m.group(1)), float(m.group(2))
        phase_m = _PHASE.search(line)
        rows.append(
            {
                "step": step,
                "loss": loss,
                "phase": phase_m.group(1) if phase_m else "",
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset="step", keep="last")


def _load_loss_curves() -> pd.DataFrame:
    frames = []
    for variant, fname in LOG_FILES.items():
        df = _parse_log(LOG_DIR / fname)
        if df.empty:
            continue
        df["variant"] = variant
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["step", "loss", "phase", "variant"])
    return pd.concat(frames, ignore_index=True)


def _phase_boundary(df: pd.DataFrame) -> Optional[int]:
    """Largest step in l1_dominant phase = curriculum boundary (B3 puts it ~5000)."""
    l1 = df[df["phase"] == "l1_dominant"]
    return int(l1["step"].max()) if not l1.empty else None


# ── Figure 1: training loss curves ─────────────────────────────────────────
def plot_loss_curves() -> Path:
    apply_acl_style()
    df = _load_loss_curves()
    if df.empty:
        print("WARNING: no loss data parsed yet; skipping loss-curve figure.")
        return FIG_DIR / "fig_loss_curve_100m.pdf"

    fig, ax = fig_one_half(height=2.6)
    for variant in VARIANT_ORDER:
        sub = df[df["variant"] == variant].sort_values("step")
        if sub.empty:
            continue
        ax.plot(
            sub["step"].clip(lower=1),
            sub["loss"],
            color=VARIANT_COLORS[variant],
            marker=VARIANT_MARKERS[variant],
            markersize=3.5,
            linewidth=1.2,
            label=variant,
        )

    boundary = _phase_boundary(df)
    if boundary is not None and boundary > 0:
        ax.axvline(boundary, color="0.55", linestyle=":", linewidth=0.7, zorder=0)
        ax.annotate(
            "L1→L2 boundary",
            xy=(boundary, ax.get_ylim()[1]),
            xytext=(2, -8),
            textcoords="offset points",
            fontsize=6,
            color="0.4",
            ha="left",
            va="top",
        )

    ax.set_xscale("log")
    style_axes(ax, xlabel="training step (log)", ylabel="train loss")
    despine(ax)
    compact_legend(ax, ncol=2, loc="upper right")
    ax.set_title("100M FineWeb · nld L1 → eng L2 (B3) · loss curves", fontsize=8)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_loss_curve_100m"
    savefig_pub(fig, out, formats=("pdf", "png"))
    plt.close(fig)
    return out.with_suffix(".pdf")


# ── Figure 2: FLORES PPL per checkpoint ────────────────────────────────────
def plot_flores_ppl() -> Path:
    apply_acl_style()
    if not FLORES_SUMMARY.exists():
        print(f"WARNING: {FLORES_SUMMARY} missing; run eval_flores_per_ckpt.py first.")
        return FIG_DIR / "fig_flores_ppl_100m.pdf"

    df = pd.read_csv(FLORES_SUMMARY)
    df["ppl"] = df["mean_nll"].apply(math.exp)

    fig, axes = fig_one_half(height=2.6, ncols=2)
    for ax, iso in zip(axes, ["nld", "eng"]):
        sub_iso = df[df["iso"] == iso]
        for variant in VARIANT_ORDER:
            sub = sub_iso[sub_iso["variant"] == variant].sort_values("step")
            if sub.empty:
                continue
            ax.plot(
                sub["step"].clip(lower=1),
                sub["ppl"],
                color=VARIANT_COLORS[variant],
                marker=VARIANT_MARKERS[variant],
                markersize=3.5,
                linewidth=1.2,
                label=variant if iso == "nld" else None,
            )
        boundary = 5000  # B3 curriculum L1→L2 boundary (canonical)
        ax.axvline(boundary, color="0.55", linestyle=":", linewidth=0.7, zorder=0)

        ax.set_xscale("log")
        ax.set_yscale("log")
        style_axes(
            ax,
            xlabel="training step (log)",
            ylabel="FLORES PPL (log)" if iso == "nld" else None,
            title=f"{iso}",
        )
        despine(ax)

    compact_legend(axes[0], ncol=2, loc="upper right")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_flores_ppl_100m"
    savefig_pub(fig, out, formats=("pdf", "png"))
    plt.close(fig)
    return out.with_suffix(".pdf")


def main() -> None:
    p1 = plot_loss_curves()
    p2 = plot_flores_ppl()
    print(f"loss curves   → {p1}")
    print(f"FLORES PPL    → {p2}")


if __name__ == "__main__":
    main()
