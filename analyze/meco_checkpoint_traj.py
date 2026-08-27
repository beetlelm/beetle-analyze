"""meco_checkpoint_traj.py — ΔlogLik trajectory across training checkpoints.

Plots ΔlogLik vs training step for the universal-best Beetle nld-eng model
(HumanScale B3 sequential 33:67 nld-eng), one panel per MECO language.
The shape mirrors the BLiMP checkpoint-trajectory grid so the appendix
reads consistently.

Inputs
------
``results/meco/canonical_deltalogl_b3_nld_eng_traj.csv`` produced by the
sister fit script.

Outputs
-------
``results/figures/main/fig_meco_b3_nld_eng_traj.{pdf,png}``
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

from analyze.plot_style import apply_acl_style, savefig_pub

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "meco" / "canonical_deltalogl_b3_nld_eng_traj.csv"
MAIN_CACHE = ROOT / "results" / "meco" / "canonical_deltalogl.csv"
MAIN_DIR = ROOT / "results" / "figures" / "main"

LANG_LABEL = {"du": "Dutch", "ge": "German", "ch_s": "Chinese"}
LANG_ORDER = ["du", "ge", "ch_s"]
COLOR = "#1b9e77"

_STEP_RE = re.compile(r"@step-(\d+)$")


def _step(model: str) -> int | None:
    m = _STEP_RE.search(model)
    return int(m.group(1)) if m else None


def render(cache: Path = CACHE, main_cache: Path = MAIN_CACHE,
           output_dir: Path = MAIN_DIR) -> Path:
    apply_acl_style()
    df = pd.read_csv(cache).dropna(subset=["delta_loglik"])
    df["step"] = df["model"].apply(_step)
    df = df.dropna(subset=["step"])
    df["step"] = df["step"].astype(int)

    # Final-checkpoint ΔlogLik for the same model (from the main cache),
    # plotted as a horizontal reference dotted line per language.
    final = pd.read_csv(main_cache).dropna(subset=["delta_loglik"])
    final = final[final["model"] ==
                  "beetle-humanscale/beetle-bilingual-l2-50-sequential-33-67-b3-humanscale-nld-eng"]
    final = final.set_index("lang")["delta_loglik"].to_dict()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
    Y_MAX = 0
    for ax, lang in zip(axes, LANG_ORDER):
        sub = df[df["lang"] == lang].sort_values("step")
        if sub.empty:
            ax.text(0.5, 0.5, "no fits yet", ha="center", va="center",
                    transform=ax.transAxes, color="#aaa")
            continue
        ax.plot(sub["step"].clip(lower=1), sub["delta_loglik"],
                color=COLOR, linestyle="-", linewidth=1.4,
                marker="o", markersize=3.4, alpha=0.95, zorder=3)
        # Final-checkpoint reference line.
        if lang in final:
            ax.axhline(final[lang], color=COLOR, linestyle=":",
                       linewidth=0.9, alpha=0.7, zorder=2)
            ax.text(1.05, final[lang],
                    f" main = {final[lang]:.0f}",
                    color=COLOR, fontsize=7.2, va="center", ha="left",
                    transform=ax.get_yaxis_transform())
        Y_MAX = max(Y_MAX, sub["delta_loglik"].max(), final.get(lang, 0))
        ax.set_xscale("log")
        ax.axhline(0, color="#bbb", linestyle=":", linewidth=0.5, zorder=1)
        ax.set_xlabel("training step (log)")
        if lang == "du":
            ax.set_ylabel("ΔlogLik (full vs baseline)")
        ax.set_title(f"{LANG_LABEL[lang]} (MECO)", fontsize=9.5)
        ax.grid(True, axis="y", linestyle=":", color="#cfcfcf", zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    # Common y-limit, with a little padding above the global max.
    if Y_MAX > 0:
        for ax in axes:
            ax.set_ylim(-15, Y_MAX * 1.08)
    fig.suptitle(
        "MECO ΔlogLik across training — Beetle HumanScale B3 sequential 33:67 nld-eng",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    out = output_dir / "fig_meco_b3_nld_eng_traj"
    savefig_pub(fig, out)
    plt.close(fig)
    return out.with_suffix(".pdf")


def main() -> None:
    print(f"[meco_checkpoint_traj] wrote {render()}")


if __name__ == "__main__":
    main()
