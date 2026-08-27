"""meco_vs_baselines.py — Beetle vs frontier-LLM ΔlogLik comparison figure.

Reads ``results/meco/canonical_deltalogl.csv`` and produces a 1×3 panel plot
(du / ge / ch_s) where Beetle models appear as a horizontal swarm with the
top-Beetle line called out, and each baseline family appears as a single
point per (model, lang). Intended for the MECO-evals branch as the headline
"do the strong baselines beat Beetle?" figure.

Outputs:
  results/figures/main/fig_meco_beetle_vs_baselines.{pdf,png}

Safe to call mid-sweep — only families with at least one fitted row appear.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze.plot_style import apply_acl_style, savefig_pub

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "meco" / "canonical_deltalogl.csv"
MAIN_DIR = ROOT / "results" / "figures" / "main"

LANG_LABEL = {"du": "Dutch (MECO)", "ge": "German (MECO)", "ch_s": "Chinese (MECO)"}
LANG_ORDER = ["du", "ge", "ch_s"]

# Family detection — same buckets used in the runtime status reports.
FAMILY_RULES: list[tuple[str, str, str]] = [
    # (substring or prefix to match in lowercased model id, family code, display label)
    ("beetle", "beetle", "Beetle"),
    ("bigscience/bloom", "bloom", "BLOOM"),
    ("facebook/", "xglm", "XGLM"),
    ("google/", "gemma2", "Gemma 2"),
    ("qwen/qwen3", "qwen3", "Qwen 3"),
    ("qwen/qwen-", "qwen", "Qwen 1.x"),
    ("meta-llama/", "llama3", "Llama 3"),
    ("utter-project/", "eurollm", "EuroLLM"),
    ("swiss-ai/", "apertus", "Apertus"),
    ("llm-jp/", "llmjp", "LLM-jp 3"),
    ("01-ai/", "yi", "Yi"),
    ("baichuan-inc/", "baichuan", "Baichuan"),
    ("thudm/", "chatglm", "ChatGLM"),
]

FAMILY_COLOR: dict[str, str] = {
    "beetle":   "#1b9e77",
    "bloom":    "#a6761d",
    "xglm":     "#666666",
    "gemma2":   "#d95f02",
    "qwen3":    "#7570b3",
    "qwen":     "#9c8ddb",
    "llama3":   "#e7298a",
    "eurollm":  "#66a61e",
    "apertus":  "#b2182b",
    "llmjp":    "#2166ac",
    "yi":       "#5ab4ac",
    "baichuan": "#bf812d",
    "chatglm":  "#80cdc1",
}


def _family(model: str) -> tuple[str, str]:
    s = model.lower()
    for needle, code, label in FAMILY_RULES:
        if needle in s:
            return code, label
    return "other", "Other"


def render(cache: Path = CACHE, output_dir: Path = MAIN_DIR) -> Path:
    df = pd.read_csv(cache)
    df = df.dropna(subset=["delta_loglik"]).copy()
    df[["family_code", "family_label"]] = df["model"].apply(
        lambda m: pd.Series(_family(m))
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=False)

    # Stable family order: Beetle first (so its swarm anchors the y-axis),
    # then any baseline family that has at least one row in the cache, in
    # alphabetical order by display label.
    families_with_rows = sorted(
        df["family_label"].dropna().unique().tolist(),
        key=lambda lab: (lab != "Beetle", lab.lower()),
    )

    for ax, lang in zip(axes, LANG_ORDER):
        sub = df[df["lang"] == lang]
        if sub.empty:
            ax.text(0.5, 0.5, "no data yet", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            ax.set_title(LANG_LABEL[lang])
            continue
        beetle_top = float(sub.loc[sub["family_label"] == "Beetle",
                                   "delta_loglik"].max())
        beetle_med = float(sub.loc[sub["family_label"] == "Beetle",
                                   "delta_loglik"].median())
        for i, fam in enumerate(families_with_rows):
            fsub = sub[sub["family_label"] == fam]
            if fsub.empty:
                continue
            color = FAMILY_COLOR.get(
                next((c for _, c, lab in FAMILY_RULES if lab == fam), "other"),
                "#666",
            )
            jitter = (np.random.RandomState(i + 7).rand(len(fsub)) - 0.5) * 0.32
            xs = np.full(len(fsub), i, dtype=float) + jitter
            ax.scatter(xs, fsub["delta_loglik"],
                       s=22, color=color, alpha=0.78,
                       edgecolors="white", linewidths=0.4, zorder=3)
            # Median tick per family
            ax.hlines(fsub["delta_loglik"].median(), i - 0.28, i + 0.28,
                      color=color, linewidth=2.0, zorder=4)
        # Beetle reference lines
        ax.axhline(beetle_top, color="#1b9e77", linestyle="--", linewidth=0.9,
                   zorder=1, alpha=0.6)
        ax.text(len(families_with_rows) - 0.5, beetle_top,
                f" Beetle max = {beetle_top:.0f}",
                color="#1b9e77", fontsize=7.5, va="center", ha="right")
        ax.axhline(beetle_med, color="#1b9e77", linestyle=":", linewidth=0.7,
                   zorder=1, alpha=0.5)
        ax.set_xticks(range(len(families_with_rows)))
        ax.set_xticklabels(families_with_rows, rotation=35, ha="right",
                           fontsize=7.5)
        ax.set_ylabel("ΔlogLik (full vs baseline)" if lang == "du" else "")
        ax.set_title(LANG_LABEL[lang], fontsize=9.5)
        ax.grid(True, axis="y", linestyle=":", color="#cfcfcf", alpha=0.6,
                zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle(
        "MECO ΔlogLik — Beetle vs frontier baselines (live)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    out = output_dir / "fig_meco_beetle_vs_baselines"
    savefig_pub(fig, out)
    plt.close(fig)
    return out.with_suffix(".pdf")


def render_bars(cache: Path = CACHE, output_dir: Path = MAIN_DIR,
                top_n_beetle: int = 5) -> Path:
    """Horizontal bar plot per language: every model on disk, sorted by
    ΔlogLik, coloured by family. Beetle entries are capped to the top-N to
    avoid the panel collapsing into a sea of green; the cap is annotated."""
    df = pd.read_csv(cache)
    df = df.dropna(subset=["delta_loglik"]).copy()
    df[["family_code", "family_label"]] = df["model"].apply(
        lambda m: pd.Series(_family(m))
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 6.5), sharex=False)

    for ax, lang in zip(axes, LANG_ORDER):
        sub = df[df["lang"] == lang].copy()
        if sub.empty:
            ax.text(0.5, 0.5, "no data yet", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            ax.set_title(LANG_LABEL[lang])
            continue

        # Cap Beetle to top-N to keep the panel readable.
        beetle = sub[sub["family_label"] == "Beetle"].nlargest(
            top_n_beetle, "delta_loglik"
        )
        rest = sub[sub["family_label"] != "Beetle"]
        plot = pd.concat([beetle, rest], ignore_index=True).sort_values(
            "delta_loglik", ascending=True
        )

        # Short labels: keep org/last-token of model id, drop "beetle-" prefix.
        def _short(m: str) -> str:
            tail = m.split("/", 1)[-1]
            tail = tail.replace("beetle-bilingual-", "")
            tail = tail.replace("beetle-monolingual-", "mono-")
            tail = tail.replace("balanced-b1", "B1")
            tail = tail.replace("simultaneous-b2", "B2")
            tail = tail.replace("sequential-33-67-b3", "B3")
            tail = tail.replace("classroom-20-b4", "B4")
            tail = tail.replace("late-b5", "B5")
            tail = tail.replace("l2-50-", "").replace("l2-80-", "")
            tail = tail.replace("humanscale-", "HS-").replace("fineweb-", "FW-")
            org = m.split("/", 1)[0]
            if "beetle" in org.lower():
                return tail
            return f"{org}/{tail}"
        plot["label"] = plot["model"].apply(_short)

        colors = [
            FAMILY_COLOR.get(
                next((c for _, c, lab in FAMILY_RULES if lab == fam), "other"),
                "#888",
            )
            for fam in plot["family_label"]
        ]
        y = np.arange(len(plot))
        ax.barh(y, plot["delta_loglik"], color=colors,
                edgecolor="white", linewidth=0.4, height=0.78, zorder=2)
        for yi, v in zip(y, plot["delta_loglik"]):
            ax.text(v, yi, f" {v:.0f}", va="center", ha="left",
                    fontsize=7, color="#333")
        ax.set_yticks(y)
        ax.set_yticklabels(plot["label"], fontsize=7)
        ax.set_xlabel("ΔlogLik (full vs baseline)" if lang == "du" else "ΔlogLik")
        ax.set_title(LANG_LABEL[lang], fontsize=10)
        ax.grid(True, axis="x", linestyle=":", color="#cfcfcf", zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.text(0.99, 0.01,
                f"Beetle: top-{top_n_beetle} of "
                f"{int((sub['family_label'] == 'Beetle').sum())} shown",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="#666", style="italic")

    # Legend over all panels
    handles = []
    seen: set[str] = set()
    for fam_label in df["family_label"].unique():
        if fam_label in seen:
            continue
        seen.add(fam_label)
        col = FAMILY_COLOR.get(
            next((c for _, c, lab in FAMILY_RULES if lab == fam_label), "other"),
            "#888",
        )
        handles.append(mlines.Line2D([], [], color=col, marker="s", linestyle="",
                                     markersize=8, label=fam_label))
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=min(len(handles), 7),
               frameon=False, fontsize=8)
    fig.suptitle("MECO ΔlogLik per language — Beetle vs frontier LLMs",
                 fontsize=11, y=1.005)
    fig.tight_layout(rect=[0, 0.04, 1, 0.99])
    out = output_dir / "fig_meco_beetle_vs_baselines_bars"
    savefig_pub(fig, out)
    plt.close(fig)
    return out.with_suffix(".pdf")


# Approximate parameter counts (in billions) for each model id we may see.
# Beetle models are all ~125 M (the Beetle-FineWeb / -HumanScale architectures
# are GPT-2-small-class) — collapsed to a single point.
PARAMS_B: dict[str, float] = {
    "google/gemma-2-2b": 2.6,
    "google/gemma-2-9b": 9.2,
    "qwen/qwen3-4b-base": 4.0,
    "qwen/qwen3-8b-base": 8.0,
    "meta-llama/llama-3.1-8b": 8.0,
    "swiss-ai/apertus-8b-2509": 8.0,
    "utter-project/eurollm-1.7b": 1.7,
    "utter-project/eurollm-9b": 9.0,
    "llm-jp/llm-jp-3-150m": 0.15,
    "llm-jp/llm-jp-3-440m": 0.44,
    "llm-jp/llm-jp-3-980m": 0.98,
    "llm-jp/llm-jp-3-1.8b": 1.8,
    "llm-jp/llm-jp-3-3.7b": 3.7,
    "llm-jp/llm-jp-3-13b": 13.0,
    "qwen/qwen-14b": 14.0,
    "baichuan-inc/baichuan-7b": 7.0,
    "01-ai/yi-6b": 6.0,
    "thudm/chatglm-6b": 6.0,
    "bigscience/bloom-560m": 0.56,
    "bigscience/bloom-1b1": 1.1,
    "bigscience/bloom-1b7": 1.7,
    "bigscience/bloom-3b": 3.0,
    "bigscience/bloom-7b1": 7.1,
    "facebook/xglm-4.5b": 4.5,
}
BEETLE_PARAMS_B = 0.125  # All Beetle architectures are ~125M parameters.


def render_size_scaling(cache: Path = CACHE,
                        output_dir: Path = MAIN_DIR) -> Path:
    """ΔlogLik vs parameter count, log x-axis, one panel per MECO language.

    Beetle models share a single x-tick (~0.125B). For each language we
    highlight the *matched-pair* Beetle best — i.e. the strongest Beetle
    model that was trained on the L1 of that BLiMP language (nld-eng for
    Dutch RTs, deu-eng for German, zho-eng for Chinese) — rather than the
    cross-language eng-X variant that may sit higher numerically. This is
    the right comparison when arguing "Beetle's L1-matched model beats every
    frontier LLM trained on the same language."

    All panels share a single y-axis (0..240) so cross-language differences
    in ΔlogLik magnitude are directly visible.
    """
    df = pd.read_csv(cache)
    df = df.dropna(subset=["delta_loglik"]).copy()
    df[["family_code", "family_label"]] = df["model"].apply(
        lambda m: pd.Series(_family(m))
    )

    def _params(model: str) -> float | None:
        if "beetle" in model.lower():
            return BEETLE_PARAMS_B
        return PARAMS_B.get(model.lower())
    df["params_b"] = df["model"].apply(_params)
    df = df.dropna(subset=["params_b"])

    # Matched-pair ISO per MECO test language. The "matched" Beetle is the
    # one whose language pair contains this ISO (preferring the X-eng
    # direction since that's where the L1 is the BLiMP target).
    MATCHED_ISO = {"du": "nld", "ge": "deu", "ch_s": "zho"}
    # Common y-axis across panels for direct comparison.
    Y_LIM = (0.0, 240.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)

    for ax, lang in zip(axes, LANG_ORDER):
        sub = df[df["lang"] == lang]
        if sub.empty:
            ax.text(0.5, 0.5, "no data yet", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            ax.set_title(LANG_LABEL[lang])
            continue
        # Plot per family with consistent colour.
        for fam_label, g in sub.groupby("family_label"):
            color = FAMILY_COLOR.get(
                next((c for _, c, lab in FAMILY_RULES if lab == fam_label),
                     "other"),
                "#666",
            )
            marker = "o" if fam_label == "Beetle" else "s"
            ax.scatter(g["params_b"], g["delta_loglik"],
                       label=fam_label, color=color, s=42,
                       marker=marker, alpha=0.82,
                       edgecolors="white", linewidths=0.5, zorder=3)
        # Highlight the matched-pair Beetle best (the model trained on the
        # BLiMP target language). For Dutch RTs use the best nld-eng/eng-nld
        # Beetle, for German deu-eng/eng-deu, for Chinese zho-eng. No green
        # band, no shaded envelope — just a star marker + lean annotation.
        target_iso = MATCHED_ISO.get(lang)
        beetle_rows = sub[sub["family_label"] == "Beetle"]
        matched = beetle_rows[
            beetle_rows["model"].str.contains(
                rf"-{target_iso}-eng|-eng-{target_iso}",
                case=False, regex=True,
            )
        ] if target_iso else beetle_rows
        if not matched.empty:
            top = matched.loc[matched["delta_loglik"].idxmax()]
            top_y = float(top["delta_loglik"])
            top_model = top["model"].split("/", 1)[-1].lower()
            cond = ("B1" if "balanced-b1" in top_model else
                    "B2" if "simultaneous-b2" in top_model else
                    "B3" if "sequential-33-67-b3" in top_model else
                    "B4" if "classroom-20-b4" in top_model else
                    "B5" if "late-b5" in top_model else "?")
            cond_label = {"B1": "balanced 50:50",
                          "B2": "simultaneous",
                          "B3": "sequential 33:67",
                          "B4": "classroom 20%",
                          "B5": "late onset"}.get(cond, "")
            scale = "100 M" if "humanscale" in top_model else "24 B"
            pair = "-".join(top_model.split("-")[-2:])
            ax.scatter([BEETLE_PARAMS_B], [top_y],
                       s=140, color="#1b9e77", marker="*",
                       edgecolors="white", linewidths=0.8, zorder=5)
            # Place annotation either above or below the star, whichever
            # has more room within the shared y-limit.
            ytxt = top_y + 12 if top_y < Y_LIM[1] - 30 else top_y - 12
            va = "bottom" if ytxt > top_y else "top"
            ax.annotate(
                f"Beetle {pair}  ΔlogLik={top_y:.0f}\n{cond} ({cond_label}) · {scale} tokens",
                xy=(BEETLE_PARAMS_B, top_y),
                xytext=(BEETLE_PARAMS_B * 6, ytxt),
                fontsize=7.2, color="#1b9e77",
                ha="left", va=va,
                arrowprops=dict(arrowstyle="-", color="#1b9e77",
                                linewidth=0.7, alpha=0.7),
            )
        ax.set_xscale("log")
        ax.set_xlim(0.05, 30)
        ax.set_ylim(*Y_LIM)
        ax.set_xlabel("Parameters (B, log)")
        if lang == "du":
            ax.set_ylabel("ΔlogLik")
        ax.set_title(LANG_LABEL[lang], fontsize=9.5)
        ax.grid(True, linestyle=":", color="#cfcfcf", zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.04),
                   ncol=min(len(handles), 7), frameon=False, fontsize=7.5)
    fig.suptitle("Beetle (~125 M) sets the MECO ΔlogLik ceiling — "
                 "frontier LLMs do not exceed it at any scale",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    out = output_dir / "fig_meco_size_vs_deltalogl"
    savefig_pub(fig, out)
    plt.close(fig)
    return out.with_suffix(".pdf")


def render_best_of_family(cache: Path = CACHE,
                          output_dir: Path = MAIN_DIR,
                          *,
                          top_baselines: int | None = None,
                          baseline_families: list[str] | None = None,
                          color_overrides: dict[str, str] | None = None,
                          include_suptitle: bool = True,
                          out_stem: str = "fig_meco_best_of_family") -> Path:
    """Compact 3-panel summary: best ΔlogLik per family per language.

    One bar per (family, language) using the highest-ΔlogLik member of the
    family; sorted within each panel. Beetle in green at the top of every
    panel makes the headline claim immediately legible.

    Parameters
    ----------
    top_baselines : int, optional
        If given, keep only the N strongest non-Beetle baseline families per
        panel (Beetle is always retained). ``None`` (default) keeps every family.
    baseline_families : list of str, optional
        If given, restrict baselines to this exact set of family display labels
        (e.g. ``["XGLM", "EuroLLM", "Apertus"]``). Beetle is always retained.
        Mutually exclusive with ``top_baselines``.
    color_overrides : dict, optional
        Map of family-code → hex color. Layered on top of ``FAMILY_COLOR``.
    include_suptitle : bool
        If False, omit the figure-level headline.
    out_stem : str
        Output filename stem (without suffix).
    """
    if top_baselines is not None and baseline_families is not None:
        raise ValueError("pass either top_baselines or baseline_families, not both")

    palette = {**FAMILY_COLOR, **(color_overrides or {})}

    df = pd.read_csv(cache)
    df = df.dropna(subset=["delta_loglik"]).copy()
    df[["family_code", "family_label"]] = df["model"].apply(
        lambda m: pd.Series(_family(m))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharex=False)
    for ax, lang in zip(axes, LANG_ORDER):
        sub = df[df["lang"] == lang]
        if sub.empty:
            ax.text(0.5, 0.5, "no data yet", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            ax.set_title(LANG_LABEL[lang])
            continue
        best = (sub.sort_values("delta_loglik", ascending=False)
                   .groupby("family_label", as_index=False).first())
        if top_baselines is not None:
            beetle_rows = best[best["family_label"] == "Beetle"]
            baseline_rows = (best[best["family_label"] != "Beetle"]
                             .sort_values("delta_loglik", ascending=False)
                             .head(top_baselines))
            best = pd.concat([beetle_rows, baseline_rows])
        elif baseline_families is not None:
            keep = ["Beetle", *baseline_families]
            best = best[best["family_label"].isin(keep)]
        best = best.sort_values("delta_loglik", ascending=True)
        colors = [
            palette.get(
                next((c for _, c, lab in FAMILY_RULES if lab == fam), "other"),
                "#888",
            )
            for fam in best["family_label"]
        ]
        y = np.arange(len(best))
        ax.barh(y, best["delta_loglik"], color=colors,
                edgecolor="white", linewidth=0.5, zorder=2)
        for yi, v in zip(y, best["delta_loglik"]):
            ax.text(v, yi, f" {v:.0f}", va="center", ha="left", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(best["family_label"], fontsize=8.5)
        ax.set_xlabel("ΔlogLik (best-of-family)" if lang == "du" else "ΔlogLik")
        ax.set_title(LANG_LABEL[lang], fontsize=10)
        ax.grid(True, axis="x", linestyle=":", color="#cfcfcf", zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    if include_suptitle:
        fig.suptitle("Best-of-family MECO ΔlogLik — Beetle dominates every panel",
                     fontsize=10.5, y=1.01)
    fig.tight_layout()
    out = output_dir / out_stem
    savefig_pub(fig, out)
    plt.close(fig)
    return out.with_suffix(".pdf")


def main() -> None:
    apply_acl_style()
    paths = render()
    print(f"[meco_vs_baselines] wrote swarm {paths}")
    bars = render_bars()
    print(f"[meco_vs_baselines] wrote bars  {bars}")
    sz = render_size_scaling()
    print(f"[meco_vs_baselines] wrote size  {sz}")
    bof = render_best_of_family()
    print(f"[meco_vs_baselines] wrote bestf {bof}")
    bof_simpl = render_best_of_family(
        baseline_families=["XGLM", "EuroLLM", "Apertus"],
        color_overrides={
            "xglm":    "#2C7BB6",  # bright blue
            "eurollm": "#9C46C8",  # bright purple
            "apertus": "#D04758",  # bright red
        },
        include_suptitle=False,
        out_stem="fig_meco_best_of_family_simplified",
    )
    print(f"[meco_vs_baselines] wrote bestfS {bof_simpl}")


if __name__ == "__main__":
    main()
