"""
paper_figures.py — ACL main-body figures for the BeetleLM paper.

Each function reads from ``results/ppl/ppl_summary.csv`` and (optionally)
MECO alignment / convergence CSVs, and emits one PDF + PNG pair under
``results/figures/main/``. Companion CSVs for the paper's ``\\input{}`` tables
are written to ``results/figures/tables/``.

Figure → paper section mapping (as planned in
``~/.claude/plans/quizzical-twirling-scroll.md`` Phase 3.4):

    fig_mainbody_1_factor_summary        Six-factor effect-size heatmap
    fig_mainbody_2_onset_ablation        B2 vs B5        (L2 onset timing)
    fig_mainbody_3_ratio_ablation        B1 vs B3 vs B3_full  (exposure ratio)
    fig_mainbody_4_transition_ablation   B2 vs B2_smooth (transition sharpness)
    fig_mainbody_5_burst_ablation        B2 vs B4        (burstiness)
    fig_mainbody_6_episodic_ablation     B1 vs B1b vs B4 (episodic clustering)
    fig_mainbody_7_cl_mechanism          EWC / LAMOL / LR pairs
    fig_mainbody_8_family_bridge         HumanScale vs FineWeb, per factor
    fig_mainbody_9_minimal_sweep_forest  14-run minimal sweep forest plot
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from analyze import _path  # noqa: F401 — adds repo root to sys.path

from plot_style import (
    apply_acl_style, savefig_pub,
    color_for_condition, color_for_lang,
    WIDTH_SINGLE, WIDTH_DOUBLE,
    PALETTE_ACL,
)

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

apply_acl_style()
log = logging.getLogger("paper_figures")

MAIN_DIR   = Path("results/figures/main")
TABLES_DIR = Path("results/figures/tables")
PPL_SUMMARY = Path("results/ppl/ppl_summary.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_ppl() -> pd.DataFrame:
    if not PPL_SUMMARY.is_file():
        raise FileNotFoundError(f"{PPL_SUMMARY} not found — run --ppl first")
    df = pd.read_csv(PPL_SUMMARY)
    # Paper figures are all final-snapshot by nature. When CKPT_SCOPE=all|both
    # produces a multi-revision ppl_summary.csv, collapse to one row per
    # (repo, iso_code) via the "main-or-max-step-N" rule.
    try:
        from ppl_utils import final_revision_row_mask
        if "revision" in df.columns:
            mask = final_revision_row_mask(df)
            if mask.any():
                df = df[mask].copy()
    except Exception:
        pass
    from analyze.model_naming import parse_repo
    meta = df["repo"].map(parse_repo)
    df["family"]    = meta.map(lambda m: m.family)
    df["condition"] = meta.map(lambda m: m.condition)
    df["variant"]   = meta.map(lambda m: m.variant)
    df["cl"]        = meta.map(lambda m: m.cl_variant)
    df["l1"]        = meta.map(lambda m: m.l1)
    df["l2"]        = meta.map(lambda m: m.l2)
    df["seed"]      = meta.map(lambda m: m.seed)
    return df


def _agg_mean_sem(df: pd.DataFrame, group_cols: list[str], val: str) -> pd.DataFrame:
    g = df.groupby(group_cols)[val].agg(["mean", "sem", "count"]).reset_index()
    g["sem"] = g["sem"].fillna(0.0)
    return g


# Module-level hook set by ``analyze/run.py::_run_paper_figures`` to tag the
# output-filename variant (always "final" for paper figures, since they are
# final-snapshot by nature). None disables suffixing.
CURRENT_FIGURE_VARIANT: str | None = None


def _variant_stem(stem: str) -> str:
    """Append ``_<variant>`` to *stem* when a variant hook is active."""
    if CURRENT_FIGURE_VARIANT and not stem.endswith(
        ("_final", "_trajectory")
    ):
        return f"{stem}_{CURRENT_FIGURE_VARIANT}"
    return stem


def _save_table(df: pd.DataFrame, stem: str) -> Path:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    stem = _variant_stem(stem)
    out = TABLES_DIR / f"{stem}.csv"
    df.to_csv(out, index=False)
    return out


def _save_fig(fig, stem: str) -> Path:
    stem = _variant_stem(stem)
    paths = savefig_pub(fig, MAIN_DIR / stem)
    log.info("saved %s.{pdf,png}", MAIN_DIR / stem)
    return paths["pdf"] if isinstance(paths, dict) else paths[0]


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — factor summary heatmap (6 factors × {B1..B5})
# ─────────────────────────────────────────────────────────────────────────────

def fig_mainbody_1_factor_summary(langs: Sequence[str]) -> Optional[Path]:
    """Effect size (Cohen's d vs mono) of each condition on PPL per language."""
    df = _load_ppl()
    df = df[df["iso_code"].isin(langs)]
    mono = df[df["condition"] == "mono"].groupby("iso_code")["mean_ppl"].mean()
    if mono.empty:
        log.warning("fig1: no mono baselines found")
        return None

    conditions = ["B1", "B2", "B3", "B4", "B5"]
    rows = []
    for cond in conditions:
        sub = df[df["condition"] == cond]
        for lang in langs:
            v = sub[sub["iso_code"] == lang]["mean_ppl"]
            if v.empty or lang not in mono:
                rows.append({"condition": cond, "lang": lang, "delta": np.nan})
                continue
            # Cohen's d ≈ (mean_cond - mean_mono) / pooled_sd
            sd = v.std(ddof=1) if len(v) > 1 else v.mean() * 0.1
            delta = (v.mean() - mono[lang]) / max(sd, 1e-6)
            rows.append({"condition": cond, "lang": lang, "delta": float(delta)})
    effect = pd.DataFrame(rows)
    pivot = effect.pivot(index="condition", columns="lang", values="delta")

    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE, 2.4))
    norm = TwoSlopeNorm(vcenter=0.0) if np.nanmin(pivot.values) < 0 else None
    im = ax.imshow(pivot.values, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i, c in enumerate(pivot.index):
        for j, lang in enumerate(pivot.columns):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        color="black", fontsize=6)
    ax.set_title("Effect size (Cohen's d) vs mono")
    fig.colorbar(im, ax=ax, shrink=0.8, label=r"$d$")

    _save_table(effect, "fig_mainbody_1_factor_summary")
    return _save_fig(fig, "fig_mainbody_1_factor_summary")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: paired comparison bar chart with seed-aggregated SE
# ─────────────────────────────────────────────────────────────────────────────

def _paired_bar(
    df: pd.DataFrame,
    group_col: str,
    order: list[str],
    lang: str,
    title: str,
    stem: str,
) -> Optional[Path]:
    sub = df[df["iso_code"] == lang].copy()
    sub = sub[sub[group_col].isin(order)]
    if sub.empty:
        log.warning("%s: no data for lang=%s", stem, lang)
        return None
    agg = _agg_mean_sem(sub, [group_col], "mean_ppl")
    agg = agg.set_index(group_col).reindex(order).reset_index().dropna(subset=["mean"])

    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE, 2.3))
    colors = [color_for_condition(g.split("_")[0]) for g in agg[group_col]]
    ax.bar(range(len(agg)), agg["mean"], yerr=agg["sem"],
           color=colors, edgecolor="black", linewidth=0.6,
           width=0.6, capsize=2)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(agg[group_col], rotation=15, fontsize=7)
    ax.set_ylabel(f"mean PPL ({lang})")
    ax.set_title(title)

    _save_table(agg, stem)
    return _save_fig(fig, stem)


def fig_mainbody_2_onset_ablation(lang: str = "nld") -> Optional[Path]:
    df = _load_ppl()
    return _paired_bar(
        df, "condition", ["B2", "B5"], lang,
        "L2 onset timing: B2 (mid) vs B5 (late)",
        "fig_mainbody_2_onset_ablation",
    )


def fig_mainbody_3_ratio_ablation(lang: str = "nld") -> Optional[Path]:
    df = _load_ppl()
    # B3_full models have variant == "B3_full"; merge into a single ordering
    df["ratio_key"] = df["variant"].where(
        df["variant"].notna(), df["condition"]
    )
    return _paired_bar(
        df, "ratio_key", ["B1", "B3", "B3_full"], lang,
        "L2 exposure ratio: B1 (50:50) → B3 (33:67) → B3_full (0:100)",
        "fig_mainbody_3_ratio_ablation",
    )


def fig_mainbody_4_transition_ablation(lang: str = "nld") -> Optional[Path]:
    df = _load_ppl()
    df["key"] = df["variant"].where(df["variant"].notna(), df["condition"])
    return _paired_bar(
        df, "key", ["B2", "B2_smooth"], lang,
        "Transition sharpness: B2 (sharp) vs B2_smooth",
        "fig_mainbody_4_transition_ablation",
    )


def fig_mainbody_5_burst_ablation(lang: str = "nld") -> Optional[Path]:
    df = _load_ppl()
    df["key"] = df["variant"].where(df["variant"].notna(), df["condition"])
    return _paired_bar(
        df, "key", ["B2", "B2_burst", "B4"], lang,
        "Burst exposure: B2 vs B2_burst vs B4",
        "fig_mainbody_5_burst_ablation",
    )


def fig_mainbody_6_episodic_ablation(lang: str = "nld") -> Optional[Path]:
    df = _load_ppl()
    df["key"] = df["variant"].where(df["variant"].notna(), df["condition"])
    return _paired_bar(
        df, "key", ["B1", "B1b", "B4"], lang,
        "Episodic clustering: B1 vs B1b vs B4",
        "fig_mainbody_6_episodic_ablation",
    )


def fig_mainbody_7_cl_mechanism(lang: str = "nld") -> Optional[Path]:
    """EWC vs LAMOL under matched settings (paper tab:mechanisms)."""
    df = _load_ppl()
    df["key"] = df["variant"].where(df["variant"].notna(), df["condition"])
    pairs = [
        ("B2_ewc",    "B2_lamol",    "balanced interference"),
        ("B4",        "B4b_lamol",   "strong interference"),
        ("B5_ewc",    "B5_lamol",    "late L2 (maximal forgetting)"),
    ]
    rows = []
    for left, right, label in pairs:
        for key in (left, right):
            v = df[(df["iso_code"] == lang) & (df["key"] == key)]["mean_ppl"]
            if v.empty:
                rows.append({"pair": label, "variant": key, "mean": np.nan, "sem": 0.0})
            else:
                rows.append({
                    "pair": label, "variant": key,
                    "mean": float(v.mean()),
                    "sem": float(v.sem() if len(v) > 1 else 0.0),
                })
    agg = pd.DataFrame(rows).dropna(subset=["mean"])
    if agg.empty:
        log.warning("fig7: no CL-mechanism data for lang=%s", lang)
        return None

    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE, 2.6))
    x = np.arange(len(pairs))
    width = 0.35
    lefts  = [agg[agg["variant"] == p[0]]["mean"].sum() for p in pairs]
    rights = [agg[agg["variant"] == p[1]]["mean"].sum() for p in pairs]
    lefts_e  = [agg[agg["variant"] == p[0]]["sem"].sum() for p in pairs]
    rights_e = [agg[agg["variant"] == p[1]]["sem"].sum() for p in pairs]
    ax.bar(x - width/2, lefts, width, yerr=lefts_e,
           color=PALETTE_ACL["B2"], edgecolor="black", linewidth=0.6,
           label="EWC", capsize=2)
    ax.bar(x + width/2, rights, width, yerr=rights_e,
           color=PALETTE_ACL["B5"], edgecolor="black", linewidth=0.6,
           label="LAMOL", capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([p[2] for p in pairs], fontsize=7)
    ax.set_ylabel(f"mean PPL ({lang})")
    ax.set_title("Continual-learning mechanism comparison")
    ax.legend(fontsize=7, frameon=False)

    _save_table(agg, "fig_mainbody_7_cl_mechanism")
    return _save_fig(fig, "fig_mainbody_7_cl_mechanism")


def fig_mainbody_8_family_bridge(langs: Sequence[str]) -> Optional[Path]:
    """HumanScale vs FineWeb, grouped by condition."""
    df = _load_ppl()
    df = df[df["family"].isin(["humanscale", "fineweb"])]
    df = df[df["iso_code"].isin(langs)]
    if df.empty:
        log.warning("fig8: no humanscale/fineweb rows; family bridge skipped")
        return None
    agg = _agg_mean_sem(df, ["condition", "family"], "mean_ppl")
    conds = ["B1", "B2", "B3", "B4", "B5"]
    agg = agg[agg["condition"].isin(conds)]

    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE, 2.6))
    x = np.arange(len(conds))
    w = 0.38
    hs = agg[agg["family"] == "humanscale"].set_index("condition").reindex(conds)
    fw = agg[agg["family"] == "fineweb"].set_index("condition").reindex(conds)
    ax.bar(x - w/2, hs["mean"].fillna(0), w,
           yerr=hs["sem"].fillna(0), label="HumanScale",
           color=[color_for_condition(c) for c in conds],
           edgecolor="black", linewidth=0.6, capsize=2)
    ax.bar(x + w/2, fw["mean"].fillna(0), w,
           yerr=fw["sem"].fillna(0), label="FineWeb",
           color=[color_for_condition(c) for c in conds],
           edgecolor="black", linewidth=0.6, hatch="///", capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylabel("mean PPL")
    ax.set_title("HumanScale vs FineWeb (per condition)")
    ax.legend(fontsize=7, frameon=False)

    _save_table(agg, "fig_mainbody_8_family_bridge")
    return _save_fig(fig, "fig_mainbody_8_family_bridge")


def fig_mainbody_9_minimal_sweep_forest(lang: str = "nld") -> Optional[Path]:
    """Forest plot: effect of each minimal-sweep variant vs B1 baseline."""
    df = _load_ppl()
    sub = df[(df["iso_code"] == lang) & df["seed"].notna()]
    if sub.empty:
        log.warning("fig9: no seeded ablation data; minimal-sweep forest skipped")
        return None
    sub["key"] = sub["variant"].where(sub["variant"].notna(), sub["condition"])
    baseline = sub[sub["key"] == "B1"]["mean_ppl"]
    if baseline.empty:
        log.warning("fig9: no B1 baseline seeds for lang=%s", lang)
        return None
    base_mean = baseline.mean()

    agg = _agg_mean_sem(sub, ["key"], "mean_ppl")
    agg["delta"] = agg["mean"] - base_mean
    agg = agg.sort_values("delta")

    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE, 3.0))
    y = np.arange(len(agg))
    ax.errorbar(agg["delta"], y, xerr=agg["sem"],
                fmt="o", color="black", ecolor="grey", capsize=2)
    ax.axvline(0, color="grey", lw=0.6, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(agg["key"], fontsize=7)
    ax.set_xlabel(r"$\Delta$ mean PPL vs B1")
    ax.set_title(f"Minimal ablation sweep — {lang}")

    _save_table(agg, "fig_mainbody_9_minimal_sweep_forest")
    return _save_fig(fig, "fig_mainbody_9_minimal_sweep_forest")


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def fig_c_blimp_delta_table(
    langs: Sequence[str] = ("deu", "nld", "zho", "eng"),
    blimp_tasks: Sequence[str] = ("blimp_eng",),
) -> Path:
    """Fig C — BLiMP accuracy-delta table.

    Produces two complementary Δ-columns per L1, following the paper's
    Table 3.1:

    * **Effect of L2 Input** (L1 held constant): Δ = acc(L2 = large)
      − acc(L2 = small). Uses the highest vs lowest-step checkpoints for
      each (repo, benchmark, lang) as a proxy for "more L2 training".
    * **Effect of L1 Input** (L2 held constant): Δ = acc(L1 = large)
      − acc(L1 = small). For Beetle-HumanScale vs Beetle-FineWeb pairs
      (same L1, matched condition, different L1-budget), this is a direct
      L1-budget ablation.

    The table is written to
    ``results/figures/tables/fig_c_blimp_delta_table.csv`` and, when
    pandas is built with ``to_latex`` support, ``.tex`` as well. Returns
    the path to the CSV.

    Missing inputs (no BLiMP result CSVs on disk) raise ``FileNotFoundError``
    so ``render_all`` can skip the figure gracefully.
    """
    import os
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Locate BLiMP result CSVs produced by ``legacy/eval_model.py``. Two
    # layouts are supported (in order of preference):
    #   results/<bm>_results.csv            (flat, legacy)
    #   results/<bm>/<slug>.csv             (sharded, current)
    result_root = Path("results")
    rows: list[dict] = []
    found_any = False

    for bm in blimp_tasks:
        flat = result_root / f"{bm}_results.csv"
        if flat.is_file():
            found_any = True
            df_bm = pd.read_csv(flat)
            df_bm["benchmark"] = bm
            rows.append(df_bm)
            continue
        sharded = result_root / bm
        if sharded.is_dir():
            shard_rows = []
            for csv in sharded.glob("*.csv"):
                try:
                    df_i = pd.read_csv(csv)
                    df_i["benchmark"] = bm
                    shard_rows.append(df_i)
                except Exception:
                    pass
            if shard_rows:
                found_any = True
                rows.append(pd.concat(shard_rows, ignore_index=True))

    if not found_any:
        raise FileNotFoundError(
            "No BLiMP result CSVs found under results/. "
            "Run the stage-2 BLiMP suite before rendering Fig C."
        )

    df = pd.concat(rows, ignore_index=True)
    df = df[df["lang_code"].isin(set(langs))] if "lang_code" in df.columns else df

    # Join model metadata.
    from analyze.model_naming import parse_repo, training_budget
    meta = df["model"].map(parse_repo)
    df["family"]    = meta.map(lambda m: m.family)
    df["condition"] = meta.map(lambda m: m.condition)
    df["l1"]        = meta.map(lambda m: m.l1)
    df["l2"]        = meta.map(lambda m: m.l2)
    df["l1_tokens"] = meta.map(lambda m: training_budget(m)[0])
    df["l2_tokens"] = meta.map(lambda m: training_budget(m)[1])
    df["step_int"]  = pd.to_numeric(
        df["checkpoint"].astype(str).str.extract(r"step-(\d+)")[0],
        errors="coerce",
    )

    # ── Effect of L2 input (L1 constant): high-step vs low-step per repo.
    l2_rows = []
    for (repo, bm, lang_code), grp in df.groupby(
        ["model", "benchmark", "lang_code"]
    ):
        grp2 = grp.dropna(subset=["step_int"])
        if grp2["step_int"].nunique() < 2:
            continue
        lo = grp2.loc[grp2["step_int"].idxmin()]
        hi = grp2.loc[grp2["step_int"].idxmax()]
        l2_rows.append({
            "repo": repo, "benchmark": bm, "lang": lang_code,
            "l1": lo["l1"], "l2": lo["l2"], "family": lo["family"],
            "l1_tokens": lo["l1_tokens"],
            "low_step":  int(lo["step_int"]),
            "high_step": int(hi["step_int"]),
            "acc_low":  float(lo["accuracy"]),
            "acc_high": float(hi["accuracy"]),
            "delta_l2": float(hi["accuracy"]) - float(lo["accuracy"]),
        })
    df_l2 = pd.DataFrame(l2_rows)

    # ── Effect of L1 input (L2 constant): HumanScale vs FineWeb pairs.
    # Match on (condition, l1, l2, benchmark, lang) with same final-step
    # accuracy.
    df_final = df.dropna(subset=["step_int"]).sort_values("step_int")
    df_final = df_final.drop_duplicates(
        subset=["model", "benchmark", "lang_code"], keep="last",
    )
    pivot = df_final.pivot_table(
        index=["condition", "l1", "l2", "benchmark", "lang_code"],
        columns="family", values="accuracy", aggfunc="mean",
    ).reset_index()
    l1_rows = []
    if {"humanscale", "fineweb"} <= set(pivot.columns):
        for _, row in pivot.iterrows():
            hs = row.get("humanscale")
            fw = row.get("fineweb")
            if pd.notnull(hs) and pd.notnull(fw):
                l1_rows.append({
                    "condition": row["condition"],
                    "l1":        row["l1"],
                    "l2":        row["l2"],
                    "benchmark": row["benchmark"],
                    "lang":      row["lang_code"],
                    "acc_humanscale": float(hs),
                    "acc_fineweb":    float(fw),
                    "delta_l1":       float(fw) - float(hs),
                })
    df_l1 = pd.DataFrame(l1_rows)

    # ── Write artefacts.
    out_csv = TABLES_DIR / "fig_c_blimp_delta_table.csv"
    with out_csv.open("w", encoding="utf-8") as fh:
        fh.write("# fig_c_blimp_delta_table\n")
        fh.write("# \n")
        fh.write("# (1) Effect of L2 Input (L1 held constant): "
                 "delta = acc(high_step) - acc(low_step)\n")
        df_l2.to_csv(fh, index=False)
        fh.write("\n# (2) Effect of L1 Input (L2 held constant): "
                 "delta = acc(fineweb) - acc(humanscale)\n")
        df_l1.to_csv(fh, index=False)

    try:
        tex_path = TABLES_DIR / "fig_c_blimp_delta_table.tex"
        tex_path.write_text(
            "% Effect of L2 Input (L1 held constant)\n"
            + df_l2.to_latex(index=False, float_format="%.4f")
            + "\n\n% Effect of L1 Input (L2 held constant)\n"
            + (df_l1.to_latex(index=False, float_format="%.4f") if not df_l1.empty else "% (no humanscale/fineweb matched pairs)\n")
        )
    except Exception as e:  # pragma: no cover — to_latex missing
        log.warning("fig_c_blimp_delta_table: .tex not written (%s)", e)

    # ── Companion figure ───────────────────────────────────────────────
    # Two stacked panels: (A) Effect of L2 Input per L1; (B) Effect of L1
    # Input per condition. Uses plot_style so line weights, font, palette
    # all match the rest of the main-body figures.
    try:
        from plot_style import (
            fig_double, savefig_pub, style_axes, add_panel_label,
            compact_legend, color_for_lang, hatch_for_condition,
            PALETTE_ACL, LW_AXIS, SIZE_ANNOTATION, COLOR_ZEROLINE,
        )
        if not df_l2.empty:
            fig, (axA, axB) = fig_double(height=4.6, nrows=2, ncols=1,
                                         sharex=False)

            # Panel A — Effect of L2 Input by L1, mean over benchmarks.
            l2_by_l1 = (
                df_l2.groupby("l1")["delta_l2"]
                .agg(["mean", "sem", "count"]).reset_index()
                .sort_values("mean")
            )
            l1s = l2_by_l1["l1"].tolist()
            xs = np.arange(len(l1s))
            colours = [color_for_lang(l) for l in l1s]
            bars = axA.bar(
                xs, l2_by_l1["mean"], yerr=l2_by_l1["sem"].fillna(0),
                color=colours, edgecolor="black",
                linewidth=LW_AXIS, capsize=2, zorder=3,
            )
            for bar, n in zip(bars, l2_by_l1["count"]):
                axA.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1e-4,
                    f"n={int(n)}",
                    ha="center", va="bottom",
                    fontsize=SIZE_ANNOTATION, color="#555555",
                )
            axA.set_xticks(xs)
            axA.set_xticklabels(l1s)
            style_axes(
                axA,
                ylabel=r"$\Delta$ accuracy (high $-$ low step)",
                title="Effect of L2 input (L1 held constant)",
                grid=True, grid_axis="y", zero_line=True,
            )
            add_panel_label(axA, "A")

            # Panel B — Effect of L1 Input: HumanScale vs FineWeb matched
            # pairs, grouped by condition.
            if not df_l1.empty:
                by_cond = (
                    df_l1.groupby("condition")["delta_l1"]
                    .agg(["mean", "sem", "count"]).reset_index()
                    .sort_values("mean")
                )
                xs = np.arange(len(by_cond))
                colours = [PALETTE_ACL.get(c, PALETTE_ACL["unknown"])
                           for c in by_cond["condition"]]
                hatches = [hatch_for_condition(c)
                           for c in by_cond["condition"]]
                bars = axB.bar(
                    xs, by_cond["mean"], yerr=by_cond["sem"].fillna(0),
                    color=colours, edgecolor="black",
                    linewidth=LW_AXIS, capsize=2, zorder=3,
                )
                for bar, h in zip(bars, hatches):
                    bar.set_hatch(h)
                axB.set_xticks(xs)
                axB.set_xticklabels(by_cond["condition"])
                style_axes(
                    axB,
                    ylabel=r"$\Delta$ accuracy (FineWeb $-$ HumanScale)",
                    xlabel="Condition",
                    title="Effect of L1 input (L2 held constant)",
                    grid=True, grid_axis="y", zero_line=True,
                )
            else:
                axB.text(
                    0.5, 0.5,
                    "No HumanScale/FineWeb matched pairs "
                    "available in current results.",
                    ha="center", va="center",
                    transform=axB.transAxes, fontsize=7,
                    color="#777777",
                )
                style_axes(axB, xlabel="", ylabel="",
                           title="Effect of L1 input (L2 held constant)",
                           grid=False)
            add_panel_label(axB, "B")

            savefig_pub(fig, MAIN_DIR / "fig_c_blimp_delta_table")
    except Exception as e:  # pragma: no cover — figure is best-effort
        log.warning("fig_c_blimp_delta_table figure skipped — %s", e)

    log.info(
        "fig_c_blimp_delta_table: %d L2-effect rows, %d L1-effect rows → %s",
        len(df_l2), len(df_l1), out_csv,
    )
    return out_csv


def render_all(langs: Sequence[str] = ("deu", "nld", "zho", "eng")) -> dict:
    """Render every figure that has enough data; log skips otherwise."""
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {}

    def _try(name, fn, *a, **kw):
        try:
            out[name] = fn(*a, **kw)
        except FileNotFoundError as e:
            log.warning("%s skipped — %s", name, e)
        except Exception as e:  # pragma: no cover
            log.exception("%s failed: %s", name, e)

    _try("fig1", fig_mainbody_1_factor_summary, langs)
    _try("fig2", fig_mainbody_2_onset_ablation, "nld")
    _try("fig3", fig_mainbody_3_ratio_ablation, "nld")
    _try("fig4", fig_mainbody_4_transition_ablation, "nld")
    _try("fig5", fig_mainbody_5_burst_ablation, "nld")
    _try("fig6", fig_mainbody_6_episodic_ablation, "nld")
    _try("fig7", fig_mainbody_7_cl_mechanism, "nld")
    _try("fig8", fig_mainbody_8_family_bridge, langs)
    _try("fig9", fig_mainbody_9_minimal_sweep_forest, "nld")
    _try("figC", fig_c_blimp_delta_table, langs)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--langs", nargs="+", default=["deu", "nld", "zho", "eng"])
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if a.all:
        render_all(a.langs)
