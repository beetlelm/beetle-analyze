"""
d7_l1_control.py — Native-L1 control: beetle does *not* outperform on L1-native RT.

Claim (the falsifier)
---------------------
If beetle's MECO L2 advantage were "any model that's slightly more
human-like at predicting RT", it should also outperform L1-mono baselines
on L1-native readers reading their L1. We show it does *not* — the L2
advantage is L2-specific. This is the strongest counter-argument to a
reviewer who suspects we're just describing a generic well-calibrated LM.

Stat
----
Mirror the canonical pipeline on the L1-native subset of MECO L1 (the
wave-1 ``en`` cohort for English-natives reading English) using the
*same* surprisal sources used in the L2 fits, plus a matched L1-mono
beetle (e.g. ``beetle-monolingual-fineweb-eng``). Report:

  • ΔlogLik(beetle full | L2-native readers, L2 text)         ← original
  • ΔlogLik(beetle full | L1-native readers, L1 text)         ← falsifier
  • ΔlogLik(L1-mono baseline | L1-native readers, L1 text)    ← reference

Output
------
``results/meco_detailed/d7_l1_control.csv``
    columns: model, reader_group ∈ {L2, L1}, lang, delta_loglik,
             beta_surprisal, beta_surprisal_se, n_obs, n_subjects, notes

``plots/meco_detailed/d7_l1_control.pdf``
    Paired bars per L1: ΔlogLik for L2-reader cohort vs L1-reader cohort
    of the same language. Annotation: "gap" = L2 − L1.

CLI
---
    python -m analyze.meco_detailed.d7_l1_control [--lang du ge]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from analyze import _path  # noqa: F401
from analyze import meco_canonical as mc
from analyze.meco_detailed._common import (
    CANONICAL_LANGS, L1_PRETTY, OSF_L1_CODES, figure_pdf, result_csv,
)


SLUG = "d7_l1_control"

# MECO L2 release contains an English-L1 cohort coded ``en`` reading
# English text — this is our L1-control population. For Dutch / German /
# Chinese the L1-native group is the one whose ``lang`` code equals their
# native L1 code (``du`` / ``ge`` / ``ch_s`` reading the matched-L1 text).
L1_CONTROL_CODE = "en"


def compute(
    surprisal_df: pd.DataFrame,
    *,
    langs: Iterable[str] = CANONICAL_LANGS,
    verbose: bool = True,
) -> pd.DataFrame:
    from analyze.meco_detailed._common import get_rt_with_features
    rt = get_rt_with_features()

    rows = []
    for model in sorted(surprisal_df["model"].dropna().unique()):
        merged = mc.join_model_surprisal(rt, surprisal_df, model)
        if merged.empty:
            continue

        # L2-reader runs (canonical configuration)
        for lang in langs:
            sub = merged[merged["lang"] == lang].dropna(
                subset=["firstrun_dur", "Surprisal", "wlen", "freq", "subid"])
            sub = mc.attach_spillover(sub)
            sub = mc.drop_sentence_boundaries(sub)
            if len(sub) < 200:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = mc.fit_canonical_pair(sub)
                rows.append({
                    "model": model, "reader_group": "L2", "lang": lang,
                    "delta_loglik": fit["delta_loglik"],
                    "beta_surprisal": fit["beta_surprisal"],
                    "beta_surprisal_se": fit["beta_surprisal_se"],
                    "n_obs": int(len(sub)),
                    "n_subjects": int(sub["subid"].nunique()),
                    "notes": "",
                })
            except Exception as e:
                rows.append({
                    "model": model, "reader_group": "L2", "lang": lang,
                    "delta_loglik": np.nan, "beta_surprisal": np.nan,
                    "beta_surprisal_se": np.nan,
                    "n_obs": int(len(sub)),
                    "n_subjects": int(sub["subid"].nunique()),
                    "notes": f"{e.__class__.__name__}: {e}",
                })

        # L1-native control (English-native readers in MECO L2 wave-1)
        sub_l1 = merged[merged["lang"] == L1_CONTROL_CODE].dropna(
            subset=["firstrun_dur", "Surprisal", "wlen", "freq", "subid"])
        sub_l1 = mc.attach_spillover(sub_l1)
        sub_l1 = mc.drop_sentence_boundaries(sub_l1)
        if len(sub_l1) >= 200:
            if verbose:
                print(f"[d7] {model} | L1-native (en) (n={len(sub_l1)})",
                      file=sys.stderr, flush=True)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = mc.fit_canonical_pair(sub_l1)
                rows.append({
                    "model": model, "reader_group": "L1", "lang": "en",
                    "delta_loglik": fit["delta_loglik"],
                    "beta_surprisal": fit["beta_surprisal"],
                    "beta_surprisal_se": fit["beta_surprisal_se"],
                    "n_obs": int(len(sub_l1)),
                    "n_subjects": int(sub_l1["subid"].nunique()),
                    "notes": "L1-native control",
                })
            except Exception as e:
                rows.append({
                    "model": model, "reader_group": "L1", "lang": "en",
                    "delta_loglik": np.nan, "beta_surprisal": np.nan,
                    "beta_surprisal_se": np.nan,
                    "n_obs": int(len(sub_l1)),
                    "n_subjects": int(sub_l1["subid"].nunique()),
                    "notes": f"{e.__class__.__name__}: {e}",
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(df: pd.DataFrame, out_path: Path) -> None:
    """Two heatmaps side-by-side:
      (a) L2-reader ΔlogLik per (L1 × curriculum)
      (b) L1-native (en) ΔlogLik per curriculum (rows collapse to one)
    plus a third panel showing the gap L2 − L1 per curriculum.
    """
    import matplotlib.pyplot as plt
    from analyze.plot_style import apply_acl_style, savefig_pub, WIDTH_DOUBLE
    from analyze.meco_detailed._plot import (
        CURRICULUM_LABEL, CURRICULUM_ORDER, CURRICULUM_PALETTE,
        L1_ORDER, L1_PALETTE, L1_PRETTY,
        attach_model_metadata, axis_label, coverage_badge, direction_badge,
        mark_missing_cell,
    )

    apply_acl_style()
    if df.empty:
        return

    df = attach_model_metadata(df)
    # ``reader_group`` ∈ {L2, L1}. For L2 rows the lang column carries the
    # reader L1; for L1 rows it's always "en".
    df_l2 = df[df["reader_group"] == "L2"].copy()
    df_l1 = df[df["reader_group"] == "L1"].copy()
    if "lang" in df_l2.columns:
        df_l2["l1_code"] = df_l2["lang"].fillna(df_l2["l1_code"])

    used_l1s = [l for l in L1_ORDER if l in set(df_l2["l1_code"])]
    used_curr = [c for c in CURRICULUM_ORDER
                 if c != "mono" and c in set(df_l2["curriculum"])]
    if not used_l1s:
        used_l1s = ["du", "ge", "ch_s"]
    if not used_curr:
        used_curr = ["B1", "B2", "B3", "B4", "B5"]

    fig, (ax_l2, ax_l1, ax_gap) = plt.subplots(
        1, 3,
        figsize=(WIDTH_DOUBLE, max(2.6, 0.28 * len(used_l1s) + 1.5)),
        gridspec_kw={"width_ratios": [2.0, 0.6, 1.4]},
        constrained_layout=True,
    )

    # (a) L2-reader heatmap
    g_l2 = (df_l2.groupby(["l1_code", "curriculum"])["delta_loglik"]
                  .mean().unstack("curriculum"))
    g_l2 = g_l2.reindex(index=used_l1s, columns=used_curr)
    vmax_l2 = float(np.nanmax(np.abs(g_l2.to_numpy(dtype=float)))) or 1.0
    im = ax_l2.imshow(g_l2.to_numpy(dtype=float), cmap="RdBu_r",
                       vmin=-vmax_l2, vmax=vmax_l2, aspect="auto")
    for r, l1 in enumerate(used_l1s):
        for c, curr in enumerate(used_curr):
            v = g_l2.iloc[r, c]
            if not np.isfinite(v):
                mark_missing_cell(ax_l2, c, r, w=0.96, h=0.96)
            else:
                ax_l2.text(c, r, f"{v:.0f}", ha="center", va="center",
                            fontsize=5.5,
                            color="white" if abs(v) > vmax_l2 * 0.55 else "#222")
    ax_l2.set_xticks(range(len(used_curr)))
    ax_l2.set_xticklabels([CURRICULUM_LABEL.get(c, c) for c in used_curr],
                            rotation=35, ha="right", fontsize=6)
    ax_l2.set_yticks(range(len(used_l1s)))
    ax_l2.set_yticklabels([L1_PRETTY.get(l, l) for l in used_l1s], fontsize=7)
    for ytick, l1 in zip(ax_l2.get_yticklabels(), used_l1s):
        ytick.set_color(L1_PALETTE.get(l1, "#222"))
    ax_l2.set_xlabel("Curriculum", fontsize=7)
    ax_l2.set_ylabel("L1 reader cohort", fontsize=7)
    ax_l2.set_title("(a) L2-reader cohorts (matched-L1 reading English)\n"
                     "  ↑ higher = better cognitive fit",
                     fontsize=7.5)

    # (b) L1-native single column
    g_l1 = (df_l1.groupby("curriculum")["delta_loglik"]
                  .mean().reindex(used_curr))
    ax_l1.imshow(g_l1.to_numpy(dtype=float).reshape(-1, 1), cmap="RdBu_r",
                  vmin=-vmax_l2, vmax=vmax_l2, aspect="auto")
    for c, curr in enumerate(used_curr):
        v = g_l1.get(curr, np.nan)
        if not np.isfinite(v):
            mark_missing_cell(ax_l1, 0, c, w=0.96, h=0.96)
        else:
            ax_l1.text(0, c, f"{v:.0f}", ha="center", va="center",
                        fontsize=5.5,
                        color="white" if abs(v) > vmax_l2 * 0.55 else "#222")
    ax_l1.set_yticks(range(len(used_curr)))
    ax_l1.set_yticklabels([CURRICULUM_LABEL.get(c, c) for c in used_curr],
                            fontsize=6)
    ax_l1.set_xticks([0])
    ax_l1.set_xticklabels(["English-native\nreaders (en)"], fontsize=6)
    ax_l1.set_ylabel("Curriculum", fontsize=7)
    ax_l1.set_title("(b) L1-native\ncontrol cohort", fontsize=7.5)

    # (c) Gap (L2 − L1) per (L1, curriculum) — picks up the "L2-specific" claim
    gap = g_l2.copy()
    for curr in used_curr:
        v_l1 = g_l1.get(curr, np.nan)
        if np.isfinite(v_l1):
            gap[curr] = gap[curr] - v_l1
    arr = gap.to_numpy(dtype=float)
    vmax_g = float(np.nanmax(np.abs(arr))) or 1.0
    ax_gap.imshow(arr, cmap="RdBu_r", vmin=-vmax_g, vmax=vmax_g, aspect="auto")
    for r, l1 in enumerate(used_l1s):
        for c, curr in enumerate(used_curr):
            v = gap.iloc[r, c]
            if not np.isfinite(v):
                mark_missing_cell(ax_gap, c, r, w=0.96, h=0.96)
            else:
                ax_gap.text(c, r, f"{v:+.0f}", ha="center", va="center",
                             fontsize=5.5,
                             color="white" if abs(v) > vmax_g * 0.55 else "#222")
    ax_gap.set_xticks(range(len(used_curr)))
    ax_gap.set_xticklabels([CURRICULUM_LABEL.get(c, c) for c in used_curr],
                             rotation=35, ha="right", fontsize=6)
    ax_gap.set_yticks([])
    ax_gap.set_xlabel("Curriculum", fontsize=7)
    ax_gap.set_title("(c) L2 − L1 gap = the falsifier test\n"
                       "  ↑ positive = beetle helps L2 readers *specifically*",
                       fontsize=7.5)

    cb = fig.colorbar(im, ax=[ax_l2, ax_l1, ax_gap], shrink=0.6, pad=0.015,
                       aspect=18, location="right")
    cb.set_label(axis_label("ΔlogLik / L2-L1 gap", better="higher"),
                  fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    fig.suptitle("D7 — Beetle advantage is L2-specific: L2 cohort vs L1-native cohort, "
                  "per (L1 × curriculum)",
                  fontsize=9)
    direction_badge(fig, "L2 − L1 gap", better="higher",
                     zero_means="advantage is generic, not L2-specific")
    coverage_badge(fig, df_l2, dims=("l1_code", "curriculum"), text_y=-0.01)
    savefig_pub(fig, out_path.with_suffix(""))
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--langs", nargs="+", default=list(CANONICAL_LANGS))
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from plots.checkpoint_loaders import load_meco_checkpoint_files  # type: ignore

    csv_path = result_csv(SLUG)
    if csv_path.exists() and not args.overwrite:
        df = pd.read_csv(csv_path)
        print(f"[d7] using cached {csv_path}", file=sys.stderr)
    else:
        surp = load_meco_checkpoint_files()
        if surp.empty:
            print("[d7] no MECO surprisal CSVs found", file=sys.stderr)
            return 2
        last_step = surp.groupby("model")["step"].max().rename("_last")
        surp = surp.join(last_step, on="model")
        surp = surp[surp["step"] == surp["_last"]].drop(columns=["_last"])
        df = compute(surp, langs=args.langs)
        df.to_csv(csv_path, index=False)
        print(f"[d7] wrote {csv_path} ({len(df)} rows)", file=sys.stderr)

    plot(df, figure_pdf(SLUG))
    print(f"[d7] wrote {figure_pdf(SLUG)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
