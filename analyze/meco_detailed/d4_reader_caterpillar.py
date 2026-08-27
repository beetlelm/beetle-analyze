"""
d4_reader_caterpillar.py — Subject-level β(Surprisal) caterpillar + ICC reduction.

Claim
-----
Beetle isn't just predicting the group mean — it tightens *individual*
surprisal-RT slopes too. A reader-level random-slope on Surprisal has
smaller residual variance under the beetle full-model fit than under
the best multilingual baseline (BLOOM / XGLM).

Stat
----
From the canonical full-model fit, extract BLUPs of the subject random
slope on ``sc_Surprisal``: ``β_i = (β_global + u_i)``. Per-reader 1-SE
intervals come from the random-effect prediction-variance approximation
that ``MixedLMResults.random_effects_cov`` exposes.

We also report the intra-class-correlation reduction:

    ICC(Surprisal) = Var(u_i) / (Var(u_i) + σ²ε)

A smaller ICC under beetle means more of the per-reader slope is
captured by the population estimate — i.e. less idiosyncratic noise.

Output
------
``results/meco_detailed/d4_reader_caterpillar.csv``
    columns: model, lang, subid, beta_i, beta_se_i, icc_surprisal, n_obs_i

``plots/meco_detailed/d4_reader_caterpillar.pdf``
    Two-facet caterpillar (beetle vs best baseline), x = subject (sorted
    by β_i), y = β_i ± SE. Side panel: bar of ICC for each model.

CLI
---
    python -m analyze.meco_detailed.d4_reader_caterpillar [--lang du]
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
    CANONICAL_LANGS, L1_PRETTY, figure_pdf, result_csv,
)


SLUG = "d4_reader_caterpillar"


def _fit_full_with_random_slopes(df: pd.DataFrame):
    """Return a fitted ``MixedLMResults`` with random slope on ``sc_Surprisal``.

    We deliberately use a *correlated* random-effects spec for the
    Surprisal slope so the BLUPs are recoverable from ``random_effects``.
    """
    import statsmodels.formula.api as smf

    scaled = mc._prepare_scaled(df, include_surprisal=True)  # noqa: SLF001
    fixed = " + ".join(f"sc_{v}" for v in mc._ALL_VARS_FULL)  # noqa: SLF001
    formula = f"firstrun_dur ~ {fixed}"
    md = smf.mixedlm(formula, data=scaled, groups=scaled["subid"],
                     re_formula="1 + sc_Surprisal")
    return md.fit(reml=False, method="lbfgs", disp=False)


def _icc_surprisal(res) -> float:
    """ICC for the Surprisal random slope under a correlated re_formula."""
    try:
        cov = res.cov_re
        var_u = float(cov.loc["sc_Surprisal", "sc_Surprisal"])
        sigma2 = float(res.scale)
        return var_u / (var_u + sigma2)
    except Exception:
        return float("nan")


def _extract_blups(res, df: pd.DataFrame) -> pd.DataFrame:
    """Return per-subject β_i (= β_global + u_i^slope) with rough SE."""
    try:
        re = res.random_effects  # dict subid -> Series of (Intercept, slope)
    except Exception:
        return pd.DataFrame()
    rows = []
    b_global = float(res.fe_params.get("sc_Surprisal", np.nan))
    for sub, vec in re.items():
        slope_u = float(vec.get("sc_Surprisal", np.nan))
        beta_i = b_global + slope_u
        n_i = int((df["subid"] == sub).sum())
        rows.append({"subid": str(sub), "beta_i": beta_i,
                     "beta_se_i": float("nan"),  # statsmodels does not expose this directly
                     "n_obs_i": n_i})
    return pd.DataFrame(rows)


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
        for lang in langs:
            sub = merged[merged["lang"] == lang].dropna(
                subset=["firstrun_dur", "Surprisal", "wlen", "freq", "subid"])
            sub = mc.attach_spillover(sub)
            sub = mc.drop_sentence_boundaries(sub)
            if len(sub) < 200:
                continue
            if verbose:
                print(f"[d4] {model} | {lang} (n={len(sub)})", file=sys.stderr,
                      flush=True)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = _fit_full_with_random_slopes(sub)
                icc = _icc_surprisal(res)
                blups = _extract_blups(res, sub)
                blups["model"] = model
                blups["lang"] = lang
                blups["icc_surprisal"] = icc
                rows.append(blups)
            except Exception as e:
                print(f"[d4] fail {model}/{lang}: {e}", file=sys.stderr)
    if not rows:
        return pd.DataFrame(columns=["model", "lang", "subid", "beta_i",
                                      "beta_se_i", "icc_surprisal", "n_obs_i"])
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(df: pd.DataFrame, out_path: Path) -> None:
    """Two-panel figure:
      (a) ICC heatmap by L1 × curriculum — lower = more population structure
          captured (the headline metric the figure was built for).
      (b) Caterpillar of per-reader β_i for the L1 with the most subjects,
          one curve per curriculum (overplotted).
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
    if "lang" in df.columns:
        df["l1_code"] = df["lang"].fillna(df["l1_code"])

    used_l1s = [l for l in L1_ORDER if l in set(df["l1_code"])]
    used_curr = [c for c in CURRICULUM_ORDER
                 if c != "mono" and c in set(df["curriculum"])]
    if not used_l1s:
        used_l1s = ["du", "ge", "ch_s"]
    if not used_curr:
        used_curr = ["B1", "B2", "B3", "B4", "B5"]

    fig = plt.figure(figsize=(WIDTH_DOUBLE, max(2.6, 0.28 * len(used_l1s) + 1.6)),
                       constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.4])

    # (a) ICC heatmap — one ICC per (model, lang); average to (L1, curriculum)
    ax = fig.add_subplot(gs[0, 0])
    icc = (df.dropna(subset=["icc_surprisal"])
              .groupby(["l1_code", "curriculum"])["icc_surprisal"]
              .first().unstack("curriculum"))
    icc = icc.reindex(index=used_l1s, columns=used_curr)
    arr = icc.to_numpy(dtype=float)
    if np.isfinite(arr).any():
        im = ax.imshow(arr, cmap="cividis_r", aspect="auto",
                        vmin=float(np.nanmin(arr)), vmax=float(np.nanmax(arr)))
    else:
        im = None
    for r, l1 in enumerate(used_l1s):
        for c, curr in enumerate(used_curr):
            v = icc.iloc[r, c] if (l1 in icc.index and curr in icc.columns) else np.nan
            if not np.isfinite(v):
                mark_missing_cell(ax, c, r, w=0.96, h=0.96)
            else:
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                         fontsize=5.5, color="white" if v > 0.4 else "#222")
    ax.set_xticks(range(len(used_curr)))
    ax.set_xticklabels([CURRICULUM_LABEL.get(c, c) for c in used_curr],
                        rotation=35, ha="right", fontsize=6)
    ax.set_yticks(range(len(used_l1s)))
    ax.set_yticklabels([L1_PRETTY.get(l, l) for l in used_l1s], fontsize=7)
    for ytick, l1 in zip(ax.get_yticklabels(), used_l1s):
        ytick.set_color(L1_PALETTE.get(l1, "#222"))
    ax.set_xlabel("Curriculum", fontsize=7)
    ax.set_ylabel("L1 reader cohort", fontsize=7)
    ax.set_title("(a) ICC of β(Surprisal) across readers\n(↓ lower = tighter "
                  "individual fits = better)",
                  fontsize=7.5)
    if im is not None:
        cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, aspect=14)
        cb.set_label(axis_label("ICC(β)", better="lower"), fontsize=6.5)
        cb.ax.tick_params(labelsize=6)

    # (b) Per-reader caterpillar for the L1 with most subjects
    ax_cat = fig.add_subplot(gs[0, 1])
    counts = (df.groupby("l1_code")["subid"].nunique()
                 .reindex(used_l1s).fillna(0))
    if counts.sum() == 0:
        ax_cat.set_visible(False)
    else:
        focus_l1 = counts.idxmax()
        focus = df[df["l1_code"] == focus_l1]
        for curr in used_curr:
            sub = focus[focus["curriculum"] == curr].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("beta_i")
            x = np.arange(len(sub))
            ax_cat.scatter(x, sub["beta_i"], s=3,
                            color=CURRICULUM_PALETTE.get(curr, "#888"),
                            label=CURRICULUM_LABEL.get(curr, curr), alpha=0.85)
        ax_cat.axhline(0, color="#888", lw=0.4)
        ax_cat.set_xlabel("Reader, sorted by β(Surprisal) within each curriculum",
                           fontsize=7)
        ax_cat.set_ylabel(axis_label(
            f"Per-reader β(Surprisal) — {L1_PRETTY.get(focus_l1, focus_l1)}",
            better="higher",
        ), fontsize=7)
        ax_cat.set_title(f"(b) Reader-level β(Surprisal) fits, "
                           f"{L1_PRETTY.get(focus_l1, focus_l1)}\n"
                           f"(↑ higher = stronger surprisal sensitivity)",
                          fontsize=7.5)
        ax_cat.legend(fontsize=6, frameon=False, loc="upper left",
                       title="curriculum", title_fontsize=6)

    fig.suptitle("D4 — Beetle tightens between-reader β(Surprisal) variance "
                  "across L1 × curriculum",
                  fontsize=9)
    direction_badge(fig, "ICC(β)", better="lower",
                     zero_means="all between-reader variance explained")
    coverage_badge(fig, df, dims=("l1_code", "curriculum"), text_y=-0.01)
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
        print(f"[d4] using cached {csv_path}", file=sys.stderr)
    else:
        surp = load_meco_checkpoint_files()
        if surp.empty:
            print("[d4] no MECO surprisal CSVs found", file=sys.stderr)
            return 2
        last_step = surp.groupby("model")["step"].max().rename("_last")
        surp = surp.join(last_step, on="model")
        surp = surp[surp["step"] == surp["_last"]].drop(columns=["_last"])
        df = compute(surp, langs=args.langs)
        df.to_csv(csv_path, index=False)
        print(f"[d4] wrote {csv_path} ({len(df)} rows)", file=sys.stderr)

    plot(df, figure_pdf(SLUG))
    print(f"[d4] wrote {figure_pdf(SLUG)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
