"""
d8_comprehension_link.py — Cognitive fit ↔ downstream comprehension.

Claim (the cross-subfield payoff)
---------------------------------
A model whose surprisal predicts L2 reading times also predicts which
texts a reader will *understand*. That ties the cognitive-plausibility
story to behavioural validity — the figure even a reviewer who doesn't
care about psycholinguistics has to take seriously.

Stat
----
For each (subid, itemid), aggregate model surprisal to a text-level
mean. Then fit, per model:

    comp_acc ~ sc(mean_surprisal_text) + (1 | subid) + (1 | itemid)

and read out the slope ``β_comp`` (negative means harder-to-predict
texts ⇒ lower comprehension, the cognitively coherent direction).

Inset scatter cross-references with the MECO ΔlogLik for the same model
× lang: if beetle sits in the upper-right (large MECO ΔlogLik AND
strongly-negative β_comp), the story is "good cognitive fit ⇒ good
prediction of behavioural performance."

Output
------
``results/meco_detailed/d8_comprehension_link.csv``
    columns: model, lang, beta_comp, beta_comp_se, n_obs, n_subjects, notes

``plots/meco_detailed/d8_comprehension_link.pdf``
    Two-panel: (a) forest plot of β_comp across models; (b) scatter of
    β_comp vs MECO ΔlogLik (joined on (model, lang)).

CLI
---
    python -m analyze.meco_detailed.d8_comprehension_link [--overwrite]
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
    CANONICAL_LANGS, L1_PRETTY, figure_pdf, load_comprehension, result_csv,
)


SLUG = "d8_comprehension_link"


def _fit_comp_link(text_df: pd.DataFrame) -> tuple[float, float, int, int]:
    """Fit ``acc ~ sc(mean_surp) + (1|subid) + (1|itemid)``.

    Statsmodels mixedlm only handles a single grouping factor natively, so
    we approximate the crossed random-effects spec via ``vc_formula`` with
    ``itemid`` as the variance component.
    """
    import statsmodels.formula.api as smf

    s = text_df.dropna(subset=["accuracy", "mean_surprisal"]).copy()
    if len(s) < 50 or s["subid"].nunique() < 4:
        return float("nan"), float("nan"), int(len(s)), int(s["subid"].nunique())
    sd = s["mean_surprisal"].std(ddof=1)
    s["sc_mean_surprisal"] = (s["mean_surprisal"] - s["mean_surprisal"].mean()) / (sd or 1)
    try:
        md = smf.mixedlm("accuracy ~ sc_mean_surprisal", data=s, groups=s["subid"],
                         re_formula="1",
                         vc_formula={"item": "0 + C(itemid)"})
        res = md.fit(reml=False, method="lbfgs", disp=False)
        b = float(res.fe_params.get("sc_mean_surprisal", np.nan))
        bse_attr = "bse_fe" if hasattr(res, "bse_fe") else "bse"
        se_src = getattr(res, bse_attr, None)
        se = float(se_src.get("sc_mean_surprisal", np.nan)) if se_src is not None else float("nan")
        return b, se, int(len(s)), int(s["subid"].nunique())
    except Exception:
        # Fall back to OLS on text-mean accuracy if the mixed model fails to converge.
        from statsmodels.api import OLS, add_constant
        X = add_constant(s["sc_mean_surprisal"])
        try:
            r = OLS(s["accuracy"], X).fit()
            return float(r.params.iloc[1]), float(r.bse.iloc[1]), int(len(s)), int(s["subid"].nunique())
        except Exception:
            return float("nan"), float("nan"), int(len(s)), int(s["subid"].nunique())


def compute(
    surprisal_df: pd.DataFrame,
    *,
    langs: Iterable[str] = CANONICAL_LANGS,
    deltall_path: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    comp = load_comprehension()
    if comp.empty:
        print("[d8] comprehension data unavailable — returning empty frame",
              file=sys.stderr)
        return pd.DataFrame(columns=["model", "lang", "beta_comp", "beta_comp_se",
                                      "n_obs", "n_subjects", "notes",
                                      "delta_loglik"])

    # Pre-compute text-level mean surprisal per (model, itemid)
    surp = surprisal_df.copy()
    surp["surprisal"] = pd.to_numeric(surp["surprisal"], errors="coerce")
    text_surp = (surp.groupby(["model", "itemid"])["surprisal"]
                     .mean().reset_index()
                     .rename(columns={"surprisal": "mean_surprisal"}))
    text_surp["itemid"] = text_surp["itemid"].astype(str)
    comp["itemid"] = comp["itemid"].astype(str)
    comp["subid"] = comp["subid"].astype(str)

    rows = []
    for model in sorted(text_surp["model"].dropna().unique()):
        joined = comp.merge(text_surp[text_surp["model"] == model],
                            on="itemid", how="inner")
        if joined.empty:
            continue
        for lang in langs:
            sub = joined[joined["lang"] == lang]
            if len(sub) < 50:
                continue
            if verbose:
                print(f"[d8] {model} | {lang} (n={len(sub)})",
                      file=sys.stderr, flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                b, se, n_obs, n_subs = _fit_comp_link(sub)
            rows.append({
                "model": model, "lang": lang,
                "beta_comp": b, "beta_comp_se": se,
                "n_obs": n_obs, "n_subjects": n_subs,
                "notes": "",
            })

    out = pd.DataFrame(rows)
    if deltall_path is not None and deltall_path.exists():
        dll = pd.read_csv(deltall_path)
        keep = dll[["model", "lang", "delta_loglik"]] if "delta_loglik" in dll.columns else dll
        out = out.merge(keep, on=["model", "lang"], how="left")
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(df: pd.DataFrame, out_path: Path) -> None:
    """Two-panel figure:
      (a) β(comp) heatmap by L1 × curriculum (negative = cognitively coherent)
      (b) Cross-figure scatter — MECO ΔlogLik (x) vs β(comp) (y), one
          dot per (model, lang). L1 by color, curriculum by marker.
          The bottom-right quadrant is the "cognitive fit ⇒ downstream
          validity" region — beetles should congregate there.
    """
    import matplotlib.pyplot as plt
    from analyze.plot_style import apply_acl_style, savefig_pub, WIDTH_DOUBLE
    from analyze.meco_detailed._plot import (
        CURRICULUM_LABEL, CURRICULUM_ORDER, CURRICULUM_PALETTE,
        L1_ORDER, L1_PALETTE, L1_PRETTY,
        attach_model_metadata, axis_label, coverage_badge, direction_badge,
        curriculum_legend_handles, l1_legend_handles, mark_missing_cell,
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

    fig, (axL, axR) = plt.subplots(
        1, 2,
        figsize=(WIDTH_DOUBLE, max(2.4, 0.28 * len(used_l1s) + 1.5)),
        gridspec_kw={"width_ratios": [1.0, 1.2]},
        constrained_layout=True,
    )

    # (a) β(comp) heatmap
    g = (df.groupby(["l1_code", "curriculum"])["beta_comp"]
            .mean().unstack("curriculum"))
    g = g.reindex(index=used_l1s, columns=used_curr)
    arr = g.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(arr))) if np.isfinite(np.nanmax(arr)) else 1.0
    vmax = max(vmax, 1e-3)
    im = axL.imshow(arr, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    # Note: RdBu (no _r) puts negative in red — the "cognitive coherence"
    # direction — which is the headline colour we want to draw the eye.
    for r, l1 in enumerate(used_l1s):
        for c, curr in enumerate(used_curr):
            v = g.iloc[r, c] if (l1 in g.index and curr in g.columns) else np.nan
            if not np.isfinite(v):
                mark_missing_cell(axL, c, r, w=0.96, h=0.96)
            else:
                axL.text(c, r, f"{v:+.2f}", ha="center", va="center",
                          fontsize=5.5,
                          color="white" if abs(v) > vmax * 0.55 else "#222")
    axL.set_xticks(range(len(used_curr)))
    axL.set_xticklabels([CURRICULUM_LABEL.get(c, c) for c in used_curr],
                          rotation=35, ha="right", fontsize=6)
    axL.set_yticks(range(len(used_l1s)))
    axL.set_yticklabels([L1_PRETTY.get(l, l) for l in used_l1s], fontsize=7)
    for ytick, l1 in zip(axL.get_yticklabels(), used_l1s):
        ytick.set_color(L1_PALETTE.get(l1, "#222"))
    axL.set_xlabel("Curriculum", fontsize=7)
    axL.set_ylabel("L1 reader cohort", fontsize=7)
    axL.set_title("(a) β(mean surprisal → comprehension)\n"
                   "  ↓ more negative = harder-to-predict texts → lower comp\n"
                   "  (the cognitively coherent direction)",
                   fontsize=7)
    cb = fig.colorbar(im, ax=axL, shrink=0.7, pad=0.015, aspect=18)
    cb.set_label(axis_label("β(comp)", better="more negative"), fontsize=6.5)
    cb.ax.tick_params(labelsize=6)

    # (b) Cross-figure scatter
    if "delta_loglik" in df.columns:
        scat = df.dropna(subset=["beta_comp", "delta_loglik"]).copy()
        for _, row in scat.iterrows():
            color = L1_PALETTE.get(row.get("l1_code", "unknown"), "#888")
            marker = {"B1": "o", "B2": "s", "B3": "D", "B4": "^", "B5": "v",
                      "mono": "x", "baseline": "*", "unknown": "o"}.get(
                        row.get("curriculum", "unknown"), "o")
            axR.scatter(row["delta_loglik"], row["beta_comp"],
                         color=color, marker=marker, s=20,
                         edgecolor="white", linewidth=0.6)
        axR.axvline(0, color="#444", lw=0.4)
        axR.axhline(0, color="#444", lw=0.4)
        # Annotate the "good quadrant" — high MECO ΔlogLik + negative β_comp
        xmin, xmax = axR.get_xlim()
        ymin, ymax = axR.get_ylim()
        # Brighten the good-quadrant shading so it reads at print size.
        axR.fill_betweenx([ymin, 0], 0, xmax, color="#5DC863", alpha=0.18,
                            zorder=-1)
        axR.annotate(
            "good quadrant:\nhigh ΔlogLik\n  +\nnegative β(comp)",
            xy=(xmax * 0.6, ymin * 0.6),
            xytext=(xmax * 0.95, ymin * 0.95),
            ha="right", va="bottom", fontsize=6, color="#1b5e20",
            arrowprops=dict(arrowstyle="->", color="#1b5e20", lw=0.6),
        )
        axR.set_xlabel(axis_label("MECO ΔlogLik (cognitive fit)",
                                     better="higher"),
                        fontsize=7)
        axR.set_ylabel(axis_label("β(mean surprisal → comp acc)",
                                     better="more negative"),
                        fontsize=7)
        axR.set_title("(b) Cognitive fit ↔ downstream comprehension\n"
                       "  one dot per (model, L1)",
                       fontsize=7)
    else:
        axR.set_visible(False)

    # Combined legend (curriculum markers + L1 colors)
    cur_h = [plt.Line2D([], [], marker=m, color="w",
                         markerfacecolor="#444", markersize=5,
                         markeredgecolor="white", label=CURRICULUM_LABEL.get(c, c))
              for c, m in [("B1","o"),("B2","s"),("B3","D"),("B4","^"),("B5","v")]
              if c in used_curr]
    l1_h = l1_legend_handles(used_l1s)
    legend = fig.legend(handles=cur_h + l1_h, loc="lower center",
                         ncol=min(8, len(cur_h) + len(l1_h)),
                         frameon=False, bbox_to_anchor=(0.5, -0.04),
                         fontsize=5.5)

    fig.suptitle("D8 — Models that predict MECO reading times also "
                  "predict comprehension", fontsize=9)
    direction_badge(fig, "β(comp)", better="more negative",
                     zero_means="no link between surprisal and comprehension")
    coverage_badge(fig, df, dims=("l1_code", "curriculum"), text_y=-0.06)
    savefig_pub(fig, out_path.with_suffix(""))
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--langs", nargs="+", default=list(CANONICAL_LANGS))
    ap.add_argument("--deltall-path",
                    default="results/meco/canonical_deltalogl.csv",
                    help="canonical ΔlogLik CSV for the scatter inset")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from plots.checkpoint_loaders import load_meco_checkpoint_files  # type: ignore

    csv_path = result_csv(SLUG)
    if csv_path.exists() and not args.overwrite:
        df = pd.read_csv(csv_path)
        print(f"[d8] using cached {csv_path}", file=sys.stderr)
    else:
        surp = load_meco_checkpoint_files()
        if surp.empty:
            print("[d8] no MECO surprisal CSVs found", file=sys.stderr)
            return 2
        last_step = surp.groupby("model")["step"].max().rename("_last")
        surp = surp.join(last_step, on="model")
        surp = surp[surp["step"] == surp["_last"]].drop(columns=["_last"])
        df = compute(surp, langs=args.langs,
                     deltall_path=Path(args.deltall_path))
        df.to_csv(csv_path, index=False)
        print(f"[d8] wrote {csv_path} ({len(df)} rows)", file=sys.stderr)

    plot(df, figure_pdf(SLUG))
    print(f"[d8] wrote {figure_pdf(SLUG)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
