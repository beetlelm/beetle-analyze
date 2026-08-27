"""
d3_bilingual_decomp.py — Nested L1-mono / L2-mono / bilingual surprisal.

Claim (the mechanism figure)
----------------------------
Beetle's MECO-L2 advantage comes from *bilingual context*, not merely
from being a competent L2 (English) language model. We decompose the
canonical full model's surprisal regressor into three sources:

  • L2-mono     surprisal: from a monolingual English Beetle / GPT
  • L1-mono     surprisal: from a monolingual L1 Beetle (Dutch / German / Chinese)
  • Bilingual   surprisal: from the matched beetle bilingual model

and fit five nested models (controls = wordnum, wlen{,1p,2p}, freq{,1p,2p}):

    M_C   : controls only
    M_L2  : controls + L2-mono Surprisal (+1p +2p)
    M_L1  : controls + L1-mono Surprisal (+1p +2p)
    M_L1L2: controls + L1-mono Surprisal + L2-mono Surprisal (+1p +2p)
    M_BI  : controls + L1-mono + L2-mono + Bilingual Surprisal (+1p +2p)

Decomposition (per language):
  ΔLL_L2_unique  = LL(M_L1L2) − LL(M_L1)        # what L2-mono adds above L1
  ΔLL_L1_unique  = LL(M_L1L2) − LL(M_L2)        # what L1-mono adds above L2
  ΔLL_L1L2_share = LL(M_L1L2) − LL(M_C) − ΔLL_L2_unique − ΔLL_L1_unique
  ΔLL_BI_unique  = LL(M_BI)   − LL(M_L1L2)      # the bilingual *unique* gain

The headline number is ``ΔLL_BI_unique`` — if it is positive and CI
excludes zero, the bilingual model is contributing signal beyond what
L1-mono and L2-mono *together* explain. That's the strongest possible
mechanistic test of the matched-bilingual hypothesis.

Output
------
``results/meco_detailed/d3_bilingual_decomp.csv``
    columns: lang, beetle_model, l1_mono, l2_mono,
             ll_controls, ll_l1, ll_l2, ll_l1l2, ll_bi,
             dll_l2_unique, dll_l1_unique, dll_l1l2_share, dll_bi_unique,
             n_obs, n_subjects, notes

``plots/meco_detailed/d3_bilingual_decomp.pdf``
    stacked horizontal bars per L1 — L2-unique, L1-unique, shared, and
    Bilingual-unique stacked left-to-right. The bilingual-unique sliver
    is the headline.

CLI
---
    python -m analyze.meco_detailed.d3_bilingual_decomp \\
        --beetle BeetleLM/beetlelm_nld_L1-eng_L2_balanced \\
        --l1-mono BeetleLM/beetlelm_nld_mono \\
        --l2-mono BeetleLM/beetlelm_eng_mono \\
        --lang du
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


SLUG = "d3_bilingual_decomp"

# Default decomposition triplets. Override on the CLI.
DEFAULT_TRIPLETS: dict[str, dict[str, str]] = {
    "du": {
        "beetle":  "BeetleLM/beetlelm_eng-nld_balanced",
        "l1_mono": "BeetleLM/beetlelm_nld_mono",
        "l2_mono": "BeetleLM/beetlelm_eng_mono",
    },
    "ge": {
        "beetle":  "BeetleLM/beetlelm_eng-deu_simultaneous",
        "l1_mono": "BeetleLM/beetlelm_deu_mono",
        "l2_mono": "BeetleLM/beetlelm_eng_mono",
    },
    "ch_s": {
        "beetle":  "BeetleLM/beetlelm_eng-zho_balanced",
        "l1_mono": "BeetleLM/beetlelm_zho_mono",
        "l2_mono": "BeetleLM/beetlelm_eng_mono",
    },
}


# ---------------------------------------------------------------------------
# Multi-source surprisal join
# ---------------------------------------------------------------------------

def _join_three_surprisals(
    rt: pd.DataFrame,
    surprisal_df: pd.DataFrame,
    *,
    l1_mono: str,
    l2_mono: str,
    beetle: str,
) -> pd.DataFrame:
    """Attach three surprisal columns: ``S_L1``, ``S_L2``, ``S_BI``."""
    def _sub(model: str, col: str) -> pd.DataFrame:
        s = (surprisal_df[surprisal_df["model"] == model]
             [["itemid", "wordnum", "surprisal"]]
             .rename(columns={"surprisal": col}).copy())
        s["itemid"] = s["itemid"].astype(str)
        s["wordnum"] = s["wordnum"].astype(int)
        return s.drop_duplicates(["itemid", "wordnum"], keep="last")

    out = rt
    for model, col in [(l1_mono, "S_L1"), (l2_mono, "S_L2"), (beetle, "S_BI")]:
        out = out.merge(_sub(model, col), on=["itemid", "wordnum"], how="inner")
    return out


def _attach_spillover_three(df: pd.DataFrame) -> pd.DataFrame:
    """Spillover for each of S_L1, S_L2, S_BI plus the canonical wlen/freq lags."""
    base_cols = ["itemid", "wordnum", "S_L1", "S_L2", "S_BI", "wlen", "freq"]
    base = df[base_cols].drop_duplicates().copy()
    for shift, suffix in ((1, "1p"), (2, "2p")):
        ren = {c: f"{c}{suffix}" for c in ("S_L1", "S_L2", "S_BI", "wlen", "freq")}
        shifted = (base.rename(columns=ren)
                       .assign(wordnum=lambda d: d["wordnum"] - shift))
        shifted = shifted[shifted["wordnum"] >= 1]
        df = df.merge(shifted, on=["itemid", "wordnum"], how="inner")
    return df


# ---------------------------------------------------------------------------
# Nested fitter
# ---------------------------------------------------------------------------

_CTRL = ["wordnum", "wlen", "wlen1p", "wlen2p", "freq", "freq1p", "freq2p"]

_FAMILIES: dict[str, list[str]] = {
    # name → list of surprisal-family terms to add to the controls
    "C":    [],
    "L2":   ["S_L2", "S_L21p", "S_L22p"],
    "L1":   ["S_L1", "S_L11p", "S_L12p"],
    "L1L2": ["S_L1", "S_L11p", "S_L12p", "S_L2", "S_L21p", "S_L22p"],
    "BI":   ["S_L1", "S_L11p", "S_L12p",
             "S_L2", "S_L21p", "S_L22p",
             "S_BI", "S_BI1p", "S_BI2p"],
}


def _zscale(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - x.mean()) / sd


def _fit_family(df: pd.DataFrame, family_terms: list[str]) -> float:
    """Fit one nested family; return log-likelihood (NaN on fail)."""
    import statsmodels.formula.api as smf

    terms = _CTRL + family_terms
    sub = df.copy()
    for t in terms:
        sub[f"sc_{t}"] = _zscale(sub[t])
    formula = "firstrun_dur ~ " + " + ".join(f"sc_{t}" for t in terms)
    vc = {f"rs_{t}": f"0 + sc_{t}" for t in terms}
    for vc_arg, label in ((vc, "full"),
                          ({f"rs_{t}": f"0 + sc_{t}" for t in family_terms}, "surp"),
                          (None, "ri")):
        try:
            md = smf.mixedlm(formula, data=sub, groups=sub["subid"],
                             re_formula="1", vc_formula=vc_arg)
            res = md.fit(reml=False, method="lbfgs", disp=False)
            ll = float(res.llf)
            if np.isfinite(ll):
                return ll
        except Exception:
            continue
    return float("nan")


def fit_decomposition_one(df: pd.DataFrame) -> dict:
    """Fit all five nested models on a prepared dataframe; return ΔLL parts."""
    lls = {name: _fit_family(df, terms) for name, terms in _FAMILIES.items()}
    dll_l2_unique = lls["L1L2"] - lls["L1"]
    dll_l1_unique = lls["L1L2"] - lls["L2"]
    dll_total_l1l2 = lls["L1L2"] - lls["C"]
    dll_l1l2_share = dll_total_l1l2 - dll_l2_unique - dll_l1_unique
    dll_bi_unique = lls["BI"] - lls["L1L2"]
    return {
        "ll_controls": lls["C"], "ll_l1": lls["L1"], "ll_l2": lls["L2"],
        "ll_l1l2": lls["L1L2"], "ll_bi": lls["BI"],
        "dll_l2_unique": dll_l2_unique,
        "dll_l1_unique": dll_l1_unique,
        "dll_l1l2_share": dll_l1l2_share,
        "dll_bi_unique": dll_bi_unique,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute(
    surprisal_df: pd.DataFrame,
    triplets: dict[str, dict[str, str]] | None = None,
    *,
    langs: Iterable[str] = CANONICAL_LANGS,
    verbose: bool = True,
) -> pd.DataFrame:
    triplets = triplets or DEFAULT_TRIPLETS
    from analyze.meco_detailed._common import get_rt_with_features
    rt = get_rt_with_features()
    rows = []
    for lang in langs:
        if lang not in triplets:
            continue
        trip = triplets[lang]
        merged = _join_three_surprisals(
            rt, surprisal_df,
            l1_mono=trip["l1_mono"], l2_mono=trip["l2_mono"],
            beetle=trip["beetle"],
        )
        sub = merged[merged["lang"] == lang].dropna(
            subset=["firstrun_dur", "S_L1", "S_L2", "S_BI",
                    "wlen", "freq", "subid"])
        sub = _attach_spillover_three(sub)
        sub = mc.drop_sentence_boundaries(sub)
        if len(sub) < 200:
            rows.append({
                "lang": lang, **trip,
                "ll_controls": np.nan, "ll_l1": np.nan, "ll_l2": np.nan,
                "ll_l1l2": np.nan, "ll_bi": np.nan,
                "dll_l2_unique": np.nan, "dll_l1_unique": np.nan,
                "dll_l1l2_share": np.nan, "dll_bi_unique": np.nan,
                "n_obs": int(len(sub)),
                "n_subjects": int(sub["subid"].nunique()) if len(sub) else 0,
                "notes": f"n={len(sub)} < 200 — possibly missing model surprisals",
            })
            continue
        if verbose:
            print(f"[d3] {lang}: fitting 5 nested LMs (n={len(sub)})",
                  file=sys.stderr, flush=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = fit_decomposition_one(sub)
            rows.append({
                "lang": lang, **trip, **fit,
                "n_obs": int(len(sub)),
                "n_subjects": int(sub["subid"].nunique()),
                "notes": "",
            })
        except Exception as e:
            rows.append({
                "lang": lang, **trip,
                "ll_controls": np.nan, "ll_l1": np.nan, "ll_l2": np.nan,
                "ll_l1l2": np.nan, "ll_bi": np.nan,
                "dll_l2_unique": np.nan, "dll_l1_unique": np.nan,
                "dll_l1l2_share": np.nan, "dll_bi_unique": np.nan,
                "n_obs": int(len(sub)),
                "n_subjects": int(sub["subid"].nunique()),
                "notes": f"{e.__class__.__name__}: {e}",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(df: pd.DataFrame, out_path: Path) -> None:
    """Stacked-bar decomposition per (L1, curriculum) cell.

    Layout: rows = L1 (typological order), columns = curriculum. Each cell
    shows the four ΔLL contributions stacked left-to-right; the rightmost
    red slice is the **bilingual-unique** gain — the mechanism headline.
    Empty (L1, curriculum) cells are explicitly marked.
    """
    import matplotlib.pyplot as plt
    from analyze.plot_style import apply_acl_style, savefig_pub, WIDTH_DOUBLE
    from analyze.meco_detailed._plot import (
        CURRICULUM_LABEL, CURRICULUM_ORDER, L1_ORDER, L1_PALETTE, L1_PRETTY,
        attach_model_metadata, axis_label, coverage_badge, direction_badge,
        mark_missing_cell,
    )

    apply_acl_style()
    if df.empty:
        return

    # The d3 CSV uses 'beetle' as the model column. Attach metadata via it.
    work = df.copy()
    if "model" not in work.columns and "beetle" in work.columns:
        work["model"] = work["beetle"]
    work = attach_model_metadata(work)
    if "lang" in work.columns:
        work["l1_code"] = work["lang"].fillna(work["l1_code"])

    used_l1s = [l for l in L1_ORDER if l in set(work["l1_code"])]
    used_curr = [c for c in CURRICULUM_ORDER
                 if c != "mono" and c in set(work["curriculum"])]
    if not used_l1s:
        used_l1s = ["du", "ge", "ch_s"]
    if not used_curr:
        used_curr = ["B1", "B3", "B5"]

    parts = [
        ("dll_l1_unique",  "L1-mono unique",   "#4c72b0"),
        ("dll_l2_unique",  "L2-mono unique",   "#dd8452"),
        ("dll_l1l2_share", "L1+L2 shared",     "#cccccc"),
        ("dll_bi_unique",  "Bilingual unique", "#c44e52"),
    ]

    # apply_acl_style sets constrained_layout=True globally; manual GridSpec
    # placement sidesteps the layout engine entirely so the figure stays at
    # exactly the requested ``figsize`` even when stacked-bar cells have
    # extreme xlim:ylim ratios.
    fig_h = max(3.6, 0.55 * len(used_l1s) + 1.5)
    fig = plt.figure(figsize=(WIDTH_DOUBLE, fig_h))
    fig.set_layout_engine("none")
    gs = fig.add_gridspec(
        nrows=len(used_l1s), ncols=len(used_curr),
        left=0.10, right=0.98, top=0.82, bottom=0.20,
        hspace=0.35, wspace=0.10,
    )
    axes: list[list] = [
        [fig.add_subplot(gs[r, c]) for c in range(len(used_curr))]
        for r in range(len(used_l1s))
    ]

    # Common x-axis: max stacked Δlogℒ across all populated cells.
    pop = work.dropna(subset=[c for c, *_ in parts], how="all")
    if not pop.empty:
        xmax = float(np.nanmax(pop[[c for c, *_ in parts]].sum(axis=1))) * 1.10
    else:
        xmax = 1.0
    xmax = max(xmax, 1e-3)

    for r, l1 in enumerate(used_l1s):
        for c, curr in enumerate(used_curr):
            ax = axes[r][c]
            ax.set_yticks([])
            ax.set_ylim(-0.5, 0.5)
            ax.axvline(0, color="#bbb", lw=0.4)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)

            cell = work[(work["l1_code"] == l1) & (work["curriculum"] == curr)]
            if cell.empty or cell[[c for c, *_ in parts]].isna().all().all():
                ax.set_xticks([])
                ax.set_xlim(0, xmax)  # lock xlim BEFORE drawing the rect
                mark_missing_cell(ax, xmax / 2, 0.0, w=xmax * 0.95, h=0.8,
                                   label="not trained")
                ax.set_xlim(0, xmax)  # re-clamp after rect added
            else:
                # Use first-row decomposition (one decomp per cell)
                row = cell.iloc[0]
                left = 0.0
                for col, label, color in parts:
                    v = float(row.get(col, np.nan))
                    if not np.isfinite(v):
                        continue
                    ax.barh(0, v, left=left, color=color,
                             edgecolor="white", linewidth=0.4, height=0.55,
                             label=label if (r == 0 and c == 0) else None)
                    left += v
                ax.set_xlim(0, xmax)

            if r == 0:
                ax.set_title(CURRICULUM_LABEL.get(curr, curr),
                              fontsize=7,
                              color="#222")
            if c == 0:
                ax.set_ylabel(L1_PRETTY.get(l1, l1), fontsize=7, rotation=0,
                               ha="right", va="center",
                               color=L1_PALETTE.get(l1, "#000"))
            if r == len(used_l1s) - 1:
                ax.tick_params(axis="x", labelsize=6)
                ax.set_xlabel(axis_label("ΔlogLik above controls",
                                          better="higher"), fontsize=6.5)
            else:
                ax.set_xticks([])

    # Highlight the largest bilingual-unique slice with a colored box border
    # (more layout-friendly than a free-text annotation arrow).
    pop = work.dropna(subset=["dll_bi_unique"])
    if not pop.empty:
        top = pop.loc[pop["dll_bi_unique"].idxmax()]
        try:
            r_top = used_l1s.index(top["l1_code"])
            c_top = used_curr.index(top["curriculum"])
            ax_top = axes[r_top][c_top]
            for spine in ("top", "bottom", "left", "right"):
                ax_top.spines[spine].set_visible(True)
                ax_top.spines[spine].set_color("#c44e52")
                ax_top.spines[spine].set_linewidth(1.4)
        except (ValueError, KeyError):
            pass

    # Single shared legend, placed in the reserved bottom margin
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                    frameon=False, fontsize=7,
                    bbox_to_anchor=(0.5, 0.04),
                    title="Surprisal source contributing to ΔlogLik",
                    title_fontsize=7)
    fig.suptitle("D3 — Bilingual surprisal decomposition by L1 (rows) × curriculum (cols)\n"
                  "red-bordered cell = largest bilingual-unique ΔLL; "
                  "rightmost red slice in every bar = the headline metric",
                  fontsize=8.5)
    direction_badge(fig, "bilingual-unique ΔLL", better="higher",
                     zero_means="bilingual model adds nothing above L1+L2 mono")
    coverage_badge(fig, work, dims=("l1_code", "curriculum"), text_y=0.015)
    savefig_pub(fig, out_path.with_suffix(""))
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--langs", nargs="+", default=list(CANONICAL_LANGS))
    ap.add_argument("--beetle", help="override beetle model (single-lang mode)")
    ap.add_argument("--l1-mono", help="override L1-mono model")
    ap.add_argument("--l2-mono", help="override L2-mono model")
    ap.add_argument("--lang", help="single-language override")
    args = ap.parse_args(argv)

    triplets = dict(DEFAULT_TRIPLETS)
    if args.lang and (args.beetle or args.l1_mono or args.l2_mono):
        base = triplets.get(args.lang, {})
        triplets = {args.lang: {
            "beetle":  args.beetle  or base.get("beetle"),
            "l1_mono": args.l1_mono or base.get("l1_mono"),
            "l2_mono": args.l2_mono or base.get("l2_mono"),
        }}

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from plots.checkpoint_loaders import load_meco_checkpoint_files  # type: ignore

    csv_path = result_csv(SLUG)
    if csv_path.exists() and not args.overwrite:
        df = pd.read_csv(csv_path)
        print(f"[d3] using cached {csv_path}", file=sys.stderr)
    else:
        surp = load_meco_checkpoint_files()
        if surp.empty:
            print("[d3] no MECO surprisal CSVs found", file=sys.stderr)
            return 2
        last_step = surp.groupby("model")["step"].max().rename("_last")
        surp = surp.join(last_step, on="model")
        surp = surp[surp["step"] == surp["_last"]].drop(columns=["_last"])
        df = compute(surp, triplets, langs=args.langs)
        df.to_csv(csv_path, index=False)
        print(f"[d3] wrote {csv_path} ({len(df)} rows)", file=sys.stderr)

    plot(df, figure_pdf(SLUG))
    print(f"[d3] wrote {figure_pdf(SLUG)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
