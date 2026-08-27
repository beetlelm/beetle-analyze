"""
d2_spillover_decay.py — Per-model spillover signature on MECO L2.

Claim
-----
Beetle reproduces the human-like *temporal* surprisal profile: large
immediate effect at lag 0, smaller but non-zero spillover at lag-1 and
lag-2. Frontier multilingual LLMs (BLOOM-7B / XGLM-4.5B) over-weight
lag 0 and decay too sharply — their surprisal is "too eager" relative to
the human reading-time signal.

Stat
----
From the canonical full-model fit (``meco_canonical.fit_canonical_pair``),
extract the fixed-effect betas for

    sc_Surprisal     (lag  0  — current word)
    sc_Surprisal1p   (lag -1 — preceding word, immediate spillover)
    sc_Surprisal2p   (lag -2 — two words back, delayed spillover)

with their SEs. Standardise within model (divide by Σ|β| so the area
under the lag curve is fixed; this makes shape comparable across models
with very different baseline ΔlogLik).

Output
------
``results/meco_detailed/d2_spillover_decay.csv``
    columns: model, lang, lag, beta, beta_se, beta_norm

``plots/meco_detailed/d2_spillover_decay.pdf``
    line plot, x ∈ {0, −1, −2}, y = β_norm, one line per model,
    faceted by L1. Baseline LLMs in dashed grey, beetle in bright.

CLI
---
    python -m analyze.meco_detailed.d2_spillover_decay [--overwrite]
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


SLUG = "d2_spillover_decay"

LAG_NAMES = {0: "sc_Surprisal", 1: "sc_Surprisal1p", 2: "sc_Surprisal2p"}


def _fit_full_and_extract_betas(df: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Fit the canonical full model; return {lag: (beta, se)} for lags 0/1/2.

    Uses the same statsmodels backend selection as the canonical pipeline
    so reported betas are directly comparable across modules.
    """
    import statsmodels.formula.api as smf

    scaled = mc._prepare_scaled(df, include_surprisal=True)  # noqa: SLF001
    fixed = " + ".join(f"sc_{v}" for v in mc._ALL_VARS_FULL)  # noqa: SLF001
    formula = f"firstrun_dur ~ {fixed}"
    vc_full = {f"rs_{v}": f"0 + sc_{v}" for v in mc._ALL_VARS_FULL}  # noqa: SLF001

    res = None
    for vc, label in ((vc_full, "full"),
                      ({f"rs_{v}": f"0 + sc_{v}" for v in mc._SURPRISAL_VARS}, "surp"),  # noqa: SLF001
                      (None, "ri")):
        try:
            md = smf.mixedlm(formula, data=scaled, groups=scaled["subid"],
                             re_formula="1", vc_formula=vc)
            res = md.fit(reml=False, method="lbfgs", disp=False)
            if np.isfinite(float(res.llf)):
                break
        except Exception:
            res = None
            continue
    if res is None:
        return {k: (float("nan"), float("nan")) for k in (0, 1, 2)}

    out: dict[int, tuple[float, float]] = {}
    bse_attr = "bse_fe" if hasattr(res, "bse_fe") else "bse"
    se_src = getattr(res, bse_attr, None)
    for lag, term in LAG_NAMES.items():
        b = float(res.fe_params.get(term, np.nan))
        se = float(se_src.get(term, np.nan)) if se_src is not None else float("nan")
        out[lag] = (b, se)
    return out


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
                print(f"[d2] {model} | {lang} (n={len(sub)})",
                      file=sys.stderr, flush=True)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    betas = _fit_full_and_extract_betas(sub)
            except Exception as e:
                print(f"[d2] fail {model}/{lang}: {e}", file=sys.stderr)
                continue
            total = sum(abs(b) for b, _ in betas.values() if np.isfinite(b))
            for lag, (b, se) in betas.items():
                rows.append({
                    "model": model, "lang": lang, "lag": lag,
                    "beta": b, "beta_se": se,
                    "beta_norm": (b / total) if total > 0 else float("nan"),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(df: pd.DataFrame, out_path: Path) -> None:
    """Spillover signature per L1 (rows) × curriculum (cols).

    Empty cells (no model trained for that (L1, curriculum)) are kept as
    placeholders with an 'n/a' overlay so the matrix structure stays
    readable even when many cells aren't populated.
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

    fig, axes = plt.subplots(len(used_l1s), len(used_curr),
                              figsize=(WIDTH_DOUBLE,
                                       max(3.4, 0.65 * len(used_l1s) + 1.4)),
                              sharex=True, sharey=True, squeeze=False,
                              constrained_layout=True)

    # Average β_norm per (L1, curriculum, lag) — collapse over models in cell
    grouped = (df.groupby(["l1_code", "curriculum", "lag"])["beta_norm"]
                  .mean().reset_index())

    # Baselines (non-beetle) plotted as a single grey reference per panel
    base_df = df[df["is_beetle"] == False]
    base_lag_means = (base_df.groupby(["lang", "lag"])["beta_norm"].mean()
                            .reset_index() if not base_df.empty else None)

    for r, l1 in enumerate(used_l1s):
        for c, curr in enumerate(used_curr):
            ax = axes[r][c]
            ax.axhline(0, color="#bbb", lw=0.4)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["0", "-1", "-2"], fontsize=6)
            ax.set_ylim(-0.05, 1.05)
            cell = grouped[(grouped["l1_code"] == l1) &
                            (grouped["curriculum"] == curr)].sort_values("lag")
            if cell.empty:
                # Empty (L1, curriculum) cell — overlay placeholder
                ax.set_xticks([])
                ax.set_yticks([])
                mark_missing_cell(ax, 1.0, 0.5, w=2.6, h=1.1, label="not trained")
                continue
            ax.plot(cell["lag"], cell["beta_norm"],
                     color=CURRICULUM_PALETTE.get(curr, "#444"),
                     lw=1.4, marker="o", ms=3.0,
                     mec="white", mew=0.5)
            # Baseline reference for this L1 (if available)
            if base_lag_means is not None:
                bsl = base_lag_means[base_lag_means["lang"] == l1].sort_values("lag")
                if not bsl.empty:
                    ax.plot(bsl["lag"], bsl["beta_norm"],
                             color="#888888", lw=0.7, linestyle="--",
                             alpha=0.7, zorder=0)
            if r == 0:
                ax.set_title(CURRICULUM_LABEL.get(curr, curr),
                              fontsize=7, color=CURRICULUM_PALETTE.get(curr, "#000"))
            if c == 0:
                ax.set_ylabel(L1_PRETTY.get(l1, l1), fontsize=7, rotation=0,
                               ha="right", va="center",
                               color=L1_PALETTE.get(l1, "#000"))
                ax.tick_params(axis="y", labelsize=6)
            else:
                ax.set_yticks([])
            if r == len(used_l1s) - 1:
                ax.set_xlabel("lag (words before current)", fontsize=6)

    # Shared axis labels for the whole grid
    fig.supylabel("normalised β(Surprisal_lag-k)   (Σ|β|=1; shape, not size)",
                   fontsize=7, color="#444")
    fig.supxlabel("Lag k:  0 = current word,  -1 / -2 = one / two words back  "
                   "(human reading: large at 0, decays at -1/-2)",
                   fontsize=7, color="#444")

    fig.suptitle("D2 — Surprisal-RT spillover signature across L1 (rows) × curriculum (cols)",
                  fontsize=9)
    direction_badge(fig, "spillover shape",
                     better="monotonic decay (human-like)",
                     zero_means="no surprisal effect at that lag")
    coverage_badge(fig, df.assign(curriculum=df["curriculum"]),
                    dims=("l1_code", "curriculum"), text_y=-0.01)
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
        print(f"[d2] using cached {csv_path}", file=sys.stderr)
    else:
        surp = load_meco_checkpoint_files()
        if surp.empty:
            print("[d2] no MECO surprisal CSVs found", file=sys.stderr)
            return 2
        last_step = surp.groupby("model")["step"].max().rename("_last")
        surp = surp.join(last_step, on="model")
        surp = surp[surp["step"] == surp["_last"]].drop(columns=["_last"])
        df = compute(surp, langs=args.langs)
        df.to_csv(csv_path, index=False)
        print(f"[d2] wrote {csv_path} ({len(df)} rows)", file=sys.stderr)

    plot(df, figure_pdf(SLUG))
    print(f"[d2] wrote {figure_pdf(SLUG)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
