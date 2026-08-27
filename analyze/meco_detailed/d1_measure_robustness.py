"""
d1_measure_robustness.py — ΔlogLik across 5 eye-tracking measures.

Claim
-----
The beetle advantage on MECO L2 is not a first-pass-duration artefact:
ΔlogLik is positive across early measures (firstrun.dur, gaze.dur),
late measures (tt.dur, reg.in), and re-reading (refix).

Stat
----
For each (model, L1, measure):

    Δ_m = logLik(full | DV = measure) − logLik(baseline | DV = measure)

where (full, baseline) are the canonical fixed-effects specs from
``analyze.meco_canonical``. Bootstrap 95 % CI over subjects (B=500).

Output
------
``results/meco_detailed/d1_measure_robustness.csv``
    columns: model, lang, measure, delta_loglik, ci_lo, ci_hi,
             beta_surprisal, beta_surprisal_se, n_obs, n_subjects, notes

``plots/meco_detailed/d1_measure_robustness.pdf``
    5-panel small-multiples — one panel per measure, x = curriculum/model,
    y = ΔlogLik with 95 % CIs; dashed BLOOM-7B reference line.

CLI
---
    python -m analyze.meco_detailed.d1_measure_robustness \\
        [--overwrite] [--langs du ge ch_s] [--bootstrap 500]
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
    CANONICAL_LANGS, ET_LABELS, ET_MEASURES, L1_PRETTY,
    figure_pdf, load_joined_rt_full, result_csv,
)


SLUG = "d1_measure_robustness"


def _fit_pair_with_dv(df: pd.DataFrame, dv: str) -> dict:
    """Run ``meco_canonical.fit_canonical_pair`` with an arbitrary DV.

    The canonical fitter hard-codes ``firstrun_dur`` as the response. We
    transparently swap the DV column name inside a copy of the frame so
    the rest of the canonical pipeline (scaling, vc_formula, optimisers)
    is reused verbatim.
    """
    if dv not in df.columns:
        raise KeyError(f"DV column {dv!r} not in dataframe")
    sub = df.copy()
    # Swap the requested DV into the canonical name.
    if dv != "firstrun_dur":
        sub["firstrun_dur"] = sub[dv]
    sub = sub.dropna(subset=["firstrun_dur"])
    if len(sub) < 200:
        return {"delta_loglik": np.nan, "loglik_full": np.nan,
                "loglik_baseline": np.nan, "beta_surprisal": np.nan,
                "beta_surprisal_se": np.nan, "fit_backend": "skipped",
                "fit_backend_baseline": "skipped", "notes": f"n={len(sub)} < 200"}
    return mc.fit_canonical_pair(sub)


def _bootstrap_delta_ll(
    df: pd.DataFrame, dv: str, *, n_boot: int, rng: np.random.Generator,
) -> tuple[float, float]:
    """Cluster bootstrap over subjects → 2.5/97.5 percentile of Δlogℒ."""
    subs = df["subid"].unique()
    deltas: list[float] = []
    for _ in range(n_boot):
        pick = rng.choice(subs, size=len(subs), replace=True)
        boot = pd.concat([df[df["subid"] == s] for s in pick], ignore_index=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = _fit_pair_with_dv(boot, dv)
            if np.isfinite(fit.get("delta_loglik", np.nan)):
                deltas.append(float(fit["delta_loglik"]))
        except Exception:
            continue
    if not deltas:
        return float("nan"), float("nan")
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def compute(
    surprisal_df: pd.DataFrame,
    *,
    langs: Iterable[str] = CANONICAL_LANGS,
    measures: Sequence[str] = tuple(ET_MEASURES.keys()),
    n_boot: int = 0,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fit (full, baseline) per (model, lang, measure) and collect a long-form
    DataFrame. Bootstrap CIs are computed when ``n_boot > 0``.
    """
    rng = np.random.default_rng(seed)
    # D1 needs the *full* set of fixation measures, so we load via the
    # multi-measure loader (cached separately from the canonical loader).
    from analyze.meco_detailed._common import get_rt_full_with_features
    rt = get_rt_full_with_features()

    rows = []
    for model in sorted(surprisal_df["model"].dropna().unique()):
        merged_all = mc.join_model_surprisal(rt, surprisal_df, model)
        if merged_all.empty:
            continue
        for lang in langs:
            sub = merged_all[merged_all["lang"] == lang]
            sub = mc.attach_spillover(sub)
            sub = mc.drop_sentence_boundaries(sub)
            for py_meas in measures:
                if py_meas not in sub.columns:
                    continue
                this = sub.dropna(subset=[py_meas, "Surprisal", "wlen", "freq", "subid"])
                if len(this) < 200:
                    rows.append({
                        "model": model, "lang": lang, "measure": py_meas,
                        "delta_loglik": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                        "beta_surprisal": np.nan, "beta_surprisal_se": np.nan,
                        "n_obs": int(len(this)),
                        "n_subjects": int(this["subid"].nunique()),
                        "notes": f"n={len(this)} < 200",
                    })
                    continue
                if verbose:
                    print(f"[d1] {model} | {lang} | {py_meas} "
                          f"(n={len(this)}, subs={this['subid'].nunique()})",
                          file=sys.stderr, flush=True)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit = _fit_pair_with_dv(this, py_meas)
                    ci_lo = ci_hi = float("nan")
                    if n_boot > 0 and np.isfinite(fit.get("delta_loglik", np.nan)):
                        ci_lo, ci_hi = _bootstrap_delta_ll(
                            this, py_meas, n_boot=n_boot, rng=rng,
                        )
                    rows.append({
                        "model": model, "lang": lang, "measure": py_meas,
                        "delta_loglik": fit["delta_loglik"],
                        "ci_lo": ci_lo, "ci_hi": ci_hi,
                        "beta_surprisal": fit["beta_surprisal"],
                        "beta_surprisal_se": fit["beta_surprisal_se"],
                        "n_obs": int(len(this)),
                        "n_subjects": int(this["subid"].nunique()),
                        "notes": "",
                    })
                except Exception as e:
                    rows.append({
                        "model": model, "lang": lang, "measure": py_meas,
                        "delta_loglik": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                        "beta_surprisal": np.nan, "beta_surprisal_se": np.nan,
                        "n_obs": int(len(this)),
                        "n_subjects": int(this["subid"].nunique()),
                        "notes": f"{e.__class__.__name__}: {e}",
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(df: pd.DataFrame, out_path: Path) -> None:
    """One heatmap per fixation measure: rows = L1 (typological order),
    columns = curriculum. Cells with no data are hatched grey + 'n/a'.
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

    # Attach curriculum/L1 metadata from model names, then merge L1 from the
    # MECO ``lang`` column where the parser couldn't recover it.
    df = attach_model_metadata(df)
    # If the canonical L1 was already in the data (MECO `lang` col), prefer it
    # since it's the *reader* group, not the *model*'s training L1.
    if "lang" in df.columns:
        df["l1_code"] = df["lang"].fillna(df["l1_code"])

    measures = [m for m in ET_MEASURES if m in df["measure"].unique()]
    used_l1s = [l for l in L1_ORDER if l in set(df["l1_code"])]
    used_curr = [c for c in CURRICULUM_ORDER
                 if c != "mono" and c in set(df["curriculum"])]
    if not used_curr:
        used_curr = ["B1", "B2", "B3", "B4", "B5"]
    if not used_l1s:
        used_l1s = ["du", "ge", "ch_s"]

    n_meas = len(measures)
    fig, axes = plt.subplots(1, n_meas, figsize=(WIDTH_DOUBLE,
                                                  max(2.6, 0.28 * len(used_l1s) + 1.5)),
                              sharey=True, squeeze=False,
                              constrained_layout=True)

    # Global vmin/vmax for shared colour scale.
    vals = df["delta_loglik"].to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(vals))) if np.isfinite(np.nanmax(vals)) else 1.0
    vmax = max(vmax, 1e-6)

    for ax_idx, meas in enumerate(measures):
        ax = axes[0][ax_idx]
        sub = df[df["measure"] == meas]
        grid = (sub.groupby(["l1_code", "curriculum"])["delta_loglik"]
                   .mean().unstack("curriculum"))
        grid = grid.reindex(index=used_l1s, columns=used_curr)
        # Heatmap
        im = ax.imshow(grid.to_numpy(dtype=float),
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
                       interpolation="nearest")
        # Mark missing cells
        for r, l1 in enumerate(used_l1s):
            for c, curr in enumerate(used_curr):
                v = grid.iloc[r, c] if (l1 in grid.index and curr in grid.columns) else np.nan
                if not np.isfinite(v):
                    mark_missing_cell(ax, c, r, w=0.96, h=0.96)
                else:
                    ax.text(c, r, f"{v:.0f}",
                            ha="center", va="center", fontsize=5.5,
                            color="white" if abs(v) > vmax * 0.55 else "#333")

        ax.set_xticks(range(len(used_curr)))
        ax.set_xticklabels([CURRICULUM_LABEL.get(c, c) for c in used_curr],
                            rotation=35, ha="right", fontsize=6)
        if ax_idx == 0:
            ax.set_yticks(range(len(used_l1s)))
            ax.set_yticklabels([L1_PRETTY.get(l, l) for l in used_l1s],
                                fontsize=7,
                                color="#222")
            # Colour the y-tick labels with the L1 palette for fast scanning.
            for ytick, l1 in zip(ax.get_yticklabels(), used_l1s):
                ytick.set_color(L1_PALETTE.get(l1, "#222"))
        else:
            ax.set_yticks([])
        ax.set_title(ET_LABELS.get(meas, meas), fontsize=7)

    # Single colourbar with explicit direction indicator
    cbar = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.55, pad=0.015,
                         aspect=18)
    cbar.set_label(axis_label("ΔlogLik", better="higher"), fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Left and bottom edge axis labels — explicit, paper-ready
    axes[0][0].text(-0.55, 0.5, "L1 reader cohort\n(typological distance to English ↓)",
                     transform=axes[0][0].transAxes,
                     fontsize=6.5, ha="right", va="center", rotation=90,
                     color="#444")
    fig.supxlabel("Curriculum (B1 balanced → B5 late-L2 onset)",
                   fontsize=7, color="#444")

    fig.suptitle("D1 — Beetle ΔlogLik across 5 fixation measures, "
                 "by L1 (rows) × curriculum (cols)", fontsize=9)
    direction_badge(fig, "ΔlogLik", better="higher",
                     zero_means="surprisal adds no info above lexical controls")
    coverage_badge(fig, df, dims=("l1_code", "curriculum"), text_y=-0.02)
    savefig_pub(fig, out_path.with_suffix(""))
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--langs", nargs="+", default=list(CANONICAL_LANGS))
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="bootstrap iterations for 95%% CIs (0 = skip)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from plots.checkpoint_loaders import load_meco_checkpoint_files  # type: ignore

    csv_path = result_csv(SLUG)
    if csv_path.exists() and not args.overwrite:
        df = pd.read_csv(csv_path)
        print(f"[d1] using cached {csv_path}", file=sys.stderr)
    else:
        surp = load_meco_checkpoint_files()
        if surp.empty:
            print("[d1] no MECO surprisal CSVs found", file=sys.stderr)
            return 2
        # Final-step surprisal per model (matches canonical pipeline)
        last_step = surp.groupby("model")["step"].max().rename("_last")
        surp = surp.join(last_step, on="model")
        surp = surp[surp["step"] == surp["_last"]].drop(columns=["_last"])
        df = compute(surp, langs=args.langs, n_boot=args.bootstrap, seed=args.seed)
        df.to_csv(csv_path, index=False)
        print(f"[d1] wrote {csv_path} ({len(df)} rows)", file=sys.stderr)

    plot(df, figure_pdf(SLUG))
    print(f"[d1] wrote {figure_pdf(SLUG)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
