"""
d6_proficiency_mod.py — LEAP-Q proficiency moderation of beetle ΔlogLik.

Claim
-----
Beetle predicts low-proficiency L2 readers *best*. Lower-proficiency
readers have noisier RTs and are more surprisal-sensitive; the
matched-bilingual model's tighter cognitive fit should manifest
disproportionately for them.

Stat
----
Per subject, join LEAP-Q self-rated English-reading proficiency
(``english_reading_proficiency``, 1–10) onto the canonical MECO frame.
Then split subjects into quartiles by proficiency *within each L1 cohort*
(so the comparison isn't confounded by L1-level differences in mean
proficiency), and refit the canonical ΔlogLik pipeline per quartile.

We also report a continuous-moderator alternative:

    firstrun.dur ~ sc(Surprisal) * sc(proficiency)
                    + (canonical controls + spillover)
                    + (1 + … || subid)

and read out the ``sc_Surprisal:sc_proficiency`` interaction β. A
negative interaction means low-proficiency readers have a larger
surprisal slope (= more sensitivity), which is the expected sign.

Output
------
``results/meco_detailed/d6_proficiency_mod.csv``
    columns: model, lang, prof_quartile (1–4 or "continuous"),
             delta_loglik, beta_surprisal, beta_interaction,
             n_obs, n_subjects, notes

``plots/meco_detailed/d6_proficiency_mod.pdf``
    4-panel forest by proficiency quartile + a continuous-moderator
    overlay (one subplot showing the interaction β per model).

CLI
---
    python -m analyze.meco_detailed.d6_proficiency_mod [--lang du]
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
    CANONICAL_LANGS, L1_PRETTY, figure_pdf, load_leapq_all, result_csv,
)


SLUG = "d6_proficiency_mod"

# Map MECO lang code → LEAP-Q L1 xlsx code (they happen to share most codes
# but ch_s isn't in the LEAP-Q release — note this in the output).
LANG_TO_LEAPQ = {
    "du": "du", "ge": "ge", "ch_s": None, "fi": "fi", "ge_de": "ge",
    "en": "en", "ru": "ru", "sp": "sp", "tr": "tr", "no": "no",
    "it": "it", "gr": "gr", "he": "he", "ee": "ee",
}


def _join_proficiency(rt: pd.DataFrame, leapq: pd.DataFrame) -> pd.DataFrame:
    """Inner-join per-subject English-reading proficiency to RT rows."""
    if leapq.empty:
        return rt.assign(proficiency=np.nan)
    cols = [c for c in ("subid", "english_reading_proficiency") if c in leapq.columns]
    if "english_reading_proficiency" not in cols:
        return rt.assign(proficiency=np.nan)
    p = leapq[cols].drop_duplicates("subid").rename(
        columns={"english_reading_proficiency": "proficiency"})
    p["subid"] = p["subid"].astype(str)
    rt2 = rt.copy()
    rt2["subid"] = rt2["subid"].astype(str)
    return rt2.merge(p, on="subid", how="left")


def _fit_interaction(df: pd.DataFrame) -> tuple[float, float]:
    """Fit Surprisal × Proficiency interaction; return (β_inter, β_surprisal)."""
    import statsmodels.formula.api as smf

    s = df.dropna(subset=["proficiency"]).copy()
    if len(s) < 200 or s["subid"].nunique() < 8:
        return float("nan"), float("nan")

    scaled = mc._prepare_scaled(s, include_surprisal=True)  # noqa: SLF001
    # Build prof scale outside the canonical helper.
    sd = s["proficiency"].std(ddof=1)
    scaled["sc_proficiency"] = (s["proficiency"] - s["proficiency"].mean()) / (sd or 1)

    fixed = " + ".join(f"sc_{v}" for v in mc._ALL_VARS_FULL)  # noqa: SLF001
    formula = (f"firstrun_dur ~ {fixed} "
               f"+ sc_proficiency + sc_Surprisal:sc_proficiency")
    try:
        md = smf.mixedlm(formula, data=scaled, groups=scaled["subid"],
                         re_formula="1")
        res = md.fit(reml=False, method="lbfgs", disp=False)
        b_int = float(res.fe_params.get("sc_Surprisal:sc_proficiency", np.nan))
        b_surp = float(res.fe_params.get("sc_Surprisal", np.nan))
        return b_int, b_surp
    except Exception:
        return float("nan"), float("nan")


def compute(
    surprisal_df: pd.DataFrame,
    *,
    langs: Iterable[str] = CANONICAL_LANGS,
    verbose: bool = True,
) -> pd.DataFrame:
    from analyze.meco_detailed._common import get_rt_with_features
    rt = get_rt_with_features()
    leapq = load_leapq_all()

    rows = []
    for model in sorted(surprisal_df["model"].dropna().unique()):
        merged = mc.join_model_surprisal(rt, surprisal_df, model)
        if merged.empty:
            continue
        merged = _join_proficiency(merged, leapq)
        for lang in langs:
            sub = merged[merged["lang"] == lang].dropna(
                subset=["firstrun_dur", "Surprisal", "wlen", "freq", "subid"])
            sub = mc.attach_spillover(sub)
            sub = mc.drop_sentence_boundaries(sub)
            if len(sub) < 200:
                continue
            # --- per-quartile ΔlogLik
            quart = pd.qcut(sub.groupby("subid")["proficiency"].first(),
                            q=4, labels=[1, 2, 3, 4], duplicates="drop")
            sub_with_q = sub.merge(quart.rename("prof_q").reset_index(),
                                    on="subid", how="left")
            for q in [1, 2, 3, 4]:
                qdf = sub_with_q[sub_with_q["prof_q"] == q]
                if len(qdf) < 200 or qdf["subid"].nunique() < 4:
                    rows.append({
                        "model": model, "lang": lang, "prof_quartile": q,
                        "delta_loglik": np.nan, "beta_surprisal": np.nan,
                        "beta_interaction": np.nan,
                        "n_obs": int(len(qdf)),
                        "n_subjects": int(qdf["subid"].nunique()),
                        "notes": "too few subjects in quartile",
                    })
                    continue
                if verbose:
                    print(f"[d6] {model} | {lang} | Q{q} "
                          f"(n={len(qdf)}, subs={qdf['subid'].nunique()})",
                          file=sys.stderr, flush=True)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit = mc.fit_canonical_pair(qdf)
                except Exception as e:
                    rows.append({
                        "model": model, "lang": lang, "prof_quartile": q,
                        "delta_loglik": np.nan, "beta_surprisal": np.nan,
                        "beta_interaction": np.nan,
                        "n_obs": int(len(qdf)),
                        "n_subjects": int(qdf["subid"].nunique()),
                        "notes": f"fit error: {e}",
                    })
                    continue
                rows.append({
                    "model": model, "lang": lang, "prof_quartile": q,
                    "delta_loglik": fit["delta_loglik"],
                    "beta_surprisal": fit["beta_surprisal"],
                    "beta_interaction": np.nan,
                    "n_obs": int(len(qdf)),
                    "n_subjects": int(qdf["subid"].nunique()),
                    "notes": "",
                })
            # --- continuous interaction
            b_int, b_surp = _fit_interaction(sub)
            rows.append({
                "model": model, "lang": lang, "prof_quartile": "continuous",
                "delta_loglik": np.nan,
                "beta_surprisal": b_surp, "beta_interaction": b_int,
                "n_obs": int(len(sub)),
                "n_subjects": int(sub["subid"].nunique()),
                "notes": "Surprisal × proficiency interaction",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(df: pd.DataFrame, out_path: Path) -> None:
    """Trend across LEAP-Q proficiency quartiles.

    Layout: rows = L1, single panel showing ΔlogLik vs proficiency
    quartile (Q1 low → Q4 high), one line per curriculum. The continuous
    interaction β is reported as a side bar at the right.

    Cells with no proficiency data (e.g. Chinese ch_s has no LEAP-Q in
    the OSF release) are marked 'no LEAP-Q' rather than left blank.
    """
    import matplotlib.pyplot as plt
    from analyze.plot_style import apply_acl_style, savefig_pub, WIDTH_DOUBLE
    from analyze.meco_detailed._plot import (
        CURRICULUM_LABEL, CURRICULUM_ORDER, CURRICULUM_PALETTE,
        L1_ORDER, L1_PALETTE, L1_PRETTY,
        attach_model_metadata, axis_label, coverage_badge, direction_badge,
        curriculum_legend_handles, mark_missing_cell,
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
        used_curr = ["B1", "B3", "B5"]

    fig, axes = plt.subplots(len(used_l1s), 2,
                              figsize=(WIDTH_DOUBLE,
                                       max(3.4, 0.7 * len(used_l1s) + 1.4)),
                              gridspec_kw={"width_ratios": [3.0, 1.0]},
                              sharex="col", squeeze=False,
                              constrained_layout=True)

    # Per L1, plot ΔlogLik vs quartile per curriculum
    for r, l1 in enumerate(used_l1s):
        ax = axes[r][0]
        ax_int = axes[r][1]
        ax.axhline(0, color="#bbb", lw=0.4)
        ax_int.axvline(0, color="#bbb", lw=0.4)
        sub_l1 = df[df["l1_code"] == l1]
        quart_sub = sub_l1[sub_l1["prof_quartile"].isin([1, 2, 3, 4,
                                                          "1", "2", "3", "4"])]
        if quart_sub.empty:
            mark_missing_cell(ax, 2.5, 0, w=4.0, h=1.0,
                               label="no LEAP-Q data for this L1")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            quart_sub = quart_sub.copy()
            quart_sub["q"] = pd.to_numeric(quart_sub["prof_quartile"],
                                            errors="coerce")
            for curr in used_curr:
                cs = quart_sub[quart_sub["curriculum"] == curr].sort_values("q")
                if cs.empty:
                    continue
                # Aggregate over (model, q) to one value
                agg = cs.groupby("q")["delta_loglik"].mean().reset_index()
                ax.plot(agg["q"], agg["delta_loglik"], marker="o", ms=3.0, lw=1.1,
                         color=CURRICULUM_PALETTE.get(curr, "#888"),
                         label=CURRICULUM_LABEL.get(curr, curr)
                               if r == 0 else None)
            ax.set_xticks([1, 2, 3, 4])
            ax.tick_params(axis="both", labelsize=6)

        # Continuous interaction β as horizontal scatter
        cont = sub_l1[sub_l1["prof_quartile"].astype(str) == "continuous"]
        if not cont.empty:
            yvals = np.arange(len(used_curr))
            for i, curr in enumerate(used_curr):
                v = cont[cont["curriculum"] == curr]["beta_interaction"].mean()
                if not np.isfinite(v):
                    continue
                ax_int.scatter(v, i,
                                color=CURRICULUM_PALETTE.get(curr, "#888"),
                                s=14, edgecolor="white", linewidth=0.5)
            ax_int.set_yticks(yvals)
            ax_int.set_yticklabels([CURRICULUM_LABEL.get(c, c) for c in used_curr],
                                     fontsize=5)
            ax_int.tick_params(axis="x", labelsize=6)
        else:
            ax_int.set_xticks([])
            ax_int.set_yticks([])
            mark_missing_cell(ax_int, 0, 0, w=1.0, h=1.0)

        # Row label: L1
        axes[r][0].set_ylabel(L1_PRETTY.get(l1, l1), fontsize=7, rotation=0,
                               ha="right", va="center",
                               color=L1_PALETTE.get(l1, "#000"))

    axes[-1][0].set_xlabel("LEAP-Q proficiency quartile  (Q1 lowest → Q4 highest)",
                             fontsize=7)
    axes[-1][1].set_xlabel(axis_label("β(Surprisal × Proficiency)",
                                         better="more negative"),
                             fontsize=7)
    axes[0][0].set_title("ΔlogLik per quartile  (one line per curriculum)\n"
                          "predicted: ΔlogLik decreases with proficiency "
                          "(low-prof readers benefit most)",
                          fontsize=7)
    axes[0][1].set_title("Surp × Prof interaction\n"
                          "(↓ negative = predicted direction)", fontsize=7)
    # Shared y-axis label for the left column
    axes[0][0].set_ylabel(axis_label("ΔlogLik", better="higher"), fontsize=7)

    handles = curriculum_legend_handles(used_curr)
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                    frameon=False, fontsize=6.5,
                    title="curriculum", title_fontsize=6.5)
    fig.suptitle("D6 — LEAP-Q proficiency moderation of beetle ΔlogLik, "
                  "by L1 × curriculum",
                  fontsize=9)
    direction_badge(fig, "Surp × Prof β", better="more negative",
                     zero_means="proficiency does not moderate surprisal effect")
    coverage_badge(fig, df, dims=("l1_code", "curriculum"), text_y=-0.04)
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
        print(f"[d6] using cached {csv_path}", file=sys.stderr)
    else:
        surp = load_meco_checkpoint_files()
        if surp.empty:
            print("[d6] no MECO surprisal CSVs found", file=sys.stderr)
            return 2
        last_step = surp.groupby("model")["step"].max().rename("_last")
        surp = surp.join(last_step, on="model")
        surp = surp[surp["step"] == surp["_last"]].drop(columns=["_last"])
        df = compute(surp, langs=args.langs)
        df.to_csv(csv_path, index=False)
        print(f"[d6] wrote {csv_path} ({len(df)} rows)", file=sys.stderr)

    plot(df, figure_pdf(SLUG))
    print(f"[d6] wrote {figure_pdf(SLUG)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
