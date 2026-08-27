"""
d5_item_granularity.py — Where does beetle help? freq × cognate × position.

Claim
-----
Beetle's advantage on MECO L2 concentrates exactly where L2 reading is
hardest:

  (a) low-frequency words (large RT residuals under the baseline LMs);
  (b) non-cognates (no L1-form shortcut);
  (c) mid-sentence positions (full processing demand, not edge effects).

Stat
----
We compute the per-word *predicted-RT residual* under both the canonical
beetle full model and a baseline LM full model, then take

    Δresid_w = |resid_baseline_w| − |resid_beetle_w|

(positive ⇒ beetle predicts that word better). We then bin Δresid by

  • log-frequency decile of the word in English
  • cognate vs non-cognate (Dutch-English cognate list)
  • normalised sentence position (wordnum / sentence-length)

Output
------
``results/meco_detailed/d5_item_granularity.csv``
    columns: model_beetle, model_baseline, lang,
             bin_kind ∈ {freq, cognate, position}, bin_key,
             mean_delta_resid, ci_lo, ci_hi, n

``plots/meco_detailed/d5_item_granularity.pdf``
    1×3 panel — (a) Δresid vs freq decile; (b) cognate vs non-cognate;
    (c) Δresid vs normalised sentence position. Bootstrap-clustered
    over (itemid, wordnum) for CIs.

CLI
---
    python -m analyze.meco_detailed.d5_item_granularity \\
        --beetle BeetleLM/beetlelm_eng-nld_balanced \\
        --baseline bigscience/bloom-7b1 --lang du
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
    CANONICAL_LANGS, L1_PRETTY, figure_pdf, load_cognate_table, result_csv,
)


SLUG = "d5_item_granularity"


def _fit_and_predict(df: pd.DataFrame) -> np.ndarray:
    """Fit the canonical full model on ``df`` and return per-row residuals."""
    import statsmodels.formula.api as smf

    scaled = mc._prepare_scaled(df, include_surprisal=True)  # noqa: SLF001
    fixed = " + ".join(f"sc_{v}" for v in mc._ALL_VARS_FULL)  # noqa: SLF001
    formula = f"firstrun_dur ~ {fixed}"
    vc = {f"rs_{v}": f"0 + sc_{v}" for v in mc._SURPRISAL_VARS}  # noqa: SLF001
    md = smf.mixedlm(formula, data=scaled, groups=scaled["subid"],
                     re_formula="1", vc_formula=vc)
    res = md.fit(reml=False, method="lbfgs", disp=False)
    return np.asarray(res.resid)


def _bootstrap_mean(x: pd.Series, n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    if x.empty:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = x.to_numpy()
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def compute(
    surprisal_df: pd.DataFrame,
    *,
    beetle: str, baseline: str,
    langs: Iterable[str] = CANONICAL_LANGS,
    n_boot: int = 500,
    verbose: bool = True,
) -> pd.DataFrame:
    from analyze.meco_detailed._common import get_rt_with_features
    rt = get_rt_with_features()

    rows = []
    for lang in langs:
        rt_l = rt[rt["lang"] == lang]
        # Join each model's surprisal on the *same* RT subset; then attach
        # spillover + drop boundaries from each independently.
        df_b = mc.join_model_surprisal(rt_l, surprisal_df, beetle).dropna(
            subset=["firstrun_dur", "Surprisal", "wlen", "freq", "subid"])
        df_b = mc.drop_sentence_boundaries(mc.attach_spillover(df_b))
        df_x = mc.join_model_surprisal(rt_l, surprisal_df, baseline).dropna(
            subset=["firstrun_dur", "Surprisal", "wlen", "freq", "subid"])
        df_x = mc.drop_sentence_boundaries(mc.attach_spillover(df_x))
        if len(df_b) < 200 or len(df_x) < 200:
            continue
        if verbose:
            print(f"[d5] {lang}: fitting beetle (n={len(df_b)}) + "
                  f"baseline (n={len(df_x)})", file=sys.stderr, flush=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resid_b = _fit_and_predict(df_b)
                resid_x = _fit_and_predict(df_x)
        except Exception as e:
            print(f"[d5] fail {lang}: {e}", file=sys.stderr)
            continue
        df_b = df_b.assign(resid_beetle=resid_b)
        df_x = df_x.assign(resid_base=resid_x)
        # Inner-join on the shared (itemid, wordnum, subid) triple so each row
        # has matched residuals from both models.
        keys = ["itemid", "wordnum", "subid"]
        merged = df_b[keys + ["resid_beetle", "word", "freq"]].merge(
            df_x[keys + ["resid_base"]], on=keys, how="inner")
        merged["delta_resid"] = merged["resid_base"].abs() - merged["resid_beetle"].abs()
        merged["word_lc"] = merged["word"].astype(str).str.lower()

        # ---- (a) frequency decile
        merged["freq_dec"] = pd.qcut(merged["freq"], q=10, duplicates="drop",
                                     labels=False).astype("Int64")
        for k, g in merged.groupby("freq_dec", dropna=True):
            lo, hi = _bootstrap_mean(g["delta_resid"], n_boot=n_boot)
            rows.append({
                "model_beetle": beetle, "model_baseline": baseline, "lang": lang,
                "bin_kind": "freq", "bin_key": int(k),
                "mean_delta_resid": float(g["delta_resid"].mean()),
                "ci_lo": lo, "ci_hi": hi, "n": int(len(g)),
            })

        # ---- (b) cognate vs non-cognate (Dutch-English table — extendable)
        cog = load_cognate_table("nld")
        cog_set = set(cog.loc[cog["is_cognate"], "word_en"])
        merged["is_cognate"] = merged["word_lc"].isin(cog_set)
        for k, g in merged.groupby("is_cognate"):
            lo, hi = _bootstrap_mean(g["delta_resid"], n_boot=n_boot)
            rows.append({
                "model_beetle": beetle, "model_baseline": baseline, "lang": lang,
                "bin_kind": "cognate", "bin_key": "cognate" if k else "non_cognate",
                "mean_delta_resid": float(g["delta_resid"].mean()),
                "ci_lo": lo, "ci_hi": hi, "n": int(len(g)),
            })

        # ---- (c) normalised sentence position
        if "sentnum" in merged.columns and merged["sentnum"].notna().any():
            pos = merged.copy()
            sent_max = pos.groupby(["itemid", "sentnum"])["wordnum"].transform("max")
            sent_min = pos.groupby(["itemid", "sentnum"])["wordnum"].transform("min")
            denom = (sent_max - sent_min).replace(0, np.nan)
            pos["pos_norm"] = (pos["wordnum"] - sent_min) / denom
            pos["pos_bin"] = pd.cut(pos["pos_norm"], bins=5, labels=False,
                                     include_lowest=True).astype("Int64")
            for k, g in pos.groupby("pos_bin", dropna=True):
                lo, hi = _bootstrap_mean(g["delta_resid"], n_boot=n_boot)
                rows.append({
                    "model_beetle": beetle, "model_baseline": baseline, "lang": lang,
                    "bin_kind": "position", "bin_key": int(k),
                    "mean_delta_resid": float(g["delta_resid"].mean()),
                    "ci_lo": lo, "ci_hi": hi, "n": int(len(g)),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(df: pd.DataFrame, out_path: Path) -> None:
    """3 rows (freq / cognate / position) × N L1 columns. Lines per
    curriculum within each panel; missing curricula are silently dropped
    but the panel itself is drawn (so the grid stays uniform).
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

    # D5 has model_beetle column; use it for metadata.
    work = df.copy()
    if "model" not in work.columns and "model_beetle" in work.columns:
        work["model"] = work["model_beetle"]
    work = attach_model_metadata(work)
    if "lang" in work.columns:
        work["l1_code"] = work["lang"].fillna(work["l1_code"])

    used_l1s = [l for l in L1_ORDER if l in set(work["l1_code"])][:4]  # cap @ 4 cols
    if not used_l1s:
        used_l1s = ["du", "ge", "ch_s"]
    used_curr = [c for c in CURRICULUM_ORDER
                 if c != "mono" and c in set(work["curriculum"])]

    fig, axes = plt.subplots(3, len(used_l1s),
                              figsize=(WIDTH_DOUBLE, 4.8),
                              sharex=False, sharey="row", squeeze=False,
                              constrained_layout=True)
    row_titles = {"freq":     "(a) low-freq → high-freq words",
                  "cognate":  "(b) non-cognates vs cognates",
                  "position": "(c) early → late sentence position"}
    row_xlabels = {"freq":     "Log-frequency decile  (0 = rarest, 9 = commonest)",
                   "cognate":  "Cognate status",
                   "position": "Normalised sentence position  (0 = first word, 4 = last)"}
    bin_kinds = ["freq", "cognate", "position"]

    for r, kind in enumerate(bin_kinds):
        for c, l1 in enumerate(used_l1s):
            ax = axes[r][c]
            ax.axhline(0, color="#bbb", lw=0.4)
            cell = work[(work["l1_code"] == l1) & (work["bin_kind"] == kind)]
            if cell.empty:
                mark_missing_cell(ax, 0, 0, w=1.0, h=1.0, label="n/a")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            if kind == "cognate":
                pivot = (cell.pivot_table(
                            index="curriculum", columns="bin_key",
                            values="mean_delta_resid", aggfunc="mean")
                          .reindex(columns=["non_cognate", "cognate"]))
                x = np.arange(2)
                w = 0.16
                offset = 0
                for curr in used_curr:
                    if curr not in pivot.index:
                        continue
                    vals = pivot.loc[curr].to_numpy()
                    ax.bar(x + (offset - len(used_curr) / 2 + 0.5) * w, vals, w,
                            color=CURRICULUM_PALETTE.get(curr, "#888"),
                            label=CURRICULUM_LABEL.get(curr, curr)
                                  if (r == 0 and c == 0) else None,
                            edgecolor="white", linewidth=0.4)
                    offset += 1
                ax.set_xticks(x)
                ax.set_xticklabels(["non-cog", "cog"], fontsize=6)
            else:
                # numeric bin → line per curriculum
                for curr in used_curr:
                    s = cell[cell["curriculum"] == curr].sort_values("bin_key")
                    if s.empty:
                        continue
                    xs = pd.to_numeric(s["bin_key"], errors="coerce")
                    # Clamp yerr to non-negative — bootstrap CIs may straddle
                    # the point estimate in either direction.
                    yerr_lo = (s["mean_delta_resid"] - s["ci_lo"]).abs()
                    yerr_hi = (s["ci_hi"] - s["mean_delta_resid"]).abs()
                    ax.errorbar(xs, s["mean_delta_resid"],
                                 yerr=[yerr_lo, yerr_hi],
                                 fmt="o-", lw=1.0, ms=2.8,
                                 color=CURRICULUM_PALETTE.get(curr, "#888"),
                                 label=CURRICULUM_LABEL.get(curr, curr)
                                       if (r == 0 and c == 0) else None)
                ax.tick_params(axis="x", labelsize=6)

            if r == 0:
                ax.set_title(L1_PRETTY.get(l1, l1), fontsize=7,
                              color=L1_PALETTE.get(l1, "#000"))
            if c == 0:
                ax.set_ylabel(row_titles[kind] + "\n" +
                               axis_label("beetle's RT-residual reduction",
                                           better="higher"),
                               fontsize=6.5)
            if r == 2:
                ax.set_xlabel(row_xlabels[kind], fontsize=6.5)

    # Single legend
    handles = curriculum_legend_handles(used_curr)
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                    frameon=False, fontsize=6.5,
                    title="curriculum", title_fontsize=6.5)
    fig.suptitle("D5 — Beetle helps where L2 reading is hardest, by L1 × curriculum",
                  fontsize=9)
    direction_badge(fig, "RT-residual reduction", better="higher",
                     zero_means="beetle and baseline predict that word equally")
    coverage_badge(fig, work, dims=("l1_code", "curriculum"), text_y=-0.05)
    savefig_pub(fig, out_path.with_suffix(""))
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--langs", nargs="+", default=list(CANONICAL_LANGS))
    ap.add_argument("--beetle", default="BeetleLM/beetlelm_eng-nld_balanced")
    ap.add_argument("--baseline", default="bigscience/bloom-7b1")
    ap.add_argument("--bootstrap", type=int, default=500)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from plots.checkpoint_loaders import load_meco_checkpoint_files  # type: ignore

    csv_path = result_csv(SLUG)
    if csv_path.exists() and not args.overwrite:
        df = pd.read_csv(csv_path)
        print(f"[d5] using cached {csv_path}", file=sys.stderr)
    else:
        surp = load_meco_checkpoint_files()
        if surp.empty:
            print("[d5] no MECO surprisal CSVs found", file=sys.stderr)
            return 2
        last_step = surp.groupby("model")["step"].max().rename("_last")
        surp = surp.join(last_step, on="model")
        surp = surp[surp["step"] == surp["_last"]].drop(columns=["_last"])
        df = compute(surp, beetle=args.beetle, baseline=args.baseline,
                     langs=args.langs, n_boot=args.bootstrap)
        df.to_csv(csv_path, index=False)
        print(f"[d5] wrote {csv_path} ({len(df)} rows)", file=sys.stderr)

    plot(df, figure_pdf(SLUG))
    print(f"[d5] wrote {figure_pdf(SLUG)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
