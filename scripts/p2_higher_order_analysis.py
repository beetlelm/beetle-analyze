#!/usr/bin/env python3
"""scripts/p2_higher_order_analysis.py

Build the higher-order pico ↔ downstream analysis for the nld-eng Beetle
trajectory grid:

    pico metrics  (gini, hoyer, per, condition_number, norm)
        × scale  (FineWeb-100M, FineWeb-2B, FineWeb-24B, HumanScale)
        × curriculum  (B1, B2, B3, B4, B5)
    vs
    downstream metrics
        BLiMP-eng / BLiMP-nld accuracy
        MECO Δlog-likelihood (Dutch-L1 readers)
        Priming Δlogit (Bernolet, between-language)
        BLiSS RP_at_0 / NGS  (overall + per-CEFR slope = pedagogical alignment)

Outputs:
    results/evals/cross_intrinsic/nld_eng_unified.csv          # tidy long
    results/evals/cross_intrinsic/nld_eng_wide.csv             # one row / (run,step)
    results/evals/cross_intrinsic/predictiveness.csv           # pico → downstream R²
    results/evals/cross_intrinsic/lead_lag.csv                 # lag-correlation
    results/figures/main/fig_intrinsic_eval_predictiveness.{pdf,png}
    results/figures/main/fig_intrinsic_eval_factor_biplot.{pdf,png}
    results/figures/appendix/fig_intrinsic_eval_lead_lag.{pdf,png}
    results/figures/appendix/fig_intrinsic_eval_curriculum_grid.{pdf,png}
    results/figures/appendix/fig_intrinsic_eval_scale_consistency.{pdf,png}

Designed to be re-runnable: skips analyses for which there is no aligned data.
"""
from __future__ import annotations

import os
import json
import logging
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("p2_higher_order")
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s %(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO = Path(__file__).resolve().parents[1]
PICO_ROOT = Path(os.environ.get("PICO_ROOT", str(Path.home() / "pico-analyze")))
PICO_ANALYSIS = PICO_ROOT / "analysis_results"

DOWNSTREAM = {
    "blimp":   REPO / "results/evals_checkpoints/blimp/blimp_results.csv",
    "bliss":   REPO / "results/evals_checkpoints/bliss/bliss_results.csv",
    "meco_dll": REPO / "results/meco/canonical_deltalogl_nld_eng_traj.csv",
    "priming_dir": REPO / "results/evals/priming/tier2_headline_trajectory",
}

OUT_DIR = REPO / "results/evals/cross_intrinsic"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_MAIN = REPO / "results/figures/main"
FIG_APP = REPO / "results/figures/appendix"
FIG_MAIN.mkdir(parents=True, exist_ok=True)
FIG_APP.mkdir(parents=True, exist_ok=True)

# ── Run-name parsing ──────────────────────────────────────────────────────

CURRICULUM_RE = re.compile(
    r"beetle-bilingual-(?:"
    r"(?P<balanced>balanced-(?P<bal_b>b\d))|"
    r"(?P<l250>l2-50-(?P<l250_kind>simultaneous|sequential-\d+-\d+|classroom-\d+))-(?P<l250_b>b\d)|"
    r"(?P<l280>l2-80-late)-(?P<l280_b>b\d)"
    r")"
)
SCALE_TOKENS = {
    "fineweb-100m": "100M",
    "humanscale": "HumanScale",
    "fineweb-2b": "2B",
    # fall through "fineweb" without 100m / 2b → 24B
}


def parse_run_name(name: str) -> dict:
    """Return {curriculum, scale, lang_pair}.

    name examples:
        beetle-bilingual-balanced-b1-fineweb-100m-nld-eng
        beetle-bilingual-l2-50-classroom-20-b4-fineweb-2b-nld-eng
        beetle-bilingual-l2-80-late-b5-humanscale-nld-eng
        beetle-bilingual-balanced-b1-fineweb-nld-eng        (24B)
    """
    n = name.lower()
    # curriculum bucket
    if "l2-80-late" in n:
        curr = "B5_late"
    elif "classroom" in n:
        curr = "B4_classroom"
    elif "sequential" in n:
        curr = "B3_sequential"
    elif "simultaneous" in n:
        curr = "B2_simultaneous"
    elif "balanced-b1" in n or n.endswith("-b1") or "balanced-b1-" in n:
        curr = "B1_balanced"
    elif "balanced-b4" in n:
        curr = "B4_classroom"   # FineWeb-24B labels classroom as 'balanced-b4'
    elif "balanced-b5" in n:
        curr = "B5_late"        # likewise late
    else:
        curr = "UNK"

    # scale
    if "fineweb-100m" in n:
        scale = "100M"
    elif "humanscale" in n:
        scale = "HumanScale"
    elif "fineweb-2b" in n:
        scale = "2B"
    elif "fineweb" in n:
        scale = "24B"
    else:
        scale = "UNK"

    # lang_pair
    m = re.search(r"(nld-eng|eng-nld|deu-eng|eng-deu|zho-eng|eng-zho)", n)
    lang_pair = m.group(1) if m else "UNK"
    return {"curriculum": curr, "scale": scale, "lang_pair": lang_pair, "name": name}


# ── 1.  Pico metric loader ────────────────────────────────────────────────

PICO_METRICS = [
    "gini-ov_circuit_train",
    "hoyer-swiglu_train",
    "per-ov_circuit_train",
    "condition_number-ov_circuit_train",
    "condition_number-swiglu_train",
    "norm-ov_circuit-frobenius_train",
    "norm-swiglu-nuclear_train",
    "norm-attention_norm_swiglu_norm-frobenius_train",
    "gini-swiglu_train",
]


def _layer_index(key: str) -> Optional[int]:
    m = re.match(r"layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def _aggregate_metric_json(path: Path) -> dict:
    """Reduce a per-layer/per-head JSON to scalar features.

    Returns: {mean, std, layer0, layerL, head_mean, head_var, layer_slope}
    The point is to capture coarse properties without leaking n_layers × n_heads
    columns into the predictiveness regression.
    """
    with path.open() as f:
        d = json.load(f)
    if not d:
        return {}
    # Separate per-head keys from per-layer aggregates.
    per_head_vals = []
    per_layer_vals = {}
    for k, v in d.items():
        if not isinstance(v, (int, float)):
            continue
        if "heads." in k:
            per_head_vals.append(v)
        else:
            li = _layer_index(k)
            if li is not None:
                per_layer_vals[li] = v
    layers_sorted = sorted(per_layer_vals.items())
    layer_arr = np.array([v for _, v in layers_sorted]) if layers_sorted else np.array([])
    head_arr = np.array(per_head_vals) if per_head_vals else np.array([])

    out = {}
    if layer_arr.size:
        out["layer_mean"] = float(layer_arr.mean())
        out["layer_std"] = float(layer_arr.std())
        out["layer0"] = float(layer_arr[0])
        out["layerL"] = float(layer_arr[-1])
        if layer_arr.size >= 2:
            x = np.arange(layer_arr.size, dtype=float)
            slope = float(np.polyfit(x, layer_arr, 1)[0])
            out["layer_slope"] = slope
    if head_arr.size:
        out["head_mean"] = float(head_arr.mean())
        out["head_var"] = float(head_arr.var())
    return out


def load_pico_features() -> pd.DataFrame:
    """One row per (run_name, step), columns = aggregated pico features."""
    rows = []
    if not PICO_ANALYSIS.exists():
        logger.warning("pico analysis dir missing: %s", PICO_ANALYSIS)
        return pd.DataFrame()
    for run_dir in sorted(PICO_ANALYSIS.iterdir()):
        if not run_dir.is_dir() or "_pico_analysis" not in run_dir.name:
            continue
        run_name = run_dir.name.replace("_pico_analysis", "")
        meta = parse_run_name(run_name)
        if meta["lang_pair"] != "nld-eng":
            continue
        for step_dir in sorted(run_dir.glob("step_*")):
            mstep = re.match(r"step_(\d+)", step_dir.name)
            if not mstep:
                continue
            step = int(mstep.group(1))
            row = {**meta, "step": step}
            any_metric = False
            for metric_name in PICO_METRICS:
                fp = step_dir / f"{metric_name}.json"
                if not fp.exists():
                    continue
                feats = _aggregate_metric_json(fp)
                for k, v in feats.items():
                    row[f"pico::{metric_name}::{k}"] = v
                if feats:
                    any_metric = True
            if any_metric:
                rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["scale", "curriculum", "step"]).reset_index(drop=True)
    return df


# ── 2.  Downstream metric loaders ─────────────────────────────────────────

def _branch_to_step(b: str) -> Optional[int]:
    m = re.match(r"step-(\d+)$", str(b))
    if m:
        return int(m.group(1))
    if str(b).lower() == "main":
        return -1
    return None


def _model_to_run_name(model: str) -> str:
    """e.g. 'Beetle-FineWeb/beetle-bilingual-balanced-b1-fineweb-nld-eng' → run name."""
    return str(model).split("/", 1)[-1].lower()


def load_blimp() -> pd.DataFrame:
    """One row per (run_name, step), columns blimp_eng, blimp_nld."""
    p = DOWNSTREAM["blimp"]
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["run_name"] = df["model"].map(_model_to_run_name)
    df["step"] = df["branch"].map(_branch_to_step)
    df = df[df["step"].notna() & (df["step"] >= 0)]
    df["step"] = df["step"].astype(int)
    # Filter to nld-eng runs with the canonical English/Dutch BLiMP benches.
    df = df[df["run_name"].str.contains("nld-eng")]
    df = df[df["benchmark"].isin(["blimp_eng", "blimp_nl"])]
    pivoted = df.pivot_table(
        index=["run_name", "step"], columns="benchmark", values="accuracy"
    ).reset_index()
    pivoted.columns.name = None
    pivoted = pivoted.rename(columns={"blimp_eng": "blimp_eng_acc",
                                      "blimp_nl":  "blimp_nld_acc"})
    return pivoted


def load_bliss() -> pd.DataFrame:
    """One row per (run_name, step):
        bliss_RP0_mean      mean RP_at_0 across (l1, cefr) cells
        bliss_RP0_dyn_slope slope of RP_at_0 across CEFR levels (predicts pedagogical
                            grading: positive = better on harder items, ~0 = flat).
        bliss_NGS_mean
    """
    p = DOWNSTREAM["bliss"]
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["run_name"] = df["model"].map(_model_to_run_name)
    df["step"] = df["branch"].map(_branch_to_step)
    df = df[df["step"].notna() & (df["step"] >= 0)]
    df["step"] = df["step"].astype(int)
    df = df[df["run_name"].str.contains("nld-eng")]

    cefr_order = {"A1": 1, "A2": 2, "B1": 3, "B1+": 3.5, "B2": 4, "C1": 5, "C2": 6}
    df["cefr_rank"] = df["auto_cefr"].map(cefr_order)
    df_rp = df.dropna(subset=["RP_at_0", "cefr_rank"])

    rows = []
    for (run, step), grp in df_rp.groupby(["run_name", "step"]):
        rp_mean = grp["RP_at_0"].mean()
        ngs_mean = grp["NGS"].mean() if "NGS" in grp else np.nan
        # Slope across CEFR within run/step
        if grp["cefr_rank"].nunique() >= 2:
            slope = np.polyfit(grp["cefr_rank"].astype(float),
                               grp["RP_at_0"].astype(float), 1)[0]
        else:
            slope = np.nan
        rows.append({
            "run_name": run, "step": step,
            "bliss_RP0_mean": rp_mean,
            "bliss_NGS_mean": ngs_mean,
            "bliss_RP0_cefr_slope": slope,
        })
    return pd.DataFrame(rows)


def load_meco_dll() -> pd.DataFrame:
    p = DOWNSTREAM["meco_dll"]
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df = df[df["lang"] == "du"].copy()
    df["run_name"] = df["model"].apply(lambda m: m.split("@")[0].split("/", 1)[-1].lower())
    df["step"] = df["model"].apply(
        lambda m: int(m.split("@step-")[-1]) if "@step-" in m else None
    )
    df = df[df["step"].notna()]
    df["step"] = df["step"].astype(int)
    df = df[df["run_name"].str.contains("nld-eng")]
    return df[["run_name", "step", "delta_loglik"]].rename(
        columns={"delta_loglik": "meco_dll_du"}
    )


# Priming
def _base(t: str) -> str:
    return re.sub(r"_(Prime|Target)$", "", str(t))


def _per_item_priming(tsv: Path, between_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep="\t")
    if between_only:
        df = df[df["PrimeLanguage"] != df["TargetLanguage"]].copy()
    df["target_base"] = df["TargetType"].map(_base)
    df["prime_base"] = df["PrimeType"].map(_base)
    df["match"] = df["target_base"] == df["prime_base"]
    keys = ["Set", "ExtraCondition", "PrimeLanguage", "TargetLanguage",
            "TargetSentence", "TargetType"]
    m = df[df["match"]].groupby(keys, as_index=False)["Surprisal"].mean().rename(
        columns={"Surprisal": "s_match"})
    nm = df[~df["match"]].groupby(keys, as_index=False)["Surprisal"].mean().rename(
        columns={"Surprisal": "s_mis"})
    out = m.merge(nm, on=keys, how="inner")
    out["priming"] = out["s_mis"] - out["s_match"]
    return out


def load_priming() -> pd.DataFrame:
    pr_dir = DOWNSTREAM["priming_dir"]
    if not pr_dir.exists():
        return pd.DataFrame()
    rows = []
    for tsv in sorted(pr_dir.glob("bernolet__*__step-*.tsv")):
        m = re.match(r"bernolet__(.+?)__step-(\d+)\.tsv$", tsv.name)
        if not m:
            continue
        repo_part, step = m.group(1), int(m.group(2))
        # repo_part = e.g. beetle-fineweb_beetle-bilingual-balanced-b1-fineweb-nld-eng
        # collapse the 'beetle-fineweb_' or 'beetle-humanscale_' prefix to the run name
        run_name = repo_part.split("_", 1)[-1]
        if "nld-eng" not in run_name:
            continue
        try:
            ip = _per_item_priming(tsv)
        except Exception as e:
            logger.warning("priming parse failed %s: %s", tsv.name, e)
            continue
        if ip.empty:
            continue
        rows.append({
            "run_name": run_name,
            "step": step,
            "priming_bernolet_mean": float(ip["priming"].mean()),
            "priming_bernolet_n": int(len(ip)),
        })
    return pd.DataFrame(rows)


# ── 3.  Build unified frame ───────────────────────────────────────────────

def build_unified() -> pd.DataFrame:
    pico = load_pico_features()
    blimp = load_blimp()
    bliss = load_bliss()
    meco = load_meco_dll()
    priming = load_priming()

    logger.info("rows: pico=%d blimp=%d bliss=%d meco=%d priming=%d",
                len(pico), len(blimp), len(bliss), len(meco), len(priming))

    if pico.empty:
        logger.error("no pico data — aborting")
        return pd.DataFrame()
    pico = pico.rename(columns={"name": "run_name"})
    df = pico.copy()
    for d, label in [(blimp, "blimp"), (bliss, "bliss"),
                     (meco, "meco"), (priming, "priming")]:
        if d.empty:
            logger.warning("downstream %s empty — skipping merge", label)
            continue
        df = df.merge(d, on=["run_name", "step"], how="left")
    return df


# ── 4.  Predictiveness (linear, leave-one-curriculum-out) ─────────────────

DOWNSTREAM_TARGETS = [
    "blimp_eng_acc", "blimp_nld_acc",
    "meco_dll_du",
    "priming_bernolet_mean",
    "bliss_RP0_mean", "bliss_NGS_mean", "bliss_RP0_cefr_slope",
]


def predictiveness(df: pd.DataFrame) -> pd.DataFrame:
    """Per (downstream_target, pico_feature, scale): predictive R² of a univariate
    linear regression pooled across (curriculum, step), evaluated by leaving one
    curriculum out at a time (LOOcv across curricula → robust generalisation).

    Also reports Spearman ρ for monotonic alignment.
    """
    from scipy.stats import spearmanr

    pico_cols = [c for c in df.columns if c.startswith("pico::")]
    out = []
    for scale in sorted(df["scale"].dropna().unique()):
        sub = df[df["scale"] == scale]
        for tgt in DOWNSTREAM_TARGETS:
            if tgt not in sub.columns:
                continue
            sub_t = sub.dropna(subset=[tgt])
            if sub_t["curriculum"].nunique() < 2 or len(sub_t) < 6:
                continue
            for pc in pico_cols:
                ss = sub_t.dropna(subset=[pc])
                if len(ss) < 6:
                    continue
                # Pearson, Spearman over pooled
                x = ss[pc].astype(float).values
                y = ss[tgt].astype(float).values
                if np.std(x) == 0 or np.std(y) == 0:
                    continue
                rho, prho = spearmanr(x, y)
                # LOOcv across curricula
                preds, trues = [], []
                for cv_curr in ss["curriculum"].unique():
                    train = ss[ss["curriculum"] != cv_curr]
                    test = ss[ss["curriculum"] == cv_curr]
                    if len(train) < 4 or len(test) < 1:
                        continue
                    a, b = np.polyfit(train[pc].astype(float),
                                      train[tgt].astype(float), 1)
                    pred = a * test[pc].astype(float).values + b
                    preds.extend(pred.tolist())
                    trues.extend(test[tgt].astype(float).tolist())
                if not preds:
                    continue
                preds = np.array(preds)
                trues = np.array(trues)
                ss_res = np.mean((preds - trues) ** 2)
                ss_tot = np.var(trues) if np.var(trues) > 0 else 1.0
                r2_loo = 1.0 - ss_res / ss_tot
                out.append({
                    "scale": scale,
                    "downstream": tgt,
                    "pico_feature": pc,
                    "n": len(ss),
                    "spearman_rho": rho,
                    "spearman_p": prho,
                    "r2_loocv_curriculum": r2_loo,
                })
    return pd.DataFrame(out)


def lead_lag(df: pd.DataFrame, max_lag_steps: int = 4) -> pd.DataFrame:
    """For each (curriculum, scale), compute Spearman correlation between pico
    feature at step t and downstream target at step t+k (k=−max..+max). The
    optimal k tells us whether pico signatures lead, lag, or co-evolve with
    downstream behaviour.
    """
    from scipy.stats import spearmanr
    pico_cols = [c for c in df.columns if c.startswith("pico::")]
    out = []
    for (curr, scale), g in df.groupby(["curriculum", "scale"]):
        g = g.sort_values("step").reset_index(drop=True)
        if len(g) < 6:
            continue
        for tgt in DOWNSTREAM_TARGETS:
            if tgt not in g.columns:
                continue
            y = g[tgt].astype(float).values
            for pc in pico_cols:
                x = g[pc].astype(float).values
                if np.all(np.isnan(x)) or np.all(np.isnan(y)):
                    continue
                for k in range(-max_lag_steps, max_lag_steps + 1):
                    if k >= 0:
                        a, b = x[:len(x) - k], y[k:]
                    else:
                        a, b = x[-k:], y[:len(y) + k]
                    mask = ~(np.isnan(a) | np.isnan(b))
                    if mask.sum() < 4:
                        continue
                    rho, p = spearmanr(a[mask], b[mask])
                    out.append({
                        "curriculum": curr, "scale": scale,
                        "downstream": tgt, "pico_feature": pc,
                        "lag_steps": k, "rho": rho, "p": p, "n": int(mask.sum()),
                    })
    return pd.DataFrame(out)


# ── 5.  Plots ─────────────────────────────────────────────────────────────

def _safe_savefig(fig, base: Path):
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=180)
    logger.info("wrote %s.{pdf,png}", base)


def fig_predictiveness_heatmap(pred: pd.DataFrame, base: Path):
    import matplotlib.pyplot as plt
    if pred.empty:
        logger.warning("predictiveness empty — skipping heatmap")
        return
    # collapse across scales: use mean R² across scales (or whichever scale exists)
    sel = pred.groupby(["downstream", "pico_feature"])["r2_loocv_curriculum"].mean().reset_index()
    sel = sel.pivot(index="downstream", columns="pico_feature", values="r2_loocv_curriculum")
    # Reorder rows by mean predictiveness for readability
    sel = sel.reindex(sel.mean(axis=1).sort_values(ascending=False).index)
    # Take top-N pico columns by max
    top_cols = sel.max(axis=0).sort_values(ascending=False).index[:24]
    sel = sel[top_cols]

    fig, ax = plt.subplots(figsize=(0.45 * len(sel.columns) + 2, 0.5 * len(sel.index) + 2))
    vmax = max(0.8, np.nanmax(sel.values))
    im = ax.imshow(sel.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(sel.columns)))
    ax.set_xticklabels([c.replace("pico::", "").replace("_train", "") for c in sel.columns],
                       rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(sel.index)))
    ax.set_yticklabels(sel.index, fontsize=8)
    ax.set_title("Predictiveness of pico signatures for downstream tasks (LOOcv R², nld-eng)",
                 fontsize=10)
    for i in range(sel.shape[0]):
        for j in range(sel.shape[1]):
            v = sel.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if abs(v) < 0.5 else "white")
    fig.colorbar(im, ax=ax, fraction=0.025, label="LOOcv R² (across curricula)")
    _safe_savefig(fig, base)
    plt.close(fig)


def fig_factor_biplot(df: pd.DataFrame, base: Path):
    """PCA on the downstream metric matrix, plot trajectory points coloured by
    curriculum and shaped by scale, plus loading vectors."""
    import matplotlib.pyplot as plt
    cols = [c for c in DOWNSTREAM_TARGETS if c in df.columns]
    sub = df.dropna(subset=cols, how="any")[cols + ["curriculum", "scale", "step"]]
    if len(sub) < 8 or len(cols) < 3:
        logger.warning("too few rows for factor biplot (n=%d, cols=%d)", len(sub), len(cols))
        return
    X = sub[cols].astype(float).values
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-9)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc = U[:, :2] * s[:2]
    load = (Vt[:2].T) * s[:2]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    palette = {
        "B1_balanced": "#1f77b4", "B2_simultaneous": "#2ca02c",
        "B3_sequential": "#ff7f0e", "B4_classroom": "#9467bd",
        "B5_late": "#d62728",
    }
    markers = {"100M": "o", "2B": "s", "24B": "D", "HumanScale": "^"}
    for (curr, scale), g in sub.groupby(["curriculum", "scale"]):
        idx = g.index.tolist()
        ax.scatter(pc[idx, 0], pc[idx, 1],
                   c=palette.get(curr, "#888"),
                   marker=markers.get(scale, "x"),
                   s=30, alpha=0.7, edgecolors="white", linewidths=0.4)
    # loadings
    scale_arrow = 0.8 * np.max(np.abs(pc)) / max(1e-9, np.max(np.abs(load)))
    for i, c in enumerate(cols):
        ax.arrow(0, 0, load[i, 0] * scale_arrow, load[i, 1] * scale_arrow,
                 color="black", width=0.005, head_width=0.05, alpha=0.7)
        ax.text(load[i, 0] * scale_arrow * 1.07, load[i, 1] * scale_arrow * 1.07,
                c.replace("_", " "), fontsize=8, color="black")
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    var_exp = (s ** 2) / (s ** 2).sum()
    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
    ax.set_title("Task factor structure across curricula & scales (nld-eng)")

    # legend
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=v, marker="o", linestyle="", label=k)
               for k, v in palette.items()] + \
              [Line2D([], [], color="black", marker=v, linestyle="", label=k)
               for k, v in markers.items()]
    ax.legend(handles=handles, fontsize=7, loc="best", ncol=2)
    _safe_savefig(fig, base)
    plt.close(fig)


def fig_lead_lag(ll: pd.DataFrame, base: Path):
    """Median rho across (curriculum, scale) at each lag, per (downstream × pico-feature
    family). Reduces pico features to a handful of families for readability."""
    import matplotlib.pyplot as plt
    if ll.empty:
        logger.warning("lead_lag empty"); return
    ll = ll.copy()
    ll["pico_family"] = ll["pico_feature"].apply(
        lambda s: s.replace("pico::", "").split("::")[0])
    grouped = ll.groupby(["downstream", "pico_family", "lag_steps"])["rho"].median().reset_index()
    families = sorted(grouped["pico_family"].unique())
    targets = [t for t in DOWNSTREAM_TARGETS if t in grouped["downstream"].unique()]
    fig, axes = plt.subplots(len(targets), 1,
                             figsize=(7, 1.3 * max(1, len(targets))),
                             sharex=True)
    if len(targets) == 1:
        axes = [axes]
    for ax, tgt in zip(axes, targets):
        sub = grouped[grouped["downstream"] == tgt]
        for fam in families:
            ss = sub[sub["pico_family"] == fam].sort_values("lag_steps")
            ax.plot(ss["lag_steps"], ss["rho"], marker="o", ms=3,
                    lw=1, label=fam.replace("_train", ""))
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel(tgt, fontsize=7)
    axes[-1].set_xlabel("Pico-step lag (positive = pico leads downstream)")
    axes[0].legend(fontsize=6, loc="upper right", ncol=2)
    fig.suptitle("Lead-lag: median Spearman ρ between pico features and downstream metrics",
                 fontsize=9, y=1.0)
    _safe_savefig(fig, base)
    plt.close(fig)


def fig_curriculum_grid(df: pd.DataFrame, base: Path):
    """Grid of trajectories: rows = downstream metrics, cols = curricula, lines per scale."""
    import matplotlib.pyplot as plt
    metrics = [m for m in DOWNSTREAM_TARGETS if m in df.columns and df[m].notna().any()]
    currs = sorted(df["curriculum"].dropna().unique())
    if not metrics or not currs:
        logger.warning("nothing to plot in curriculum grid"); return
    fig, axes = plt.subplots(len(metrics), len(currs),
                             figsize=(2.0 * len(currs), 1.4 * len(metrics)),
                             sharex=True)
    if len(metrics) == 1:
        axes = np.array([axes])
    if len(currs) == 1:
        axes = axes.reshape(-1, 1)
    scale_color = {"100M": "#888", "2B": "#1f77b4",
                   "24B": "#2ca02c", "HumanScale": "#d62728"}
    for i, m in enumerate(metrics):
        for j, c in enumerate(currs):
            ax = axes[i, j]
            sub = df[(df["curriculum"] == c)].dropna(subset=[m, "step"])
            for sc, g in sub.groupby("scale"):
                gs = g.sort_values("step")
                ax.plot(gs["step"].clip(lower=1), gs[m],
                        color=scale_color.get(sc, "k"), lw=1, marker=".",
                        ms=3, label=sc, alpha=0.8)
            ax.set_xscale("log")
            if i == 0:
                ax.set_title(c, fontsize=8)
            if j == 0:
                ax.set_ylabel(m, fontsize=7)
            ax.tick_params(labelsize=6)
    axes[0, -1].legend(fontsize=6, loc="best", ncol=1)
    fig.suptitle("Per-curriculum trajectories across scales (nld-eng)", fontsize=10, y=1.00)
    fig.tight_layout()
    _safe_savefig(fig, base)
    plt.close(fig)


def fig_scale_consistency(pred: pd.DataFrame, base: Path):
    """Within-task correlation of predictiveness rankings across scales."""
    import matplotlib.pyplot as plt
    if pred.empty: return
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot = pred.pivot_table(index=["downstream", "pico_feature"],
                              columns="scale", values="r2_loocv_curriculum")
    pivot = pivot.dropna(how="any")
    if pivot.empty or pivot.shape[1] < 2:
        logger.warning("scale consistency: not enough scale overlap"); return
    cols = list(pivot.columns)
    # Plot pairwise scatter for the two most-populated scales
    sa, sb = cols[0], cols[1]
    for tgt in pivot.index.get_level_values(0).unique():
        sub = pivot.xs(tgt, level=0)
        ax.scatter(sub[sa], sub[sb], s=14, alpha=0.6, label=tgt)
    lim = (-0.5, 1.0)
    ax.plot(lim, lim, "k--", lw=0.5)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"R² @ {sa}"); ax.set_ylabel(f"R² @ {sb}")
    ax.set_title("Pico→downstream predictiveness across scales")
    ax.legend(fontsize=6, loc="best")
    _safe_savefig(fig, base)
    plt.close(fig)


# ── 6.  Driver ────────────────────────────────────────────────────────────

def main():
    df = build_unified()
    if df.empty:
        logger.error("unified dataframe empty")
        return 1
    df.to_csv(OUT_DIR / "nld_eng_wide.csv", index=False)
    logger.info("wide unified dataframe → %s (%d rows, %d cols)",
                OUT_DIR / "nld_eng_wide.csv", len(df), df.shape[1])

    pred = predictiveness(df)
    pred.to_csv(OUT_DIR / "predictiveness.csv", index=False)
    logger.info("predictiveness → %s (%d rows)", OUT_DIR / "predictiveness.csv", len(pred))

    ll = lead_lag(df)
    ll.to_csv(OUT_DIR / "lead_lag.csv", index=False)
    logger.info("lead_lag → %s (%d rows)", OUT_DIR / "lead_lag.csv", len(ll))

    fig_predictiveness_heatmap(pred, FIG_MAIN / "fig_intrinsic_eval_predictiveness")
    fig_factor_biplot(df, FIG_MAIN / "fig_intrinsic_eval_factor_biplot")
    fig_lead_lag(ll, FIG_APP / "fig_intrinsic_eval_lead_lag")
    fig_curriculum_grid(df, FIG_APP / "fig_intrinsic_eval_curriculum_grid")
    fig_scale_consistency(pred, FIG_APP / "fig_intrinsic_eval_scale_consistency")
    logger.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
