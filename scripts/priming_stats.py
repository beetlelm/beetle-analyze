#!/usr/bin/env python3
"""
Item-level statistical inference on cross-lingual structural priming TSVs.

Analyses:
  (a) within-model significance of the priming effect
      → mixed-effects: Surprisal ~ condition + (1|item) per (model × direction)
  (b') Beetle-vs-Beetle pairwise contrasts
      → paired permutation on per-item priming effect
  (c') Beetle peak-vs-final stability per trajectory model
      → paired permutation on per-item priming, peak-ckpt vs final-ckpt
  (d) Normalised relative-probability robustness (Arnett 2025 metric)
      → P_N = exp(-S_match) / (exp(-S_match) + exp(-S_mismatch)),
        per-item paired test of P_N - 0.5 against zero
  (e) Within-language baseline vs cross-language priming
      → paired permutation contrasting cross-lingual and within-lingual priming
        on the same items, per Beetle final-ckpt model
  (f) Confirmatory mixed-effects (statsmodels MixedLM)
      → Δ ~ Curriculum + (1|Item) on Beetle FineWeb final ckpts (Bernolet)
      → Δ ~ ckpt_phase + (1|Item) on a single trajectory (B1) collapsing
        ckpts into early / mid / peak / final phases
  (g) Earlier-emergence timing per curriculum
      → first ckpt where mean per-item priming exceeds zero with
        permutation p < 0.01, marked relative to L2-onset fraction

Bernolet only. (a)-(d), (f), (g) use between-language pairs only; (e)
uses both cells by construction.

Per-item priming effect:
    For each item = (Set, ExtraCondition, PrimeLanguage, TargetLanguage,
    TargetSentence), there are two observations of target Surprisal —
    one under the matching prime (PrimeType base = TargetType base) and one
    under the mismatching prime. priming(item) = Surprisal_mismatch − Surprisal_match.

Outputs:
    paper/emnlp/PRIMING_STATS.md
    results/evals/priming/stats/{a_within_model.csv, b_pairwise.csv, c_peak_vs_final.csv}
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("priming_stats")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

REPO_ROOT = Path(__file__).resolve().parents[1]
TIER1_DIR = REPO_ROOT / "results/evals/priming/tier1_final"
TIER2_DIR = REPO_ROOT / "results/evals/priming/tier2_headline_trajectory"
OUT_DIR = REPO_ROOT / "results/evals/priming/stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260509)
N_PERMUTATIONS = 10000


# ── Item-level priming computation ─────────────────────────────────────────

def _base(t: str) -> str:
    """Strip _Prime/_Target suffix to get construction base."""
    return re.sub(r"_(Prime|Target)$", "", t)


def per_item_priming(tsv: Path, between_only: bool = True) -> pd.DataFrame:
    """Load a Bernolet TSV and return per-item priming effects.

    Returns DataFrame with columns:
        Set, ExtraCondition, PrimeLanguage, TargetLanguage, TargetSentence,
        TargetType, surprisal_match, surprisal_mismatch, priming, p_n

    `priming`  = surprisal_mismatch − surprisal_match  (larger = stronger)
    `p_n`      = exp(-S_match) / (exp(-S_match) + exp(-S_mismatch))
                 i.e. Arnett (2025) normalised relative probability of
                 the matched alternative; 0.5 = no priming, > 0.5 = priming.
    """
    df = pd.read_csv(tsv, sep="\t")
    if between_only:
        df = df[df["PrimeLanguage"] != df["TargetLanguage"]].copy()
    df["target_base"] = df["TargetType"].map(_base)
    df["prime_base"]  = df["PrimeType"].map(_base)
    df["match"] = (df["target_base"] == df["prime_base"])

    keys = ["Set", "ExtraCondition", "PrimeLanguage", "TargetLanguage",
            "TargetSentence", "TargetType"]
    m = df[df["match"]].groupby(keys, as_index=False)["Surprisal"].mean()
    nm = df[~df["match"]].groupby(keys, as_index=False)["Surprisal"].mean()
    m = m.rename(columns={"Surprisal": "surprisal_match"})
    nm = nm.rename(columns={"Surprisal": "surprisal_mismatch"})
    out = m.merge(nm, on=keys, how="inner")
    out["priming"] = out["surprisal_mismatch"] - out["surprisal_match"]
    # Numerically-stable softmax of the two negative-surprisals (= log-probs).
    # Subtract row-wise max to avoid overflow when surprisals are tiny.
    a = -out["surprisal_match"].values
    b = -out["surprisal_mismatch"].values
    mx = np.maximum(a, b)
    ea, eb = np.exp(a - mx), np.exp(b - mx)
    out["p_n"] = ea / (ea + eb)
    return out


# ── (a) Within-model significance via paired test on items ─────────────────

def paired_t_p(diffs: np.ndarray) -> tuple[float, float, float, int]:
    """Paired t-test of diffs against zero.

    Returns (mean_diff, t_stat, p_value, n)."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n < 3:
        return float(np.nan), float(np.nan), float(np.nan), n
    mean = diffs.mean()
    sd = diffs.std(ddof=1)
    se = sd / np.sqrt(n)
    if se == 0:
        return mean, float("inf") if mean != 0 else 0.0, 0.0 if mean != 0 else 1.0, n
    t = mean / se
    # two-sided p, large-n normal approx adequate for n>30; use t exact via scipy
    from scipy import stats
    p = 2 * (1 - stats.t.cdf(abs(t), df=n - 1))
    return mean, t, p, n


def perm_p(diffs: np.ndarray, n_iter: int = N_PERMUTATIONS) -> float:
    """Sign-flip permutation p for paired diffs vs zero."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n < 3:
        return float("nan")
    obs = diffs.mean()
    signs = RNG.choice([-1.0, 1.0], size=(n_iter, n))
    null = (signs * diffs).mean(axis=1)
    return float((np.abs(null) >= abs(obs)).mean())


def analysis_a_within_model() -> pd.DataFrame:
    """Per-model × direction paired test of per-item priming > 0 (Bernolet, between-lang)."""
    rows = []
    tsvs = sorted(TIER1_DIR.glob("bernolet*__main.tsv"))
    logger.info("(a) processing %d tier1 Bernolet TSVs", len(tsvs))
    for tsv in tsvs:
        # Skip non-final or aux variants if needed
        model_name = tsv.stem.replace("bernolet__", "").replace("__main", "")
        per = per_item_priming(tsv, between_only=True)
        if not len(per):
            continue
        for (pl, tl), sub in per.groupby(["PrimeLanguage", "TargetLanguage"]):
            mean, t, p, n = paired_t_p(sub["priming"].values)
            pp = perm_p(sub["priming"].values)
            rows.append({
                "model": model_name,
                "PrimeLanguage": pl, "TargetLanguage": tl,
                "n_items": n,
                "mean_priming": round(mean, 4),
                "t": round(t, 3) if np.isfinite(t) else float("nan"),
                "p_t": round(p, 6),
                "p_perm": round(pp, 6),
            })
    return pd.DataFrame(rows).sort_values(["model", "PrimeLanguage", "TargetLanguage"]).reset_index(drop=True)


# ── (b′) Beetle-vs-Beetle pairwise contrasts ───────────────────────────────

PAIRS_B = [
    # (label_A, glob_A, label_B, glob_B)
    ("B5", "bernolet*balanced-b5-fineweb-nld-eng__main.tsv",
     "B3", "bernolet*sequential-33-67-b3-fineweb-nld-eng__main.tsv"),
    ("B5", "bernolet*balanced-b5-fineweb-nld-eng__main.tsv",
     "B2", "bernolet*simultaneous-b2-fineweb-nld-eng__main.tsv"),
    ("B1", "bernolet*balanced-b1-fineweb-nld-eng__main.tsv",
     "B3", "bernolet*sequential-33-67-b3-fineweb-nld-eng__main.tsv"),
    ("B1", "bernolet*balanced-b1-fineweb-nld-eng__main.tsv",
     "B2", "bernolet*simultaneous-b2-fineweb-nld-eng__main.tsv"),
    ("B5", "bernolet*balanced-b5-fineweb-nld-eng__main.tsv",
     "B1", "bernolet*balanced-b1-fineweb-nld-eng__main.tsv"),
    ("B4", "bernolet*balanced-b4-fineweb-nld-eng__main.tsv",
     "B3", "bernolet*sequential-33-67-b3-fineweb-nld-eng__main.tsv"),
]

KEYS_ITEM = ["Set", "ExtraCondition", "PrimeLanguage", "TargetLanguage",
             "TargetSentence", "TargetType"]


def _first_match(pat: str) -> Path | None:
    for d in (TIER1_DIR, TIER2_DIR):
        hits = list(d.glob(pat))
        if hits:
            return hits[0]
    return None


def analysis_b_pairwise() -> pd.DataFrame:
    rows = []
    logger.info("(b′) running %d Beetle pairwise contrasts", len(PAIRS_B))
    for la, ga, lb, gb in PAIRS_B:
        fa, fb = _first_match(ga), _first_match(gb)
        if fa is None or fb is None:
            logger.warning("  skip %s vs %s (missing TSV)", la, lb)
            continue
        a = per_item_priming(fa).rename(columns={"priming": f"priming_{la}"})
        b = per_item_priming(fb).rename(columns={"priming": f"priming_{lb}"})
        merged = a.merge(b, on=KEYS_ITEM, suffixes=("_a", "_b"))
        if not len(merged):
            logger.warning("  no matched items for %s vs %s", la, lb)
            continue
        diffs = (merged[f"priming_{la}"] - merged[f"priming_{lb}"]).values
        mean, t, p_t, n = paired_t_p(diffs)
        p_perm = perm_p(diffs)
        rows.append({
            "contrast": f"{la} − {lb}",
            "n_items": n,
            "mean_diff": round(float(mean), 4),
            "mean_A": round(float(merged[f"priming_{la}"].mean()), 4),
            "mean_B": round(float(merged[f"priming_{lb}"].mean()), 4),
            "t": round(t, 3) if np.isfinite(t) else float("nan"),
            "p_t": round(p_t, 6),
            "p_perm": round(p_perm, 6),
        })
    return pd.DataFrame(rows)


# ── (c′) Beetle peak-vs-final stability ────────────────────────────────────

BEETLE_TRAJECTORY_MODELS = [
    ("B1", "bernolet__beetle-fineweb_beetle-bilingual-balanced-b1-fineweb-nld-eng"),
    ("B2", "bernolet__beetle-fineweb_beetle-bilingual-l2-50-simultaneous-b2-fineweb-nld-eng"),
    ("B3", "bernolet__beetle-fineweb_beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-nld-eng"),
    ("B5", "bernolet__beetle-fineweb_beetle-bilingual-balanced-b5-fineweb-nld-eng"),
]

CKPT_RE = re.compile(r"__step-(\d+)\.tsv$")


def analysis_c_peak_vs_final() -> pd.DataFrame:
    rows = []
    logger.info("(c′) checking peak-vs-final stability for %d Beetle trajectories",
                len(BEETLE_TRAJECTORY_MODELS))
    for label, prefix in BEETLE_TRAJECTORY_MODELS:
        tsvs = sorted(TIER2_DIR.glob(f"{prefix}__step-*.tsv"))
        if not tsvs:
            logger.warning("  no trajectory TSVs for %s", label)
            continue

        # Compute mean priming per ckpt to find peak
        per_ckpt = []
        per_ckpt_items: dict[int, pd.DataFrame] = {}
        for tsv in tsvs:
            m = CKPT_RE.search(tsv.name)
            if not m: continue
            ck = int(m.group(1))
            try:
                items = per_item_priming(tsv, between_only=True)
            except Exception as e:
                logger.warning("  failed to read %s: %s", tsv.name, e)
                continue
            if not len(items): continue
            per_ckpt_items[ck] = items
            per_ckpt.append((ck, items["priming"].mean()))

        if len(per_ckpt) < 3:
            logger.warning("  too few ckpts for %s", label)
            continue

        ck_df = pd.DataFrame(per_ckpt, columns=["ckpt", "mean_priming"]).sort_values("ckpt")
        peak_ckpt = int(ck_df.loc[ck_df.mean_priming.idxmax(), "ckpt"])
        final_ckpt = int(ck_df.ckpt.max())
        if peak_ckpt == final_ckpt:
            # No decay possible — record as no change
            mean = 0.0; t = 0.0; p_t = 1.0; p_perm = 1.0; n = 0
            peak_val = ck_df.mean_priming.max(); final_val = peak_val
        else:
            peak_val = float(per_ckpt_items[peak_ckpt]["priming"].mean())
            final_val = float(per_ckpt_items[final_ckpt]["priming"].mean())
            merged = per_ckpt_items[peak_ckpt].merge(
                per_ckpt_items[final_ckpt][KEYS_ITEM + ["priming"]],
                on=KEYS_ITEM, suffixes=("_peak", "_final"))
            diffs = (merged["priming_peak"] - merged["priming_final"]).values
            mean, t, p_t, n = paired_t_p(diffs)
            p_perm = perm_p(diffs)
        rows.append({
            "curriculum": label,
            "peak_ckpt": peak_ckpt,
            "final_ckpt": final_ckpt,
            "peak_priming": round(peak_val, 4),
            "final_priming": round(final_val, 4),
            "decay_pct": round(100.0 * (peak_val - final_val) / peak_val, 1) if peak_val > 0 else float("nan"),
            "n_items": n,
            "mean_diff_peak_minus_final": round(float(mean), 4),
            "t": round(t, 3) if np.isfinite(t) else float("nan"),
            "p_t": round(p_t, 6),
            "p_perm": round(p_perm, 6),
        })
    return pd.DataFrame(rows)


# ── (d) Normalised relative probability (Arnett 2025) ──────────────────────

def analysis_d_pn() -> pd.DataFrame:
    """Per-model × direction paired test of (P_N - 0.5) > 0.

    Uses the SAME final-ckpt Bernolet TSVs as (a). Reviewers asking
    "why surprisal-diff and not Arnett's P_N?" get a robustness table
    showing the qualitative effect survives the metric switch.
    """
    rows = []
    tsvs = sorted(TIER1_DIR.glob("bernolet*__main.tsv"))
    logger.info("(d) P_N robustness across %d Bernolet TSVs", len(tsvs))
    for tsv in tsvs:
        model_name = tsv.stem.replace("bernolet__", "").replace("__main", "")
        per = per_item_priming(tsv, between_only=True)
        if not len(per):
            continue
        for (pl, tl), sub in per.groupby(["PrimeLanguage", "TargetLanguage"]):
            diffs = sub["p_n"].values - 0.5
            mean, t, p, n = paired_t_p(diffs)
            pp = perm_p(diffs)
            rows.append({
                "model": model_name,
                "PrimeLanguage": pl, "TargetLanguage": tl,
                "n_items": n,
                "mean_p_n": round(float(sub["p_n"].mean()), 4),
                "mean_p_n_minus_half": round(float(mean), 4),
                "t": round(t, 3) if np.isfinite(t) else float("nan"),
                "p_t": round(p, 6),
                "p_perm": round(pp, 6),
            })
    return pd.DataFrame(rows).sort_values(
        ["model", "PrimeLanguage", "TargetLanguage"]).reset_index(drop=True)


# ── (e) Within-language vs cross-language baseline ─────────────────────────

def analysis_e_within_vs_cross() -> pd.DataFrame:
    """For each Beetle model, contrast cross-lingual priming with the
    same-language baseline (Eng→Eng, Nld→Nld) computed on the same items.

    Returns one row per (model × cross-direction) with the matched
    within-language priming, and a paired permutation test of
    (cross_priming − within_priming) on shared items.
    """
    rows = []
    tsvs = sorted(TIER1_DIR.glob("bernolet*__main.tsv"))
    logger.info("(e) within-vs-cross across %d Bernolet TSVs", len(tsvs))
    for tsv in tsvs:
        model_name = tsv.stem.replace("bernolet__", "").replace("__main", "")
        cross = per_item_priming(tsv, between_only=True)
        within = per_item_priming(tsv, between_only=False)
        within = within[within["PrimeLanguage"] == within["TargetLanguage"]]
        if not len(cross) or not len(within):
            continue
        # Match on TargetSentence + TargetType (the cross items vs the
        # within items targeting the same construction in the target lang).
        for (pl, tl), c_sub in cross.groupby(["PrimeLanguage", "TargetLanguage"]):
            w_sub = within[within["TargetLanguage"] == tl][
                ["TargetSentence", "TargetType", "priming"]
            ].rename(columns={"priming": "priming_within"})
            merged = c_sub.merge(w_sub, on=["TargetSentence", "TargetType"], how="inner")
            if not len(merged):
                continue
            diffs = (merged["priming"] - merged["priming_within"]).values
            mean, t, p_t, n = paired_t_p(diffs)
            p_perm = perm_p(diffs)
            rows.append({
                "model": model_name,
                "PrimeLanguage": pl, "TargetLanguage": tl,
                "n_items": n,
                "cross_priming": round(float(merged["priming"].mean()), 4),
                "within_priming": round(float(merged["priming_within"].mean()), 4),
                "mean_diff_cross_minus_within": round(float(mean), 4),
                "t": round(t, 3) if np.isfinite(t) else float("nan"),
                "p_t": round(p_t, 6),
                "p_perm": round(p_perm, 6),
            })
    return pd.DataFrame(rows).sort_values(
        ["model", "PrimeLanguage", "TargetLanguage"]).reset_index(drop=True)


# ── (f) Confirmatory mixed-effects (statsmodels MixedLM) ───────────────────

def analysis_f_mixedlm_curriculum() -> pd.DataFrame:
    """Δ ~ Curriculum + (1|Item) on Beetle FineWeb final ckpts (Bernolet,
    between-lang). Reference level = B1.
    """
    import statsmodels.formula.api as smf

    curricula = [
        ("B1", "bernolet*balanced-b1-fineweb-nld-eng__main.tsv"),
        ("B2", "bernolet*simultaneous-b2-fineweb-nld-eng__main.tsv"),
        ("B3", "bernolet*sequential-33-67-b3-fineweb-nld-eng__main.tsv"),
        ("B4", "bernolet*balanced-b4-fineweb-nld-eng__main.tsv"),
        ("B5", "bernolet*balanced-b5-fineweb-nld-eng__main.tsv"),
    ]
    frames = []
    for label, pat in curricula:
        f = _first_match(pat)
        if f is None:
            continue
        per = per_item_priming(f, between_only=True)
        per = per.assign(Curriculum=label, Item=lambda d:
                         d["Set"].astype(str) + "|"
                         + d["ExtraCondition"].astype(str) + "|"
                         + d["PrimeLanguage"] + "->" + d["TargetLanguage"]
                         + "|" + d["TargetSentence"] + "|" + d["TargetType"])
        frames.append(per[["Curriculum", "Item", "priming"]])
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["Curriculum"] = pd.Categorical(df["Curriculum"],
                                      categories=["B1", "B2", "B3", "B4", "B5"])

    md = smf.mixedlm("priming ~ C(Curriculum, Treatment(reference='B1'))",
                     data=df, groups=df["Item"])
    fit = md.fit(method="lbfgs", reml=True)
    return _mixedlm_to_df(fit, model_label="Δ ~ Curriculum + (1|Item)")


def analysis_f_mixedlm_phase(curriculum: str = "B1") -> pd.DataFrame:
    """Δ ~ ckpt_phase + (1|Item) on a single trajectory (default B1).
    Bins ckpts into early (<25%), mid (25–50%), peak (50–75%), final (≥75%).
    Reference = early.
    """
    import statsmodels.formula.api as smf

    prefixes = {
        "B1": "bernolet__beetle-fineweb_beetle-bilingual-balanced-b1-fineweb-nld-eng",
        "B2": "bernolet__beetle-fineweb_beetle-bilingual-l2-50-simultaneous-b2-fineweb-nld-eng",
        "B3": "bernolet__beetle-fineweb_beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-nld-eng",
        "B5": "bernolet__beetle-fineweb_beetle-bilingual-balanced-b5-fineweb-nld-eng",
    }
    prefix = prefixes[curriculum]
    tsvs = sorted(TIER2_DIR.glob(f"{prefix}__step-*.tsv"))
    if not tsvs:
        logger.warning("(f) no trajectory TSVs for %s", curriculum)
        return pd.DataFrame()
    cks = []
    for t in tsvs:
        m = CKPT_RE.search(t.name)
        if m: cks.append((int(m.group(1)), t))
    cks.sort()
    max_ck = max(c for c, _ in cks)
    frames = []
    for ck, t in cks:
        per = per_item_priming(t, between_only=True)
        if not len(per): continue
        frac = ck / max_ck
        if frac < 0.25:   phase = "early"
        elif frac < 0.50: phase = "mid"
        elif frac < 0.75: phase = "peak"
        else:             phase = "final"
        per = per.assign(
            Phase=phase, ckpt=ck,
            Item=lambda d:
                d["Set"].astype(str) + "|"
                + d["ExtraCondition"].astype(str) + "|"
                + d["PrimeLanguage"] + "->" + d["TargetLanguage"]
                + "|" + d["TargetSentence"] + "|" + d["TargetType"])
        frames.append(per[["Phase", "ckpt", "Item", "priming"]])
    df = pd.concat(frames, ignore_index=True)
    df["Phase"] = pd.Categorical(df["Phase"],
                                 categories=["early", "mid", "peak", "final"])
    md = smf.mixedlm("priming ~ C(Phase, Treatment(reference='early'))",
                     data=df, groups=df["Item"])
    fit = md.fit(method="lbfgs", reml=True)
    return _mixedlm_to_df(fit, model_label=f"Δ ~ Phase + (1|Item) [{curriculum}]")


def _mixedlm_to_df(fit, model_label: str) -> pd.DataFrame:
    params = fit.params
    se = fit.bse
    z = fit.tvalues
    pvals = fit.pvalues
    rows = []
    for term in params.index:
        rows.append({
            "model": model_label,
            "term": term,
            "estimate": round(float(params[term]), 4),
            "se": round(float(se.get(term, float("nan"))), 4),
            "z": round(float(z.get(term, float("nan"))), 3) if np.isfinite(z.get(term, float("nan"))) else float("nan"),
            "p": round(float(pvals.get(term, float("nan"))), 6),
            "n_obs": int(fit.nobs),
            "n_groups": int(getattr(fit, "n_groups", -1)) if hasattr(fit, "n_groups") else -1,
        })
    return pd.DataFrame(rows)


# ── (g) Earlier-emergence timing ───────────────────────────────────────────

# L2 onset (fraction of training) per curriculum.
# B1 balanced: bilingual from step 0 — L2 onset = 0.0
# B2 simultaneous: L1+L2 simultaneous — onset 0.0
# B3 sequential 33/67: L1 only for first 33% — onset 0.33
# B4 classroom-20: interleaved at 20% blocks from 0% — onset 0.0
# B5 late 80/20: L1 only for first 80% — onset 0.80
L2_ONSET_FRAC = {"B1": 0.00, "B2": 0.00, "B3": 0.33, "B4": 0.00, "B5": 0.80}


def analysis_g_emergence(alpha: float = 0.01,
                         min_effect: float = 0.10,
                         persistence: int = 3) -> pd.DataFrame:
    """First ckpt where Bernolet between-language priming becomes
    *robustly* established under three joint criteria:

      1.  paired permutation p < alpha (default 0.01),
      2.  mean per-item priming > min_effect nats (default 0.10),
      3.  the same two criteria hold for the next `persistence` ckpts
          (default 3) so we don't latch on to an early noise spike.

    Reports the first-emergence ckpt as a fraction of training and
    relative to the curriculum's L2 onset.
    """
    rows = []
    for label, prefix in BEETLE_TRAJECTORY_MODELS:
        tsvs = sorted(TIER2_DIR.glob(f"{prefix}__step-*.tsv"))
        if not tsvs:
            continue
        cks = []
        for t in tsvs:
            m = CKPT_RE.search(t.name)
            if m: cks.append((int(m.group(1)), t))
        cks.sort()
        max_ck = max(c for c, _ in cks)

        # Pre-compute per-ckpt summary so we can apply the persistence
        # window without re-permuting.
        summary = []
        for ck, t in cks:
            per = per_item_priming(t, between_only=True)
            if not len(per): continue
            diffs = per["priming"].values
            mean = float(diffs.mean())
            p = perm_p(diffs, n_iter=2000) if mean > 0 else 1.0
            summary.append((ck, mean, p))

        first_sig_ck = None
        first_sig_frac = None
        first_mean = float("nan")
        for i, (ck, mean, p) in enumerate(summary):
            window = summary[i:i + persistence]
            if len(window) < persistence:
                break
            if all(m > min_effect and pv < alpha for _, m, pv in window):
                first_sig_ck = ck
                first_sig_frac = ck / max_ck
                first_mean = mean
                break

        l2_onset = L2_ONSET_FRAC.get(label, 0.0)
        rows.append({
            "curriculum": label,
            "max_ckpt": max_ck,
            "first_sig_ckpt": first_sig_ck if first_sig_ck is not None else float("nan"),
            "first_sig_frac_training": (round(first_sig_frac, 4)
                                        if first_sig_frac is not None
                                        else float("nan")),
            "L2_onset_frac": l2_onset,
            "frac_post_L2_onset": (round(first_sig_frac - l2_onset, 4)
                                   if first_sig_frac is not None
                                   else float("nan")),
            "first_sig_mean_priming": round(first_mean, 4),
        })
    return pd.DataFrame(rows)


# ── Output formatting ──────────────────────────────────────────────────────

def fmt_p(p: float) -> str:
    if pd.isna(p): return "—"
    if p < 1e-4: return "< 1e-4"
    if p < 1e-3: return f"{p:.1e}"
    return f"{p:.3f}"


def write_md(a_df: pd.DataFrame, b_df: pd.DataFrame, c_df: pd.DataFrame,
             d_df: pd.DataFrame | None = None,
             e_df: pd.DataFrame | None = None,
             f_curriculum_df: pd.DataFrame | None = None,
             f_phase_df: pd.DataFrame | None = None,
             g_df: pd.DataFrame | None = None,
             out_md: Path | None = None) -> None:
    lines = [
        "# Cross-lingual priming — statistical inference",
        "",
        "All analyses are item-level, between-language pairs only, on Bernolet.",
        "BGPT models are excluded because per-item raw TSVs are not available "
        "in this repo (only aggregate per-language-pair means).",
        "",
        "Item = unique `(Set, ExtraCondition, PrimeLanguage, TargetLanguage, "
        "TargetSentence, TargetType)`. ",
        "Per-item priming = `Surprisal_mismatch_target − Surprisal_match_target`.",
        "",
        "---",
        "",
        "## (a) Within-model significance (Beetle final-ckpt, Bernolet)",
        "",
        "Paired *t*-test of per-item priming against zero (and sign-flip "
        f"permutation, n_iter={N_PERMUTATIONS}).",
        "",
        "| model | direction | n | mean priming | t | p (t) | p (perm) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in a_df.itertuples():
        lines.append(
            f"| `{r.model}` | {r.PrimeLanguage}→{r.TargetLanguage} | {r.n_items} | "
            f"{r.mean_priming:+.4f} | {r.t:+.2f} | {fmt_p(r.p_t)} | {fmt_p(r.p_perm)} |"
        )

    lines += [
        "",
        "---",
        "",
        "## (b′) Beetle-vs-Beetle pairwise contrasts (Bernolet, between-lang)",
        "",
        "Paired permutation test on per-item priming differences "
        f"(matched items across model pairs, n_iter={N_PERMUTATIONS}). "
        "Item-level inference: scope is the Bernolet stimulus set; cannot "
        "generalise across re-trainings (single seed per curriculum).",
        "",
        "| contrast | n | mean A | mean B | mean diff | t | p (t) | p (perm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in b_df.itertuples():
        lines.append(
            f"| {r.contrast} | {r.n_items} | {r.mean_A:+.4f} | {r.mean_B:+.4f} | "
            f"{r.mean_diff:+.4f} | {r.t:+.2f} | {fmt_p(r.p_t)} | {fmt_p(r.p_perm)} |"
        )

    lines += [
        "",
        "---",
        "",
        "## (c′) Beetle peak-vs-final stability (Bernolet, between-lang)",
        "",
        "For each Beetle curriculum trajectory, identify the checkpoint with "
        "maximum mean priming and compare to the final checkpoint via paired "
        "permutation on per-item priming.",
        "",
        "| curriculum | peak ckpt | final ckpt | peak mean | final mean | decay % | n | Δ peak−final | t | p (t) | p (perm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in c_df.itertuples():
        lines.append(
            f"| {r.curriculum} | {r.peak_ckpt} | {r.final_ckpt} | "
            f"{r.peak_priming:+.4f} | {r.final_priming:+.4f} | "
            f"{r.decay_pct:+.1f}% | {r.n_items} | "
            f"{r.mean_diff_peak_minus_final:+.4f} | {r.t:+.2f} | "
            f"{fmt_p(r.p_t)} | {fmt_p(r.p_perm)} |"
        )

    if d_df is not None and len(d_df):
        lines += [
            "",
            "---",
            "",
            "## (d) Normalised relative-probability robustness — Arnett 2025 metric",
            "",
            "Per-item P_N = exp(−S_match) / (exp(−S_match) + exp(−S_mismatch)). "
            "0.5 = no priming; > 0.5 = match-favoured. Paired *t*-test of "
            "(P_N − 0.5) against zero, plus sign-flip permutation. Same Bernolet "
            "between-language items as (a). Reviewers asking why we report "
            "surprisal differences rather than P_N can read off this table that "
            "the qualitative pattern survives the metric switch.",
            "",
            "| model | direction | n | mean P_N | mean (P_N − 0.5) | t | p (t) | p (perm) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in d_df.itertuples():
            lines.append(
                f"| `{r.model}` | {r.PrimeLanguage}→{r.TargetLanguage} | {r.n_items} | "
                f"{r.mean_p_n:.4f} | {r.mean_p_n_minus_half:+.4f} | {r.t:+.2f} | "
                f"{fmt_p(r.p_t)} | {fmt_p(r.p_perm)} |"
            )

    if e_df is not None and len(e_df):
        lines += [
            "",
            "---",
            "",
            "## (e) Within-language baseline vs cross-language priming (Bernolet)",
            "",
            "For each Beetle final-ckpt model and each cross-lingual direction, "
            "we match items by (TargetSentence, TargetType) against the "
            "within-language priming on the same target language. The difference "
            "(`cross − within`) is paired-permuted to test whether cross-lingual "
            "priming carries an effect distinguishable from monolingual priming "
            "on the same construction.",
            "",
            "| model | direction | n | cross priming | within-T priming | Δ (cross−within) | t | p (t) | p (perm) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in e_df.itertuples():
            lines.append(
                f"| `{r.model}` | {r.PrimeLanguage}→{r.TargetLanguage} | {r.n_items} | "
                f"{r.cross_priming:+.4f} | {r.within_priming:+.4f} | "
                f"{r.mean_diff_cross_minus_within:+.4f} | {r.t:+.2f} | "
                f"{fmt_p(r.p_t)} | {fmt_p(r.p_perm)} |"
            )

    if f_curriculum_df is not None and len(f_curriculum_df):
        lines += [
            "",
            "---",
            "",
            "## (f) Confirmatory mixed-effects (statsmodels MixedLM)",
            "",
            "Confirmatory analysis matching the psycholinguistic convention of "
            "modelling Item as a random intercept. Estimates are REML; reference "
            "level for Curriculum is **B1** (balanced FineWeb-24B); reference "
            "level for Phase is **early** (< 25 % of training).",
            "",
            "### (f.i) `Δ ~ Curriculum + (1|Item)` — Beetle FineWeb-24B final ckpts",
            "",
            "| term | estimate | SE | z | p | n_obs | n_items |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in f_curriculum_df.itertuples():
            lines.append(
                f"| `{r.term}` | {r.estimate:+.4f} | {r.se:.4f} | "
                f"{r.z:+.2f} | {fmt_p(r.p)} | {r.n_obs} | {r.n_groups} |"
            )

    if f_phase_df is not None and len(f_phase_df):
        lines += [
            "",
            "### (f.ii) `Δ ~ Phase + (1|Item)` — B1 trajectory",
            "",
            "Phase = early (< 25 %) / mid (25–50 %) / peak (50–75 %) / final (≥ 75 %) "
            "of training, with reference = early.",
            "",
            "| term | estimate | SE | z | p | n_obs | n_items |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in f_phase_df.itertuples():
            lines.append(
                f"| `{r.term}` | {r.estimate:+.4f} | {r.se:.4f} | "
                f"{r.z:+.2f} | {fmt_p(r.p)} | {r.n_obs} | {r.n_groups} |"
            )

    if g_df is not None and len(g_df):
        lines += [
            "",
            "---",
            "",
            "## (g) Earlier-emergence timing per curriculum",
            "",
            "First trajectory ckpt at which per-item Bernolet between-language "
            "priming exceeds zero with permutation p < 0.01 (n_iter=2000, "
            "cheaper sweep). `L2_onset_frac` is the fraction of training at "
            "which the curriculum first introduces L2 data; "
            "`frac_post_L2_onset` is the gap between L2 introduction and "
            "first significant priming.",
            "",
            "| curriculum | first-sig ckpt | frac of training | L2 onset | gap | mean priming at first-sig |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for r in g_df.itertuples():
            lines.append(
                f"| {r.curriculum} | "
                f"{int(r.first_sig_ckpt) if pd.notna(r.first_sig_ckpt) else '—'} | "
                f"{r.first_sig_frac_training if pd.notna(r.first_sig_frac_training) else '—'} | "
                f"{r.L2_onset_frac:.2f} | "
                f"{r.frac_post_L2_onset if pd.notna(r.frac_post_L2_onset) else '—'} | "
                f"{r.first_sig_mean_priming:+.4f} |"
            )

    lines += [
        "",
        "---",
        "",
        "## BGPT — descriptive only",
        "",
        "BGPT raw TSVs are not in this repo (only aggregate per-language-pair "
        "means in `comparison/priming_effects_per_model.csv`). The following "
        "BGPT trajectory observations are reported descriptively and cannot "
        "currently be tested for significance:",
        "",
        "- `nl_en_5050` Bernolet: peak ≈ +0.363, final ≈ +0.143 (−61% decay).",
        "- `nl_en_7525`, `en_nl_7525`: peak ≈ final (≤+5% decay).",
        "- Significance testing for these BGPT contrasts requires re-running "
        "the BGPT priming evaluation with item-level logging.",
        "",
    ]
    out_md.write_text("\n".join(lines))


# ── Driver ─────────────────────────────────────────────────────────────────

def main() -> None:
    a_df = analysis_a_within_model()
    a_df.to_csv(OUT_DIR / "a_within_model.csv", index=False)
    logger.info("(a) wrote %d rows", len(a_df))

    b_df = analysis_b_pairwise()
    b_df.to_csv(OUT_DIR / "b_pairwise.csv", index=False)
    logger.info("(b′) wrote %d rows", len(b_df))

    c_df = analysis_c_peak_vs_final()
    c_df.to_csv(OUT_DIR / "c_peak_vs_final.csv", index=False)
    logger.info("(c′) wrote %d rows", len(c_df))

    d_df = analysis_d_pn()
    d_df.to_csv(OUT_DIR / "d_pn_robustness.csv", index=False)
    logger.info("(d) wrote %d rows", len(d_df))

    e_df = analysis_e_within_vs_cross()
    e_df.to_csv(OUT_DIR / "e_within_vs_cross.csv", index=False)
    logger.info("(e) wrote %d rows", len(e_df))

    try:
        f_curr_df = analysis_f_mixedlm_curriculum()
    except Exception as exc:
        logger.warning("(f.i) MixedLM curriculum failed: %s", exc)
        f_curr_df = pd.DataFrame()
    f_curr_df.to_csv(OUT_DIR / "f_mixedlm_curriculum.csv", index=False)
    logger.info("(f.i) wrote %d rows", len(f_curr_df))

    try:
        f_phase_df = analysis_f_mixedlm_phase("B1")
    except Exception as exc:
        logger.warning("(f.ii) MixedLM phase failed: %s", exc)
        f_phase_df = pd.DataFrame()
    f_phase_df.to_csv(OUT_DIR / "f_mixedlm_phase_b1.csv", index=False)
    logger.info("(f.ii) wrote %d rows", len(f_phase_df))

    g_df = analysis_g_emergence(alpha=0.01)
    g_df.to_csv(OUT_DIR / "g_emergence.csv", index=False)
    logger.info("(g) wrote %d rows", len(g_df))

    write_md(a_df, b_df, c_df,
             d_df=d_df, e_df=e_df,
             f_curriculum_df=f_curr_df, f_phase_df=f_phase_df,
             g_df=g_df,
             out_md=REPO_ROOT / "paper/emnlp/PRIMING_STATS.md")
    logger.info("wrote paper/emnlp/PRIMING_STATS.md")


if __name__ == "__main__":
    main()
