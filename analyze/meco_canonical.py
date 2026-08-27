"""
analyze/meco_canonical.py — Canonical MECO ΔlogLik pipeline.

Python port of the `beetlelm-rts-private/stats/meco_l2.Rmd` `lmer()` pipeline
(also present verbatim in `beetle-analyze/analyze/meco_l2.Rmd`). The canonical
per-(model, language) metric is

    ΔlogLik(model, lang) = logLik(full) − logLik(baseline)

where both models are fit on the joined MECO L2 wave-1 + wave-2 data, boundary
words per (itemid, sentnum) dropped, subset to the canonical L2 language codes
(du, ch_s, ge), with these fixed effects:

    full:      firstrun.dur ~ scale(Surprisal) + scale(Surprisal1p) + scale(Surprisal2p)
                             + scale(wordnum)
                             + scale(wlen)  + scale(wlen1p)  + scale(wlen2p)
                             + scale(freq)  + scale(freq1p)  + scale(freq2p)
                             + (1 + <all above> || subid)
    baseline:  firstrun.dur ~ scale(wordnum)
                             + scale(wlen)  + scale(wlen1p)  + scale(wlen2p)
                             + scale(freq)  + scale(freq1p)  + scale(freq2p)
                             + (1 + <all above> || subid)

Both fit with REML=F. Uncorrelated random slopes (lme4's `||`) are approximated
in `statsmodels.mixedlm` via one `vc_formula` entry per slope term; this is the
closest native statsmodels equivalent. If `pymer4` is importable a true `lme4`
fit is used instead, for bit-exact parity with the R pipeline.

Outputs a long-form dataframe with columns:

    model, lang, delta_loglik, loglik_full, loglik_baseline,
    beta_surprisal, beta_surprisal_se, n_obs, n_subjects, fit_backend, notes

cached under `results/meco/canonical_deltalogl.csv`.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# Canonical L2 language codes (Dutch, Mandarin-Simplified, German) — the only
# three subsets the canonical Rmd ever fits. Anything else is out of scope.
CANONICAL_LANGS: tuple[str, ...] = ("du", "ch_s", "ge")

# Columns produced by the canonical R loader.
_W1_RDA = "data/meco_l2/joint_data_l2_trimmed_version2.2.rda"
_W2_RDA = "data/meco_l2/joint_data_trimmed_L2_wave2_version2.2.rda"

_CACHE_PATH = Path("results/meco/canonical_deltalogl.csv")

# Fixed-effect term lists (pre-scaled).
_CONTROL_VARS = [
    "wordnum",
    "wlen", "wlen1p", "wlen2p",
    "freq", "freq1p", "freq2p",
]
_SURPRISAL_VARS = ["Surprisal", "Surprisal1p", "Surprisal2p"]
_ALL_VARS_FULL = _SURPRISAL_VARS + _CONTROL_VARS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_joined_rt(
    w1_path: str | Path = _W1_RDA,
    w2_path: str | Path = _W2_RDA,
) -> pd.DataFrame:
    """Replicate the canonical R join:

        all_data = w1 %>% add_row(w2 %>% select(-supplementary_id))

    Returns a dataframe with the `firstrun.dur` column renamed to
    `firstrun_dur` (Python-safe) and with `itemid` / `wordnum` normalised.
    """
    import rdata

    w1 = list(rdata.read_rda(str(w1_path)).values())[0]
    w2 = list(rdata.read_rda(str(w2_path)).values())[0]

    # Rdata returns numpy string dtypes for column names — normalise to str.
    w1.columns = [str(c) for c in w1.columns]
    w2.columns = [str(c) for c in w2.columns]

    if "supplementary_id" in w2.columns:
        w2 = w2.drop(columns=["supplementary_id"])

    joined = pd.concat([w1, w2], ignore_index=True, sort=False)

    rename = {"firstrun.dur": "firstrun_dur"}
    joined = joined.rename(columns=rename)

    joined["itemid"] = joined["itemid"].astype(str)
    joined["wordnum"] = pd.to_numeric(joined["wordnum"], errors="coerce")
    joined["sentnum"] = pd.to_numeric(joined["sentnum"], errors="coerce")
    joined["firstrun_dur"] = pd.to_numeric(joined["firstrun_dur"], errors="coerce")
    # Only require wordnum — sentnum is NULL for some L2 wave-2 subsets (e.g.
    # ch_s), in which case drop_sentence_boundaries() falls back to item-level
    # boundary exclusion. Dropping on sentnum would silently eliminate all rows
    # for those language groups.
    joined = joined.dropna(subset=["wordnum"])
    joined["wordnum"] = joined["wordnum"].astype(int)
    # sentnum: keep as nullable float; cast only where non-null.
    mask_sn = joined["sentnum"].notna()
    joined.loc[mask_sn, "sentnum"] = joined.loc[mask_sn, "sentnum"].astype(int)
    return joined


def add_word_features(rt_df: pd.DataFrame) -> pd.DataFrame:
    """Attach `wlen` (character count) and `freq` (Zipf, English) to each row.

    Mirrors the R `nchar(word)` + `wordfreq$zipf_frequency(word, "en")`.
    The Zipf corpus is English for every L2 language — this is the canonical
    choice from `stats/meco_l2.Rmd` and is intentional (the L2 readers are
    reading English text).
    """
    import wordfreq

    out = rt_df.copy()
    out["word"] = out["word"].astype(str)
    out["wlen"] = out["word"].str.len().astype(float)

    uniq = out["word"].drop_duplicates()
    zipf = {w: float(wordfreq.zipf_frequency(w, "en")) for w in uniq}
    out["freq"] = out["word"].map(zipf).astype(float)
    return out


def join_model_surprisal(
    rt_df: pd.DataFrame,
    surprisal_df: pd.DataFrame,
    model: str,
) -> pd.DataFrame:
    """Restrict `rt_df` to one model's per-word surprisals, joining on
    (itemid, wordnum). Returns a new frame with column `Surprisal`.
    """
    sub = (surprisal_df[surprisal_df["model"] == model]
           [["itemid", "wordnum", "surprisal"]]
           .rename(columns={"surprisal": "Surprisal"})
           .copy())
    sub["itemid"] = sub["itemid"].astype(str)
    sub["wordnum"] = sub["wordnum"].astype(int)
    sub = sub.drop_duplicates(subset=["itemid", "wordnum"], keep="last")
    return rt_df.merge(sub, on=["itemid", "wordnum"], how="inner")


def attach_spillover(df: pd.DataFrame) -> pd.DataFrame:
    """Build Surprisal1p / Surprisal2p / wlen1p / wlen2p / freq1p / freq2p.

    Replicates the R `get_prev()` helper: for each (itemid, wordnum) row we
    attach the values one and two words earlier **in the same item**. Rows
    whose spillover context is incomplete are dropped (inner join).
    """
    cols = ["itemid", "wordnum", "Surprisal", "wlen", "freq"]
    base = df[cols].drop_duplicates().copy()

    back1 = (base.rename(columns={"Surprisal": "Surprisal1p",
                                  "wlen": "wlen1p", "freq": "freq1p"})
                 .assign(wordnum=lambda d: d["wordnum"] - 1))
    back1 = back1[back1["wordnum"] >= 1]

    back2 = (base.rename(columns={"Surprisal": "Surprisal2p",
                                  "wlen": "wlen2p", "freq": "freq2p"})
                 .assign(wordnum=lambda d: d["wordnum"] - 2))
    back2 = back2[back2["wordnum"] >= 1]

    out = (df.merge(back1, on=["itemid", "wordnum"], how="inner")
             .merge(back2, on=["itemid", "wordnum"], how="inner")
             .drop_duplicates())
    return out


def drop_sentence_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate R's `filter(wordnum != min(wordnum)) %>% filter(wordnum != max(wordnum))`
    grouped by (itemid, sentnum).

    When sentnum is missing (NaN) for a row — as in the wave-2 ch_s subset —
    we fall back to item-level boundary exclusion, grouping only by itemid.
    This is the closest approximation available without sentence segmentation.
    """
    out = df.copy()
    has_sn = out["sentnum"].notna()

    # Rows WITH sentnum: canonical sentence-level exclusion.
    if has_sn.any():
        part_sn = out[has_sn]
        g = part_sn.groupby(["itemid", "sentnum"])["wordnum"]
        mins = g.transform("min")
        maxs = g.transform("max")
        keep_sn = (part_sn["wordnum"] != mins) & (part_sn["wordnum"] != maxs)
        out_sn = part_sn[keep_sn]
    else:
        out_sn = out.iloc[:0]  # empty

    # Rows WITHOUT sentnum: item-level boundary exclusion (first/last word per item).
    if (~has_sn).any():
        part_no = out[~has_sn]
        g2 = part_no.groupby("itemid")["wordnum"]
        mins2 = g2.transform("min")
        maxs2 = g2.transform("max")
        keep_no = (part_no["wordnum"] != mins2) & (part_no["wordnum"] != maxs2)
        out_no = part_no[keep_no]
    else:
        out_no = out.iloc[:0]  # empty

    return pd.concat([out_sn, out_no], ignore_index=True)


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def _zscale(x: pd.Series) -> pd.Series:
    """R's `scale()` default (mean 0, sd 1 using n-1 denominator)."""
    x = x.astype(float)
    mu = x.mean()
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sd


def _prepare_scaled(df: pd.DataFrame, *, include_surprisal: bool) -> pd.DataFrame:
    """Return a new frame with `sc_<var>` z-scored copies of every fixed term."""
    vars_ = (_ALL_VARS_FULL if include_surprisal else _CONTROL_VARS)
    out = df.copy()
    for v in vars_:
        out[f"sc_{v}"] = _zscale(out[v])
    # firstrun_dur stays on its natural ms scale (matches R).
    return out


def _extract_surp_params(res, *, include_surprisal: bool) -> tuple[float, float]:
    """Pull β_Surprisal and its SE from a fitted MixedLMResults."""
    if not include_surprisal:
        return float("nan"), float("nan")
    b = float(res.fe_params.get("sc_Surprisal", np.nan))
    bse_attr = "bse_fe" if hasattr(res, "bse_fe") else "bse"
    se_src = getattr(res, bse_attr, None)
    se = float(se_src.get("sc_Surprisal", np.nan)) if se_src is not None else float("nan")
    return b, se


def _fit_with_statsmodels(
    df: pd.DataFrame, *, include_surprisal: bool,
) -> tuple[float, float, float, str]:
    """Fit the canonical mixed model under statsmodels.

    Uncorrelated random slopes (lme4's ``||``) are emulated via one
    ``vc_formula`` entry per slope term. We try three tiers of
    random-effects complexity in order, retaining whichever converges:

      1. Full vc (all fixed terms as uncorrelated random slopes) — exact
         match to the canonical ``(1 + <same> || subid)`` spec.
      2. Reduced vc (surprisal slopes only) — preserves the key parameter
         estimation while reducing the variance-component space.
      3. Random-intercept only — last resort; noted in ``fit_backend``.

    Within each tier we try three optimizers (lbfgs, bfgs, powell) and
    keep the solution with the highest log-likelihood.

    Returns (loglik, beta_surprisal, beta_surprisal_se, notes).
    """
    import statsmodels.formula.api as smf

    scaled = _prepare_scaled(df, include_surprisal=include_surprisal)
    vars_ = (_ALL_VARS_FULL if include_surprisal else _CONTROL_VARS)
    fixed_terms = " + ".join(f"sc_{v}" for v in vars_)
    formula = f"firstrun_dur ~ {fixed_terms}"

    _METHODS = ("lbfgs", "bfgs", "powell")
    # ``BEETLE_FAST_FIT=1`` opts out of the expensive Tier-1 attempt (full
    # uncorrelated random slopes on every fixed term) and of the multi-
    # optimizer search inside ``_best_fit``. Tier-1 attempts cost ~2 min each
    # and frequently fail on subject pools <30; skipping them is roughly a
    # 4× speedup for the orchestrated sweep.
    _FAST = os.environ.get("BEETLE_FAST_FIT") == "1"
    if _FAST:
        _METHODS = ("lbfgs",)

    def _best_fit(md) -> tuple[object, float]:
        """Try multiple optimizers; return (result, llf) of the best."""
        best_res, best_ll = None, -np.inf
        for meth in _METHODS:
            try:
                res = md.fit(reml=False, method=meth, disp=False)
                ll = float(res.llf)
                if np.isfinite(ll) and ll > best_ll:
                    best_ll, best_res = ll, res
            except Exception:
                pass
        return best_res, best_ll

    if not _FAST:
        # Tier 1: full uncorrelated random slopes.
        vc_full = {f"rs_{v}": f"0 + sc_{v}" for v in vars_}
        try:
            md = smf.mixedlm(formula, data=scaled, groups=scaled["subid"],
                             re_formula="1", vc_formula=vc_full)
            res, ll = _best_fit(md)
            if res is not None and np.isfinite(ll):
                b, se = _extract_surp_params(res, include_surprisal=include_surprisal)
                return ll, b, se, "statsmodels|re=1+vc(||)"
        except Exception:
            pass

    # Tier 2: surprisal-only uncorrelated random slopes.
    if include_surprisal:
        vc_surp = {f"rs_{v}": f"0 + sc_{v}" for v in _SURPRISAL_VARS}
        try:
            md = smf.mixedlm(formula, data=scaled, groups=scaled["subid"],
                             re_formula="1", vc_formula=vc_surp)
            res, ll = _best_fit(md)
            if res is not None and np.isfinite(ll):
                b, se = _extract_surp_params(res, include_surprisal=include_surprisal)
                return ll, b, se, "statsmodels|re=1+vc(surp-only)"
        except Exception:
            pass

    # Tier 3: random intercept only.
    try:
        md = smf.mixedlm(formula, data=scaled, groups=scaled["subid"], re_formula="1")
        res, ll = _best_fit(md)
        if res is not None and np.isfinite(ll):
            b, se = _extract_surp_params(res, include_surprisal=include_surprisal)
            return ll, b, se, "statsmodels|re=1 only"
    except Exception as e:
        raise RuntimeError(f"all three statsmodels tiers failed: {e}") from e

    raise RuntimeError("no statsmodels tier produced a finite log-likelihood")


def _fit_with_pymer4(
    df: pd.DataFrame, *, include_surprisal: bool,
) -> tuple[float, float, float, str]:
    """True lme4 parity via pymer4. Silent ImportError if unavailable."""
    from pymer4.models import Lmer  # noqa: F401

    scaled = _prepare_scaled(df, include_surprisal=include_surprisal)
    vars_ = (_ALL_VARS_FULL if include_surprisal else _CONTROL_VARS)
    fixed = " + ".join(f"sc_{v}" for v in vars_)
    slopes = " + ".join(f"sc_{v}" for v in vars_)
    formula = f"firstrun_dur ~ {fixed} + (1 + {slopes} || subid)"

    m = Lmer(formula, data=scaled)
    m.fit(REML=False, summarize=False)
    ll = float(m.logLike)
    if include_surprisal and "sc_Surprisal" in m.coefs.index:
        row = m.coefs.loc["sc_Surprisal"]
        b = float(row["Estimate"])
        se = float(row["SE"])
    else:
        b, se = float("nan"), float("nan")
    return ll, b, se, "pymer4|lme4 parity"


def fit_canonical_pair(
    df: pd.DataFrame,
    *,
    backend: str = "auto",
) -> dict:
    """Fit both (full + baseline) canonical models on one (model, lang) subset.

    ``backend`` is one of "auto" (pymer4 if importable else statsmodels),
    "pymer4", or "statsmodels".
    """
    use_pymer = backend == "pymer4" or (backend == "auto" and _pymer4_available())
    fit_fn = _fit_with_pymer4 if use_pymer else _fit_with_statsmodels

    ll_full, b_surp, b_surp_se, notes_full = fit_fn(df, include_surprisal=True)
    ll_base, _, _, notes_base = fit_fn(df, include_surprisal=False)
    return {
        "loglik_full": ll_full,
        "loglik_baseline": ll_base,
        "delta_loglik": ll_full - ll_base,
        "beta_surprisal": b_surp,
        "beta_surprisal_se": b_surp_se,
        "fit_backend": notes_full,
        "fit_backend_baseline": notes_base,
    }


def _pymer4_available() -> bool:
    try:
        import pymer4  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_delta_loglik_all_models(
    surprisal_df: pd.DataFrame,
    *,
    langs: Iterable[str] = CANONICAL_LANGS,
    cache_path: Path = _CACHE_PATH,
    overwrite: bool = False,
    min_rows: int = 200,
    backend: str = "auto",
    verbose: bool = True,
) -> pd.DataFrame:
    """For every model × language in ``surprisal_df`` × ``langs``, fit the
    canonical full + baseline mixed models and return ΔlogLik.

    Results are cached per (model, lang) in ``cache_path``; rerun with
    ``overwrite=True`` to refit everything.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    rt = load_joined_rt()
    rt = add_word_features(rt)

    models = sorted(surprisal_df["model"].dropna().unique().tolist())

    # Load existing cache (if any) and determine which (model, lang) pairs
    # we still need to fit.
    if cache_path.exists() and not overwrite:
        cached = pd.read_csv(cache_path)
    else:
        cached = pd.DataFrame(
            columns=["model", "lang", "delta_loglik", "loglik_full",
                     "loglik_baseline", "beta_surprisal", "beta_surprisal_se",
                     "n_obs", "n_subjects", "fit_backend", "fit_backend_baseline",
                     "notes"]
        )
    done_pairs = set(zip(cached.get("model", []), cached.get("lang", [])))

    accumulated = list(cached.to_dict("records")) if not cached.empty else []

    def _flush(new_row: dict) -> None:
        """Append new_row to the in-memory list and write the full cache atomically."""
        accumulated.append(new_row)
        df = pd.DataFrame(accumulated).drop_duplicates(
            subset=["model", "lang"], keep="last"
        )
        tmp = cache_path.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(cache_path)  # atomic on POSIX

    for model in models:
        merged = join_model_surprisal(rt, surprisal_df, model)
        if merged.empty:
            continue
        for lang in langs:
            if (model, lang) in done_pairs:
                continue
            sub = merged[merged["lang"] == lang].dropna(
                subset=["firstrun_dur", "Surprisal", "wlen", "freq", "subid"]
            )
            sub = attach_spillover(sub)
            sub = drop_sentence_boundaries(sub)
            if len(sub) < min_rows:
                row = {
                    "model": model, "lang": lang,
                    "delta_loglik": np.nan,
                    "loglik_full": np.nan, "loglik_baseline": np.nan,
                    "beta_surprisal": np.nan, "beta_surprisal_se": np.nan,
                    "n_obs": int(len(sub)), "n_subjects": int(sub["subid"].nunique()),
                    "fit_backend": "skipped",
                    "fit_backend_baseline": "skipped",
                    "notes": f"n={len(sub)} < min_rows={min_rows}",
                }
                _flush(row)
                continue

            if verbose:
                print(f"[meco-canonical] fitting {model} @ {lang} "
                      f"(n={len(sub)}, subs={sub['subid'].nunique()})",
                      file=sys.stderr, flush=True)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = fit_canonical_pair(sub, backend=backend)
                row = {
                    "model": model, "lang": lang,
                    **fit,
                    "n_obs": int(len(sub)),
                    "n_subjects": int(sub["subid"].nunique()),
                    "notes": "",
                }
            except Exception as e:
                row = {
                    "model": model, "lang": lang,
                    "delta_loglik": np.nan,
                    "loglik_full": np.nan, "loglik_baseline": np.nan,
                    "beta_surprisal": np.nan, "beta_surprisal_se": np.nan,
                    "n_obs": int(len(sub)),
                    "n_subjects": int(sub["subid"].nunique()),
                    "fit_backend": "error",
                    "fit_backend_baseline": "error",
                    "notes": f"{e.__class__.__name__}: {e}",
                }
            _flush(row)

    combined = pd.DataFrame(accumulated).drop_duplicates(
        subset=["model", "lang"], keep="last"
    )
    combined.to_csv(cache_path, index=False)
    if verbose:
        print(f"[meco-canonical] cache → {cache_path} ({len(combined)} rows)",
              file=sys.stderr)
    return combined


# ---------------------------------------------------------------------------
# Formula string reported in figure/table captions (authoritative).
# ---------------------------------------------------------------------------

CANONICAL_FORMULA = (
    "firstrun.dur ~ scale(Surprisal) + scale(Surprisal1p) + scale(Surprisal2p) "
    "+ scale(wordnum) + scale(wlen) + scale(wlen1p) + scale(wlen2p) "
    "+ scale(freq) + scale(freq1p) + scale(freq2p) "
    "+ (1 + <same> || subid), REML=FALSE"
)

CANONICAL_BASELINE_FORMULA = (
    "firstrun.dur ~ scale(wordnum) + scale(wlen) + scale(wlen1p) + scale(wlen2p) "
    "+ scale(freq) + scale(freq1p) + scale(freq2p) "
    "+ (1 + <same> || subid), REML=FALSE"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    # Local import to keep this module importable without the rest of the
    # analyze package (e.g. from plotting code that only needs the constants).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plots.checkpoint_loaders import load_meco_checkpoint_files  # type: ignore

    ap = argparse.ArgumentParser(description="Fit canonical MECO ΔlogLik pipeline")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "pymer4", "statsmodels"])
    ap.add_argument("--cache-path", default=str(_CACHE_PATH))
    args = ap.parse_args()

    surp = load_meco_checkpoint_files()
    if surp.empty:
        print("[meco-canonical] no MECO surprisal CSVs found; aborting",
              file=sys.stderr)
        sys.exit(2)
    # Reduce to final-step surprisal per model (canonical pipeline uses final checkpoint).
    last_step = surp.groupby("model")["step"].max().rename("_last")
    surp = surp.join(last_step, on="model")
    surp = surp[surp["step"] == surp["_last"]].drop(columns=["_last"])

    out = compute_delta_loglik_all_models(
        surp,
        cache_path=Path(args.cache_path),
        overwrite=args.overwrite,
        backend=args.backend,
    )
    print(out.to_string(index=False))
