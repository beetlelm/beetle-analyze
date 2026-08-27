#!/usr/bin/env python3
"""
evals/eval_meco_english_l2_participants.py — MECO surprisal with L1-matched
participant RT correlation.

Extends evals/eval_meco_english_l2.py by loading the MECO L2 participant-level
reading-time table (``data/meco_l2/joint_data_l2_trimmed_version2.2.rda``) via
``pyreadr``, filtering to participants whose L1 matches the model's L1, and
reporting Spearman / Pearson correlations between model surprisal and the
L1-matched group-mean RT.

Requires ``pyreadr`` (``pip install pyreadr``).

Usage
-----
    python evals/eval_meco_english_l2_participants.py

    python evals/eval_meco_english_l2_participants.py \\
        --rt-columns firstfix.dur gaze.dur totalobs.dur
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_ANALYZE_DIR = _PROJECT_ROOT / "analyze"
if str(_ANALYZE_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYZE_DIR))

from analyze.meco_surprisal import (  # noqa: E402
    DEFAULT_STIMULI,
    DEVICE,
    RESULTS_DIR,
)
from evals.eval_meco_english_l2 import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    IN_SCOPE_L1,
    _L1_DISPLAY,
    _tex_escape,
    load_or_score,
    select_english_l2_models,
)
from models import ALL_MODELS  # noqa: E402


# MECO participant-L1 code → BeetleLM L1 code. MECO wave 1 uses short codes
# (``du``, ``ge``, ``ch``); wave 2 uses suffixed variants (``ch_s``/``ch_t``
# for Simplified/Traditional Chinese, ``ge_po``/``ge_zu`` for the Potsdam and
# Zurich German subgroups). Both waves are merged below — see
# ``load_meco_rda``. Only the three languages covered by BeetleLM
# Human-Scale (nld/deu/zho) are retained.
DEFAULT_MECO_L1_MAP: dict[str, str] = {
    # Dutch (wave 1 only)
    "du": "nld", "nl": "nld", "dut": "nld", "nld": "nld",
    # German (wave 1: ``ge``; wave 2: ``ge_po``=Potsdam, ``ge_zu``=Zurich)
    "ge": "deu", "de": "deu", "ger": "deu", "deu": "deu",
    "ge_po": "deu", "ge_zu": "deu",
    # Chinese (wave 1 export uses ``ch``; wave 2 splits Simplified/Traditional)
    "ch": "zho", "zh": "zho", "zho": "zho",
    "ch_s": "zho", "ch_t": "zho",
}

# Candidate RT columns we will try if the user doesn't pass --rt-columns.
# Only those actually present in the data frame are used.
DEFAULT_RT_COLUMNS: tuple[str, ...] = (
    "dur", "firstfix.dur", "firstrun.dur", "gaze.dur", "totalobs.dur",
    "reread.dur", "refix",
)

# Default RDAs. Wave 1 has Dutch + German participants but no Chinese; wave 2
# has Chinese (ch_s/ch_t) + German (ge_po/ge_zu) but no Dutch. We concatenate
# both so all three in-scope L1 groups (nld/deu/zho) are populated.
DEFAULT_RDA = _PROJECT_ROOT / "data" / "meco_l2" / \
    "joint_data_l2_trimmed_version2.2.rda"
DEFAULT_RDA_WAVE2 = _PROJECT_ROOT / "data" / "meco_l2" / \
    "joint_data_trimmed_L2_wave2_version2.2.rda"


# ---------------------------------------------------------------------------
# .rda loader
# ---------------------------------------------------------------------------

def _read_single_rda(rda_path: Path) -> pd.DataFrame:
    """Read one MECO .rda archive and return the first DataFrame."""
    try:
        import pyreadr
    except ImportError as e:
        raise SystemExit(
            "pyreadr is required for participant-level evaluation. "
            "Install with: pip install pyreadr"
        ) from e
    result = pyreadr.read_r(str(rda_path))
    if not result:
        raise RuntimeError(f"No objects found in {rda_path}")
    # Pick the first (and usually only) frame.
    key = next(iter(result.keys()))
    df = result[key]
    print(f"[meco-l2] loaded {rda_path.name}: key='{key}' "
          f"shape={df.shape} cols={list(df.columns)[:12]}...")
    return df


def load_meco_rda(rda_paths: Path | list[Path]) -> pd.DataFrame:
    """Load one or more MECO L2 RT .rda archives and concatenate them.

    Wave 1 (``joint_data_l2_trimmed_version2.2.rda``) covers participants with
    L1 in {du, ge, ee, fi, gr, he, it, no, ru, sp, tr, en}. Wave 2
    (``joint_data_trimmed_L2_wave2_version2.2.rda``) covers L1 in
    {ba, bp, ch_s, ch_t, da, en_uk, ge_po, ge_zu, hi_*, ic, no, ru_mo, se,
    sp_ch, tr}. Both share the same per-word fixation columns; wave 2 has
    one extra column (``supplementary_id``). We concat with sort=False so
    the union of columns is preserved without forcing a column reordering.
    """
    if isinstance(rda_paths, (str, Path)):
        rda_paths = [Path(rda_paths)]
    else:
        rda_paths = [Path(p) for p in rda_paths]

    frames: list[pd.DataFrame] = []
    for p in rda_paths:
        df = _read_single_rda(p)
        df = df.assign(_meco_wave=p.name)
        frames.append(df)

    if len(frames) == 1:
        return frames[0]

    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(f"[meco-l2] concatenated {len(frames)} waves: "
          f"shape={combined.shape}, "
          f"per-wave row counts={[len(f) for f in frames]}")
    return combined


def _pick_word_index_column(df: pd.DataFrame) -> str:
    """MECO exports word index under several names."""
    for c in ("ianum", "wordnum", "word_index", "word_id"):
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find word-index column in MECO RT frame; looked for "
        f"ianum/wordnum/word_index/word_id. Have: {list(df.columns)}"
    )


def _pick_l1_column(df: pd.DataFrame) -> str:
    for c in ("lang", "l1", "L1", "native_lang", "native_language"):
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find participant-L1 column; looked for "
        f"lang/l1/native_lang. Have: {list(df.columns)}"
    )


def _pick_subject_column(df: pd.DataFrame) -> str:
    for c in ("uniform_id", "subid", "subject", "subj", "id"):
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find subject-id column; looked for "
        f"uniform_id/subid/subject/subj/id. Have: {list(df.columns)}"
    )


def aggregate_rts_by_l1(
    rt_df: pd.DataFrame,
    rt_columns: list[str],
    l1_map: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Return (agg_df, n_subjects_by_l1) where agg_df has columns
    [l1, itemid, wordnum, <rt_col>, ...] holding the group-mean RT over
    participants sharing that L1.
    """
    l1_col = _pick_l1_column(rt_df)
    wi_col = _pick_word_index_column(rt_df)
    subj_col = _pick_subject_column(rt_df)

    if "itemid" not in rt_df.columns:
        raise KeyError(
            f"MECO RT frame missing 'itemid' column. Have: {list(rt_df.columns)}"
        )

    # Normalise L1 codes (strip whitespace, lower-case).
    norm_l1 = rt_df[l1_col].astype(str).str.strip().str.lower()
    mapped = norm_l1.map(l1_map)
    unknown = sorted(set(norm_l1[mapped.isna()].unique()))
    if unknown:
        print(f"[meco-l2] dropping participants with unmapped L1 codes "
              f"(not in scope): {unknown}")
    rt_df = rt_df.assign(_l1_norm=mapped).dropna(subset=["_l1_norm"])

    # Only keep in-scope L1s (nld/deu/zho).
    rt_df = rt_df[rt_df["_l1_norm"].isin(IN_SCOPE_L1)].copy()

    present_rt_cols = [c for c in rt_columns if c in rt_df.columns]
    missing = [c for c in rt_columns if c not in rt_df.columns]
    if missing:
        print(f"[meco-l2] RT columns missing from frame, skipped: {missing}")
    if not present_rt_cols:
        raise RuntimeError(
            f"None of the requested RT columns are present. "
            f"Have: {list(rt_df.columns)}"
        )

    # Coerce RT columns to numeric (MECO stores some as factors).
    for c in present_rt_cols:
        rt_df[c] = pd.to_numeric(rt_df[c], errors="coerce")

    # Subject counts per L1
    n_subjects_by_l1 = {
        l1: int(rt_df.loc[rt_df["_l1_norm"] == l1, subj_col].nunique())
        for l1 in IN_SCOPE_L1
    }

    # Coerce itemid / word-index to int64 so the downstream merge with the
    # surprisal table (which reads itemid/wordnum as int64 from the
    # tab-separated stimuli file) doesn't fail with a dtype mismatch — pyreadr
    # imports R factor / character columns as Python objects.
    rt_df["itemid"] = pd.to_numeric(rt_df["itemid"], errors="coerce") \
        .astype("Int64")
    rt_df[wi_col] = pd.to_numeric(rt_df[wi_col], errors="coerce") \
        .astype("Int64")

    agg = (
        rt_df.dropna(subset=["itemid", wi_col])
        .groupby(["_l1_norm", "itemid", wi_col], dropna=False)[present_rt_cols]
        .mean()
        .reset_index()
        .rename(columns={"_l1_norm": "l1", wi_col: "wordnum"})
    )
    # Drop the nullable Int64 wrappers so plain int64 keys merge cleanly with
    # the surprisal frame.
    agg["itemid"] = agg["itemid"].astype("int64")
    agg["wordnum"] = agg["wordnum"].astype("int64")
    return agg, n_subjects_by_l1


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def correlate_surprisal_with_rt(
    surprisal_df: pd.DataFrame,
    agg_rt: pd.DataFrame,
    l1: str,
    rt_columns: list[str],
) -> dict:
    """
    Merge surprisal with L1-matched RT aggregate and return correlations for
    each RT column that has data.
    """
    rt_l1 = agg_rt[agg_rt["l1"] == l1]
    if rt_l1.empty:
        return {}

    # Keep only columns that actually exist in the aggregate.
    present = [c for c in rt_columns if c in rt_l1.columns]
    if not present:
        return {}

    merged = surprisal_df.merge(
        rt_l1[["itemid", "wordnum", *present]],
        on=["itemid", "wordnum"], how="inner",
    ).dropna(subset=["surprisal"])

    out: dict = {"n_matched": int(len(merged))}
    for col in present:
        sub = merged.dropna(subset=[col])
        if len(sub) < 10:
            out[f"spearman_rho_{col}"] = float("nan")
            out[f"spearman_p_{col}"]   = float("nan")
            out[f"pearson_r_{col}"]    = float("nan")
            out[f"n_{col}"]            = int(len(sub))
            continue
        rho, pval = stats.spearmanr(sub["surprisal"], sub[col])
        try:
            r, _ = stats.pearsonr(sub["surprisal"], sub[col])
        except Exception:
            r = float("nan")
        out[f"spearman_rho_{col}"] = float(rho)
        out[f"spearman_p_{col}"]   = float(pval)
        out[f"pearson_r_{col}"]    = float(r)
        out[f"n_{col}"]            = int(len(sub))
    return out


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

def to_latex_participants_table(
    df: pd.DataFrame, primary_rt: str,
) -> str:
    """Columns: Model | L1 | Mean surprisal | ρ(surp, RT) | N subj."""
    rho_col = f"spearman_rho_{primary_rt}"
    p_col   = f"spearman_p_{primary_rt}"

    lines: list[str] = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        f"Model & L1 & Mean surp. & $\\rho$ (surp, {primary_rt}) "
        f"& $N$ subj. \\\\",
        r"\midrule",
    ]

    # Bold largest |ρ| per L1 block.
    bold_target: dict[str, float] = {}
    for l1 in df["l1"].unique():
        sub = df[df["l1"] == l1]
        if rho_col in sub.columns:
            abs_rho = sub[rho_col].abs()
            if abs_rho.notna().any():
                bold_target[l1] = float(abs_rho.max())

    current_l1: str | None = None
    for _, row in df.iterrows():
        if current_l1 is not None and row["l1"] != current_l1:
            lines.append(r"\midrule")
        current_l1 = row["l1"]

        mean_val = row.get("mean_surprisal", float("nan"))
        rho_val  = row.get(rho_col, float("nan"))
        p_val    = row.get(p_col, float("nan"))

        mean_str = f"{mean_val:.3f}" if pd.notna(mean_val) else "---"
        if pd.notna(rho_val):
            star = ""
            if pd.notna(p_val):
                if p_val < 0.001:
                    star = "***"
                elif p_val < 0.01:
                    star = "**"
                elif p_val < 0.05:
                    star = "*"
            rho_str = f"{rho_val:+.3f}{star}"
            if (
                row["l1"] in bold_target
                and np.isclose(abs(float(rho_val)), bold_target[row["l1"]])
            ):
                rho_str = r"\textbf{" + rho_str + r"}"
        else:
            rho_str = "---"

        n_subj = row.get(f"n_subjects_{row['l1']}", float("nan"))
        n_subj_str = f"{int(n_subj)}" if pd.notna(n_subj) else "---"

        lines.append(
            f"{_tex_escape(row['model'].split('/')[-1])} & "
            f"{_L1_DISPLAY.get(row['l1'], row['l1'])} & "
            f"{mean_str} & {rho_str} & {n_subj_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--stimuli", default=str(DEFAULT_STIMULI))
    ap.add_argument(
        "--rda", nargs="+",
        default=[str(DEFAULT_RDA), str(DEFAULT_RDA_WAVE2)],
        help=("Path(s) to MECO L2 participant-level .rda file(s). Multiple "
              "files are concatenated. Default: wave 1 + wave 2 so that "
              "all three in-scope L1 groups (nld/deu/zho) are populated."),
    )
    ap.add_argument("--surprisal-cache-dir", default=str(RESULTS_DIR))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--no-reuse-cache", action="store_true")
    ap.add_argument(
        "--rt-columns", nargs="+", default=None,
        help=("RT column names to correlate with surprisal. Default: try "
              + " ".join(DEFAULT_RT_COLUMNS) + ", keep those present."),
    )
    ap.add_argument(
        "--primary-rt", default="dur",
        help="RT column used in the headline LaTeX table column.",
    )
    args = ap.parse_args()

    stimuli_path = Path(args.stimuli)
    cache_dir = Path(args.surprisal_cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select models
    requested = args.models if args.models else ALL_MODELS
    selected = select_english_l2_models(requested)
    if not selected:
        print("[meco-l2] No models matched L2=eng, L1 in {nld,deu,zho}.",
              file=sys.stderr)
        return 1

    # Load & aggregate participant RTs (one or more waves, concatenated).
    rda_paths = [Path(p) for p in (
        args.rda if isinstance(args.rda, list) else [args.rda]
    )]
    missing = [p for p in rda_paths if not p.exists()]
    if missing:
        print(f"[meco-l2] RDA not found: {missing}", file=sys.stderr)
        return 2
    rt_full = load_meco_rda(rda_paths)
    rt_columns = list(args.rt_columns) if args.rt_columns \
        else [c for c in DEFAULT_RT_COLUMNS if c in rt_full.columns]
    agg_rt, n_subjects_by_l1 = aggregate_rts_by_l1(
        rt_full, rt_columns, DEFAULT_MECO_L1_MAP,
    )
    print(f"[meco-l2] subjects per L1: {n_subjects_by_l1}")

    # Score / load surprisals, correlate per model
    rows: list[dict] = []
    for repo, l1, cond in selected:
        try:
            df = load_or_score(
                repo, cache_dir, stimuli_path, args.device,
                reuse_cache=not args.no_reuse_cache,
            )
        except Exception as e:
            print(f"[meco-l2]   FAILED {repo}: {e}", file=sys.stderr)
            continue

        valid = df["surprisal"].dropna() if "surprisal" in df.columns \
            else pd.Series(dtype=float)
        row = {
            "model": repo,
            "l1": l1,
            "condition": cond,
            "n_words": int(len(valid)),
            "mean_surprisal": float(valid.mean()) if len(valid) else float("nan"),
            "median_surprisal": float(valid.median()) if len(valid) else float("nan"),
            f"n_subjects_{l1}": n_subjects_by_l1.get(l1, 0),
        }
        corr = correlate_surprisal_with_rt(df, agg_rt, l1, rt_columns)
        row.update(corr)
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        ["l1", "condition", "model"]).reset_index(drop=True)
    summary_path = output_dir / "summary_by_model_participants.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n[meco-l2] Summary CSV → {summary_path}")

    if args.primary_rt not in rt_columns:
        # Fall back to the first available RT column
        args.primary_rt = rt_columns[0] if rt_columns else "dur"
    tex_path = output_dir / "table_english_l2_participants.tex"
    tex_path.write_text(to_latex_participants_table(summary, args.primary_rt))
    print(f"[meco-l2] LaTeX table → {tex_path}")

    print("\n[meco-l2] Summary:")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
