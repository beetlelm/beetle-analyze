"""
analyze/meco_l1_loader.py — MECO L1 release loader.

The existing pipeline (`meco_surprisal.py`) consumes only the L2 release at
`data/meco_l2_stims.tsv`.  Paper-A RQ-A1 needs *first-language* eye-tracking
data per language: this module builds an analogous TSV from the public MECO
L1 wave 1 release (Siegelman et al. 2022) — the per-language fixation files
in JOINT_DATA / SUPP_TEXT_format.

Each L1 stimulus row is normalized to the same schema as `meco_l2_stims.tsv`:

    word_id          unique within stim/sentence
    sent_id          sentence index within passage
    word             surface form (target)
    FullText         the (un-marked) sentence
    FullTextMarked   sentence with target word delimited by * for minicons
    RT               reading time (first-pass dur, ms)  — averaged across subjects
    n_subjects       count of subjects contributing to RT
    language         ISO-639-3

Loader handles per-language CSV variants in MECO L1 since the column names
are not perfectly uniform.  An offline build is the recommended workflow:

    python -m analyze.meco_l1_loader \
        --raw-root /path/to/meco_l1/JOINT_DATA \
        --out      /path/to/beetle-analyze/data/meco_l1_stims.tsv

The result drops into the existing `meco_surprisal.score_checkpoint()`
pipeline simply by passing ``--stimuli data/meco_l1_stims.tsv``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "meco_l1_stims.tsv"

# Map MECO's two-letter language codes to ISO-639-3 used elsewhere in Beetle.
LANG_MAP = {
    "en": "eng", "ge": "deu", "it": "ita", "du": "nld",
    "sp": "spa", "ru": "rus", "tr": "tur", "no": "nor",
    "fi": "fin", "ee": "est", "ko": "kor", "he": "heb",
    "gr": "ell", "po": "pol", "pt": "por", "zh": "zho",
}


# ---------------------------------------------------------------------------
# Per-language ingest
# ---------------------------------------------------------------------------

def _read_lang_csv(path: Path) -> pd.DataFrame:
    """Read a single MECO L1 per-language CSV and normalise column names."""
    df = pd.read_csv(path)
    # Lower-case column lookup for robustness against MECO's mixed casing.
    lc = {c.lower(): c for c in df.columns}
    def col(*names):
        for n in names:
            if n in lc:
                return lc[n]
        return None

    word_col   = col("ia", "word", "ia_label", "ia_name")
    rt_col     = col("dur", "first_pass_dur", "fpd", "fpf_dur", "rt")
    text_col   = col("text", "passage", "sent", "sentence")
    sent_col   = col("sent_n", "sentnum", "sent_id", "sentence_id")
    word_idx   = col("ianum", "ia_num", "wordnum", "word_id")
    subj_col   = col("subj", "subject", "subjid")

    if not (word_col and rt_col and text_col):
        raise ValueError(
            f"[meco_l1] {path.name}: missing required columns "
            f"(found: {list(df.columns)})"
        )

    out = pd.DataFrame({
        "word":     df[word_col].astype(str),
        "RT":       pd.to_numeric(df[rt_col], errors="coerce"),
        "FullText": df[text_col].astype(str),
        "sent_id":  df[sent_col].astype(int) if sent_col else 0,
        "word_id":  df[word_idx].astype(int) if word_idx else df.index,
        "subj":     df[subj_col].astype(str) if subj_col else "0",
    })
    return out


def _aggregate_subjects(df: pd.DataFrame) -> pd.DataFrame:
    """Average RT across subjects per (sent_id, word_id, word)."""
    agg = (df.groupby(["sent_id", "word_id", "word", "FullText"])
             .agg(RT=("RT", "mean"), n_subjects=("subj", "nunique"))
             .reset_index())
    return agg


def _mark_target(text: str, word: str, occ: int = 1) -> str:
    """Insert *…* delimiters around the `occ`-th occurrence of `word` in `text`.

    Mirrors the FullTextMarked convention used by `meco_surprisal._get_surprisal`.
    """
    parts = text.split(word)
    if len(parts) <= occ:
        return text  # word not found; minicons will return NaN
    pre  = word.join(parts[:occ])
    post = word.join(parts[occ:])
    return f"{pre}*{word}*{post}"


def build_meco_l1_tsv(
    raw_root: Path,
    out_path: Path = DEFAULT_OUT,
    languages: Optional[list[str]] = None,
) -> Path:
    """Convert MECO L1 fixation CSVs under `raw_root` to a unified TSV.

    `raw_root` is assumed to contain per-language folders or files named
    after the MECO short codes (en, ge, it, …).  Both layouts are searched.

    Languages restricts the output if provided.
    """
    raw_root = Path(raw_root)
    candidates: dict[str, Path] = {}
    for code in LANG_MAP:
        sub = raw_root / code
        if sub.is_dir():
            csvs = sorted(sub.glob("*.csv"))
            if csvs:
                candidates[code] = csvs[0]
                continue
        flat = list(raw_root.glob(f"*{code}*.csv"))
        if flat:
            candidates[code] = flat[0]

    if not candidates:
        raise FileNotFoundError(f"No MECO L1 csvs found under {raw_root}")

    keep = languages or list(LANG_MAP.values())
    keep_iso = set(keep)

    rows: list[pd.DataFrame] = []
    for code, csv_path in candidates.items():
        iso = LANG_MAP[code]
        if iso not in keep_iso:
            continue
        df = _aggregate_subjects(_read_lang_csv(csv_path))
        df["language"] = iso
        df["FullTextMarked"] = df.apply(
            lambda r: _mark_target(r["FullText"], r["word"]),
            axis=1,
        )
        rows.append(df)
        print(f"[meco_l1] {iso} ← {csv_path.name}: {len(df)} word-level rows")

    if not rows:
        raise RuntimeError("[meco_l1] No language matched the languages filter.")

    combined = pd.concat(rows, ignore_index=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, sep="\t", index=False)
    print(f"[meco_l1] Wrote {len(combined)} rows → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Loader for downstream code
# ---------------------------------------------------------------------------

def load_meco_l1(
    path: Path = DEFAULT_OUT,
    language: Optional[str] = None,
) -> pd.DataFrame:
    """Load the prebuilt L1 TSV; optionally filter to a single ISO-639-3 lang."""
    df = pd.read_csv(path, sep="\t")
    if language:
        df = df[df["language"] == language].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build the MECO L1 stimulus TSV.")
    p.add_argument("--raw-root", required=True,
                   help="MECO L1 release root (containing per-language CSVs).")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--languages", nargs="+", default=None,
                   help="ISO-639-3 filter, e.g. eng deu ita")
    args = p.parse_args()
    build_meco_l1_tsv(Path(args.raw_root), Path(args.out), args.languages)
