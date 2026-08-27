"""
analyze.meco_detailed._common — Shared loaders + paths for D1–D8.

Centralises:
  • paths to the OSF MECO L2 release (``meco-l2-complete/``)
  • the L1 ISO-2 codes used in the canonical R loader (du, ch_s, ge, …)
  • a richer ``load_joined_rt_full`` that retains the full set of fixation
    measures (firstrun.dur, gaze.dur, tt.dur, refix, reg.in, reg.out, …)
    rather than the canonical-pipeline subset of ``firstrun.dur`` only
  • LEAP-Q proficiency loader
  • comprehension data loader
  • output path helpers
"""

from __future__ import annotations

import functools
import os
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from analyze import _path  # noqa: F401  — repo-root bootstrap


# Conservative default for cpu-bound multiprocessing parallelism. Each worker
# loads a copy of the RT cache (≈60 MB) so we cap at the number of cores the
# host actually has — but never below 1 and never above 16, because the lmer
# fits saturate well before that.
OS_CPU_COUNT = max(1, min(16, (os.cpu_count() or 4)))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
OSF_ROOT = REPO_ROOT / "meco-l2-complete" / "osfstorage" / "release 1.0"
OSF_V11 = OSF_ROOT / "version 1.1"
OSF_V10 = OSF_ROOT / "version 1.0"

OSF_EYE = OSF_V11 / "primary data" / "eye tracking data" / "joint_data_l2_trimmed.rda"
OSF_COMP = OSF_V11 / "primary data" / "comprehension data" / "joint.comp.l2.rda"
OSF_INDDIFF = OSF_V11 / "primary data" / "individual differences data" / "joint.ind.diff.l2.rda"
OSF_LEAPQ_DIR = OSF_V11 / "primary data" / "individual differences data" / "leap-q"

# Wave-2 sits only in version 1.0 alongside the canonical R port; reuse the
# in-tree copy under data/meco_l2/ (kept in sync with version 1.0).
INTREE_W1 = REPO_ROOT / "data" / "meco_l2" / "joint_data_l2_trimmed_version2.2.rda"
INTREE_W2 = REPO_ROOT / "data" / "meco_l2" / "joint_data_trimmed_L2_wave2_version2.2.rda"

RESULTS_DIR = REPO_ROOT / "results" / "meco_detailed"
PLOTS_DIR = REPO_ROOT / "plots" / "meco_detailed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Language codes (canonical 3 + the broader OSF set)
# ---------------------------------------------------------------------------

CANONICAL_LANGS: tuple[str, ...] = ("du", "ch_s", "ge")  # Dutch / Mandarin / German

OSF_L1_CODES: tuple[str, ...] = (
    "du", "ee", "en", "fi", "ge", "gr", "he", "it", "no", "ru", "sp", "tr",
)  # Dutch / Estonian / English-L1 / Finnish / German / Greek / Hebrew /
# Italian / Norwegian / Russian / Spanish / Turkish

L1_PRETTY = {
    "du": "Dutch", "ee": "Estonian", "en": "English (L1)",
    "fi": "Finnish", "ge": "German", "gr": "Greek", "he": "Hebrew",
    "it": "Italian", "no": "Norwegian", "ru": "Russian",
    "sp": "Spanish", "tr": "Turkish", "ch_s": "Chinese (S)",
}


# ---------------------------------------------------------------------------
# Eye-tracking measures used by D1 (column names match the MECO R release)
# ---------------------------------------------------------------------------

ET_MEASURES: dict[str, str] = {
    # python-safe key → original R column name
    "firstrun_dur": "firstrun.dur",   # first-pass duration (early)
    "gaze_dur":     "gaze.dur",        # gaze duration (early)
    "tt_dur":       "tt.dur",          # total fixation time (late)
    "refix":        "refix",           # refixation probability (re-reading)
    "reg_in":       "reg.in",          # regression-in probability (late)
}

ET_LABELS: dict[str, str] = {
    "firstrun_dur": "First-pass duration (ms)",
    "gaze_dur":     "Gaze duration (ms)",
    "tt_dur":       "Total fixation time (ms)",
    "refix":        "Refixation prob.",
    "reg_in":       "Regression-in prob.",
}


# ---------------------------------------------------------------------------
# Rich RT loader (preserves every fixation measure we might want)
# ---------------------------------------------------------------------------

def load_joined_rt_full(
    w1_path: str | Path = INTREE_W1,
    w2_path: str | Path = INTREE_W2,
    *,
    keep_extra_measures: bool = True,
) -> pd.DataFrame:
    """Like ``meco_canonical.load_joined_rt`` but keeps every fixation
    measure available in the joined wave-1 + wave-2 release.

    The canonical pipeline renames ``firstrun.dur`` → ``firstrun_dur`` and
    drops the rest. We rename every measure in ``ET_MEASURES`` to its
    python-safe equivalent so D1 can swap DV by lookup.
    """
    import rdata

    w1 = list(rdata.read_rda(str(w1_path)).values())[0]
    w2 = list(rdata.read_rda(str(w2_path)).values())[0]
    w1.columns = [str(c) for c in w1.columns]
    w2.columns = [str(c) for c in w2.columns]
    if "supplementary_id" in w2.columns:
        w2 = w2.drop(columns=["supplementary_id"])
    joined = pd.concat([w1, w2], ignore_index=True, sort=False)

    # Rename every available R-style fixation column to a python-safe alias.
    renames: dict[str, str] = {}
    for py_name, r_name in ET_MEASURES.items():
        if r_name in joined.columns:
            renames[r_name] = py_name
    joined = joined.rename(columns=renames)

    joined["itemid"] = joined["itemid"].astype(str)
    joined["wordnum"] = pd.to_numeric(joined["wordnum"], errors="coerce")
    joined["sentnum"] = pd.to_numeric(joined["sentnum"], errors="coerce")
    for py_name in ET_MEASURES:
        if py_name in joined.columns:
            joined[py_name] = pd.to_numeric(joined[py_name], errors="coerce")

    joined = joined.dropna(subset=["wordnum"])
    joined["wordnum"] = joined["wordnum"].astype(int)
    mask_sn = joined["sentnum"].notna()
    joined.loc[mask_sn, "sentnum"] = joined.loc[mask_sn, "sentnum"].astype(int)

    if not keep_extra_measures:
        keep = ["itemid", "wordnum", "sentnum", "subid", "lang", "word",
                "firstrun_dur"]
        joined = joined[[c for c in keep if c in joined.columns]]
    return joined


# ---------------------------------------------------------------------------
# LEAP-Q proficiency loader (D6)
# ---------------------------------------------------------------------------

# LEAP-Q sheets vary in field naming across L1 cohorts but share the
# following standardised proficiency self-ratings. We only require the
# fields we use for moderation; missing values are NaN.
LEAPQ_KEYS = (
    "subid",
    "english_reading_proficiency",
    "english_listening_proficiency",
    "english_speaking_proficiency",
    "english_age_acquisition",
    "current_english_exposure_pct",
)


def load_leapq(l1: str) -> Optional[pd.DataFrame]:
    """Load one L1 cohort's LEAP-Q xlsx as a normalised DataFrame.

    Returns ``None`` if the file is missing. The xlsx schemas were
    standardised by the MECO L2 team but column names differ slightly
    across cohorts; this loader does a best-effort normalisation to
    ``LEAPQ_KEYS``.
    """
    path = OSF_LEAPQ_DIR / f"{l1}.xlsx"
    if not path.exists():
        return None
    try:
        df = pd.read_excel(path)
    except ImportError:
        # openpyxl is in requirements.txt but may be missing in a stripped
        # virtual-env; surface a clear message rather than a stack trace.
        import warnings as _w
        _w.warn("openpyxl missing — install it to read LEAP-Q (D6) sheets.",
                RuntimeWarning, stacklevel=2)
        return None
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Best-effort key mapping — accept several common variants.
    rename = {}
    for col in df.columns:
        if col in ("subid", "subject", "id", "subject_id", "participant"):
            rename[col] = "subid"
        elif "read" in col and "english" in col and "rating" in col:
            rename[col] = "english_reading_proficiency"
        elif "listen" in col and "english" in col and "rating" in col:
            rename[col] = "english_listening_proficiency"
        elif "speak" in col and "english" in col and "rating" in col:
            rename[col] = "english_speaking_proficiency"
        elif "age" in col and "english" in col:
            rename[col] = "english_age_acquisition"
        elif "exposure" in col and "english" in col:
            rename[col] = "current_english_exposure_pct"
    df = df.rename(columns=rename)
    df["l1"] = l1

    keep = [k for k in LEAPQ_KEYS if k in df.columns] + ["l1"]
    return df[keep].copy()


def load_leapq_all() -> pd.DataFrame:
    """Concatenate every available LEAP-Q xlsx into one long frame.

    Returns a frame with whichever ``LEAPQ_KEYS`` are present per cohort;
    missing fields are NaN. Index is reset.
    """
    frames = []
    for l1 in OSF_L1_CODES:
        f = load_leapq(l1)
        if f is not None and not f.empty:
            frames.append(f)
    if not frames:
        return pd.DataFrame(columns=list(LEAPQ_KEYS) + ["l1"])
    return pd.concat(frames, ignore_index=True, sort=False)


# ---------------------------------------------------------------------------
# Comprehension data loader (D8)
# ---------------------------------------------------------------------------

def load_comprehension() -> pd.DataFrame:
    """Per-(subid, text) comprehension accuracy from the OSF L2 release.

    Returns columns: ``subid, lang, itemid, accuracy`` where ``accuracy``
    is the mean per-question correctness within an item for that subject.
    Falls back to ``joint.comp.l2.rda`` shape (one row per question) when
    the breakdown table is missing.
    """
    import rdata

    # Prefer the explicit breakdown (one row per (sub, item, question))
    if OSF_COMP.exists():
        raw = list(rdata.read_rda(str(OSF_COMP)).values())[0]
        raw.columns = [str(c) for c in raw.columns]
        # Normalise expected columns
        for c in ("subid", "itemid", "lang"):
            if c in raw.columns:
                raw[c] = raw[c].astype(str)
        if "accuracy" not in raw.columns:
            # Map common alternatives.
            for alt in ("acc", "correct", "is_correct", "score"):
                if alt in raw.columns:
                    raw = raw.rename(columns={alt: "accuracy"})
                    break
        if {"subid", "itemid", "accuracy"}.issubset(raw.columns):
            agg = (raw.groupby(["subid", "lang", "itemid"], dropna=False)
                       ["accuracy"].mean().reset_index())
            return agg
    return pd.DataFrame(columns=["subid", "lang", "itemid", "accuracy"])


# ---------------------------------------------------------------------------
# Cognate annotation (D5)
# ---------------------------------------------------------------------------

def load_cognate_table(l1: str = "nld") -> pd.DataFrame:
    """Lower-cased Dutch-English cognate / false-friend tables shipped with
    the repo under ``data/cognates/``. The shipped tables are Dutch-English;
    extension to other L1s is a no-op for now (returns empty for non-nld).
    """
    if l1 != "nld":
        return pd.DataFrame(columns=["word_en", "is_cognate", "is_false_friend"])
    cog_path = REPO_ROOT / "data" / "cognates" / "dutch_english_cognates.tsv"
    ff_path = REPO_ROOT / "data" / "cognates" / "dutch_english_false_friends.tsv"
    cog = pd.read_csv(cog_path, sep="\t") if cog_path.exists() else pd.DataFrame()
    ff = pd.read_csv(ff_path, sep="\t") if ff_path.exists() else pd.DataFrame()
    en_col_cog = next((c for c in cog.columns if "en" in c.lower()), None)
    en_col_ff = next((c for c in ff.columns if "en" in c.lower()), None)
    cog_words = set(cog[en_col_cog].astype(str).str.lower()) if en_col_cog else set()
    ff_words = set(ff[en_col_ff].astype(str).str.lower()) if en_col_ff else set()
    union = sorted(cog_words | ff_words)
    return pd.DataFrame({
        "word_en": union,
        "is_cognate": [w in cog_words for w in union],
        "is_false_friend": [w in ff_words for w in union],
    })


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def result_csv(slug: str) -> Path:
    """Return the canonical CSV path for module ``dN_<slug>``."""
    return RESULTS_DIR / f"{slug}.csv"


def figure_pdf(slug: str) -> Path:
    """Return the canonical PDF path for module ``dN_<slug>``."""
    return PLOTS_DIR / f"{slug}.pdf"


# ---------------------------------------------------------------------------
# Worker-local caches (used by the optimised pipeline orchestrator)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def get_rt_full_with_features() -> pd.DataFrame:
    """Multi-measure variant of ``get_rt_with_features`` used by D1.

    Carries every fixation measure in ``ET_MEASURES``, not just
    ``firstrun_dur``. Cached at the process level just like its sibling.
    """
    from analyze import meco_canonical as mc  # local import: avoid cycle
    rt = load_joined_rt_full()
    rt = mc.add_word_features(rt)
    return rt


@functools.lru_cache(maxsize=1)
def get_rt_with_features() -> pd.DataFrame:
    """Load + feature-attach the joined MECO L2 wave-1 + wave-2 RT data **once**
    per process.

    Every D-module calls ``mc.load_joined_rt()`` followed by
    ``mc.add_word_features(rt)``. With 8 modules × 81 models = 648 calls per
    sweep that's roughly 90 minutes of redundant rdata + wordfreq work; this
    cache collapses it to one ~10 s load per worker process.

    The result is returned by reference — callers MUST treat it as read-only
    or call ``.copy()`` before mutating.
    """
    from analyze import meco_canonical as mc  # local import: avoid cycle
    rt = mc.load_joined_rt()
    rt = mc.add_word_features(rt)
    return rt


def upsert_csv(
    path: Path,
    new_rows: pd.DataFrame,
    *,
    key_cols: Sequence[str],
    timeout: float = 60.0,
) -> int:
    """Atomically merge ``new_rows`` into the CSV at ``path`` keyed on
    ``key_cols``.

    Process-safe: takes an ``fcntl`` exclusive lock on a sidecar lockfile so
    multiple workers can write the same CSV without clobbering each other.
    Returns the total row count after the merge.

    Behaviour:
      * if ``path`` does not exist, the new rows are written verbatim
      * else the existing CSV is read, rows whose key matches a row in
        ``new_rows`` are dropped, and ``new_rows`` is concatenated
      * a temp file is written then atomically moved into place

    The lock is released even on exception.
    """
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    deadline = time.time() + timeout
    with lock_path.open("a+") as lockf:
        while True:
            try:
                fcntl.flock(lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() > deadline:
                    raise TimeoutError(f"upsert_csv lock timeout on {path}")
                time.sleep(0.05)
        try:
            if path.exists():
                old = pd.read_csv(path)
                # Drop rows whose composite key matches a new row
                if not old.empty and not new_rows.empty:
                    join_keys = list(key_cols)
                    new_keys = (new_rows[join_keys].astype(str)
                                .agg("\x00".join, axis=1).tolist())
                    new_key_set = set(new_keys)
                    old_keys = (old[join_keys].astype(str)
                                .agg("\x00".join, axis=1))
                    old = old[~old_keys.isin(new_key_set)]
                merged = pd.concat([old, new_rows], ignore_index=True, sort=False)
            else:
                merged = new_rows
            tmp = path.with_suffix(path.suffix + ".tmp")
            merged.to_csv(tmp, index=False)
            tmp.replace(path)
            return len(merged)
        finally:
            try:
                fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def is_fast_fit() -> bool:
    """Honour ``BEETLE_FAST_FIT=1`` env var.

    When set, the canonical lmer pipeline skips Tier-1 (full uncorrelated
    random slopes on every fixed term) and starts from Tier-2 (surprisal-only
    random slopes). Tier-1 attempts cost ~2 min each and frequently fail to
    converge on smaller subject pools, so dropping them is a ~4× speedup for
    a sweep where we know we'll have to fall back anyway.
    """
    return os.environ.get("BEETLE_FAST_FIT") == "1"
