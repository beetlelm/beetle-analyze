"""
evals/priming_stimuli.py — Loader for Arnett (2025) structural-priming stimuli.

The OSF release at osf.io/5cw2e ships five stimulus TSVs with two schemas:

  Schema A (Schoonbaert 2007, Bernolet 2013):
      Set, ExtraCondition, PrimeLanguage, PrimeType, TargetType,
      TargetLanguage, PrimeSentence, TargetSentence, FullSentence
      (PrimeLanguage / TargetLanguage carry "_Prime" / "_Target" suffix.)

  Schema B (more_{spanish,greek,polish}_stims_144):
      <unnamed-index>, Set, PrimeLanguage, TargetLanguage, PrimeType,
      TargetType, Prime, Target
      (PrimeType ∈ {"active", "passive"}; PrimeLanguage is bare "English"/"Polish".)

This loader reads either schema and returns a normalised long-form
DataFrame with the columns the Beetle eval pipeline expects:

  dataset           — short tag {schoonbaert, bernolet, hartsuiker, kotzochampou, fleischer}
  Set               — stimulus item id (joins per-item rows together)
  ExtraCondition    — schema-A only; "" for schema-B
  PrimeLanguage     — bare lang in title case ("English", "Dutch", "Spanish", "Greek", "Polish")
  TargetLanguage    — bare lang in title case
  PrimeType         — e.g. DO_Prime / PO_Prime / Active_Prime / s-genitive_Prime
  TargetType        — e.g. DO_Target / PO_Target / Active_Target / s-genitive_Target
  PrimeSentence     — prime sentence text
  TargetSentence    — target sentence text
  FullSentence      — concatenation used as the joined string for scoring,
                      matches Arnett's `FullText` column.

Per Arnett's Rmd preprocess loop (final_structural_priming_analysis.Rmd):
  full sentence = "{PrimeSentence}. {TargetSentence}." for schemas where
  the OSF file already provides FullSentence; for the *_144 files the Rmd
  reconstructs FullText as "{Prime} {Target}". We mirror that exactly.

The loader also maps each stimulus file to the (l1_iso3, l2_iso3) language pair
the model under test must cover, so the eval driver can skip stimuli files
that don't apply to a given model (e.g. don't run Schoonbaert on a deu-eng
model).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# ISO-639-3 codes used throughout the Beetle pipeline.
LANG_TITLE_TO_ISO3: dict[str, str] = {
    "English":  "eng",
    "Dutch":    "nld",
    "Spanish":  "spa",
    "Greek":    "ell",
    "Polish":   "pol",
    "German":   "deu",
}
ISO3_TO_TITLE: dict[str, str] = {v: k for k, v in LANG_TITLE_TO_ISO3.items()}


@dataclass(frozen=True)
class StimulusFile:
    """One Arnett stimulus dataset as it lives on disk."""

    dataset: str               # tag used in output filenames + ledger
    path: Path                 # location of the TSV
    schema: str                # "A" or "B"
    language_pair: tuple[str, str]   # (iso3, iso3); order is (English, Other)


# Default file layout — assumes ``arnett-2025-priming-raw/`` is at repo root.
ARNETT_DIR = Path(__file__).resolve().parent.parent / "arnett-2025-priming-raw"
STIMULI_DIR = ARNETT_DIR / "stimuli"

ARNETT_STIMULI: tuple[StimulusFile, ...] = (
    StimulusFile("schoonbaert",   STIMULI_DIR / "schoonbaert_2007.tsv",         "A", ("eng", "nld")),
    StimulusFile("bernolet",      STIMULI_DIR / "bernolet_2013.tsv",            "A", ("eng", "nld")),
    StimulusFile("hartsuiker",    STIMULI_DIR / "more_spanish_stims_144.tsv",   "B", ("eng", "spa")),
    StimulusFile("kotzochampou",  STIMULI_DIR / "more_greek_stims_144.tsv",     "B", ("eng", "ell")),
    StimulusFile("fleischer",     STIMULI_DIR / "more_polish_stims_144.tsv",    "B", ("eng", "pol")),
    # RQ-A: German↔English cross-lingual priming, sourced from real published
    # items in van Dijk & Hopp (2025), "Structural preferences in language
    # learning across languages", Language, Cognition and Neuroscience,
    # DOI 10.1080/23273798.2025.2476703. Open materials at OSF project ypsvn
    # (https://osf.io/ypsvn/), file WP4_stimuli_OSF_20240826.xlsx.
    # Build via: ``python scripts/build_rq_a_stimuli.py``.
    # Each pair is naturally deu→eng (per study design); we do not synthesise
    # an eng→deu direction from these items.
    StimulusFile("vandijk_hopp_dat",    STIMULI_DIR / "vandijk_hopp_2025_german_dat.tsv",    "B", ("eng", "deu")),
    StimulusFile("vandijk_hopp_actpas", STIMULI_DIR / "vandijk_hopp_2025_german_actpas.tsv", "B", ("eng", "deu")),
)


def _strip_suffix(s: str, suffix: str) -> str:
    return s[: -len(suffix)] if s.endswith(suffix) else s


def _normalise_schema_a(df: pd.DataFrame) -> pd.DataFrame:
    """Schoonbaert / Bernolet — already long with the right column names.

    We strip the "_Prime"/"_Target" suffixes from PrimeLanguage / TargetLanguage
    so they hold bare language names, matching Schema B after normalisation.
    """
    out = df.copy()
    out["PrimeLanguage"]  = out["PrimeLanguage"].astype(str).map(lambda s: _strip_suffix(s, "_Prime"))
    out["TargetLanguage"] = out["TargetLanguage"].astype(str).map(lambda s: _strip_suffix(s, "_Target"))
    if "ExtraCondition" not in out.columns:
        out["ExtraCondition"] = ""
    cols = [
        "Set", "ExtraCondition",
        "PrimeLanguage", "PrimeType", "TargetType", "TargetLanguage",
        "PrimeSentence", "TargetSentence", "FullSentence",
    ]
    return out[cols]


def _normalise_schema_b(df: pd.DataFrame) -> pd.DataFrame:
    """Spanish / Greek / Polish 144-stim files.

    The Rmd preprocesses these by:
      - dropping the unnamed index column
      - renaming Prime → PrimeSentence, Target → TargetSentence
      - reconstructing FullText as ``"{Prime} {Target}"``
      - title-casing PrimeType ("active" → "Active_Prime") so the values match
        the schema-A convention used downstream (DO_Prime / PO_Prime / etc.)
    """
    out = df.copy()
    # Drop unnamed index column if present.
    if "" in out.columns:
        out = out.drop(columns=[""])
    if out.columns[0] in ("Unnamed: 0", "index"):
        out = out.drop(columns=[out.columns[0]])

    rename = {"Prime": "PrimeSentence", "Target": "TargetSentence"}
    out = out.rename(columns=rename)

    # PrimeType / TargetType: lowercase active/passive → Active_Prime / Active_Target.
    def _suffix_type(val: str, kind: str) -> str:
        v = str(val)
        if v.endswith("_Prime") or v.endswith("_Target"):
            return v
        return f"{v.capitalize()}_{kind}"

    out["PrimeType"]  = out["PrimeType"].map(lambda v: _suffix_type(v, "Prime"))
    out["TargetType"] = out["TargetType"].map(lambda v: _suffix_type(v, "Target"))

    out["FullSentence"] = out["PrimeSentence"].astype(str) + " " + out["TargetSentence"].astype(str)
    out["ExtraCondition"] = ""
    cols = [
        "Set", "ExtraCondition",
        "PrimeLanguage", "PrimeType", "TargetType", "TargetLanguage",
        "PrimeSentence", "TargetSentence", "FullSentence",
    ]
    return out[cols]


def load_stimuli(stim: StimulusFile) -> pd.DataFrame:
    """Load and normalise one Arnett stimulus file.

    Adds a ``dataset`` column tagged with ``stim.dataset``.
    """
    if not stim.path.exists():
        raise FileNotFoundError(
            f"Arnett stimulus file not found: {stim.path}. "
            "Run scripts/download_arnett_osf.sh or check arnett-2025-priming-raw/stimuli/."
        )
    df = pd.read_csv(stim.path, sep="\t")
    if stim.schema == "A":
        norm = _normalise_schema_a(df)
    elif stim.schema == "B":
        norm = _normalise_schema_b(df)
    else:
        raise ValueError(f"Unknown schema {stim.schema!r}")
    norm.insert(0, "dataset", stim.dataset)
    return norm


def stimuli_for_model_pair(l1: str, l2: str) -> list[StimulusFile]:
    """Return the Arnett stimulus files compatible with a Beetle model's
    (l1, l2) ISO-639-3 language pair.

    A stimulus file is compatible iff its ``language_pair`` set equals the
    model's. Order doesn't matter — Arnett's stimuli include both directions.

    Returns ``[]`` for pairs Arnett doesn't cover (e.g. eng-deu, eng-zho).
    """
    pair = frozenset((l1, l2))
    return [s for s in ARNETT_STIMULI if frozenset(s.language_pair) == pair]


def all_stimuli_languages() -> set[str]:
    """ISO-639-3 codes for every language Arnett's stimuli cover."""
    out: set[str] = set()
    for s in ARNETT_STIMULI:
        out.update(s.language_pair)
    return out
