"""
models.py — Central registry of all BeetleLM model repos.

MODEL_GROUPS       : dict  lang_code -> list of repo strings
ALL_MODELS         : list  deduplicated across all groups, stable order
TRILINGUAL_MODELS  : list  all trilingual repos (balanced + sequential variants)
ALL_TRILINGUAL_MODELS : list  deduplicated trilingual-only list for rank slicing
"""

from __future__ import annotations

import re

DUTCH_MODELS = [
    "BeetleLM/beetlelm_nld_mono",
    # Balanced
    "BeetleLM/beetlelm_eng-nld_balanced",
    "BeetleLM/beetlelm_nld_L1-eng_L2_balanced",
    "BeetleLM/beetlelm_nld-ukr_balanced",
    "BeetleLM/beetlelm_nld-fra_balanced",
    "BeetleLM/beetlelm_nld-deu_balanced",
    "BeetleLM/beetlelm_nld-ind_balanced",
    "BeetleLM/beetlelm_nld-bul_balanced",
    "BeetleLM/beetlelm_zho-nld_balanced",
    # Simultaneous
    "BeetleLM/beetlelm_eng-nld_simultaneous",
    "BeetleLM/beetlelm_nld_L1-eng_L2_simultaneous",
    "BeetleLM/beetlelm_fas-nld_simultaneous",
    # Sequential
    "BeetleLM/beetlelm_fas-nld_sequential",
    # Part-Time
    "BeetleLM/beetlelm_eng-nld_part_time",
    "BeetleLM/beetlelm_fas-nld_part_time",
    "BeetleLM/beetlelm_nld-deu_part_time",
    # Late
    "BeetleLM/beetlelm_nld-ukr_late",
    "BeetleLM/beetlelm_nld-bul_late",
    "BeetleLM/beetlelm_fas-nld_late",
    "BeetleLM/beetlelm_eng-nld_late",
    # Heritage
    "BeetleLM/beetlelm_nld-fra_heritage",
    "BeetleLM/beetlelm_nld-deu_heritage",
    "BeetleLM/beetlelm_nld-bul_heritage",
    "BeetleLM/beetlelm_nld-ukr_heritage",
    "BeetleLM/beetlelm_fas-nld_heritage",
]

GERMAN_MODELS = [
    "BeetleLM/beetlelm_deu_mono",
    # Balanced
    "BeetleLM/beetlelm_deu_L1-eng_L2_balanced",
    "BeetleLM/beetlelm_eng_L1-deu_L2_balanced",
    # Simultaneous
    "BeetleLM/beetlelm_deu_L1-eng_L2_simultaneous",
    "BeetleLM/beetlelm_eng_L1-deu_L2_simultaneous",
    "BeetleLM/beetlelm_eng-deu_simultaneous",
    "BeetleLM/beetlelm_deu-ukr_simultaneous",
    "BeetleLM/beetlelm_bul-deu_simultaneous",
    # Sequential
    "BeetleLM/beetlelm_deu-ukr_sequential",
    "BeetleLM/beetlelm_bul-deu_sequential",
    # Part-Time
    "BeetleLM/beetlelm_nld-deu_part_time",
    "BeetleLM/beetlelm_deu-ukr_part_time",
    "BeetleLM/beetlelm_bul-deu_part_time",
    # Late
    "BeetleLM/beetlelm_zho-deu_late",
    "BeetleLM/beetlelm_fas-deu_late",
    "BeetleLM/beetlelm_deu-ukr_late",
    "BeetleLM/beetlelm_bul-deu_late",
    # Heritage
    "BeetleLM/beetlelm_zho-deu_heritage",
    "BeetleLM/beetlelm_ind-deu_heritage",
    "BeetleLM/beetlelm_fra-deu_heritage",
    "BeetleLM/beetlelm_fas-deu_heritage",
    "BeetleLM/beetlelm_eng-deu_heritage",
    "BeetleLM/beetlelm_deu-ukr_heritage",
    "BeetleLM/beetlelm_bul-deu_heritage",
]

CHINESE_MODELS = [
    "BeetleLM/beetlelm_zho_mono",
    # Balanced
    "BeetleLM/beetlelm_zho_L1-eng_L2_balanced",
    "BeetleLM/beetlelm_zho-eng_balanced",
    "BeetleLM/beetlelm_zho-ukr_balanced",
    "BeetleLM/beetlelm_zho-fra_balanced",
    "BeetleLM/beetlelm_zho-fas_balanced",
    "BeetleLM/beetlelm_zho-ind_balanced",
    "BeetleLM/beetlelm_zho-deu_balanced",
    "BeetleLM/beetlelm_zho-nld_balanced",
    "BeetleLM/beetlelm_zho-bul_balanced",
    # Simultaneous
    "BeetleLM/beetlelm_zho_L1-eng_L2_simultaneous",
    # Sequential
    "BeetleLM/beetlelm_zho-fra_sequential",
    # Part-Time
    "BeetleLM/beetlelm_zho-fra_part_time",
    # Late
    "BeetleLM/beetlelm_zho-ukr_late",
    "BeetleLM/beetlelm_zho-fas_late",
    "BeetleLM/beetlelm_zho-ind_late",
    "BeetleLM/beetlelm_zho-deu_late",
    "BeetleLM/beetlelm_zho-fra_late",
    # Heritage
    "BeetleLM/beetlelm_zho-fas_heritage",
    "BeetleLM/beetlelm_zho-deu_heritage",
    "BeetleLM/beetlelm_zho-fra_heritage",
    "BeetleLM/beetlelm_zho-bul_heritage",
]

FRENCH_MODELS = [
    "BeetleLM/beetlelm_fra_mono",
    # Balanced
    "BeetleLM/beetlelm_zho-fra_balanced",
    "BeetleLM/beetlelm_nld-fra_balanced",
    "BeetleLM/beetlelm_fra-ind_balanced",
    "BeetleLM/beetlelm_fra-deu_balanced",
    "BeetleLM/beetlelm_fra-ukr_balanced",
    "BeetleLM/beetlelm_fas-fra_balanced",
    "BeetleLM/beetlelm_eng-fra_balanced",
    "BeetleLM/beetlelm_bul-fra_balanced",
    # Simultaneous
    "BeetleLM/beetlelm_bul-fra_simultaneous",
    # Sequential
    "BeetleLM/beetlelm_fra-ukr_sequential",
    "BeetleLM/beetlelm_bul-fra_sequential",
    # Part-Time
    "BeetleLM/beetlelm_zho-fra_part_time",
    "BeetleLM/beetlelm_fra-ukr_part_time",
    "BeetleLM/beetlelm_eng-fra_part_time",
    "BeetleLM/beetlelm_bul-fra_part_time",
    # Late
    "BeetleLM/beetlelm_zho-fra_late",
    "BeetleLM/beetlelm_fra-ind_late",
    "BeetleLM/beetlelm_fra-ukr_late",
    "BeetleLM/beetlelm_bul-fra_late",
    # Heritage
    "BeetleLM/beetlelm_nld-fra_heritage",
    "BeetleLM/beetlelm_zho-fra_heritage",
    "BeetleLM/beetlelm_fra-ukr_heritage",
    "BeetleLM/beetlelm_fra-ind_heritage",
    "BeetleLM/beetlelm_eng-fra_heritage",
    "BeetleLM/beetlelm_fra-deu_heritage",
    "BeetleLM/beetlelm_fas-fra_heritage",
    "BeetleLM/beetlelm_bul-fra_heritage",
]

PERSIAN_MODELS = [
    "BeetleLM/beetlelm_fas_mono",
    # Balanced
    "BeetleLM/beetlelm_zho-fas_balanced",
    "BeetleLM/beetlelm_fas-ukr_balanced",
    "BeetleLM/beetlelm_fas-eng_balanced",
    "BeetleLM/beetlelm_fas-nld_balanced",
    "BeetleLM/beetlelm_fas-ind_balanced",
    "BeetleLM/beetlelm_fas-bul_balanced",
    "BeetleLM/beetlelm_fas-fra_balanced",
    "BeetleLM/beetlelm_fas-deu_balanced",
    # Simultaneous
    "BeetleLM/beetlelm_fas-deu_simultaneous",
    "BeetleLM/beetlelm_fas-nld_simultaneous",
    # Sequential
    "BeetleLM/beetlelm_fas-nld_sequential",
    "BeetleLM/beetlelm_fas-deu_sequential",
    # Part-Time
    "BeetleLM/beetlelm_fas-ind_part_time",
    "BeetleLM/beetlelm_fas-bul_part_time",
    "BeetleLM/beetlelm_fas-eng_part_time",
    "BeetleLM/beetlelm_fas-nld_part_time",
    "BeetleLM/beetlelm_fas-deu_part_time",
    # Late
    "BeetleLM/beetlelm_zho-fas_late",
    "BeetleLM/beetlelm_fas-ind_late",
    "BeetleLM/beetlelm_fas-eng_late",
    "BeetleLM/beetlelm_fas-ukr_late",
    "BeetleLM/beetlelm_fas-nld_late",
    "BeetleLM/beetlelm_fas-deu_late",
    # Heritage
    "BeetleLM/beetlelm_zho-fas_heritage",
    "BeetleLM/beetlelm_fas-ukr_heritage",
    "BeetleLM/beetlelm_fas-ind_heritage",
    "BeetleLM/beetlelm_fas-bul_heritage",
    "BeetleLM/beetlelm_fas-fra_heritage",
    "BeetleLM/beetlelm_fas-nld_heritage",
    "BeetleLM/beetlelm_fas-deu_heritage",
]

BULGARIAN_MODELS = [
    "BeetleLM/beetlelm_bul_mono",
    # Balanced
    "BeetleLM/beetlelm_zho-bul_balanced",
    "BeetleLM/beetlelm_nld-bul_balanced",
    "BeetleLM/beetlelm_bul-ukr_balanced",
    "BeetleLM/beetlelm_bul-ind_balanced",
    "BeetleLM/beetlelm_bul-fra_balanced",
    "BeetleLM/beetlelm_bul-deu_balanced",
    # Simultaneous
    "BeetleLM/beetlelm_bul-fra_simultaneous",
    "BeetleLM/beetlelm_eng-bul_simultaneous",
    "BeetleLM/beetlelm_bul-deu_simultaneous",
    # Sequential
    "BeetleLM/beetlelm_bul-ukr_sequential",
    "BeetleLM/beetlelm_bul-fra_sequential",
    "BeetleLM/beetlelm_bul-deu_sequential",
    # Part-Time
    "BeetleLM/beetlelm_fas-bul_part_time",
    "BeetleLM/beetlelm_bul-ukr_part_time",
    "BeetleLM/beetlelm_bul-fra_part_time",
    "BeetleLM/beetlelm_eng-bul_part_time",
    "BeetleLM/beetlelm_bul-deu_part_time",
    # Late
    "BeetleLM/beetlelm_nld-bul_late",
    "BeetleLM/beetlelm_zho-bul_late",
    "BeetleLM/beetlelm_bul-ukr_late",
    "BeetleLM/beetlelm_bul-fra_late",
    "BeetleLM/beetlelm_bul-deu_late",
    # Heritage
    "BeetleLM/beetlelm_zho-bul_heritage",
    "BeetleLM/beetlelm_nld-bul_heritage",
    "BeetleLM/beetlelm_fas-bul_heritage",
    "BeetleLM/beetlelm_eng-bul_heritage",
    "BeetleLM/beetlelm_bul-ukr_heritage",
    "BeetleLM/beetlelm_bul-fra_heritage",
    "BeetleLM/beetlelm_bul-ind_heritage",
]

# English: all models that contain English in the language pair.
ENGLISH_MODELS = [
    # Balanced
    "BeetleLM/beetlelm_eng-nld_balanced",
    "BeetleLM/beetlelm_nld_L1-eng_L2_balanced",
    "BeetleLM/beetlelm_eng_L1-deu_L2_balanced",
    "BeetleLM/beetlelm_zho_L1-eng_L2_balanced",
    "BeetleLM/beetlelm_zho-eng_balanced",
    "BeetleLM/beetlelm_fas-eng_balanced",
    # Simultaneous
    "BeetleLM/beetlelm_eng-nld_simultaneous",
    "BeetleLM/beetlelm_nld_L1-eng_L2_simultaneous",
    "BeetleLM/beetlelm_deu_L1-eng_L2_simultaneous",
    "BeetleLM/beetlelm_eng_L1-deu_L2_simultaneous",
    "BeetleLM/beetlelm_eng-deu_simultaneous",
    "BeetleLM/beetlelm_zho_L1-eng_L2_simultaneous",
    "BeetleLM/beetlelm_eng-bul_simultaneous",
    # Part-Time
    "BeetleLM/beetlelm_eng-nld_part_time",
    "BeetleLM/beetlelm_fas-eng_part_time",
    "BeetleLM/beetlelm_eng-fra_part_time",
    "BeetleLM/beetlelm_eng-bul_part_time",
    # Late
    "BeetleLM/beetlelm_eng-nld_late",
    "BeetleLM/beetlelm_fas-eng_late",
    # Heritage
    "BeetleLM/beetlelm_eng-deu_heritage",
    "BeetleLM/beetlelm_eng-fra_heritage",
    "BeetleLM/beetlelm_eng-bul_heritage",
]

# ─────────────────────────────────────────────────────────────────────────────
# TRILINGUAL MODELS
# All trilingual models involve the eng / nld / zho triad and are evaluated
# on the blimp_eng, blimp_nl, zhoblimp, xcomps (zho), and xnli (en, zh)
# benchmarks.  They live under the "tri" group key.
# ─────────────────────────────────────────────────────────────────────────────

# Balanced and Heritage trilinguals
TRILINGUAL_BALANCED_MODELS = [
    "BeetleLM/beetlelm_eng_L1-zho_L2-nld_L3_balanced",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_balanced",
    "BeetleLM/beetlelm_nld_L1-eng_L2-zho_L3_balanced",
    "BeetleLM/beetlelm_nld_L1-zho_L2-eng_L3_balanced",
    "BeetleLM/beetlelm_zho_L1-nld_L2-eng_L3_balanced",
    "BeetleLM/beetlelm_zho_L1-eng_L2-nld_L3_balanced",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_heritage",
    "BeetleLM/beetlelm_eng_L1-zho_L2-nld_L3_heritage",
    "BeetleLM/beetlelm_nld_L1-eng_L2-zho_L3_heritage",
    "BeetleLM/beetlelm_nld_L1-zho_L2-eng_L3_heritage",
    "BeetleLM/beetlelm_zho_L1-eng_L2-nld_L3_heritage",
    "BeetleLM/beetlelm_zho_L1-nld_L2-eng_L3_heritage",
]

# Sequential variants: MAML, EWC, combined, and baseline
TRILINGUAL_SEQUENTIAL_MODELS = [
    # EWC + MAML combined
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l5_maml_h25",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l5_maml_h50",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l5_maml_h100",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l20_maml_h25",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l20_maml_h50",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l20_maml_h75",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l20_maml_h100",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l50_maml_h25",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l50_maml_h50",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l50_maml_h75",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l50_maml_h100",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l150_maml_h25",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l150_maml_h50",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l150_maml_h75",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l150_maml_h100",
    # EWC only
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l5",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l20",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l50",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l150",
    # MAML only
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_maml_h25",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_maml_h50",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_maml_h75",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_maml_h100",
    # Baseline
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_baseline",
]

TRILINGUAL_MODELS = TRILINGUAL_BALANCED_MODELS + TRILINGUAL_SEQUENTIAL_MODELS

# ─────────────────────────────────────────────────────────────────────────────
# FINEWEB BILINGUAL MODELS
# FineWeb-scale (24B tokens) bilingual models: L1-English pairs with 3
# curriculum conditions. Published under the Beetle-FineWeb-Bilingual org.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# LEGACY FINEWEB BILINGUAL MODELS (original 3-condition runs)
# ─────────────────────────────────────────────────────────────────────────────

FINEWEB_LEGACY_CONDITIONS = ["balanced", "simultaneous", "sequential"]
FINEWEB_CORE_LANGS = [
    "pol", "nld", "spa", "ell", "jpn", "fra", "zho", "deu",
    "ita", "eus", "tur", "ind", "tgl", "fas", "hin", "tam",
    "swe", "rus", "cat", "arb",
]

FINEWEB_LEGACY_MODELS = [
    f"Beetle-FineWeb-Bilingual/beetlelm_fineweb_{lang}_L1-eng_L2_{cond}"
    for lang in FINEWEB_CORE_LANGS
    for cond in FINEWEB_LEGACY_CONDITIONS
]

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2: FINEWEB BIDIRECTIONAL MODELS (new org, balanced+simultaneous)
# Published under Beetle-FineWeb org, includes both L1->L2 and L2->L1.
# ─────────────────────────────────────────────────────────────────────────────

FINEWEB_EXP2_CONDITIONS = ["balanced", "simultaneous"]
FINEWEB_EXP2_LANGS = [
    "nld", "ita", "rus", "pol", "tur",  # Set A
    "fra", "deu", "spa", "zho", "jpn",  # Set B
]

FINEWEB_BIDIRECTIONAL_MODELS = [
    f"Beetle-FineWeb/beetlelm_fineweb_{lang}_L1-eng_L2_{cond}"
    for lang in FINEWEB_EXP2_LANGS
    for cond in FINEWEB_EXP2_CONDITIONS
] + [
    f"Beetle-FineWeb/beetlelm_fineweb_eng_L1-{lang}_L2_{cond}"
    for lang in FINEWEB_EXP2_LANGS
    for cond in FINEWEB_EXP2_CONDITIONS
]

# Combined FineWeb models (legacy + new)
FINEWEB_MODELS = FINEWEB_LEGACY_MODELS + FINEWEB_BIDIRECTIONAL_MODELS

# ─────────────────────────────────────────────────────────────────────────────
# HUMAN-SCALE SWEEP MODELS (Experiment 1)
# Published under Beetle-HumanScale org.
# These are dynamically discovered via HfApi, but we define the group key.
# ─────────────────────────────────────────────────────────────────────────────

HUMAN_SCALE_SWEEP_ORG = "Beetle-HumanScale"
HUMAN_SCALE_ORG = "Beetle-HumanScale"
FINEWEB_ORG = "Beetle-FineWeb"
BEETLELM_ORG = "BeetleLM"
# Tokenizer publishing org for the fineweb regime. Note this is distinct from
# FINEWEB_ORG: FineWeb *models* are published under Beetle-FineWeb, but their
# paired tokenizers live under Beetle-Data (tokenizer-{lang}-mono,
# tokenizer-{l1}-{l2}).
BEETLE_DATA_ORG = "Beetle-Data"

# ─────────────────────────────────────────────────────────────────────────────
# ISO 639-3 → 2-letter tokenizer code. Single source of truth; re-exported by
# aoa/* modules that used to keep their own duplicate copies.
# ─────────────────────────────────────────────────────────────────────────────
ISO3_TO_TOK2: dict[str, str] = {
    "deu": "de", "jpn": "ja", "spa": "es", "zho": "zh",
    "nld": "nl", "rus": "ru", "por": "pt", "tur": "tr",
    "fas": "fa", "pol": "pl", "ell": "el", "ita": "it",
    "fra": "fr", "swe": "sv", "hun": "hu", "ces": "cs",
    "cat": "ca", "dan": "da", "fin": "fi", "eng": "en",
    "bul": "bg",
}


# Explicit list of published Human-Scale models (user-provided 2026-04-18).
# New models should be appended here; the registry merges them into
# MODEL_GROUPS via _humanscale_models_for() below.
HUMAN_SCALE_MODELS = [
    # Monolingual baselines
    "Beetle-HumanScale/beetle-monolingual-humanscale-deu",
    "Beetle-HumanScale/beetle-monolingual-humanscale-eng",
    "Beetle-HumanScale/beetle-monolingual-humanscale-nld",
    "Beetle-HumanScale/beetle-monolingual-humanscale-zho",
    # B1: balanced
    "Beetle-HumanScale/beetle-bilingual-balanced-b1-humanscale-deu-eng",
    "Beetle-HumanScale/beetle-bilingual-balanced-b1-humanscale-nld-eng",
    "Beetle-HumanScale/beetle-bilingual-balanced-b1-humanscale-nld-eng-episodic",
    "Beetle-HumanScale/beetle-bilingual-balanced-b1-humanscale-zho-eng",
    # B2: simultaneous
    "Beetle-HumanScale/beetle-bilingual-l2-50-simultaneous-b2-humanscale-deu-eng",
    "Beetle-HumanScale/beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng",
    "Beetle-HumanScale/beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng-ewc-l20",
    "Beetle-HumanScale/beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng-lamol-l20",
    "Beetle-HumanScale/beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng-lr-b2",
    "Beetle-HumanScale/beetle-bilingual-l2-50-simultaneous-b2-humanscale-zho-eng",
    # B3: sequential
    "Beetle-HumanScale/beetle-bilingual-l2-50-sequential-33-67-b3-humanscale-deu-eng",
    "Beetle-HumanScale/beetle-bilingual-l2-50-sequential-33-67-b3-humanscale-nld-eng",
    # B4: classroom
    "Beetle-HumanScale/beetle-bilingual-l2-50-classroom-20-b4-humanscale-deu-eng",
    "Beetle-HumanScale/beetle-bilingual-l2-50-classroom-20-b4-humanscale-nld-eng",
    # B5: late
    "Beetle-HumanScale/beetle-bilingual-l2-80-late-b5-humanscale-deu-eng",
    "Beetle-HumanScale/beetle-bilingual-l2-80-late-b5-humanscale-nld-eng",
]


def _load_manifest(filename: str) -> list:
    """Load a JSON manifest produced by scripts/sync_*.py. Missing file → []."""
    import json
    import os
    path = os.path.join(os.path.dirname(__file__), "analyze", filename)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict) and "models" in data:
            return [str(x) for x in data["models"]]
        return []
    except (OSError, ValueError):
        return []


# Planned FineWeb models — regenerated by scripts/sync_fineweb_manifest.py.
FINEWEB_MODELS_PLANNED = _load_manifest("fineweb_model_manifest.json")

# Seeded ablation sweep repos — regenerated by scripts/sync_ablation_manifest.py.
ABLATION_SEEDED_MODELS = _load_manifest("ablation_manifest.json")


def _matches_lang(repo: str, lang: str) -> bool:
    """Check if a Beetle-{HumanScale,FineWeb} slug references the given ISO code."""
    name = repo.split("/")[-1]
    # mono: beetle-monolingual-{ds}-{lang}
    if name.startswith(f"beetle-monolingual-humanscale-{lang}"):
        return True
    if name.startswith(f"beetle-monolingual-fineweb-{lang}"):
        return True
    # bilingual: ...-{l1}-{l2}(-suffix)?
    return (f"-{lang}-" in name) or name.endswith(f"-{lang}") or \
        (f"-{lang}-ewc" in name) or (f"-{lang}-lamol" in name) or \
        (f"-{lang}-lr" in name) or (f"-{lang}-episodic" in name) or \
        (f"-{lang}-seed" in name)


def _humanscale_models_for(lang: str) -> list:
    return [m for m in HUMAN_SCALE_MODELS if _matches_lang(m, lang)]


def _fineweb_models_planned_for(lang: str) -> list:
    return [m for m in FINEWEB_MODELS_PLANNED if _matches_lang(m, lang)]


def _ablation_models_for(lang: str) -> list:
    return [m for m in ABLATION_SEEDED_MODELS if _matches_lang(m, lang)]


# Per-language FineWeb legacy model lists for MODEL_GROUPS integration
def _fineweb_models_for(lang: str) -> list:
    return [m for m in FINEWEB_MODELS if f"_{lang}_L1" in m or f"_{lang}_L2" in m]

# ─────────────────────────────────────────────────────────────────────────────
# GROUP REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

def _merge_lang(*lists) -> list:
    """Merge lists of repos, dedup, preserve order."""
    seen: set = set()
    out: list = []
    for lst in lists:
        for m in lst:
            if m not in seen:
                seen.add(m)
                out.append(m)
    return out


MODEL_GROUPS = {
    "nld": _merge_lang(
        DUTCH_MODELS, _fineweb_models_for("nld"),
        _humanscale_models_for("nld"), _fineweb_models_planned_for("nld"),
        _ablation_models_for("nld"),
    ),
    "deu": _merge_lang(
        GERMAN_MODELS, _fineweb_models_for("deu"),
        _humanscale_models_for("deu"), _fineweb_models_planned_for("deu"),
        _ablation_models_for("deu"),
    ),
    "zho": _merge_lang(
        CHINESE_MODELS, _fineweb_models_for("zho"),
        _humanscale_models_for("zho"), _fineweb_models_planned_for("zho"),
        _ablation_models_for("zho"),
    ),
    "eng": _merge_lang(
        ENGLISH_MODELS,
        _humanscale_models_for("eng"), _fineweb_models_planned_for("eng"),
        _ablation_models_for("eng"),
    ),
    "fra": FRENCH_MODELS + _fineweb_models_for("fra"),
    "fas": PERSIAN_MODELS + _fineweb_models_for("fas"),
    "bul": BULGARIAN_MODELS,
    # Trilingual eng–nld–zho triad.
    "tri": TRILINGUAL_MODELS,
    # FineWeb bilingual models for languages not in the BabyBabel groups above
    "pol": _fineweb_models_for("pol"),
    "spa": _fineweb_models_for("spa"),
    "ell": _fineweb_models_for("ell"),
    "jpn": _fineweb_models_for("jpn"),
    "ita": _fineweb_models_for("ita"),
    "eus": _fineweb_models_for("eus"),
    "tur": _fineweb_models_for("tur"),
    "ind": _fineweb_models_for("ind"),
    "tgl": _fineweb_models_for("tgl"),
    "hin": _fineweb_models_for("hin"),
    "tam": _fineweb_models_for("tam"),
    "swe": _fineweb_models_for("swe"),
    "rus": _fineweb_models_for("rus"),
    "cat": _fineweb_models_for("cat"),
    "arb": _fineweb_models_for("arb"),
    # FineWeb (legacy + new)
    "fineweb": FINEWEB_MODELS,
    "fineweb_legacy": FINEWEB_LEGACY_MODELS,
    "fineweb_bidirectional": FINEWEB_BIDIRECTIONAL_MODELS,
    # New family-indexed groups
    "humanscale":       list(HUMAN_SCALE_MODELS),
    "fineweb_planned":  list(FINEWEB_MODELS_PLANNED),
    "ablation_seeded":  list(ABLATION_SEEDED_MODELS),
    # Beetle-FineWeb-2B bilingual NLD-ENG sweep — the five B1..B5 curricula
    # used by the xling cross-lingual analysis stack (analyze/xling/).
    # The repo names match the public HF org Beetle-FineWeb-2B.
    "beetle_fineweb_bilingual": [
        "Beetle-FineWeb-2B/beetle-bilingual-balanced-b1-fineweb-2b-nld-eng",
        "Beetle-FineWeb-2B/beetle-bilingual-l2-50-simultaneous-b2-fineweb-2b-nld-eng",
        "Beetle-FineWeb-2B/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-2b-nld-eng",
        "Beetle-FineWeb-2B/beetle-bilingual-l2-50-classroom-20-b4-fineweb-2b-nld-eng",
        "Beetle-FineWeb-2B/beetle-bilingual-l2-80-late-b5-fineweb-2b-nld-eng",
    ],
}


def _filter_by(pred) -> list:
    """Filter every repo in the new families by a predicate over parse_repo()."""
    from analyze.model_naming import parse_repo  # local import to avoid cycles
    all_repos = _merge_lang(
        HUMAN_SCALE_MODELS, FINEWEB_MODELS_PLANNED, ABLATION_SEEDED_MODELS,
    )
    return [r for r in all_repos if pred(parse_repo(r))]


# Condition-indexed groups (across humanscale / fineweb / seeded ablations)
for _cond in ("B1", "B2", "B3", "B4", "B5", "mono"):
    MODEL_GROUPS[_cond] = _filter_by(lambda m, _c=_cond: m.condition == _c)
del _cond

MODEL_GROUPS["ablations"] = _filter_by(
    lambda m: (m.variant is not None)
    or (m.cl_variant != "none")
    or m.perlang_vocab
)

# Per-variant groups (empty lists retained on purpose for downstream safety)
from analyze.model_naming import ABLATION_VARIANTS as _AV  # noqa: E402
for _v in _AV:
    MODEL_GROUPS[_v] = _filter_by(lambda m, _vv=_v: m.variant == _vv)
del _v, _AV

# Minimal ablation sweep on nld-eng (24 repos: 8 variants × 3 seeds)
MODEL_GROUPS["minimal_ablation_nld_eng"] = _filter_by(
    lambda m: m.l1 == "nld" and m.l2 == "eng" and m.seed is not None
)

# Flat deduplicated lists — stable order, used for rank::world_size slicing.
ALL_MODELS = list(dict.fromkeys(
    m for group in MODEL_GROUPS.values() for m in group
))

ALL_TRILINGUAL_MODELS = list(dict.fromkeys(TRILINGUAL_MODELS))


def get_all_models() -> list:
    """Return the deduplicated list of every model registered in MODEL_GROUPS."""
    return list(ALL_MODELS)


def get_models_for_language(lang_code: str) -> list:
    return list(MODEL_GROUPS.get(lang_code, []))


def get_seed_groups(repos) -> dict:
    """Group seeded repos by (family, label, pair) for mean-across-seeds plots."""
    from analyze.model_naming import grouping_key, parse_repo
    out: dict = {}
    for r in repos:
        key = grouping_key(parse_repo(r))
        out.setdefault(key, []).append(r)
    for key, lst in out.items():
        lst.sort()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_bilingual_type(repo: str) -> str:
    """
    Return a canonical type string for a repo.

    For Beetle-HumanScale / Beetle-FineWeb repos, delegates to
    :func:`analyze.model_naming.parse_repo` and returns the condition
    label (``B1``, ``B2+EWC(λ=20)``, etc.). For legacy BeetleLM slugs
    (``beetlelm_<pair>_<type>``), falls back to the substring tag loop.
    """
    name = repo.split("/")[-1]

    # ── New-scheme slug: delegate to model_naming.parse_repo ──────────────
    if name.startswith("beetle-bilingual-") or name.startswith("beetle-monolingual-"):
        try:
            from analyze.model_naming import parse_repo, condition_label
            return condition_label(parse_repo(repo))
        except Exception:
            pass  # fall through to legacy parsing below

    # ── Trilingual sequential sub-types (check before plain "sequential") ──
    if "sequential_ewc" in name and "maml" in name:
        return "sequential_ewc_maml"
    if "sequential_ewc" in name:
        return "sequential_ewc"
    if "sequential_maml" in name:
        return "sequential_maml"
    if "sequential_baseline" in name:
        return "sequential_baseline"

    # ── Standard bilingual types ───────────────────────────────────────────
    for tag in ("mono", "balanced", "simultaneous", "sequential",
                "part_time", "late", "heritage"):
        if tag in name:
            return tag

    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# AoA LANGUAGE REGISTRY
# Languages with standardized human AoA norms (from beetle-analyze/aoa/).
# ─────────────────────────────────────────────────────────────────────────────

LANGUAGES_WITH_AOA = {
    "deu": "aoa/standardized/deu.csv",
    "jpn": "aoa/standardized/jpn.csv",
    "spa": "aoa/standardized/spa.csv",
    "zho": "aoa/standardized/zho.csv",
    "nld": "aoa/standardized/nld.csv",
    "rus": "aoa/standardized/rus.csv",
    "por": "aoa/standardized/por.csv",
    # Additional languages from Luniewska et al. (2019) cross-linguistic norms
    "fas": "aoa/standardized/fas.csv",
    "pol": "aoa/standardized/pol.csv",
    "tur": "aoa/standardized/tur.csv",
    "ell": "aoa/standardized/ell.csv",
    "hun": "aoa/standardized/hun.csv",
    "ces": "aoa/standardized/ces.csv",
    "cat": "aoa/standardized/cat.csv",
    "dan": "aoa/standardized/dan.csv",
    "swe": "aoa/standardized/swe.csv",
    "ita": "aoa/standardized/ita.csv",
    "eng": "aoa/standardized/eng.csv",
}


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING PRIORITY RANKING
# Recommended priority order for BeetleLM training configurations.
# ─────────────────────────────────────────────────────────────────────────────

TRAINING_PRIORITY = [
    {
        "rank": 1,
        "setting": "imbalanced_bilingual_curriculum",
        "label": "Imbalanced bilingual + curriculum (L2 @ 0K / 16K / 32K / 48K steps)",
        "justification": "Drives the age-of-exposure manipulation — the most novel and theoretically motivated axis.",
    },
    {
        "rank": 2,
        "setting": "vocab_full_vs_minimal",
        "label": "Vocabulary sharing: Full Overlap vs Minimal Overlap",
        "justification": "Directly isolates embedding-sharing effects on cognate/false-friend processing.",
    },
    {
        "rank": 3,
        "setting": "vocab_friends_vs_false_friends",
        "label": "Vocabulary sharing: Friends-Only vs False-Friends-Only",
        "justification": "Fine-grained lexical control; requires (2) as baseline.",
    },
    {
        "rank": 4,
        "setting": "balanced_bilingual_24b",
        "label": "Balanced bilingual (50/50 Dutch-English, FineWeb 24B)",
        "justification": "Provides the main large-scale reference; benchmarks CLI against scale.",
    },
    {
        "rank": 5,
        "setting": "monolingual_100m",
        "label": "Monolingual Dutch / Monolingual English (100M human-scale)",
        "justification": "Baseline for CLI measurement and surprisal-RT calibration.",
    },
    {
        "rank": 6,
        "setting": "codeswitching_sentence",
        "label": "Code-switching: sentence-level",
        "justification": "Covers code-mixing as intermediate between (4) and (1).",
    },
    {
        "rank": 7,
        "setting": "codeswitching_word",
        "label": "Code-switching: word-level (Poplack constraints)",
        "justification": "Most cognitively demanding; run after (6) is validated.",
    },
    {
        "rank": 8,
        "setting": "multilingual_by_speaker",
        "label": "Multilingual by-speaker vs random sampling",
        "justification": "Tests input alignment vs distributional mixing as separate factors.",
    },
    {
        "rank": 9,
        "setting": "meco_tuned_100m",
        "label": "MECO-tuned human-scale 100M",
        "justification": "Best for psycholinguistic calibration; run once (1)-(5) are complete.",
    },
    {
        "rank": 10,
        "setting": "aoa_stratified_curriculum",
        "label": "AOA-stratified curriculum",
        "justification": "Developmental sequencing; run in parallel with (9).",
    },
]

VOCAB_CONDITIONS = [
    "full_overlap",
    "minimal_overlap",
    "friends_only",
    "false_friends_only",
]


def get_all_models() -> list[str]:
    """Return all registered models (deduplicated, stable order)."""
    return list(ALL_MODELS)


def get_models_for_language(lang_code: str) -> list[str]:
    """Return all model repos that include a given language (ISO 639-3)."""
    group = MODEL_GROUPS.get(lang_code, [])
    if group:
        return list(group)
    # Fall back: search all models for the lang code in the repo name
    return [m for m in ALL_MODELS if lang_code in m.split("/")[-1]]


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER RESOLUTION
# Single source of truth mapping a model repo to the HF repo id of the
# tokenizer it was trained with. Used by aoa/* and any future stage that
# needs tokenizer identity independent of loading the full model.
# ─────────────────────────────────────────────────────────────────────────────

from functools import lru_cache as _lru_cache


def _tokenizer_candidates(family: str, langs: list[str], prefer_lang: str | None) -> list[str]:
    """Build the ordered candidate tokenizer-repo list for a parsed model.

    Mono models → one candidate. Bilingual models → both slug orderings so
    that `bpe-humanscale-nld-eng` and `bpe-humanscale-eng-nld` are both tried
    (the Hub hosts both for every pair).
    """
    if not langs:
        return []
    if len(langs) == 1:
        lang = langs[0]
        if family == "humanscale":
            return [f"{HUMAN_SCALE_ORG}/bpe-humanscale-{lang}"]
        if family == "fineweb":
            tok2 = ISO3_TO_TOK2.get(lang)
            if tok2 is None:
                return []
            return [f"{BEETLE_DATA_ORG}/tokenizer-{tok2}-mono"]
        if family == "beetlelm_legacy":
            # Legacy BeetleLM models bundle their tokenizer in the model repo;
            # the model-repo-first strategy handles this. No slug-derived
            # fallback is known to exist as a separate repo.
            return []
        return []

    # Bilingual: try slug order, then reversed. If a specific lang is
    # preferred, try candidates whose first code matches it first.
    l1, l2 = langs[0], langs[1]
    pairs: list[tuple[str, str]] = [(l1, l2), (l2, l1)]
    if prefer_lang == l2:
        pairs = [(l2, l1), (l1, l2)]

    out: list[str] = []
    for a, b in pairs:
        if family == "humanscale":
            out.append(f"{HUMAN_SCALE_ORG}/bpe-humanscale-{a}-{b}")
        elif family == "fineweb":
            ta = ISO3_TO_TOK2.get(a)
            tb = ISO3_TO_TOK2.get(b)
            if ta and tb:
                out.append(f"{BEETLE_DATA_ORG}/tokenizer-{ta}-{tb}")
        # beetlelm_legacy: no separate tokenizer repos known; rely on model-repo.
    # Dedup while preserving order.
    seen: set = set()
    deduped: list[str] = []
    for c in out:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def _hf_repo_exists(repo_id: str) -> bool:
    """Return True iff `repo_id` resolves to a model repo with a config on the Hub."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.model_info(repo_id)
        return True
    except Exception:
        return False


def resolve_held_out_repo(regime: str, lang: str) -> str | None:
    """Return the published HuggingFace dataset repo with held-out sentences
    for the given regime+language, or None if none is known.

    humanscale → ``Beetle-HumanScale/BabyBabel-{lang}-held-out``
                 (e.g. Beetle-HumanScale/BabyBabel-nld-held-out).
    fineweb    → None (no published Hub dataset yet; pass --held-out-dir).
    """
    if regime == "humanscale":
        return f"{HUMAN_SCALE_ORG}/BabyBabel-{lang}-held-out"
    return None


@_lru_cache(maxsize=256)
def resolve_tokenizer_for_model(
    model_repo: str,
    lang: str | None = None,
    probe: bool = True,
) -> str | None:
    """Return the HF repo id of the tokenizer paired with *model_repo*.

    Strategy (model-repo first, slug-derived fallback):
      1. Return *model_repo* itself if AutoTokenizer can load from it — model
         repos ship their own tokenizer.json. Probed via `HfApi.model_info` to
         avoid downloading the full tokenizer.
      2. Else derive candidates from the slug (regime + language codes), trying
         both orderings for bilingual models. Returns the first candidate that
         exists on the Hub when `probe=True`, else the first candidate unprobed.
      3. Returns None if nothing resolves.

    If *lang* is given for a bilingual model, prefers the orientation whose
    second code is *lang* (so AoA lookups for the L2 side find the closest
    match first).
    """
    from analyze.model_naming import parse_repo  # lazy: avoid import cycle

    # Step 1: the model's own tokenizer always works when the repo is public.
    if not probe or _hf_repo_exists(model_repo):
        return model_repo

    # Step 2: slug-derived fallbacks.
    meta = parse_repo(model_repo)
    langs = [c for c in (meta.l1, meta.l2) if c]
    candidates = _tokenizer_candidates(meta.family, langs, prefer_lang=lang)
    if not probe:
        return candidates[0] if candidates else None
    for c in candidates:
        if _hf_repo_exists(c):
            return c
    return None


def get_lang_pair(repo: str) -> str:
    """
    Extract the language-pair identifier from a repo name.

    Uses a regex anchor on the first known type suffix so that compound
    trilingual sequential suffixes (e.g. '_sequential_ewc_l5_maml_h25')
    are all stripped correctly, leaving 'eng_L1-nld_L2-zho_L3'.
    """
    name = repo.split("/")[-1].replace("beetlelm_", "")
    m = re.match(
        r"^(.+?)_(mono|balanced|simultaneous|sequential|part_time|late|heritage)\b",
        name,
    )
    return m.group(1) if m else name


# ─────────────────────────────────────────────────────────────────────────────
# Baseline registries for MECO alignment comparison
#
# These are used by analyze/run_meco_comparison.py and consumed by
# analyze/meco_extended.py. Every baseline is routed through the same
# preprocessing, tokenizer-agnostic surprisal alignment, and statistical
# pipeline as Beetle models — see the plan at
# ~/.claude/plans/reactive-sprouting-hamming.md Part B for the protocol.
# ─────────────────────────────────────────────────────────────────────────────

# Catherine Arnett B-GPT 0.1B bilingual baselines (en ↔ {nl, es, el, pl} ×
# {sequential, simultaneous}).
BGPT_BASELINES = [
    "catherinearnett/B-GPT_en_nl_simultaneous",
    "catherinearnett/B-GPT_nl_en_simultaneous",
    "catherinearnett/B-GPT_en_nl_sequential",
    "catherinearnett/B-GPT_nl_en_sequential",
    "catherinearnett/B-GPT_en_es_simultaneous",
    "catherinearnett/B-GPT_es_en_simultaneous",
    "catherinearnett/B-GPT_en_es_sequential",
    "catherinearnett/B-GPT_es_en_sequential",
    "catherinearnett/B-GPT_en_el_simultaneous",
    "catherinearnett/B-GPT_el_en_simultaneous",
    "catherinearnett/B-GPT_en_el_sequential",
    "catherinearnett/B-GPT_el_en_sequential",
    "catherinearnett/B-GPT_en_pl_simultaneous",
    "catherinearnett/B-GPT_pl_en_simultaneous",
    "catherinearnett/B-GPT_en_pl_sequential",
    "catherinearnett/B-GPT_pl_en_sequential",
]

# Larger bilingual LLMs for the "bilingual_llm" group in the mixed-effects model.
# GLM-130B is gated behind --large due to hardware requirements.
BILINGUAL_LLM_BASELINES = [
    "croissantllm/CroissantLLMBase",          # French-English, 1.3B
    "m-a-p/neo_7b",                           # MAP-Neo Chinese-English, 7B
    "rinna/bilingual-gpt-neox-4b",            # Japanese-English, 4B
    # "THUDM/glm-130b",                        # gated behind --large
]

# BLOOM size ladder — uniform scale comparator (560M → 7B by default).
# BLOOM-176B gated behind --large.
BLOOM_LADDER = [
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloom-3b",
    "bigscience/bloom-7b1",
    # "bigscience/bloom",                      # 176B gated behind --large
]

# EuroLLM multilingual comparators.
EUROLLM_BASELINES = [
    "utter-project/EuroLLM-1.7B",
    "utter-project/EuroLLM-1.7B-Instruct",
    "utter-project/EuroLLM-9B",
    "utter-project/EuroLLM-9B-Instruct",
]

# Other multilingual LLMs (XLM, mT5, AfriBERTa, EuroBERT, Glot500, PolyLM,
# Emma-500, xLLama-100, xBLOOM-100). Populated opportunistically — add IDs as
# they become evaluated.
MULTILINGUAL_LLM_BASELINES = (
    BLOOM_LADDER
    + EUROLLM_BASELINES
    + [
        # "facebook/xlm-roberta-base",
        # "google/mt5-base",
        # "castorini/afriberta_large",
        # "EuroBERT/EuroBERT-210m",
        # "cis-lmu/glot500-base",
        # "DAMO-NLP-MT/polylm-1.7b",
    ]
)

# Groupings used by the mixed-effects model (model_type ∈ {...}).
# Exposed as a dict so that meco_extended.py can label every model consistently.
BASELINE_GROUPS = {
    "small_bilingual":  [],                    # populated at runtime via register_checkpoints(...)
    "bgpt":             list(BGPT_BASELINES),
    "bilingual_llm":    list(BILINGUAL_LLM_BASELINES),
    "bloom_ladder":     list(BLOOM_LADDER),
    "eurollm":          list(EUROLLM_BASELINES),
    "multilingual_llm": list(MULTILINGUAL_LLM_BASELINES),
}


def register_checkpoints(
    variant: str,
    seeds,
    base_path: str,
    *,
    group: str = "small_bilingual",
):
    """
    Dynamically extend MODEL_GROUPS / BASELINE_GROUPS in memory with the
    checkpoints produced by a multi-seed ablation sweep (e.g. the B1/B2/B3
    sweep in beetlelm/script/ablation_minimal_nld_eng.sh).

    The function mutates the in-memory registry only — it does not write
    anything to disk, so repeated calls are idempotent within a process.

    Parameters
    ----------
    variant : str
        Variant name, e.g. "B1", "B2_ewc", "B3_full".
    seeds : Iterable[int]
        Seed values used in the sweep. A separate registry entry is created
        for every (variant, seed) pair.
    base_path : str
        Either an HF-style prefix (e.g. "BeetleLM/beetlelm_ablation_nld_eng")
        or an absolute local path. Seed-tagged suffixes are appended:
        ``{base_path}_{variant}_seed{seed}``.
    group : str
        Key in BASELINE_GROUPS to register these checkpoints under. Defaults
        to "small_bilingual", the comparison group used as the mixedlm
        reference level.

    Returns
    -------
    list[str]
        The list of fully-qualified checkpoint identifiers registered.
    """
    seeds = list(seeds)
    registered = []
    for seed in seeds:
        ident = f"{base_path}_{variant}_seed{seed}"
        registered.append(ident)
        BASELINE_GROUPS.setdefault(group, []).append(ident)
        # Opportunistic: if the variant name is a known lang-code prefix,
        # also surface it in MODEL_GROUPS for downstream lookups.
        for lang_code, group_models in MODEL_GROUPS.items():
            if lang_code in ident and ident not in group_models:
                group_models.append(ident)
    return registered


def lookup_baseline_group(repo: str) -> str:
    """
    Return the baseline-group label ({small_bilingual, bgpt, bilingual_llm,
    bloom_ladder, eurollm, multilingual_llm}) that ``repo`` belongs to.

    Falls back to "unknown" when the repo is not registered.
    """
    for group, repos in BASELINE_GROUPS.items():
        if repo in repos:
            return group
    return "unknown"
