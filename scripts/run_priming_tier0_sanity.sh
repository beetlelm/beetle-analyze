#!/usr/bin/env bash
#
# Tier 0 — Sanity. Final-checkpoint smoke test on:
#   • 4 BGPT repos (re-runs the pipeline against Arnett's published numbers
#     as a cross-validation of conditional_score correctness),
#   • 1 representative bilingual eng-X / X-eng Beetle from each scale
#     (HumanScale, FineWeb canonical, FineWeb-100M, FineWeb-2B + a Polish
#     pol-eng model so Fleischer is exercised at least once).
#
# All Stage-A loss curves are skipped; we only score Stage-B (§4) priming.
# Total wall-clock: ~45 min on a single A100, ~5–10 min on 8× A100,
# ~25 hours on CPU.
#
# Outputs: results/evals/priming/tier0_sanity/
#   {schoonbaert,bernolet,fleischer}__<repo>__main.tsv
#   priming_all.tsv
#   comparison/summary.md   ← BGPT vs Beetle pivot table

# GPU/RAM budget — tier-0 mixes BGPT (small) + one 2B Beetle. Reserve enough
# host RAM per rank to load the largest model in this tier.
export PER_RANK_RAM_GB="${PER_RANK_RAM_GB:-24}"
export MIN_DISK_GB="${MIN_DISK_GB:-30}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/_priming_lib.sh"

OUTPUT_DIR="${REPO_ROOT}/results/evals/priming/tier0_sanity"

run_priming_sharded "${OUTPUT_DIR}" --revisions main <<'MODELS'
# --- BGPT ground truth (Arnett 2025) --------------------------------------
catherinearnett/B-GPT_en_nl_simultaneous
catherinearnett/B-GPT_en_nl_sequential
catherinearnett/B-GPT_nl_en_simultaneous
catherinearnett/B-GPT_nl_en_sequential

# --- One representative Beetle per scale ---------------------------------
Beetle-HumanScale/beetle-bilingual-balanced-b1-humanscale-nld-eng
Beetle-FineWeb/beetle-bilingual-balanced-b1-fineweb-eng-nld
Beetle-FineWeb-100M/beetle-bilingual-balanced-b1-fineweb-100m-nld-eng
Beetle-FineWeb-2B/beetle-bilingual-balanced-b1-fineweb-2b-nld-eng

# --- Polish (Fleischer) smoke --------------------------------------------
Beetle-FineWeb-100M/beetle-bilingual-balanced-b1-fineweb-100m-pol-eng
MODELS
