#!/usr/bin/env bash
#
# Tier 2 — Headline developmental trajectory figure (E11).
#
# 8 models × every step-N HF branch + main:
#   • HumanScale: b1, b2, b3-33/67, b5  (nld-eng)
#   • FineWeb canonical: b1, b2, b3-33/67, b5  (nld-eng)
#
# Answers Q4: when in training do shared structural representations emerge,
# and how does curriculum modulate that timing?
#
# Wall-clock (assuming ~30 step-N branches per repo):
#   1× A100  : ~3.5 hours
#   8× A100  : ~30 min
#   CPU      : ~70 hours (impractical on this scale)
#
# Coarse-grained sweep variant: pass `--max-checkpoints 12` on the CLI to
# subsample to 12 evenly-spaced checkpoints + main (~1.5 hours single A100).
#
# Outputs: results/evals/priming/tier2_headline_trajectory/

# Tier 2: HumanScale + FineWeb canonical only (≤ ~500 M params).
export PER_RANK_RAM_GB="${PER_RANK_RAM_GB:-12}"
# Trajectory tiers cache every step-N branch — up to ~3 GB × 30 ckpts × 8 repos.
export MIN_DISK_GB="${MIN_DISK_GB:-200}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/_priming_lib.sh"

OUTPUT_DIR="${REPO_ROOT}/results/evals/priming/tier2_headline_trajectory"

run_priming_sharded "${OUTPUT_DIR}" "$@" <<'MODELS'
# --- HumanScale (4 curricula, nld-eng) -----------------------------------
Beetle-HumanScale/beetle-bilingual-balanced-b1-humanscale-nld-eng
Beetle-HumanScale/beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng
Beetle-HumanScale/beetle-bilingual-l2-50-sequential-33-67-b3-humanscale-nld-eng
Beetle-HumanScale/beetle-bilingual-l2-80-late-b5-humanscale-nld-eng

# --- FineWeb canonical (same 4 curricula, nld-eng) -----------------------
Beetle-FineWeb/beetle-bilingual-balanced-b1-fineweb-nld-eng
Beetle-FineWeb/beetle-bilingual-l2-50-simultaneous-b2-fineweb-nld-eng
Beetle-FineWeb/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-nld-eng
Beetle-FineWeb/beetle-bilingual-balanced-b5-fineweb-nld-eng
MODELS
