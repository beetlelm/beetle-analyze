#!/usr/bin/env bash
#
# Tier 7 — Full Stage-B trajectory sweep (appendix-only).
#
# All 41 Stage-B-eligible Beetle bilingual models × every step-N HF
# branch + main. This is the union of Tiers 1–6 unfolded across the
# full training trajectory; provides the most complete picture but
# costs ~22 GPU-hours single A100.
#
# Use --max-checkpoints 12 (or smaller) to subsample the trajectory if
# wall-clock is tight.
#
# Wall-clock (assume ~30 ckpts per repo):
#   1× A100  : ~22 hours
#   8× A100  : ~3 hours
#   CPU      : ~25 days  (impractical without GPU)
#
# Strongly recommended to run on ≥ 4× A100. Rank fan-out is automatic.
#
# Outputs: results/evals/priming/tier7_full_sweep/

# Tier 7 includes the 2 B FineWeb-2B family alongside 100 M / canonical /
# HumanScale repos. Budget for the worst case so launch_ranks caps N_GPUS
# correctly when a rank gets handed a 2 B model.
export PER_RANK_RAM_GB="${PER_RANK_RAM_GB:-32}"
# 41 repos × ~30 ckpts × ~3 GB each can exceed 1 TB. Bail early if not.
export MIN_DISK_GB="${MIN_DISK_GB:-500}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/_priming_lib.sh"

OUTPUT_DIR="${REPO_ROOT}/results/evals/priming/tier7_full_sweep"

run_priming_sharded "${OUTPUT_DIR}" "$@" <<'MODELS'
# --- Beetle-HumanScale (11) ----------------------------------------------
Beetle-HumanScale/beetle-bilingual-balanced-b1-humanscale-nld-eng
Beetle-HumanScale/beetle-bilingual-balanced-b1-humanscale-nld-eng-episodic
Beetle-HumanScale/beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng
Beetle-HumanScale/beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng-ewc-l20
Beetle-HumanScale/beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng-lamol-l20
Beetle-HumanScale/beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng-lr-b2
Beetle-HumanScale/beetle-bilingual-l2-50-sequential-33-67-b3-humanscale-nld-eng
Beetle-HumanScale/beetle-bilingual-l2-50-sequential-0-100-b3-humanscale-nld-eng
Beetle-HumanScale/beetle-bilingual-l2-50-sequential-33-67-b3-humanscale-nld-eng-perlang-safe
Beetle-HumanScale/beetle-bilingual-l2-50-classroom-20-b4-humanscale-nld-eng
Beetle-HumanScale/beetle-bilingual-l2-80-late-b5-humanscale-nld-eng

# --- Beetle-FineWeb canonical (10) ---------------------------------------
Beetle-FineWeb/beetle-bilingual-balanced-b1-fineweb-eng-nld
Beetle-FineWeb/beetle-bilingual-balanced-b1-fineweb-nld-eng
Beetle-FineWeb/beetle-bilingual-l2-50-simultaneous-b2-fineweb-eng-nld
Beetle-FineWeb/beetle-bilingual-l2-50-simultaneous-b2-fineweb-nld-eng
Beetle-FineWeb/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-eng-nld
Beetle-FineWeb/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-nld-eng
Beetle-FineWeb/beetle-bilingual-balanced-b4-fineweb-eng-nld
Beetle-FineWeb/beetle-bilingual-balanced-b4-fineweb-nld-eng
Beetle-FineWeb/beetle-bilingual-balanced-b5-fineweb-eng-nld
Beetle-FineWeb/beetle-bilingual-balanced-b5-fineweb-nld-eng

# --- Beetle-FineWeb-2B (5) -----------------------------------------------
Beetle-FineWeb-2B/beetle-bilingual-balanced-b1-fineweb-2b-nld-eng
Beetle-FineWeb-2B/beetle-bilingual-l2-50-simultaneous-b2-fineweb-2b-nld-eng
Beetle-FineWeb-2B/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-2b-nld-eng
Beetle-FineWeb-2B/beetle-bilingual-l2-50-classroom-20-b4-fineweb-2b-nld-eng
Beetle-FineWeb-2B/beetle-bilingual-l2-80-late-b5-fineweb-2b-nld-eng

# --- Beetle-FineWeb-100M nld-eng (5 + 7 ablations) -----------------------
Beetle-FineWeb-100M/beetle-bilingual-balanced-b1-fineweb-100m-nld-eng
Beetle-FineWeb-100M/beetle-bilingual-l2-50-simultaneous-b2-fineweb-100m-nld-eng
Beetle-FineWeb-100M/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-nld-eng
Beetle-FineWeb-100M/beetle-bilingual-l2-50-classroom-20-b4-fineweb-100m-nld-eng
Beetle-FineWeb-100M/beetle-bilingual-l2-80-late-b5-fineweb-100m-nld-eng
Beetle-FineWeb-100M/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-nld-eng-safe
Beetle-FineWeb-100M/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-nld-eng-ewc
Beetle-FineWeb-100M/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-nld-eng-lamol
Beetle-FineWeb-100M/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-nld-eng-maml
Beetle-FineWeb-100M/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-nld-eng-pre-pretrain
Beetle-FineWeb-100M/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-nld-eng-alibi
Beetle-FineWeb-100M/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-nld-eng-layer-growth

# --- Beetle-FineWeb-100M Polish (3) --------------------------------------
Beetle-FineWeb-100M/beetle-bilingual-balanced-b1-fineweb-100m-pol-eng
Beetle-FineWeb-100M/beetle-bilingual-l2-50-simultaneous-b2-fineweb-100m-pol-eng
Beetle-FineWeb-100M/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-pol-eng
MODELS
