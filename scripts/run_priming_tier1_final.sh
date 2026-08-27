#!/usr/bin/env bash
#
# Tier 1 — Core EMNLP final-checkpoint results (E1–E10).
#
# All 41 Stage-B-eligible Beetle bilingual configs at the 'main' revision
# only. Populates:
#   • Q1 curriculum contrast (b1/b2/b3/b4/b5) at each scale,
#   • Q2 scale × curriculum interaction,
#   • Q3 direction asymmetry (FineWeb has both eng-nld and nld-eng),
#   • Q5 continual-learning + architecture interventions,
#   • Q6 cross-typological replication (eng-pol).
#
# Total: 41 final-ckpt evals.
# Wall-clock: ~30 min on 1× A100, ~5 min on 8× A100, ~15 hours on CPU.
#
# Outputs: results/evals/priming/tier1_final/

# Tier 1 includes 5 FineWeb-2B models (largest at 2 B params, ~8 GB bf16).
# Reserve enough host RAM per rank that 8 ranks won't OOM the kernel.
export PER_RANK_RAM_GB="${PER_RANK_RAM_GB:-24}"
export MIN_DISK_GB="${MIN_DISK_GB:-80}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/_priming_lib.sh"

OUTPUT_DIR="${REPO_ROOT}/results/evals/priming/tier1_final"

run_priming_sharded "${OUTPUT_DIR}" --revisions main <<'MODELS'
# --- Beetle-HumanScale (~human-scale tokens, nld-eng only) ---------------
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

# --- Beetle-FineWeb canonical (both directions across all 5 curricula) ---
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

# --- Beetle-FineWeb-2B (2 B params, nld-eng only) ------------------------
Beetle-FineWeb-2B/beetle-bilingual-balanced-b1-fineweb-2b-nld-eng
Beetle-FineWeb-2B/beetle-bilingual-l2-50-simultaneous-b2-fineweb-2b-nld-eng
Beetle-FineWeb-2B/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-2b-nld-eng
Beetle-FineWeb-2B/beetle-bilingual-l2-50-classroom-20-b4-fineweb-2b-nld-eng
Beetle-FineWeb-2B/beetle-bilingual-l2-80-late-b5-fineweb-2b-nld-eng

# --- Beetle-FineWeb-100M (eng-nld curricula + b3 ablations) --------------
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

# --- Beetle-FineWeb-100M Polish (Fleischer A/P stimuli) ------------------
Beetle-FineWeb-100M/beetle-bilingual-balanced-b1-fineweb-100m-pol-eng
Beetle-FineWeb-100M/beetle-bilingual-l2-50-simultaneous-b2-fineweb-100m-pol-eng
Beetle-FineWeb-100M/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-pol-eng
MODELS
