#!/usr/bin/env bash
#
# Tier 5 — Continual-learning + architecture intervention trajectories (E14).
#
# 12 models × every checkpoint, two intervention families:
#
#   HumanScale b2 (simultaneous) family
#     • vanilla
#     • EWC (Elastic Weight Consolidation)
#     • LAMOL (Language Model + Episodic Memory)
#     • LR-shrink-b2 (learning-rate intervention)
#
#   FineWeb-100M b3 (sequential 33/67) family
#     • vanilla
#     • perlang-safe (per-language safe replay)
#     • EWC
#     • LAMOL
#     • MAML (Model-Agnostic Meta-Learning)
#     • Pre-pretrain warm-up
#     • ALiBi positional encoding
#     • Layer-growth schedule
#
# Tests Q5: can any continual-learning or architectural intervention recover
# the cross-lingual structural-priming benefit that vanilla sequential
# loses (per Arnett 2025)?
#
# Wall-clock (assume ~30 ckpts each):
#   1× A100  : ~2.5 hours
#   8× A100  : ~20 min
#   CPU      : ~50 hours
#
# Outputs: results/evals/priming/tier5_interventions/

export PER_RANK_RAM_GB="${PER_RANK_RAM_GB:-12}"  # HS ~125 M + 100 M
export MIN_DISK_GB="${MIN_DISK_GB:-150}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/_priming_lib.sh"

OUTPUT_DIR="${REPO_ROOT}/results/evals/priming/tier5_interventions"

run_priming_sharded "${OUTPUT_DIR}" "$@" <<'MODELS'
# --- HumanScale b2 family ------------------------------------------------
Beetle-HumanScale/beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng
Beetle-HumanScale/beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng-ewc-l20
Beetle-HumanScale/beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng-lamol-l20
Beetle-HumanScale/beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng-lr-b2

# --- FineWeb-100M b3 family ----------------------------------------------
Beetle-FineWeb-100M/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-nld-eng
Beetle-FineWeb-100M/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-nld-eng-safe
Beetle-FineWeb-100M/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-nld-eng-ewc
Beetle-FineWeb-100M/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-nld-eng-lamol
Beetle-FineWeb-100M/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-nld-eng-maml
Beetle-FineWeb-100M/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-nld-eng-pre-pretrain
Beetle-FineWeb-100M/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-nld-eng-alibi
Beetle-FineWeb-100M/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-nld-eng-layer-growth
MODELS
