#!/usr/bin/env bash
#
# Tier 4 — Cross-typological replication trajectory (E13).
#
# 3 FineWeb-100M Polish (pol-eng) models × every checkpoint:
#   b1 balanced, b2 simultaneous, b3 sequential-33/67
#
# Uses Fleischer (2012) Active/Passive stimuli. Tests whether the curriculum
# effects observed on the Germanic eng-nld pair generalise to a Slavic pair
# with substantially different word-order properties.
#
# Polish-specific Beetle releases only exist at the 100 M scale and only in
# the pol-eng direction; b4 (classroom) and b5 (late) Polish models do not
# exist as of 2026-05-08, so this tier is intentionally narrow.
#
# Wall-clock (assume ~30 ckpts each):
#   1× A100  : ~40 min
#   8× A100  : ~5 min
#   CPU      : ~7 hours
#
# Outputs: results/evals/priming/tier4_typology/

export PER_RANK_RAM_GB="${PER_RANK_RAM_GB:-8}"   # 100 M models
export MIN_DISK_GB="${MIN_DISK_GB:-50}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/_priming_lib.sh"

OUTPUT_DIR="${REPO_ROOT}/results/evals/priming/tier4_typology"

run_priming_sharded "${OUTPUT_DIR}" "$@" <<'MODELS'
Beetle-FineWeb-100M/beetle-bilingual-balanced-b1-fineweb-100m-pol-eng
Beetle-FineWeb-100M/beetle-bilingual-l2-50-simultaneous-b2-fineweb-100m-pol-eng
Beetle-FineWeb-100M/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-pol-eng
MODELS
