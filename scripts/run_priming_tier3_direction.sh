#!/usr/bin/env bash
#
# Tier 3 — Direction-asymmetry trajectory (E12).
#
# 6 FineWeb-canonical models × every checkpoint:
#   b1, b2, b3-33/67  ×  {eng-nld, nld-eng}
#
# Tests Q3: how does training direction (Eng-as-L1 vs Nld-as-L1) interact
# with curriculum to produce asymmetric structural priming, and when in
# training does that asymmetry emerge?
#
# FineWeb is the only org that ships both directions across all curricula.
#
# Wall-clock (assume ~30 ckpts each):
#   1× A100  : ~2.5 hours
#   8× A100  : ~20 min
#   CPU      : ~50 hours
#
# Outputs: results/evals/priming/tier3_direction/

export PER_RANK_RAM_GB="${PER_RANK_RAM_GB:-12}"
export MIN_DISK_GB="${MIN_DISK_GB:-150}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/_priming_lib.sh"

OUTPUT_DIR="${REPO_ROOT}/results/evals/priming/tier3_direction"

run_priming_sharded "${OUTPUT_DIR}" "$@" <<'MODELS'
Beetle-FineWeb/beetle-bilingual-balanced-b1-fineweb-eng-nld
Beetle-FineWeb/beetle-bilingual-balanced-b1-fineweb-nld-eng
Beetle-FineWeb/beetle-bilingual-l2-50-simultaneous-b2-fineweb-eng-nld
Beetle-FineWeb/beetle-bilingual-l2-50-simultaneous-b2-fineweb-nld-eng
Beetle-FineWeb/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-eng-nld
Beetle-FineWeb/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-nld-eng
MODELS
