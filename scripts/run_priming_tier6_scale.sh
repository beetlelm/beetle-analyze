#!/usr/bin/env bash
#
# Tier 6 — Scale × trajectory at fixed curriculum (E15).
#
# 4 b1-balanced nld-eng models × every checkpoint, one per training scale:
#   • Beetle-HumanScale  (~human-scale token budget)
#   • Beetle-FineWeb     (canonical FineWeb1 scale)
#   • Beetle-FineWeb-100M (100 M params)
#   • Beetle-FineWeb-2B   (2 B params)
#
# Holds curriculum (b1 balanced) and language pair (nld-eng) fixed and
# varies only model/data scale, so the resulting figure is a clean
# scaling-law-style plot of cross-lingual structural priming vs compute.
#
# Note: 2 B forward passes are ~3× slower per token than 100 M, so this
# tier is dominated by the 2 B model's wall-clock.
#
# Wall-clock (assume ~30 ckpts per repo):
#   1× A100  : ~2.5 hours (limited by FineWeb-2B)
#   8× A100  : ~25 min
#   CPU      : impractical for 2 B (~30 min/ckpt × 30 = ~15 hours single CPU)
#
# Outputs: results/evals/priming/tier6_scale/

# Tier 6 is dominated by FineWeb-2B (largest model in any tier):
# bf16 weights ~4 GB + activations + KV cache ~2-3 GB → reserve a generous
# host-RAM budget so launch_ranks doesn't oversubscribe.
export PER_RANK_RAM_GB="${PER_RANK_RAM_GB:-32}"
export MIN_DISK_GB="${MIN_DISK_GB:-200}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/_priming_lib.sh"

OUTPUT_DIR="${REPO_ROOT}/results/evals/priming/tier6_scale"

run_priming_sharded "${OUTPUT_DIR}" "$@" <<'MODELS'
Beetle-HumanScale/beetle-bilingual-balanced-b1-humanscale-nld-eng
Beetle-FineWeb/beetle-bilingual-balanced-b1-fineweb-nld-eng
Beetle-FineWeb-100M/beetle-bilingual-balanced-b1-fineweb-100m-nld-eng
Beetle-FineWeb-2B/beetle-bilingual-balanced-b1-fineweb-2b-nld-eng
MODELS
