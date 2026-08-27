#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# eval_loss_curves_flores_dutch_v2.sh
#
# Refresh of the FLORES BPT Dutch-vs-English trajectory cohort, expanded to
# every nld-eng / eng-nld Beetle repo across the four published HF orgs:
#
#   Beetle-FineWeb           (24 B tok backbone — original cohort, log-thinned
#                             to ~25 ckpts/repo from its dense 200+ saved)
#   Beetle-FineWeb-2B        (2 B tok scale)
#   Beetle-FineWeb-100M      (100 M tok scale)
#   Beetle-nld-eng-Variants  (architecture variants on the 2 B B3 backbone)
#   catherinearnett (B-GPT)  (4 baseline variants)
#
# The original eval_loss_curves_bgpt_vs_beetle_dutch.sh resolves Beetle from
# models.FINEWEB_MODELS, which is a curated subset and so silently misses the
# B2/B3-0/B3-33 simultaneous and the smaller-scale orgs entirely. This driver
# enumerates the cohort explicitly (one place to update), and forces the
# analyze.convergence list-checkpoints --log-spaced flag so the dense
# original-cohort ckpts are thinned to a publication-grade trajectory.
#
# Output CSVs:
#   results/convergence/bpt_traj_<repo-slug>_flores.csv
#
# Safety:
#   - Does NOT acquire the beetle-pipeline BP_GPU_LOCK (this is eval, not
#     training). Compatible with /babysit-beetle as long as no training is
#     in flight (caller must verify).
#   - Per-checkpoint subprocess isolation via _per_ckpt_loop in the shared
#     lib — host RAM never balloons.
#   - HF cache lives at $HF_HOME (default /root/.hf_cache), shared across
#     all venvs.
#
# Usage:
#   bash scripts/eval_loss_curves_flores_dutch_v2.sh
#   N_GPUS=4 bash scripts/eval_loss_curves_flores_dutch_v2.sh
#   OVERWRITE=1 bash scripts/eval_loss_curves_flores_dutch_v2.sh
# ─────────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")/.."

if [[ -z "${RANK:-}" ]]; then
    # shellcheck disable=SC1091
    source scripts/_sweep_safety.sh
    sweep_safety_init
    SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-logs/loss_curves_v2}"
    : "${LANG_FILTER:=eng nld}"
    : "${LOG_SPACED:=1}"
    : "${SAMPLE_SIZE:=256}"
    export OVERWRITE LANG_FILTER HF_TOKEN SAMPLE_SIZE LOG_SPACED
    echo "[flores-v2] dispatching via launch_ranks  LOG_SPACED=$LOG_SPACED"
    launch_ranks "loss_curves_flores_v2" bash "$0"
    exit $?
fi

source scripts/_loss_curves_lib.sh

LANG_FILTER="${LANG_FILTER-eng nld}"

# ─── Cohort: BGPT (4) + Beetle nld-eng across all four scales ───────────────
ALL_REPOS=(
    # B-GPT baselines (4) — already on Hub, kept as the comparison anchor.
    "catherinearnett/B-GPT_en_nl_simultaneous"
    "catherinearnett/B-GPT_en_nl_sequential"
    "catherinearnett/B-GPT_nl_en_simultaneous"
    "catherinearnett/B-GPT_nl_en_sequential"

    # Beetle-FineWeb (24 B tok) — full B1..B5 × {nld-eng, eng-nld}.
    "Beetle-FineWeb/beetle-bilingual-balanced-b1-fineweb-nld-eng"
    "Beetle-FineWeb/beetle-bilingual-balanced-b1-fineweb-eng-nld"
    "Beetle-FineWeb/beetle-bilingual-l2-50-simultaneous-b2-fineweb-nld-eng"
    "Beetle-FineWeb/beetle-bilingual-l2-50-simultaneous-b2-fineweb-eng-nld"
    "Beetle-FineWeb/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-nld-eng"
    "Beetle-FineWeb/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-eng-nld"
    "Beetle-FineWeb/beetle-bilingual-balanced-b4-fineweb-nld-eng"
    "Beetle-FineWeb/beetle-bilingual-balanced-b4-fineweb-eng-nld"
    "Beetle-FineWeb/beetle-bilingual-balanced-b5-fineweb-nld-eng"
    "Beetle-FineWeb/beetle-bilingual-balanced-b5-fineweb-eng-nld"

    # Beetle-FineWeb-2B (2 B tok scale) — full B1..B5 nld-eng.
    "Beetle-FineWeb-2B/beetle-bilingual-balanced-b1-fineweb-2b-nld-eng"
    "Beetle-FineWeb-2B/beetle-bilingual-l2-50-simultaneous-b2-fineweb-2b-nld-eng"
    "Beetle-FineWeb-2B/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-2b-nld-eng"
    "Beetle-FineWeb-2B/beetle-bilingual-l2-50-classroom-20-b4-fineweb-2b-nld-eng"
    "Beetle-FineWeb-2B/beetle-bilingual-l2-80-late-b5-fineweb-2b-nld-eng"

    # Beetle-FineWeb-100M (100 M tok scale) — full B1..B5 nld-eng.
    "Beetle-FineWeb-100M/beetle-bilingual-balanced-b1-fineweb-100m-nld-eng"
    "Beetle-FineWeb-100M/beetle-bilingual-l2-50-simultaneous-b2-fineweb-100m-nld-eng"
    "Beetle-FineWeb-100M/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-nld-eng"
    "Beetle-FineWeb-100M/beetle-bilingual-l2-50-classroom-20-b4-fineweb-100m-nld-eng"
    "Beetle-FineWeb-100M/beetle-bilingual-l2-80-late-b5-fineweb-100m-nld-eng"

    # Architecture variants on the 2B B3 nld-eng backbone.
    "Beetle-nld-eng-Variants/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-2b-nld-eng-arch-bgpt"
    "Beetle-nld-eng-Variants/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-2b-nld-eng-arch-moe"
)

echo "[flores-v2] cohort size: ${#ALL_REPOS[@]} repos"

mapfile -t MY_REPOS < <(shard_repos "${ALL_REPOS[@]}")
echo "[flores-v2] rank ${RANK}/${WORLD_SIZE} → ${#MY_REPOS[@]} repos"

# Per-repo FLORES BPT trajectories. Held-out is intentionally skipped: the
# F10 plot uses FLORES only, and a held-out probe for every smaller-scale
# org would double cost without changing the figure.
for repo in "${MY_REPOS[@]}"; do
    run_bpt_traj "$repo" flores $LANG_FILTER || \
        echo "[flores-v2] $repo: trajectory ended with non-zero rc; continuing"
done

echo "[flores-v2] rank ${RANK} done."
