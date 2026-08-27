#!/usr/bin/env bash
#
# scripts/_priming_lib.sh — shared multi-GPU fan-out for tiered Stage-B
# (structural priming) replications of Arnett (2025) on Beetle.
#
# This is a thin wrapper over scripts/_sweep_safety.sh, which provides:
#   • RAM-aware concurrency cap (PER_RANK_RAM_GB → caps N_GPUS),
#   • trap-protected rank fan-out (one rank dying SIGTERMs all peers, then
#     SIGKILL after a 10 s grace period),
#   • setsid detach (closing the SSH session does not SIGHUP the sweep),
#   • canonical HF cache (never tmpfs), MALLOC_ARENA_MAX=2,
#     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, no tokenizer
#     thread storms, long HF download timeout.
#
# Usage: source this file and call:
#
#     run_priming_sharded OUTPUT_DIR [extra args ...] <<< "$MODELS"
#
# where $MODELS is a newline-separated list of HF repo IDs (one per line,
# blank lines and lines starting with '#' are ignored). Extra args are
# forwarded verbatim to evals/eval_priming.py (e.g. --revisions main,
# --max-checkpoints 12).
#
# Environment:
#   N_GPUS               explicit override (default: auto-detect, capped by RAM)
#   PER_RANK_RAM_GB      RAM budget per rank (default: 12; tier scripts override)
#   BEETLE_HF_CACHE_ROOT canonical HF cache (default: ~/.cache/beetle_hf)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Pull in shared safety harness (RAM-cap, trap-protected fan-out, env hygiene).
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_sweep_safety.sh"
sweep_safety_init

run_priming_sharded() {
    local output_dir="$1"; shift
    local extra_args=("$@")

    # ----- read model list from stdin (skip blanks + '# comments') ---------
    local models=()
    while IFS= read -r line; do
        line="${line%%#*}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z "${line}" ]] && continue
        models+=("${line}")
    done

    if [[ ${#models[@]} -eq 0 ]]; then
        _sweep_log "no models supplied on stdin; aborting." >&2
        return 2
    fi

    mkdir -p "${output_dir}/logs"

    # ----- derive a unique stage label so SWEEP_LOG_DIR is per-tier --------
    local stage
    stage="priming_$(basename "${output_dir}")"
    export SWEEP_LOG_DIR="${output_dir}/logs"

    _sweep_log "OUTPUT_DIR=${output_dir}"
    _sweep_log "Models (${#models[@]}):"
    printf '   %s\n' "${models[@]}" >&2
    _sweep_log "Extra args: ${extra_args[*]:-}"
    _sweep_log "PER_RANK_RAM_GB=${PER_RANK_RAM_GB:-12}"

    cd "${REPO_ROOT}"

    # ----- preflight: refuse to launch if disk free is dangerously low -----
    # HF caches a copy of every checkpoint we download; trajectory tiers can
    # easily exceed 200 GB. Bail early rather than hit ENOSPC mid-run.
    local cache_avail_gb
    cache_avail_gb=$(df -BG "${BEETLE_HF_CACHE_ROOT}" 2>/dev/null \
        | awk 'NR==2 {gsub("G","",$4); print $4}' \
        || echo 0)
    cache_avail_gb=${cache_avail_gb:-0}
    local min_disk_gb="${MIN_DISK_GB:-50}"
    if (( cache_avail_gb < min_disk_gb )); then
        _sweep_log "FATAL: ${BEETLE_HF_CACHE_ROOT} has only ${cache_avail_gb} GiB free (< ${min_disk_gb} GiB)." >&2
        _sweep_log "       Free disk or override MIN_DISK_GB=N before re-launching." >&2
        return 3
    fi
    _sweep_log "preflight: ${BEETLE_HF_CACHE_ROOT} has ${cache_avail_gb} GiB free (need ${min_disk_gb})"

    # ----- build a per-rank shell command for launch_ranks -----------------
    # launch_ranks expects an argv that runs once per rank; the harness
    # itself sets RANK / WORLD_SIZE / N_GPUS in each child's env. We embed
    # the model list and extra args as positional argv to a tiny wrapper.
    local wrapper
    wrapper="${output_dir}/.run_rank.sh"
    cat > "${wrapper}" <<'WRAPPER'
#!/usr/bin/env bash
# Per-rank wrapper invoked by scripts/_sweep_safety.sh::launch_ranks.
# Inherits RANK / WORLD_SIZE / N_GPUS from the harness.
set -euo pipefail
OUTPUT_DIR="$1"; shift
# All remaining argv is split: anything before '--' is models, after is extra.
models=()
extras=()
seen_sep=0
for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then seen_sep=1; continue; fi
    if (( seen_sep )); then extras+=("$arg"); else models+=("$arg"); fi
done
exec python evals/eval_priming.py \
    --models "${models[@]}" \
    --output "${OUTPUT_DIR}" \
    --rank "${RANK}" --world-size "${WORLD_SIZE}" \
    --skip-loss \
    "${extras[@]}"
WRAPPER
    chmod +x "${wrapper}"

    # Use launch_ranks (RAM-capped, trap-protected, setsid-detached).
    if ! launch_ranks "${stage}" \
            "${wrapper}" "${output_dir}" \
            "${models[@]}" -- "${extra_args[@]}"; then
        _sweep_log "${stage}: rank fan-out failed; see ${SWEEP_LOG_DIR}/${stage}/"
        return 1
    fi

    rm -f "${wrapper}"

    _sweep_log "${stage}: all ranks finished cleanly; running BGPT comparison."
    python evals/compare_priming_to_bgpt.py \
        --beetle-dir "${output_dir}" \
        --arnett-dir "${REPO_ROOT}/arnett-2025-priming-raw" \
        --output     "${output_dir}/comparison" || \
        _sweep_log "comparison emitted warnings; see ${output_dir}/comparison/."

    _sweep_log "summary: ${output_dir}/comparison/summary.md"
}
