#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/_sweep_safety.sh
#
# Shared launcher contract for every GPU-touching sweep in this repo.
#
# Goals:
#   1. Cap concurrency by host RAM, not just GPU count.  Two GPU nodes have
#      already been killed by 8 ranks each loading a model in parallel on a
#      box with insufficient host RAM.
#   2. Contain failures.  When one rank dies, the others must die too — they
#      cannot be allowed to keep allocating against an already-stressed kernel.
#   3. Make HF / glibc / Torch behave: single canonical cache (never tmpfs),
#      MALLOC_ARENA_MAX=2, expandable_segments, no tokenizer thread storms.
#   4. Detach from SSH (setsid) so a closed session does not SIGHUP the sweep.
#
# Public functions:
#   sweep_safety_init              — must be sourced & called at the top
#   compute_safe_n_gpus            — echoes capped N_GPUS to stdout
#   launch_ranks <stage> <cmd...>  — fan-out N_GPUS pinned ranks; trap-protected
#   tail_log_pidbound <file>       — pid-bound tail that dies with the parent
#
# Env knobs (with defaults):
#   N_GPUS                — explicit override; otherwise auto-detected
#   PER_RANK_RAM_GB       — host-RAM budget per rank (default 12)
#   BEETLE_HF_CACHE_ROOT  — canonical HF cache (default ${HOME}/.cache/beetle_hf)
#   SWEEP_LOG_DIR         — per-rank log directory (default results/logs/sweep)
# ─────────────────────────────────────────────────────────────────────────────

# Idempotent: source once.
if [[ -n "${_SWEEP_SAFETY_LOADED:-}" ]]; then
    return 0 2>/dev/null || exit 0
fi
_SWEEP_SAFETY_LOADED=1

_sweep_log() {
    echo "[sweep_safety $(date -u +%H:%M:%S)] $*" >&2
}

# ─── 1. Environment hygiene ─────────────────────────────────────────────────
sweep_safety_init() {
    # glibc arena bloat is the silent killer on long-running Python+Torch
    # processes that load and free models repeatedly.  Two arenas is the
    # standard remedy.
    export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

    # Promote eval-final.sh's allocator config to every sweep.
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

    # Tokenizer threads × N ranks × M models = thread storm.  Pin to 1.
    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

    # Long downloads on flaky links should not abort the sweep.
    export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
    export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

    # One canonical HF cache, never tmpfs (RAM-backed on most Linux nodes).
    local cache_root="${BEETLE_HF_CACHE_ROOT:-${HOME}/.cache/beetle_hf}"
    mkdir -p "$cache_root"
    export BEETLE_HF_CACHE_ROOT="$cache_root"
    export HF_HOME="${HF_HOME:-$cache_root}"
    export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$cache_root/hub}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$cache_root/transformers}"
    export BEETLE_HF_CACHE="${BEETLE_HF_CACHE:-$cache_root}"
    mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

    # Per-rank logs land here unless the caller overrides.
    export SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-results/logs/sweep}"
    mkdir -p "$SWEEP_LOG_DIR"

    _sweep_log "init: MALLOC_ARENA_MAX=$MALLOC_ARENA_MAX"
    _sweep_log "init: HF_HOME=$HF_HOME"
    _sweep_log "init: PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
}

# ─── 2. Concurrency cap ─────────────────────────────────────────────────────
# Echoes the safe rank count to stdout.  Stderr gets the rationale.
compute_safe_n_gpus() {
    local requested="${N_GPUS:-}"
    local detected

    if [[ -z "$requested" ]]; then
        if command -v nvidia-smi >/dev/null 2>&1; then
            detected=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
        else
            detected=1
        fi
        requested="${detected:-1}"
    fi

    local per_rank_gb="${PER_RANK_RAM_GB:-12}"
    local mem_kb mem_gb cap

    # Linux: /proc/meminfo MemAvailable.  macOS: hw.memsize.  Else bail.
    if [[ -r /proc/meminfo ]]; then
        mem_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        mem_gb=$(( mem_kb / 1024 / 1024 ))
    elif command -v sysctl >/dev/null 2>&1; then
        mem_gb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024 ))
    else
        mem_gb=0
    fi

    if (( mem_gb <= 0 )); then
        # Cannot read host RAM; trust the caller.
        _sweep_log "could not read host RAM; not capping N_GPUS=$requested"
        echo "$requested"
        return
    fi

    cap=$(( mem_gb / per_rank_gb ))
    (( cap < 1 )) && cap=1

    local final="$requested"
    if (( cap < requested )); then
        _sweep_log "capping N_GPUS $requested -> $cap (host has ${mem_gb} GiB; budget ${per_rank_gb} GiB/rank)"
        final="$cap"
    else
        _sweep_log "N_GPUS=$requested fits (host has ${mem_gb} GiB; budget ${per_rank_gb} GiB/rank, room for $cap)"
    fi
    echo "$final"
}

# ─── 3. Pid-bound tail (auto-dies with parent) ──────────────────────────────
tail_log_pidbound() {
    local file="$1"
    [[ -z "$file" ]] && return 0
    # GNU tail supports --pid; BSD tail does not.  Try the safer form first.
    if tail --help 2>&1 | grep -q -- '--pid'; then
        tail --pid=$$ -F "$file" 2>/dev/null &
    else
        # BSD fallback: spawn a watchdog that kills the tail when we die.
        tail -F "$file" 2>/dev/null &
        local tail_pid=$!
        ( while kill -0 $$ 2>/dev/null; do sleep 5; done; kill "$tail_pid" 2>/dev/null ) &
    fi
}

# ─── 4. Trap-protected rank fan-out ─────────────────────────────────────────
# Usage:   launch_ranks <stage_label> <cmd> [arg ...]
#
# The command is run once per rank, with these env vars exported per child:
#   RANK, WORLD_SIZE, CUDA_VISIBLE_DEVICES, N_GPUS
# stdout/stderr go to "$SWEEP_LOG_DIR/<stage>_rank<R>.log".
#
# A SIGTERM in any child triggers SIGTERM to all peers, then a non-zero exit.
launch_ranks() {
    local stage="$1"
    shift
    if [[ $# -eq 0 ]]; then
        _sweep_log "launch_ranks: no command supplied for stage=$stage" >&2
        return 2
    fi

    local n_gpus
    n_gpus=$(compute_safe_n_gpus)
    export N_GPUS="$n_gpus"

    local logdir="$SWEEP_LOG_DIR/$stage"
    mkdir -p "$logdir"
    _sweep_log "launch_ranks stage=$stage N_GPUS=$n_gpus log_dir=$logdir"

    # Build pid array; install a single trap that fires once on any signal.
    local -a pids=()
    local _terminating=0
    _terminate_peers() {
        (( _terminating )) && return
        _terminating=1
        _sweep_log "stage=$stage terminating peers ${pids[*]}" >&2
        kill -TERM "${pids[@]}" 2>/dev/null || true
        # Give children 10 s to exit cleanly, then SIGKILL stragglers.
        local deadline=$(( SECONDS + 10 ))
        while (( SECONDS < deadline )); do
            local alive=0
            for p in "${pids[@]}"; do
                kill -0 "$p" 2>/dev/null && alive=1
            done
            (( alive == 0 )) && break
            sleep 1
        done
        kill -KILL "${pids[@]}" 2>/dev/null || true
    }
    trap _terminate_peers INT TERM ERR

    local r
    for r in $(seq 0 $((n_gpus - 1))); do
        local logfile="$logdir/rank_${r}.log"
        # setsid: detach from this controlling terminal so a closed SSH
        #          session (SIGHUP) does not propagate to the sweep.
        # env -i would be nicer but children expect inherited PYTHONPATH etc.
        CUDA_VISIBLE_DEVICES="$r" \
        RANK="$r" \
        WORLD_SIZE="$n_gpus" \
        N_GPUS="$n_gpus" \
        setsid "$@" > "$logfile" 2>&1 &
        pids+=($!)
        _sweep_log "  stage=$stage rank=$r pid=${pids[-1]} log=$logfile"
    done

    # Live-tail rank 0 so the foreground console isn't silent.
    tail_log_pidbound "$logdir/rank_0.log"

    # Poll for any failure.  bash's `wait -n` returns when ANY child exits
    # (success or fail); we then check whether peers are still alive.
    local exited=0 rc=0 final_rc=0 first_failure=""
    while (( exited < ${#pids[@]} )); do
        if wait -n 2>/dev/null; then
            rc=0
        else
            rc=$?
        fi
        exited=$(( exited + 1 ))
        if (( rc != 0 )) && [[ -z "$first_failure" ]]; then
            first_failure="rc=$rc"
            final_rc=$rc
            _sweep_log "stage=$stage rank exited non-zero (rc=$rc); terminating peers"
            _terminate_peers
        fi
    done

    trap - INT TERM ERR
    if (( final_rc != 0 )); then
        _sweep_log "stage=$stage finished with failure ($first_failure); see $logdir"
        return "$final_rc"
    fi
    _sweep_log "stage=$stage all $n_gpus ranks completed cleanly"
    return 0
}
