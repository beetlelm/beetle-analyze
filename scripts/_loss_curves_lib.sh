#!/usr/bin/env bash
# Shared helpers for loss-curve trajectory drivers.
#
# Per-checkpoint subprocess isolation:
#   The historical pattern was a single long-running Python process per repo
#   that iterated over every checkpoint in-process. Cumulative RSS growth
#   from glibc arena bloat + repeated HF model loads caused host-RAM
#   exhaustion that killed two GPU nodes. Each checkpoint is now scored in
#   its own short-lived subprocess via the new
#   ``analyze.convergence list-checkpoints`` and ``--checkpoint`` plumbing,
#   so the kernel reclaims everything between checkpoints.
#
# Provides:
#   - shard_repos        : split an array of repos round-robin by RANK/WORLD_SIZE
#   - run_flores_traj    : per-checkpoint FLORES NLL for one repo+langs
#   - run_heldout_traj   : per-checkpoint held-out NLL for one repo+langs
#   - run_bpt_traj       : per-checkpoint BPT for one repo+langs (FLORES or held-out)
#   - langs_for_repo     : derive eval languages from a repo name
#   - filter_repos_by_lang : keep only repos that include a given language
#
# Driver scripts source this file and supply the model lists.

set -euo pipefail

# Promote the shared launcher contract: MALLOC_ARENA_MAX, expandable_segments,
# canonical HF cache, etc. The launcher script in turn relies on these being
# applied to every rank by the time we start scoring.
_THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_THIS_DIR}/_sweep_safety.sh"
sweep_safety_init

# ---------------------------------------------------------------------------
# Shared environment
# ---------------------------------------------------------------------------
N_GPUS="${N_GPUS:-1}"
RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-${N_GPUS}}"
HF_TOKEN_ARG=""
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_ARG="--hf-token ${HF_TOKEN}"
fi

SAMPLE_SIZE="${SAMPLE_SIZE:-256}"
ALL_CKPTS_FLAG="--all-checkpoints"
if [[ "${CKPT_SCOPE:-all}" == "latest" ]]; then
  ALL_CKPTS_FLAG=""
fi
# Set LOG_SPACED=1 to thin step-N branches to a log-spaced subset (~25 ckpts
# per repo) before scoring. Used by drivers that re-evaluate the original
# Beetle-FineWeb 24B repos, which saved 200+ irrelevant linear-sampled
# checkpoints.
LOG_SPACED_FLAG=""
if [[ "${LOG_SPACED:-0}" == "1" ]]; then
  LOG_SPACED_FLAG="--log-spaced"
fi

OVERWRITE_FLAG=""
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG="--overwrite"
fi

# Sibling beetle-data root for FineWeb / BGPT proxy held-out parquet.
BEETLE_DATA="${BEETLE_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/beetle-data}"

# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------
# Usage: shard_repos repo1 repo2 repo3 ...
# Echoes the slice this rank should process.
shard_repos() {
  local i=0
  for repo in "$@"; do
    if (( i % WORLD_SIZE == RANK )); then
      echo "$repo"
    fi
    i=$((i + 1))
  done
}

# ---------------------------------------------------------------------------
# Per-checkpoint subprocess loop
# ---------------------------------------------------------------------------
# Internal: enumerate revisions for a repo via the Python side. Echoes one
# revision per line. Honours $ALL_CKPTS_FLAG / $HF_TOKEN_ARG.
_list_checkpoints_for_repo() {
  local repo="$1"
  # ``list-checkpoints`` returns "main" alone when the repo has no step-N
  # branches or when ALL_CKPTS_FLAG is empty.
  python -m analyze.convergence list-checkpoints \
      --repo "$repo" \
      $ALL_CKPTS_FLAG $LOG_SPACED_FLAG $HF_TOKEN_ARG 2>/dev/null || true
}

# Internal: drive subcommand SUB for repo+langs by spawning one Python
# process per checkpoint. Each subprocess loads the model, scores, and
# exits — letting the kernel reclaim host RAM between checkpoints.
#
# Usage:
#   _per_ckpt_loop SUB REPO LANG1 [LANG2 ...] EXTRA_FLAGS
# where SUB ∈ {flores-trajectory, heldout-trajectory, bpt-trajectory}.
_per_ckpt_loop() {
  local sub="$1"; shift
  local repo="$1"; shift
  # Split langs from trailing extra flags by walking until the first
  # token that starts with "--".
  local langs=()
  while [[ $# -gt 0 && "$1" != --* ]]; do
    langs+=("$1"); shift
  done
  local extras=( "$@" )

  mapfile -t revs < <(_list_checkpoints_for_repo "$repo")
  if (( ${#revs[@]} == 0 )); then
    echo "[loss-curves] $sub $repo: no checkpoints listed; skipping" >&2
    return 0
  fi

  local nrev=${#revs[@]}
  local i=0
  for rev in "${revs[@]}"; do
    i=$((i + 1))
    echo "[loss-curves] $sub $repo @ $rev  (${i}/${nrev})  langs=${langs[*]}"
    # NOTE: ALL_CKPTS_FLAG is intentionally *omitted* — we are pinning a
    # single revision via --checkpoint, so the Python side must not loop.
    if ! python -m analyze.convergence "$sub" \
          --repo "$repo" \
          --langs "${langs[@]}" \
          --checkpoint "$rev" \
          --sample-size "$SAMPLE_SIZE" \
          $OVERWRITE_FLAG $HF_TOKEN_ARG \
          "${extras[@]}"
    then
      local rc=$?
      echo "[loss-curves] $sub $repo @ $rev FAILED (rc=$rc); continuing to next checkpoint" >&2
      # Non-fatal per-checkpoint: a failed revision must not abort the whole
      # repo's trajectory. The Python side appends per-row to the CSV with
      # an fcntl-flock so partial outputs are safe to re-resume.
    fi
  done
}

# ---------------------------------------------------------------------------
# Trajectory invocations
# ---------------------------------------------------------------------------
# run_flores_traj REPO LANG1 [LANG2 ...]
run_flores_traj() {
  local repo="$1"; shift
  _per_ckpt_loop flores-trajectory "$repo" "$@"
}

# run_heldout_traj REPO LANG1 [LANG2 ...]
# Optional env: HELD_OUT_DIR=<path> overrides resolution for all langs.
run_heldout_traj() {
  local repo="$1"; shift
  local extra=( --beetle-data-root "$BEETLE_DATA" )
  if [[ -n "${HELD_OUT_DIR:-}" ]]; then
    extra+=( --held-out-dir "$HELD_OUT_DIR" )
  fi
  _per_ckpt_loop heldout-trajectory "$repo" "$@" "${extra[@]}"
}

# run_bpt_traj REPO SOURCE LANG1 [LANG2 ...]
# SOURCE ∈ {flores, heldout}.
# Optional env: HELD_OUT_DIR overrides resolution when SOURCE=heldout.
run_bpt_traj() {
  local repo="$1"; shift
  local source="$1"; shift
  local extra=( --source "$source" )
  if [[ "$source" == "heldout" ]]; then
    extra+=( --beetle-data-root "$BEETLE_DATA" )
    if [[ -n "${HELD_OUT_DIR:-}" ]]; then
      extra+=( --held-out-dir "$HELD_OUT_DIR" )
    fi
  fi
  _per_ckpt_loop bpt-trajectory "$repo" "$@" "${extra[@]}"
}

# Resolve eval languages from a repo name via the Python helper.
# Echoes a space-separated list; empty if not parseable.
langs_for_repo() {
  local repo="$1"
  python - <<PY
from analyze.model_naming import eval_languages_from_repo
print(" ".join(eval_languages_from_repo("$repo")))
PY
}

# Filter repos to only those including a target language. Pass --lang LANG
# at the driver level to enable; default keeps all.
filter_repos_by_lang() {
  local target="$1"; shift
  for repo in "$@"; do
    local langs
    langs=$(langs_for_repo "$repo")
    for l in $langs; do
      if [[ "$l" == "$target" ]]; then
        echo "$repo"
        break
      fi
    done
  done
}
