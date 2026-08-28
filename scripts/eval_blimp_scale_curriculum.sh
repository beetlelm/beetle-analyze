#!/usr/bin/env bash
# eval_blimp_scale_curriculum.sh — single-GPU BLiMP fill-gap sweep.
#
# Targets the four orgs that lack final-state BLiMP coverage in
# results/evals/blimp/blimp_results.csv:
#   - Beetle-FineWeb-100M
#   - Beetle-FineWeb-2B
#   - Beetle-FineWeb2-24B
#   - Beetle-nld-eng-Variants
#
# Pulls the live model list from the HF API for each org so we don't have
# to maintain a static manifest. Runs analyze.blimp_eval_oneshot for the
# canonical 5-curriculum sweep (B1..B5) on whichever pairs the org publishes,
# regenerating figures after every model. Walltime-capped (default 23h to
# leave a margin under the 24h budget).
#
# Required env:
#   CUDA_VISIBLE_DEVICES   — must be set to a single GPU index (e.g. "0").
# Optional env:
#   WALLTIME_SECONDS       — total budget (default: 82800 = 23h).
#   ORGS                   — space-separated override of the org list.
#   HF_TOKEN               — HuggingFace token for gated repos (rare here).

set -euo pipefail

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES must be set (single GPU)" >&2
    exit 1
fi
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "ERROR: this script runs on a SINGLE GPU; CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' has multiple" >&2
    exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"
WALLTIME_SECONDS="${WALLTIME_SECONDS:-82800}"  # 23h
DEFAULT_ORGS=(
    "Beetle-FineWeb-100M"
    "Beetle-FineWeb-2B"
    "Beetle-FineWeb2-24B"
    "Beetle-nld-eng-Variants"
)
read -r -a ORGS <<<"${ORGS:-${DEFAULT_ORGS[*]}}"

LOG_DIR="$ROOT/logs/blimp_scale_curriculum"
mkdir -p "$LOG_DIR"
SWEEP_LOG="$LOG_DIR/sweep_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "[$(date -u +%FT%TZ)] sweep start  GPU=$CUDA_VISIBLE_DEVICES  walltime=${WALLTIME_SECONDS}s  orgs=${ORGS[*]}" | tee -a "$SWEEP_LOG"

# ── 1. Fetch the live repo list per org ───────────────────────────────────
fetch_repos() {
    local org="$1"
    curl -fsS --max-time 20 "https://huggingface.co/api/models?author=${org}&limit=200" \
        | "$PY" -c '
import json, sys
data = json.load(sys.stdin)
keep = []
for m in data:
    mid = m.get("modelId", "")
    name = mid.split("/", 1)[-1].lower()
    # Drop tokenizers (bpe-…), datasets, monolingual baselines belong elsewhere.
    if name.startswith("bpe-"):
        continue
    if "monolingual" not in name and "bilingual" not in name:
        continue
    keep.append(mid)
keep.sort()
print("\n".join(keep))
'
}

ALL_REPOS=()
for org in "${ORGS[@]}"; do
    echo "[$(date -u +%FT%TZ)] fetching repo list: $org" | tee -a "$SWEEP_LOG"
    while IFS= read -r repo; do
        [[ -n "$repo" ]] && ALL_REPOS+=("$repo")
    done < <(fetch_repos "$org" || echo "")
done
echo "[$(date -u +%FT%TZ)] resolved ${#ALL_REPOS[@]} repos total" | tee -a "$SWEEP_LOG"
printf '  %s\n' "${ALL_REPOS[@]}" | tee -a "$SWEEP_LOG"

if [[ ${#ALL_REPOS[@]} -eq 0 ]]; then
    echo "ERROR: no repos resolved — aborting" | tee -a "$SWEEP_LOG" >&2
    exit 1
fi

# ── 2. Drive the oneshot runner across the full list ──────────────────────
# Order: smallest-org first so the cheap evals land in the CSV/branch fast,
# and the auto-push has something to publish within the first hour.
ORDER=()
for size_tag in "fineweb-100m" "fineweb-2b" "fineweb2" "nld-eng-variants"; do
    for r in "${ALL_REPOS[@]}"; do
        rl="$(echo "$r" | tr '[:upper:]' '[:lower:]')"
        case "$rl" in
          *"$size_tag"*) ORDER+=("$r") ;;
        esac
    done
done

# Dedup while preserving order.
DEDUPED=()
declare -A seen=()
for r in "${ORDER[@]}"; do
    if [[ -z "${seen[$r]:-}" ]]; then
        seen[$r]=1
        DEDUPED+=("$r")
    fi
done

echo "[$(date -u +%FT%TZ)] eval order (${#DEDUPED[@]} models):" | tee -a "$SWEEP_LOG"
printf '  %s\n' "${DEDUPED[@]}" | tee -a "$SWEEP_LOG"

export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
# Disable HF xet (its adaptive concurrency workers occasionally hang after
# large downloads — observed on llm-jp-3-3.7b @ MECO sweep).
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-0}"

# Home filesystem on this host is at 100% — point every HF cache to scratch.
SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}/beetle}"
export HF_HOME="${HF_HOME:-$SCRATCH/huggingface_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SCRATCH/hf_datasets_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"
echo "[$(date -u +%FT%TZ)] HF_HOME=$HF_HOME HF_DATASETS_CACHE=$HF_DATASETS_CACHE TMPDIR=$TMPDIR" | tee -a "$SWEEP_LOG"

"$PY" -m analyze.blimp_eval_oneshot \
    --repos "${DEDUPED[@]}" \
    --output-root "$ROOT/results" \
    --walltime-seconds "$WALLTIME_SECONDS" \
    --benchmarks blimp_eng blimp_nl multiblimp zhoblimp \
    --regen-figures \
    2>&1 | tee -a "$SWEEP_LOG"

echo "[$(date -u +%FT%TZ)] sweep complete" | tee -a "$SWEEP_LOG"

# Final figure regen so the last model's data lands in the published PDFs.
"$PY" -m analyze.blimp_scale_curriculum 2>&1 | tee -a "$SWEEP_LOG" || true
