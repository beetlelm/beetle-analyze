#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# race_24b_sweep.sh
#
# Sweep evals/eval_race_student.py over the 17-checkpoint trace + main for
# both FineWeb2-24B Beetle repos (b4 classroom-20, b5 late-80, both deu-eng).
# 8-GPU sharded; one invocation per (repo, rank) with a subset of revisions.
#
# Outputs land under results/evals_paper/03b_race_zeroshot/evals/race/ so
# they pick up the same `_race_max` lookup used by figC_pedagogical_utility.
#
# Usage
# -----
#   bash scripts/race_24b_sweep.sh
#   N_GPUS=4 bash scripts/race_24b_sweep.sh
#   MAX_QUESTIONS=512 bash scripts/race_24b_sweep.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

OUT="${OUT:-results/evals_paper/03b_race_zeroshot}"
N_GPUS="${N_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_QUESTIONS="${MAX_QUESTIONS:-256}"
LOG_DIR="${LOG_DIR:-/root/beetle_pipeline/logs/manual_stage18/race_24b}"

REPOS=(
    "Beetle-FineWeb2-24B/beetle-bilingual-l2-80-late-b5-fineweb-deu-eng"
    "Beetle-FineWeb2-24B/beetle-bilingual-l2-50-classroom-20-b4-fineweb-deu-eng"
)

# Per-repo canonical 17-ckpt traces (the "saved-near-end" microsteps + the
# late-training samples). Rev numbering differs across repos so we can't
# share one list. Override either with REVS_<slug-with-dashes>=… or via
# the global REVS env var (applies to all repos).
declare -A REPO_REVS
REPO_REVS["Beetle-FineWeb2-24B/beetle-bilingual-l2-80-late-b5-fineweb-deu-eng"]=$(cat <<'EOR'
main step-73241 step-73242 step-73243 step-73245 step-73247 step-73249 step-73257 step-73273 step-73305 step-73369 step-73497 step-73753 step-74265 step-75289 step-77337 step-81433 step-89625
EOR
)
REPO_REVS["Beetle-FineWeb2-24B/beetle-bilingual-l2-50-classroom-20-b4-fineweb-deu-eng"]=$(cat <<'EOR'
main step-45776 step-45777 step-45778 step-45780 step-45782 step-45784 step-45792 step-45808 step-45840 step-45904 step-46032 step-46288 step-46800 step-47824 step-49872 step-53968 step-62160 step-65536 step-78544
EOR
)
DEFAULT_REVS_GLOBAL=""
GLOBAL_REVS_OVERRIDE="${REVS:-}"

mkdir -p "$LOG_DIR" "$OUT/evals/race" "$OUT/logs" "$OUT/ledger"
RANK_DIR="$OUT/_rank_outputs"
mkdir -p "$RANK_DIR"
echo "[$(date -u +%FT%TZ)] race_24b_sweep: ${#REPOS[@]} repos across $N_GPUS GPU(s)" \
    | tee -a "$LOG_DIR/dispatch.log"

pids=()
for repo in "${REPOS[@]}"; do
    if [[ -n "$GLOBAL_REVS_OVERRIDE" ]]; then
        read -ra REVS <<< "$GLOBAL_REVS_OVERRIDE"
    else
        read -ra REVS <<< "${REPO_REVS[$repo]}"
    fi
    echo "[$(date -u +%FT%TZ)]   $repo: ${#REVS[@]} revs" | tee -a "$LOG_DIR/dispatch.log"
    for ((rank=0; rank<N_GPUS; rank++)); do
        # Shard revisions across ranks via round-robin.
        rank_revs=()
        for ((j=rank; j<${#REVS[@]}; j+=N_GPUS)); do
            rank_revs+=("${REVS[$j]}")
        done
        [[ ${#rank_revs[@]} -eq 0 ]] && continue
        slug=$(echo "$repo" | tr '/' '_')
        log="$LOG_DIR/${slug}__rank${rank}.log"
        echo "[$(date -u +%FT%TZ)] $repo rank=$rank revs=${rank_revs[*]}" | tee -a "$log"
        (
            # We've already sharded revisions across ranks in bash; pass
            # rank=0/world_size=1 to eval_race_student so it doesn't shard
            # again internally (which would zero out 6 of 8 ranks).
            # Per-rank output dir to avoid concurrent appends to the same
            # CSV. Merged into $OUT/evals/race/ after each repo finishes.
            rank_out="$RANK_DIR/rank${rank}__${slug}"
            mkdir -p "$rank_out"
            CUDA_VISIBLE_DEVICES=$rank \
                python -m evals.eval_race_student \
                    --repo "$repo" \
                    --revisions "${rank_revs[@]}" \
                    --output-root "$rank_out" \
                    --max-questions "$MAX_QUESTIONS" \
                    --batch-size "$BATCH_SIZE" \
                    --rank 0 --world-size 1 \
                    >>"$log" 2>&1 || \
                    echo "[$(date -u +%FT%TZ)]   FAILED: $repo rank=$rank" | tee -a "$log"
        ) &
        pids+=($!)
        sleep 3
    done
    # Let one repo finish before launching the next so we avoid 16-process
    # GPU 0 contention.
    for pid in "${pids[@]}"; do wait "$pid" || true; done
    pids=()

    # Merge per-rank CSVs for this repo into the canonical output path.
    slug=$(echo "$repo" | tr '/' '_')
    canonical="$OUT/evals/race/race_${repo//\//__}.csv"
    python - "$RANK_DIR" "$slug" "$canonical" <<'PY' >>"$LOG_DIR/dispatch.log" 2>&1
import sys, glob, os, pandas as pd
rank_dir, slug, canonical = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for csv_path in sorted(glob.glob(f"{rank_dir}/rank*__{slug}/evals/race/race_*.csv")):
    try:
        rows.append(pd.read_csv(csv_path))
    except Exception as e:
        print(f"  skip {csv_path}: {e}")
if not rows:
    print(f"no rank csvs found for {slug}")
else:
    merged = pd.concat(rows, ignore_index=True).drop_duplicates(
        subset=[c for c in ("repo", "revision", "step") if c in rows[0].columns]
    )
    os.makedirs(os.path.dirname(canonical), exist_ok=True)
    merged.to_csv(canonical, index=False)
    print(f"merged {sum(len(r) for r in rows)} rows → {canonical}")
PY
done

echo "[$(date -u +%FT%TZ)] DONE" | tee -a "$LOG_DIR/dispatch.log"
echo "Outputs:"
ls -la "$OUT/evals/race/" 2>&1 | head
