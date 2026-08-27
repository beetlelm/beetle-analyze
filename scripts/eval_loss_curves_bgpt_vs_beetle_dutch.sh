#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# eval_loss_curves_bgpt_vs_beetle_dutch.sh
#
# Side-by-side training-loss comparison between B-GPT (catherinearnett) and
# Beetle-FineWeb / Beetle-FineWeb2-24B Dutch models.
#
# Metric: mean BITS-PER-TOKEN at every checkpoint (length-normalised NLL,
# converted from nats to bits via 1/ln 2). This is tokeniser-invariant —
# directly comparable between B-GPT (custom 16k tokeniser) and Beetle
# (FineWeb tokeniser) — unlike sum-of-token NLL or PPL.
#
# Eval data per checkpoint:
#   FLORES devtest, Dutch  (out-of-distribution probe)
#   Held-out training-data slice, Dutch  (in-distribution validation)
#
# Cohort:
#   B-GPT Dutch (4)        : catherinearnett/B-GPT_{en_nl,nl_en}_{simultaneous,sequential}
#   Beetle-FineWeb Dutch    : models.FINEWEB_MODELS filtered to *-nld-eng / *-deu-eng
#                             (German share is included because BGPT-Dutch was trained
#                              on Germanic Indo-European data; restrict via
#                              LANG_FILTER=nld for strictly Dutch-only).
#   Beetle-FineWeb2-24B    : Beetle-FineWeb2-24B/beetle-bilingual-* repos
#
# Usage:
#   bash scripts/eval_loss_curves_bgpt_vs_beetle_dutch.sh           # auto-shard across all visible GPUs
#   N_GPUS=4 bash scripts/eval_loss_curves_bgpt_vs_beetle_dutch.sh  # cap parallelism
#   N_GPUS=1 bash scripts/eval_loss_curves_bgpt_vs_beetle_dutch.sh  # force single-GPU
#   RANK=0 WORLD_SIZE=8 bash scripts/eval_loss_curves_bgpt_vs_beetle_dutch.sh  # single rank only (manual mode)
#   LANG_FILTER=nld bash scripts/eval_loss_curves_bgpt_vs_beetle_dutch.sh      # Dutch only (default: eng nld)
#   SAMPLE_SIZE=512 bash scripts/eval_loss_curves_bgpt_vs_beetle_dutch.sh
# ─────────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")/.."

# ─── Self-launcher: parallelise across GPUs via the shared safety library ──
# The hand-rolled rank fan-out used to be inlined here; it is now delegated
# to scripts/_sweep_safety.sh::launch_ranks, which adds host-RAM-aware
# concurrency capping, setsid (SSH-SIGHUP immunity), trap-protected peer
# termination on first failure, and pid-bound log tailing — none of which
# the inline implementation had, and the absence of which is what crashed
# two GPU nodes during this very sweep.
if [[ -z "${RANK:-}" ]]; then
    # shellcheck disable=SC1091
    source scripts/_sweep_safety.sh
    sweep_safety_init
    SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-logs/loss_curves}"
    export OVERWRITE LANG_FILTER HF_TOKEN SAMPLE_SIZE CKPT_SCOPE HELD_OUT_DIR BEETLE_DATA
    : "${LANG_FILTER:=eng nld}"
    export LANG_FILTER
    echo "[bgpt-vs-beetle-dutch] dispatching via launch_ranks"
    launch_ranks "loss_curves_bgpt_dutch" bash "$0"
    exit $?
fi

source scripts/_loss_curves_lib.sh

# Default eval languages: English + Dutch. Override with LANG_FILTER=nld for
# Dutch-only, or LANG_FILTER= (empty) to infer per-repo from the slug.
LANG_FILTER="${LANG_FILTER-eng nld}"

# ─── Resolve B-GPT Dutch + Beetle-FineWeb Dutch rosters ─────────────────────
mapfile -t BGPT_REPOS < <(python - <<'PY'
import models as M
out = [r for r in getattr(M, "BGPT_BASELINES", [])
       if "_en_nl_" in r or "_nl_en_" in r]
for r in out:
    print(r)
PY
)

mapfile -t BEETLE_REPOS < <(python - <<'PY'
import models as M
roster = list(getattr(M, "FINEWEB_MODELS", []))
roster += list(getattr(M, "FINEWEB_MODELS_PLANNED", []))
seen = set(); out = []
for r in roster:
    if not r or r in seen:
        continue
    # Keep only repos with Dutch in the slug (covers -nld-eng and -eng-nld).
    s = r.lower()
    if "nld" not in s and "-nl-" not in s and s.endswith("-nld") is False:
        continue
    seen.add(r); out.append(r)
for r in out:
    print(r)
PY
)

ALL_REPOS=("${BGPT_REPOS[@]}" "${BEETLE_REPOS[@]}")
echo "[bgpt-vs-beetle-dutch] BGPT repos:    ${#BGPT_REPOS[@]}"
echo "[bgpt-vs-beetle-dutch] Beetle repos:  ${#BEETLE_REPOS[@]}"
echo "[bgpt-vs-beetle-dutch] Total:         ${#ALL_REPOS[@]}"

if (( ${#ALL_REPOS[@]} == 0 )); then
    echo "ERROR: cohort resolved to 0 repos" >&2
    exit 1
fi

# Shard across rank/world_size (defaults to single-rank).
mapfile -t MY_REPOS < <(shard_repos "${ALL_REPOS[@]}")
echo "[bgpt-vs-beetle-dutch] rank ${RANK}/${WORLD_SIZE} → ${#MY_REPOS[@]} repos"

# ─── Per-repo BPT trajectories (FLORES + held-out) ──────────────────────────
for repo in "${MY_REPOS[@]}"; do
    # Eval languages: prefer LANG_FILTER if set, otherwise infer from repo name.
    if [[ -n "$LANG_FILTER" ]]; then
        langs="$LANG_FILTER"
    else
        langs=$(langs_for_repo "$repo")
    fi
    if [[ -z "$langs" ]]; then
        echo "[bgpt-vs-beetle-dutch] $repo: no eval langs resolved — skipping"
        continue
    fi

    # FLORES probe (always runs).
    run_bpt_traj "$repo" flores $langs

    # Held-out probe — non-fatal if the source can't be resolved
    # (BGPT does not publish its training held-out; falls back to
    # decontaminated FineWeb-nl in convergence._resolve_heldout_source).
    run_bpt_traj "$repo" heldout $langs || \
        echo "    (non-fatal) heldout BPT traj for $repo failed; see logs"
done

echo "[bgpt-vs-beetle-dutch] rank ${RANK} done."
echo "[bgpt-vs-beetle-dutch] CSVs: results/convergence/bpt_traj_*_flores.csv"
echo "[bgpt-vs-beetle-dutch]       results/convergence/bpt_traj_*_heldout.csv"
echo "[bgpt-vs-beetle-dutch] Plot: python plots/make_bgpt_vs_beetle_dutch.py"
