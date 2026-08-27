#!/usr/bin/env bash
# =============================================================================
# scripts/run_checkpoints_evals.sh
#
# Sibling of evals/run_evals.sh. Instead of evaluating each model on the
# `main` branch only, this driver sweeps **every HuggingFace branch**
# (numeric checkpoint tags + `main`) of every model in:
#
#   • Beetle-HumanScale       (checkpoint_mode=all_branches)
#   • Beetle-FineWeb          (checkpoint_mode=all_branches)
#   • catherinearnett/B-GPT_* (checkpoint_mode=all_branches)
#
# Plus the three static baselines (eurollm, xglm, bloom) on `main` only —
# they don't publish numbered branches.
#
# Per-benchmark gating (applied when building the plan TSV):
#   • blimp_eng     → run iff `eng` appears in repo name
#   • blimp_nl      → run iff `nld` (or `nl`) appears in repo name
#   • zhoblimp      → run iff `zho` (or `zh`) appears in repo name
#   • multiblimp    → run with --langs = (names ∩ {nld, deu, fra, fas, bul, ukr})
#   • MECO / FLORES        → run on every (repo × branch), no gating
#   • BLiSS                → run on every (repo × branch), but gated to the
#                            L1/L2 parsed from the model slug (via
#                            analyze.model_naming.parse_repo). Static
#                            baselines (bloom/xglm/EuroLLM) have no L1/L2 in
#                            the name and retain the all-strata behaviour.
#
# Each eval CSV gains a `branch` column so downstream analysis can pivot on
# (model, branch, benchmark, language).
#
# ---------------------------------------------------------------------------
# Execution order (disk-bounded)
# ---------------------------------------------------------------------------
# For every (repo × branch) row we run **all enabled stages in order**:
#     blimp → meco → flores → bliss
# and then **prune** the checkpoint from the rank's HF cache before the next
# row is loaded. Each rank keeps its own HF_HOME, so disk use is bounded by
#     N_GPUS × (one checkpoint's weights).
# This replaces the prior benchmark-outer / checkpoint-inner sweep, which
# retained every downloaded checkpoint on disk for the full run.
#
# Known inefficiency: each of the four stages is a fresh Python process that
# re-loads the same weights. Consolidating into one long-lived process per
# checkpoint would require a shared `run(model, tokenizer, ...)` API in each
# evals/eval_*.py — deliberately out of scope here.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#   bash scripts/run_checkpoints_evals.sh                      # all stages, 8 GPUs
#   N_GPUS=4 bash scripts/run_checkpoints_evals.sh             # 4 GPUs
#   STAGES="blimp bliss" bash scripts/run_checkpoints_evals.sh # subset of stages
#   DRY_RUN=1 bash scripts/run_checkpoints_evals.sh            # preview plan only
#   INCLUDE_BGPT=0 bash scripts/run_checkpoints_evals.sh       # skip bGPT
#   INCLUDE_MAIN=0 bash scripts/run_checkpoints_evals.sh       # numeric branches only
#   rm -f results/evals_checkpoints/.repo_cache.json           # force re-fetch of branch lists (24h cache)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# ── Config ────────────────────────────────────────────────────────────────────
N_GPUS="${N_GPUS:-8}"
STAGES="${STAGES:-blimp meco bliss}"
BATCH_SIZE="${BATCH_SIZE:-64}"
OUT_ROOT="${OUT_ROOT:-results/evals_checkpoints}"
LOG_DIR="${OUT_ROOT}/logs"
PLAN_FILE="${OUT_ROOT}/plan.tsv"
REPO_CACHE="${OUT_ROOT}/.repo_cache.json"

INCLUDE_HUMANSCALE="${INCLUDE_HUMANSCALE:-1}"
INCLUDE_FINEWEB="${INCLUDE_FINEWEB:-1}"
INCLUDE_BGPT="${INCLUDE_BGPT:-1}"
INCLUDE_STATIC_BASELINES="${INCLUDE_STATIC_BASELINES:-1}"
INCLUDE_MAIN="${INCLUDE_MAIN:-1}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON="${PYTHON:-python3}"

HF_CACHE_ROOT="${HF_CACHE_ROOT:-${TMPDIR:-/tmp}/beetle_hf_cache}"
HF_HOME_SHARED="${HF_CACHE_ROOT}/shared"

mkdir -p "$LOG_DIR" "$HF_CACHE_ROOT" "$HF_HOME_SHARED"

echo "[$(date -u +%FT%TZ)] N_GPUS=${N_GPUS}  STAGES=${STAGES}  OUT_ROOT=${OUT_ROOT}"
echo "[$(date -u +%FT%TZ)] include: humanscale=${INCLUDE_HUMANSCALE} fineweb=${INCLUDE_FINEWEB} bgpt=${INCLUDE_BGPT} static=${INCLUDE_STATIC_BASELINES} main=${INCLUDE_MAIN}"
echo "[$(date -u +%FT%TZ)] eval-all-branches: fineweb=${EVAL_ALL_BRANCHES_FINEWEB:-0} humanscale=${EVAL_ALL_BRANCHES_HUMANSCALE:-0}"

# ── GPU utilisation monitor ────────────────────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi dmon -s u -d 10 > "${LOG_DIR}/gpu_util.log" 2>&1 &
    NVMON_PID=$!
    trap 'kill $NVMON_PID 2>/dev/null || true' EXIT
    echo "[$(date -u +%FT%TZ)] GPU monitor PID=${NVMON_PID} → ${LOG_DIR}/gpu_util.log"
fi

# ── Step 1+2+3+4: enumerate repos, branches, parse langs, build plan TSV ──────
echo ""
echo "[$(date -u +%FT%TZ)] === Building evaluation plan ==="

INCLUDE_HUMANSCALE="${INCLUDE_HUMANSCALE}" \
INCLUDE_FINEWEB="${INCLUDE_FINEWEB}" \
INCLUDE_BGPT="${INCLUDE_BGPT}" \
INCLUDE_STATIC_BASELINES="${INCLUDE_STATIC_BASELINES}" \
INCLUDE_MAIN="${INCLUDE_MAIN}" \
EVAL_ALL_BRANCHES_FINEWEB="${EVAL_ALL_BRANCHES_FINEWEB:-0}" \
EVAL_ALL_BRANCHES_HUMANSCALE="${EVAL_ALL_BRANCHES_HUMANSCALE:-0}" \
REPO_CACHE="${REPO_CACHE}" \
${PYTHON} - <<'PYEOF' > "${PLAN_FILE}"
import json, os, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from evals.baseline_models import (
    BASELINE_MODELS,
    _BEETLE_NON_MODEL_PREFIXES,
    _is_model_repo,
)
try:
    from analyze.model_naming import parse_repo  # canonical (L1, L2) parser
except Exception as _exc:
    parse_repo = None
    print(f"WARNING: analyze.model_naming unavailable ({_exc}); "
          f"falling back to parse_langs for bliss gating", file=sys.stderr)

# ── Config from env ────────────────────────────────────────────────────────
INC_HS    = os.environ.get("INCLUDE_HUMANSCALE", "1") == "1"
INC_FW    = os.environ.get("INCLUDE_FINEWEB",    "1") == "1"
INC_BGPT  = os.environ.get("INCLUDE_BGPT",       "1") == "1"
INC_STAT  = os.environ.get("INCLUDE_STATIC_BASELINES", "1") == "1"
INC_MAIN  = os.environ.get("INCLUDE_MAIN",       "1") == "1"
EVAL_ALL_FW = os.environ.get("EVAL_ALL_BRANCHES_FINEWEB",    "0") == "1"
EVAL_ALL_HS = os.environ.get("EVAL_ALL_BRANCHES_HUMANSCALE", "0") == "1"
CACHE_PATH = Path(os.environ["REPO_CACHE"])
CACHE_TTL  = 24 * 3600

# ── BeetleLM log-scale checkpoint schedules ────────────────────────────────
# Mirrors upstream BeetleLM/configs/ckpt_schedule.py. Each family uses a
# powers-of-2 base (subsampled on FineWeb because it trains 9x longer), plus
# small offsets {0,2,4,6,8} anchored at every phase_start. phase_starts
# correspond to L1-onset (always 0) and the L2-introduction step derived from
# the condition (B1-B5). The "every-500-steps" fallback also published on the
# Hub is deliberately excluded here.
_HS_POW2 = [2**k for k in range(0, 14)]                        # 1,2,...,8192
_FW_POW2 = [2**k for k in range(0, 17) if k % 2 == 0]          # 1,4,16,...,65536
_OFFSETS = [0, 2, 4, 6, 8]
_HS_TOTAL = 10_000
_FW_TOTAL = 91_552

# Empirically verified against published HuggingFace branches: every bilingual
# HumanScale condition anchors wave 2 at step 5000 (50% of 10k) except B5 which
# uses step 8000 (80%). Every bilingual FineWeb condition uses step 45776 (50%
# of 91552). Incomplete/truncated runs (maxstep < anchor) naturally collapse to
# wave 1 only via the ∩-existing-branches intersection.
_HS_PHASE_STARTS = {"B2": [0, 5000],  "B3": [0, 5000],  "B4": [0, 5000],  "B5": [0, 8000]}
_FW_PHASE_STARTS = {"B2": [0, 45776], "B3": [0, 45776], "B4": [0, 45776], "B5": [0, 45776]}

def _logscale_target(pow2, phase_starts, total):
    out = set()
    for s in phase_starts:
        if s > total: continue
        out.add(s)
        for p in pow2 + _OFFSETS:
            t = s + p
            if t <= total:
                out.add(t)
    return out

def beetle_logscale(repo: str, existing_branches: list) -> list:
    meta = parse_repo(repo) if parse_repo is not None else None
    cond = (getattr(meta, "condition", None) or "unknown").upper()
    if repo.startswith("Beetle-HumanScale/"):
        pow2, total, ps_map = _HS_POW2, _HS_TOTAL, _HS_PHASE_STARTS
    else:  # FineWeb
        pow2, total, ps_map = _FW_POW2, _FW_TOTAL, _FW_PHASE_STARTS
    ps = ps_map.get(cond, [0])
    target = _logscale_target(pow2, ps, total)
    kept = []
    for b in existing_branches:
        if b == "main":
            continue
        m = _STEP_RE.match(b)
        step = int(m.group(1)) if m else (int(b) if _NUMERIC_RE.match(b) else None)
        if step is not None and step in target:
            kept.append(b)
    # Preserve original ordering (numeric ascending) from list_numeric_branches
    if "main" in existing_branches and INC_MAIN:
        kept.append("main")
    return kept

# ── Simple JSON cache ──────────────────────────────────────────────────────
_cache: dict = {}
if CACHE_PATH.exists():
    try:
        _cache = json.loads(CACHE_PATH.read_text())
    except Exception:
        _cache = {}

def _cache_get(key: str):
    entry = _cache.get(key, {})
    if time.time() - entry.get("ts", 0) < CACHE_TTL:
        return entry.get("value")
    return None

def _cache_set(key: str, value):
    _cache[key] = {"ts": time.time(), "value": value}
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(_cache, indent=2))

# ── Repo enumeration ───────────────────────────────────────────────────────
def list_org_models(org: str) -> list[str]:
    cached = _cache_get(f"org:{org}")
    if cached is not None:
        return cached
    from huggingface_hub import list_models
    try:
        repos = [m.id for m in list_models(author=org) if _is_model_repo(org, m.id)]
    except Exception as exc:
        print(f"WARNING: could not list {org}: {exc}", file=sys.stderr)
        repos = []
    _cache_set(f"org:{org}", repos)
    return repos

def list_bgpt_repos() -> list[str]:
    cached = _cache_get("bgpt")
    if cached is not None:
        return cached
    from huggingface_hub import list_models
    try:
        repos = [
            m.id for m in list_models(author="catherinearnett")
            if m.id.split("/", 1)[-1].startswith("B-GPT_")
        ]
    except Exception as exc:
        print(f"WARNING: could not list catherinearnett B-GPT: {exc}", file=sys.stderr)
        repos = []
    _cache_set("bgpt", repos)
    return repos

_STEP_RE    = re.compile(r"^step-(\d+)$")
_NUMERIC_RE = re.compile(r"^\d+$")

def _ckpt_sort_key(name: str) -> tuple[int, int]:
    m = _STEP_RE.match(name)
    if m:
        return (0, int(m.group(1)))
    if _NUMERIC_RE.match(name):
        return (0, int(name))
    return (1, 0)

def list_numeric_branches(repo: str) -> list[str]:
    cached = _cache_get(f"branches:{repo}")
    if cached is not None:
        return cached
    from huggingface_hub import list_repo_refs
    try:
        refs = list_repo_refs(repo)
        names = [b.name for b in refs.branches]
    except Exception as exc:
        print(f"WARNING: could not list branches for {repo}: {exc}", file=sys.stderr)
        names = []
    ckpts = sorted(
        [n for n in names if _STEP_RE.match(n) or _NUMERIC_RE.match(n)],
        key=_ckpt_sort_key,
    )
    has_main = "main" in names
    result = ckpts + (["main"] if (has_main and INC_MAIN) else [])
    if not ckpts and (
        repo.startswith("Beetle-HumanScale/") or repo.startswith("Beetle-FineWeb/")
    ):
        print(
            f"WARNING: no checkpoint branches (step-N or numeric) found for "
            f"{repo}; only {'main' if has_main else 'nothing'} will be evaluated",
            file=sys.stderr,
        )
    _cache_set(f"branches:{repo}", result)
    return result

# ── ISO code parsing (accept both 2- and 3-letter codes, emit ISO-639-3) ───
ISO1_TO_ISO3 = {
    "en": "eng", "nl": "nld", "zh": "zho", "de": "deu", "fr": "fra",
    "fa": "fas", "bg": "bul", "uk": "ukr", "id": "ind", "sr": "srp",
    "ko": "kor", "es": "spa", "el": "ell", "pl": "pol",
}
KNOWN_ISO3 = {
    "eng", "nld", "deu", "zho", "fra", "fas", "bul", "ukr",
    "ind", "srp", "kor", "spa", "ell", "pol",
}
MULTIBLIMP_AVAILABLE = {"nld", "deu", "fra", "fas", "bul", "ukr"}

def parse_langs(repo: str) -> set[str]:
    name = repo.split("/", 1)[-1]
    found: set[str] = set()
    for tok in re.split(r"[^A-Za-z]+", name):
        t = tok.lower()
        if len(t) == 3 and t in KNOWN_ISO3:
            found.add(t)
        elif len(t) == 2 and t in ISO1_TO_ISO3:
            found.add(ISO1_TO_ISO3[t])
    return found

_BGPT_RE = re.compile(r"^B-GPT_([a-z]{2})_([a-z]{2})(?:_|$)")

def bliss_langs(repo: str, fallback_langs: set[str]) -> list[str]:
    """Return ordered [L1, L2] (ISO-639-3) for BLiSS gating.

    Uses analyze.model_naming.parse_repo when available (canonical for
    Beetle-HumanScale / Beetle-FineWeb / BeetleLM slugs). Also handles the
    B-GPT_<l1>_<l2>_... convention explicitly since parse_repo returns
    family=unknown for catherinearnett/B-GPT_* repos. Returns an empty
    list for repos it cannot classify (e.g. static baselines bloom/xglm/
    EuroLLM) — the driver omits --langs for those so eval_bliss.py keeps
    its default all-strata behaviour.
    """
    # B-GPT first — parse_repo does not recognise this family
    name = repo.split("/", 1)[-1]
    m = _BGPT_RE.match(name)
    if m:
        l1 = ISO1_TO_ISO3.get(m.group(1))
        l2 = ISO1_TO_ISO3.get(m.group(2))
        out: list[str] = []
        if l1: out.append(l1)
        if l2 and l2 != l1: out.append(l2)
        return out

    if parse_repo is None:
        return []
    try:
        meta = parse_repo(repo)
    except Exception:
        return []
    if meta.family == "unknown":
        return []
    out = []
    if meta.l1:
        out.append(meta.l1)
    if meta.l2 and meta.l2 != meta.l1:
        out.append(meta.l2)
    return out

def blimp_plan(langs: set[str]) -> tuple[list[str], list[str]]:
    benches, filters = [], []
    if "eng" in langs:
        benches.append("blimp_eng"); filters.append("eng")
    if "nld" in langs:
        benches.append("blimp_nl");  filters.append("nld")
    if "zho" in langs:
        benches.append("zhoblimp");  filters.append("zho")
    mb = sorted(langs & MULTIBLIMP_AVAILABLE)
    if mb:
        if "multiblimp" not in benches:
            benches.append("multiblimp")
        for lg in mb:
            if lg not in filters:
                filters.append(lg)
    return benches, filters

# ── Assemble (repo, branches, mode) list ───────────────────────────────────
plan_sources: list[tuple[str, list[str]]] = []   # (repo, branches)

if INC_HS:
    for r in list_org_models("Beetle-HumanScale"):
        all_br = list_numeric_branches(r)
        br = all_br if EVAL_ALL_HS else beetle_logscale(r, all_br)
        plan_sources.append((r, br))
if INC_FW:
    for r in list_org_models("Beetle-FineWeb"):
        all_br = list_numeric_branches(r)
        br = all_br if EVAL_ALL_FW else beetle_logscale(r, all_br)
        plan_sources.append((r, br))
if INC_BGPT:
    for r in list_bgpt_repos():
        plan_sources.append((r, list_numeric_branches(r)))
if INC_STAT:
    for fam, repos in BASELINE_MODELS.items():
        for r in repos:
            plan_sources.append((r, ["main"]))

# ── Emit plan TSV ──────────────────────────────────────────────────────────
print("repo\tbranch\tblimp_benches\tblimp_langs\tbliss_langs")
for repo, branches in plan_sources:
    if not branches:
        print(f"WARNING: no branches found for {repo}", file=sys.stderr)
        continue
    langs = parse_langs(repo)
    bench_list, lang_filters = blimp_plan(langs)
    benches_s = ",".join(bench_list)
    langs_s   = ",".join(lang_filters)
    bliss_s   = ",".join(bliss_langs(repo, langs))
    for br in branches:
        print(f"{repo}\t{br}\t{benches_s}\t{langs_s}\t{bliss_s}")
PYEOF

N_PLAN=$(( $(wc -l < "${PLAN_FILE}" | tr -d ' ') - 1 ))
echo "       → ${N_PLAN} (repo × branch) jobs → ${PLAN_FILE}"

if (( N_PLAN <= 0 )); then
    echo "ERROR: empty plan. Check include flags and HuggingFace access." >&2
    exit 1
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1 — plan written, no evaluations dispatched. Exiting."
    exit 0
fi

# ── Pre-flight: verify git state so per-model push can succeed ────────────────
# Must run BEFORE Stage 0 — the ledger backfill rewrites ledger.jsonl on every
# invocation (non-deterministic serialisation), which would otherwise make the
# dirty-tree check unreachable. Subsequent ledger changes are rolled into the
# first model's commit naturally by git_push_repo().
GIT_REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [[ -z "$GIT_REPO_ROOT" ]]; then
    echo "ERROR: not inside a git repo; per-model push requires one." >&2
    exit 1
fi
export GIT_REPO_ROOT
if ! git -C "$GIT_REPO_ROOT" rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
    echo "ERROR: current branch has no upstream; set one with 'git push -u'." >&2
    exit 1
fi
# Require tree to be clean at start so our commits only contain eval output.
if ! git -C "$GIT_REPO_ROOT" diff --quiet \
   || ! git -C "$GIT_REPO_ROOT" diff --cached --quiet; then
    echo "ERROR: working tree dirty; commit or stash before starting." >&2
    exit 1
fi
echo "[$(date -u +%FT%TZ)] git push target: $(git -C "$GIT_REPO_ROOT" rev-parse --abbrev-ref --symbolic-full-name '@{u}')"

# ── Stage 0: ledger backfill ──────────────────────────────────────────────────
echo ""
echo "[$(date -u +%FT%TZ)] === Stage 0: ledger backfill ==="
HF_HOME="${HF_HOME_SHARED}" \
${PYTHON} -m analyze.eval_ledger backfill --output "${OUT_ROOT}" \
    2>&1 | tee "${LOG_DIR}/ledger_backfill.log" || true

# ── Helper: prune a single (repo, branch) from the current rank's HF cache ───
# Uses huggingface_hub.scan_cache_dir() to delete only the matching revision,
# which lets sibling branches of the same repo share unchanged tokenizer/config
# blobs. Falls back to `rm -rf models--<org>--<name>` if the HF API breaks.
# Assumes HF_HOME, log_file are set in the calling subshell.
prune_revision() {
    local repo="$1" branch="$2"
    ${PYTHON} - "$repo" "$branch" <<'PYEOF' >> "$log_file" 2>&1 \
        || rm -rf "${HF_HOME}/hub/models--${repo//\//--}" 2>/dev/null || true
import sys
from huggingface_hub import scan_cache_dir
repo, branch = sys.argv[1], sys.argv[2]
info = scan_cache_dir()
hashes = [
    rev.commit_hash
    for r in info.repos if r.repo_id == repo
    for rev in r.revisions
    if branch in rev.refs or rev.commit_hash == branch
]
if hashes:
    info.delete_revisions(*hashes).execute()
    print(f"pruned {repo}@{branch}: {len(hashes)} revision(s)")
PYEOF
}

# ── Helper: evaluate ONE (repo, branch) on all enabled stages, then prune ────
# STAGES env var still selects which benchmarks run (e.g. STAGES="blimp bliss").
# Each stage failure is logged but does not abort the remaining stages; the
# prune call ALWAYS runs so disk stays bounded even on partial failure.
# Assumes the caller's subshell has exported: CUDA_VISIBLE_DEVICES, HF_HOME,
# HF_DATASETS_CACHE, and set a `log_file` variable.
run_one_checkpoint() {
    local repo="$1" branch="$2" benches="$3" langs="$4" bliss_lang_csv="$5" rank="$6"
    local stages_padded=" ${STAGES} "
    echo "  [$(date -u +%FT%TZ)] [rank${rank}] ${repo}@${branch}" >> "$log_file"

    for stage in blimp meco bliss; do
        [[ "$stages_padded" != *" $stage "* ]] && continue
        case "$stage" in
            blimp)
                # Skip if language gating produced an empty benchmark list
                if [[ -z "$benches" ]]; then
                    continue
                fi
                IFS=',' read -r -a bench_arr <<< "$benches"
                IFS=',' read -r -a lang_arr  <<< "$langs"
                ${PYTHON} evals/eval_blimp.py \
                    --models "$repo" \
                    --revision "$branch" \
                    --benchmarks "${bench_arr[@]}" \
                    ${lang_arr[@]:+--langs "${lang_arr[@]}"} \
                    --output "${OUT_ROOT}/blimp" \
                    --batch-size "${BATCH_SIZE}" \
                    --rank 0 --world-size 1 \
                    >> "$log_file" 2>&1 \
                    || echo "  FAIL blimp ${repo}@${branch}" >> "$log_file"
                ;;
            meco)
                ${PYTHON} evals/eval_meco.py \
                    --models "$repo" \
                    --revision "$branch" \
                    --stimuli data/meco_l2_stims.tsv \
                    --output "${OUT_ROOT}/meco" \
                    --rank 0 --world-size 1 \
                    >> "$log_file" 2>&1 \
                    || echo "  FAIL meco ${repo}@${branch}" >> "$log_file"
                ;;
            bliss)
                local -a bliss_lang_arr=()
                if [[ -n "$bliss_lang_csv" ]]; then
                    IFS=',' read -r -a bliss_lang_arr <<< "$bliss_lang_csv"
                fi
                ${PYTHON} evals/eval_bliss.py \
                    --model "$repo" \
                    --revision "$branch" \
                    --output "${OUT_ROOT}/bliss" \
                    --batch-size 32 \
                    ${bliss_lang_arr[@]:+--langs "${bliss_lang_arr[@]}"} \
                    >> "$log_file" 2>&1 \
                    || echo "  FAIL bliss ${repo}@${branch}" >> "$log_file"
                ;;
        esac
    done

    # Always prune — bounds disk regardless of stage success/failure
    prune_revision "$repo" "$branch"
}

# ── Helper: commit + push this rank's accumulated results for ONE model ────
# Serialised across ranks via flock on a single lockfile to avoid
# `.git/index.lock` contention. Only adds CSV + JSONL artefacts — never
# weights/caches. Each push captures whatever rows exist at that moment
# (BLiSS uses a single shared CSV; duplicate adds across ranks are fine).
# Assumes the caller's subshell exports GIT_REPO_ROOT and has `log_file` set.
PUSH_LOCK="${OUT_ROOT}/.git_push.lock"
: > "$PUSH_LOCK" 2>/dev/null || true

git_push_repo() {
    local repo="$1" rank="$2"
    local slug="${repo//\//__}"
    (
        flock -x 9
        cd "$GIT_REPO_ROOT"
        # Stage any artefacts that may have been written for this model.
        git add "${OUT_ROOT}/blimp/"*"${slug}"*.csv 2>/dev/null || true
        git add "${OUT_ROOT}/meco/"*"${slug}"*.csv 2>/dev/null || true
        git add "${OUT_ROOT}/bliss/bliss_results.csv" 2>/dev/null || true
        git add "${OUT_ROOT}/ledger/"*.jsonl 2>/dev/null || true
        git add "${OUT_ROOT}/blimp/.pkl_cache/"*.pkl 2>/dev/null || true
        git add "${OUT_ROOT}/bliss/.pkl_cache/"*.pkl 2>/dev/null || true
        if git diff --cached --quiet; then
            echo "  [git] nothing new to push for ${repo}" >> "$log_file"
            return 0
        fi
        git commit -m "eval(rank${rank}): ${repo}" --no-verify >> "$log_file" 2>&1 || {
            echo "  [git] commit failed for ${repo}" >> "$log_file"
            return 0
        }
        local i
        for i in 1 2 3; do
            if git pull --rebase --autostash >> "$log_file" 2>&1 \
               && git push >> "$log_file" 2>&1; then
                echo "  [git] pushed ${repo} (attempt ${i})" >> "$log_file"
                return 0
            fi
            sleep $((RANDOM % 10 + 5))
        done
        echo "  [git] WARN: push failed for ${repo} after 3 attempts" >> "$log_file"
    ) 9>>"$PUSH_LOCK"
}

# ── Fan out checkpoint-outer / benchmark-inner across N_GPUS ranks ───────────
# Each rank runs rows where (idx-1) % N_GPUS == RANK, and for every row runs
# all enabled stages sequentially before pruning. Per-rank HF_HOME isolates
# concurrent downloads/deletions. When a rank transitions to a new repo,
# push the completed previous repo to git so multi-day runs are crash-safe.
fan_out_checkpoints() {
    local -a pids=()
    for RANK in $(seq 0 $((N_GPUS - 1))); do
        (
            export CUDA_VISIBLE_DEVICES="$RANK"
            export HF_HOME="${HF_CACHE_ROOT}/rank_${RANK}"
            export HF_DATASETS_CACHE="${HF_HOME}/datasets"
            mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"
            local log_file="${LOG_DIR}/ckpt_rank${RANK}.log"
            local idx=0
            local last_repo=""
            tail -n +2 "${PLAN_FILE}" | while IFS=$'\t' read -r repo branch benches langs bliss_lang_csv; do
                idx=$((idx + 1))
                (( (idx - 1) % N_GPUS != RANK )) && continue
                if [[ -n "$last_repo" && "$repo" != "$last_repo" ]]; then
                    git_push_repo "$last_repo" "$RANK"
                fi
                run_one_checkpoint "$repo" "$branch" "$benches" "$langs" "$bliss_lang_csv" "$RANK"
                last_repo="$repo"
            done
            # Flush the final repo processed by this rank
            if [[ -n "$last_repo" ]]; then
                git_push_repo "$last_repo" "$RANK"
            fi
        ) &
        pids+=($!)
    done
    echo "  [$(date -u +%FT%TZ)] Launched ${#pids[@]} ranks (checkpoint-outer) — waiting …"
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=$((failed + 1))
    done
    if (( failed > 0 )); then
        echo "WARNING: $failed rank(s) returned non-zero" >&2
    fi
    echo "  [$(date -u +%FT%TZ)] All ranks done"
}

# ── Stages 1–4: checkpoint-outer evaluation + prune ───────────────────────────
echo ""
echo "[$(date -u +%FT%TZ)] === Stages 1–4: per-checkpoint evaluation (${STAGES}) ==="
if [[ "$STAGES" == *bliss* ]]; then
    echo "  Note: BLiSS is a private dataset — run 'huggingface-cli login' first."
fi
fan_out_checkpoints

# ── Stage 5: merge per-rank ledger shards ─────────────────────────────────────
echo ""
echo "[$(date -u +%FT%TZ)] === Stage 5: merge ledger shards ==="
HF_HOME="${HF_HOME_SHARED}" \
${PYTHON} -m analyze.eval_ledger merge --output "${OUT_ROOT}" \
    2>&1 | tee "${LOG_DIR}/ledger_merge.log" || true

# ── Stage 6: completion report ────────────────────────────────────────────────
echo ""
echo "[$(date -u +%FT%TZ)] === Stage 6: completion report ==="
HF_HOME="${HF_HOME_SHARED}" \
${PYTHON} -m analyze.eval_ledger report --output "${OUT_ROOT}" \
    2>&1 | tee "${LOG_DIR}/ledger_report.log" || true

echo ""
echo "[$(date -u +%FT%TZ)] === All checkpoint evaluations completed ==="
echo "Results under ${OUT_ROOT}/"
