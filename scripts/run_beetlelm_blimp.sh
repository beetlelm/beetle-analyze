#!/usr/bin/env bash
# =============================================================================
# scripts/run_beetlelm_blimp.sh
#
# Run BLiMP / MultiBLiMP grammaticality evaluations on every model published
# under the HuggingFace org  https://huggingface.co/BeetleLM .
#
# Target MultiBLiMP languages : fas, bul, ukr, ind, fra
# Target BLiMP  (English)    : eng   (blimp_eng)
#
# For every candidate model we parse its repo name into the set of language
# codes it contains, then evaluate ONLY on the intersection of
#
#     {languages in the model name}  ∩  {target languages available in the
#                                         relevant BLiMP / MultiBLiMP dataset}
#
# Tokenizer repositories (repo names starting with `bpe_` / `bpe-` or
# `BabyBabel-`) are filtered out — the script only ever dispatches actual
# language-model repos to `evals/eval_blimp.py`.
#
# After evaluating, we emit a LaTeX table via
# `scripts/make_beetlelm_blimp_table.py`.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#   bash scripts/run_beetlelm_blimp.sh                      # default output
#   OUTPUT_DIR=results/beetlelm_blimp bash scripts/...sh    # custom dir
#   DRY_RUN=1 bash scripts/run_beetlelm_blimp.sh            # preview only
#   BATCH_SIZE=32 bash scripts/run_beetlelm_blimp.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-results/beetlelm_blimp}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON="${PYTHON:-python3}"

# MultiBLiMP target langs requested by the user
#   NOTE: `ind` (Indonesian) is NOT present in jumelet/multiblimp configs.
#   It will be automatically skipped at dataset-load time; we still keep it
#   in the target set so that the model-name parse is consistent.
MULTIBLIMP_TARGET_LANGS=(fas bul ukr ind fra)

# BLiMP (English) iso-code  → blimp_eng benchmark
BLIMP_ENG_ISO="eng"

MODEL_LIST_FILE="${OUTPUT_DIR}/beetlelm_models.txt"
PLAN_FILE="${OUTPUT_DIR}/eval_plan.tsv"
RESULTS_CSV="${OUTPUT_DIR}/blimp_results.csv"
LATEX_OUT="${OUTPUT_DIR}/beetlelm_blimp_table.tex"

mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Step 1. Enumerate BeetleLM models, filtering out tokenizer repos.
# ---------------------------------------------------------------------------

echo "[1/4] Listing BeetleLM repos (filtering tokenizers) …"

${PYTHON} - <<'PYEOF' > "${MODEL_LIST_FILE}"
from huggingface_hub import list_models

# Tokenizer / dataset repo prefixes published alongside actual LMs.
NON_MODEL_PREFIXES = ("bpe_", "bpe-", "BabyBabel-")

for m in list_models(author="BeetleLM"):
    name = m.id.split("/", 1)[-1]
    if any(name.startswith(p) for p in NON_MODEL_PREFIXES):
        continue
    print(m.id)
PYEOF

N_MODELS=$(wc -l < "${MODEL_LIST_FILE}" | tr -d ' ')
echo "       → ${N_MODELS} language-model repos written to ${MODEL_LIST_FILE}"

if [[ "${N_MODELS}" -eq 0 ]]; then
    echo "ERROR: no BeetleLM model repos found." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 2. Build the (model, benchmark, lang-filter) evaluation plan.
#
# Parsing rules for BeetleLM/<repo>:
#   • repo names are of the form  beetlelm_<lang1>[-<lang2>][-<lang3>]_<regime>
#     or  beetlelm_<l1>_L1-<l2>_L2[-<l3>_L3]_<regime>
#     or  babylm-<l1>-<l2>-<regime>         (e.g. babylm-srp-kor-balanced)
#   • We extract EVERY ISO-639-3 code from the following set found in the
#     repo name (as a whole token between separators `_-`):
#          eng, nld, deu, zho, fra, fas, bul, ukr, ind, srp, kor
#   • The model is then evaluated on:
#         blimp_eng        if "eng" ∈ model_langs
#         multiblimp/fas   if "fas" ∈ model_langs  (same for bul, ukr, fra)
#         multiblimp/ind   → target requested but not in jumelet/multiblimp,
#                            so silently skipped (dataset load raises; handled
#                            by eval_blimp.py's per-lang try/except).
# ---------------------------------------------------------------------------

echo "[2/4] Building evaluation plan from parsed model names …"

${PYTHON} - "${MODEL_LIST_FILE}" <<'PYEOF' > "${PLAN_FILE}"
import re, sys
from pathlib import Path

KNOWN_ISO = {"eng", "nld", "deu", "zho", "fra", "fas",
             "bul", "ukr", "ind", "srp", "kor"}
MULTIBLIMP_TARGETS = {"fas", "bul", "ukr", "ind", "fra"}
# langs actually shipped in jumelet/multiblimp from our target set
MULTIBLIMP_AVAILABLE = {"fas", "bul", "ukr", "fra"}

token_re = re.compile(r"[A-Za-z]{3}")

def parse_langs(repo: str) -> set[str]:
    # keep only the repo name (strip "BeetleLM/")
    name = repo.split("/", 1)[-1]
    # tokenise on non-letters, keep 3-letter chunks that match ISO set
    found: set[str] = set()
    for tok in re.split(r"[^A-Za-z]+", name):
        t = tok.lower()
        if len(t) == 3 and t in KNOWN_ISO:
            found.add(t)
    return found

rows = []
for line in Path(sys.argv[1]).read_text().splitlines():
    repo = line.strip()
    if not repo:
        continue
    langs = parse_langs(repo)
    benches, filters = [], []
    if "eng" in langs:
        benches.append("blimp_eng"); filters.append("eng")
    mb = sorted(langs & MULTIBLIMP_TARGETS)
    # We include ind in the filter even if unavailable; eval_blimp.py will
    # warn+skip at dataset load time.
    if mb:
        benches.append("multiblimp")
        filters.extend(mb)
    if not benches:
        continue  # no overlap with any target — skip entirely
    # Deduplicate while preserving order
    seen = set(); dedup = []
    for b in benches:
        if b not in seen:
            seen.add(b); dedup.append(b)
    seenf = set(); dedup_f = []
    for f in filters:
        if f not in seenf:
            seenf.add(f); dedup_f.append(f)
    rows.append((repo, ",".join(dedup), ",".join(dedup_f)))

# TSV header + rows
print("repo\tbenchmarks\tlangs")
for r in rows:
    print("\t".join(r))
PYEOF

N_PLAN=$(( $(wc -l < "${PLAN_FILE}" | tr -d ' ') - 1 ))
echo "       → ${N_PLAN} eligible (model × target-lang) jobs written to ${PLAN_FILE}"

if [[ "${N_PLAN}" -le 0 ]]; then
    echo "No models matched any target language — nothing to do." >&2
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 3. Dispatch evaluations.
# ---------------------------------------------------------------------------

echo "[3/4] Dispatching evaluations to evals/eval_blimp.py …"
echo "      batch_size=${BATCH_SIZE}  rank=${RANK}/${WORLD_SIZE}  output=${OUTPUT_DIR}"

idx=0
while IFS=$'\t' read -r repo benchmarks langs; do
    # skip header
    if [[ "${repo}" == "repo" ]]; then continue; fi
    idx=$((idx + 1))

    # Shard across ranks if requested
    if (( WORLD_SIZE > 1 )); then
        if (( (idx - 1) % WORLD_SIZE != RANK )); then
            continue
        fi
    fi

    IFS=',' read -r -a bench_arr <<< "${benchmarks}"
    IFS=',' read -r -a lang_arr  <<< "${langs}"

    echo "  [${idx}/${N_PLAN}] ${repo}  benches=[${benchmarks}]  langs=[${langs}]"

    if [[ "${DRY_RUN}" == "1" ]]; then
        continue
    fi

    ${PYTHON} evals/eval_blimp.py \
        --models "${repo}" \
        --benchmarks "${bench_arr[@]}" \
        --langs "${lang_arr[@]}" \
        --output "${OUTPUT_DIR}" \
        --batch-size "${BATCH_SIZE}" \
        --rank 0 --world-size 1
done < "${PLAN_FILE}"

# ---------------------------------------------------------------------------
# Step 4. Emit LaTeX table.
# ---------------------------------------------------------------------------

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[4/4] DRY_RUN=1 — skipping LaTeX table generation."
    exit 0
fi

if [[ ! -f "${RESULTS_CSV}" ]]; then
    echo "[4/4] No ${RESULTS_CSV} produced — skipping LaTeX table." >&2
    exit 0
fi

echo "[4/4] Writing LaTeX table → ${LATEX_OUT}"
${PYTHON} scripts/make_beetlelm_blimp_table.py \
    --input "${RESULTS_CSV}" --output "${LATEX_OUT}"

echo "Done. Results CSV: ${RESULTS_CSV}"
echo "      LaTeX table: ${LATEX_OUT}"
