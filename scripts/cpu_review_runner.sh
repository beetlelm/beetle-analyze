#!/usr/bin/env bash
# scripts/cpu_review_runner.sh
# CPU-only pipeline that addresses reviewer points W6 (significance), W9
# (tokenisation bound), and the Table 12 / App. K BLiSS detail comments.
# Designed to run inside a tmux pane; tees everything to logs/cpu_review/.

set -u
set -o pipefail

REPO_ROOT="/Users/suchirsalhan/Documents/PHD/PHD/beetle-analyze"
VENV_PY="/Users/suchirsalhan/.venvs/eval-pipeline/bin/python"
LOG_DIR="${REPO_ROOT}/logs/cpu_review"
mkdir -p "${LOG_DIR}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"

ts() { date +%Y-%m-%dT%H:%M:%S; }
banner() { echo; echo "==== [$(ts)] $1 ===="; }

banner "env"
"${VENV_PY}" --version
"${VENV_PY}" -c "import numpy, pandas, statsmodels; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'statsmodels', statsmodels.__version__)"

# ---------------------------------------------------------------------------
# W6: Statistical apparatus (paired bootstrap + permutation + Holm-Bonferroni)
# ---------------------------------------------------------------------------
banner "W6: meco_significance — bootstrap + permutation"
LOG_W6="${LOG_DIR}/w6_meco_significance.log"
{
  # CPU budget: 12 models x 3 langs x 200 resamples ~= 7.2K MixedLM fits.
  # bootstrap-n=200 still resolves the 95% percentile CI cleanly for the
  # headline ΔlogLik claim (percentile index 195/200). Skip --three-measures
  # (focus on firstrun.dur, the canonical reading-time measure). kfold=5
  # keeps the per-fold pass tractable on CPU.
  "${VENV_PY}" -m analyze.meco_significance \
      --bootstrap-n 200 \
      --kfold 5 \
      --backend statsmodels \
      --langs du ch_s ge \
  || echo "[w6] meco_significance exited non-zero (continuing)"
} 2>&1 | tee "${LOG_W6}"

banner "W6: outputs"
ls -la "${REPO_ROOT}/results/meco/significance" 2>&1 || true

# ---------------------------------------------------------------------------
# W9: Cross-tokeniser bits-per-byte normalisation
# Computes BLOOM ladder + Beetle B1/B5 token counts on FLORES-200 nld + eng.
# Result lands in results/cache/bpt_bpb_ratios.csv and lets us re-express the
# 7.1× headline gap in tokenizer-fair BPB.
# ---------------------------------------------------------------------------
banner "W9: byte_normalize — tokeniser-fair token/byte counts"
LOG_W9="${LOG_DIR}/w9_byte_normalize.log"
REPOS=(
  "bigscience/bloom-560m"
  "bigscience/bloom-1b1"
  "bigscience/bloom-1b7"
  "bigscience/bloom-3b"
  "bigscience/bloom-7b1"
  "Beetle-FineWeb/beetle-bilingual-balanced-b1-fineweb-nld-eng"
  "Beetle-FineWeb/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-nld-eng"
  "Beetle-FineWeb/beetle-bilingual-balanced-b5-fineweb-nld-eng"
)
{
  "${VENV_PY}" -m analyze.byte_normalize \
      --repos "${REPOS[@]}" \
      --langs nld eng \
  || echo "[w9] byte_normalize exited non-zero (continuing)"
} 2>&1 | tee "${LOG_W9}"

banner "W9: cache snapshot"
"${VENV_PY}" - <<'PY' 2>&1 | tee -a "${LOG_W9}"
import pandas as pd
df = pd.read_csv("results/cache/bpt_bpb_ratios.csv")
print(df.to_string(index=False))
PY

# ---------------------------------------------------------------------------
# Detail-comment diagnostics
# ---------------------------------------------------------------------------
banner "DC: Table 12 Panel B duplicate-row probe (ewc / lamol / B2 identical)"
LOG_DC="${LOG_DIR}/dc_diagnostics.log"
{
  "${VENV_PY}" - <<'PY'
import pandas as pd, glob, sys
from pathlib import Path
candidates = sorted(set(glob.glob("results/**/meco*.csv", recursive=True))
                    | set(glob.glob("results/**/canonical*.csv", recursive=True))
                    | set(glob.glob("results/meco/*.csv")))
print("[dc] candidate frames:", len(candidates))
hits = []
for path in candidates:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        continue
    if "model" not in df.columns:
        continue
    name_col = df["model"].astype(str).str.lower()
    mask = name_col.str.contains("ewc|lamol|b2|simultaneous", regex=True)
    sub = df[mask]
    if not sub.empty:
        print(f"\n=== {path} ===  ({len(sub)} matching rows)")
        cols = [c for c in ("model","lang","scale","curriculum",
                            "delta_loglik","beta_surprisal") if c in sub.columns]
        print(sub[cols].head(12).to_string(index=False))
        hits.append(path)
if not hits:
    print("[dc] no ewc/lamol/B2 rows discovered in results/**.csv")
PY
} 2>&1 | tee "${LOG_DC}"

banner "DC: BLiSS App. K join scope"
{
  "${VENV_PY}" - <<'PY'
import pandas as pd, glob
paths = sorted(glob.glob("results/**/bliss*.csv", recursive=True))
for p in paths:
    df = pd.read_csv(p)
    print(f"=== {p} ===")
    print("rows:", len(df), "cols:", list(df.columns)[:12])
    if "model" in df.columns:
        n_models = df["model"].nunique()
        print(f"unique models: {n_models}")
        print("first 5 model names:")
        for m in df["model"].drop_duplicates().head(5).tolist():
            print(" ", m)
    print()
PY
} 2>&1 | tee -a "${LOG_DC}"

banner "DONE"
echo "Logs in ${LOG_DIR}"
ls -la "${LOG_DIR}"
