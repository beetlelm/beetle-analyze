#!/usr/bin/env bash
# fix_paths.sh — run once from the repo root to fix ModuleNotFoundError
#
# Usage:
#   cd ~/beetle-analyze
#   bash analyze/fix_paths.sh
#
# What it does:
#   1. Drops _path.py into analyze/  (the shared sys.path bootstrap)
#   2. Renames analyze/utils.py → analyze/ppl_utils.py to avoid collision
#      with the repo-root utils.py used by eval_model.py
#   3. Rewrites all `from utils import` → `from ppl_utils import` in analyze/
#   4. Adds `import _path` as the first local import in each analyze/ script

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ANALYZE_DIR="$SCRIPT_DIR"

echo "Repo root  : $REPO_ROOT"
echo "Analyze dir: $ANALYZE_DIR"

# ── 1. Write _path.py ────────────────────────────────────────────────────────
cat > "$ANALYZE_DIR/_path.py" << 'PYEOF'
"""
_path.py — Repo-root bootstrap for analyze/ scripts.
Import as the first local import:  import _path  # noqa: F401
"""
from __future__ import annotations
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
PYEOF
echo "Written: $ANALYZE_DIR/_path.py"

# ── 2. Rename utils.py → ppl_utils.py to avoid collision with root utils.py ──
if [[ -f "$ANALYZE_DIR/utils.py" && ! -f "$ANALYZE_DIR/ppl_utils.py" ]]; then
    mv "$ANALYZE_DIR/utils.py" "$ANALYZE_DIR/ppl_utils.py"
    echo "Renamed: utils.py → ppl_utils.py"
elif [[ -f "$ANALYZE_DIR/utils.py" && -f "$ANALYZE_DIR/ppl_utils.py" ]]; then
    echo "  NOTE: both utils.py and ppl_utils.py exist — removing utils.py"
    rm "$ANALYZE_DIR/utils.py"
else
    echo "  OK: ppl_utils.py already in place"
fi

# ── 3. Rewrite `from utils import` → `from ppl_utils import` in all scripts ──
TARGET_FILES=(
    "run_pipeline.py"
    "ppl_eval.py"
    "forgetting.py"
    "embedding_drift.py"
    "reading_time.py"
    "convergence.py"
    "visualise.py"
)

echo ""
echo "Rewriting utils imports → ppl_utils …"
for fname in "${TARGET_FILES[@]}"; do
    fpath="$ANALYZE_DIR/$fname"
    [[ -f "$fpath" ]] || { echo "  SKIP (not found): $fname"; continue; }
    sed -i \
        's/from utils import/from ppl_utils import/g' \
        "$fpath"
    echo "  $fname"
done

# ── 4. Add `import _path` and remove stale sys.path.insert lines ─────────────
echo ""
echo "Patching sys.path bootstraps …"
for fname in "${TARGET_FILES[@]}"; do
    fpath="$ANALYZE_DIR/$fname"
    [[ -f "$fpath" ]] || continue

    # Remove stale wrong-directory sys.path.insert lines
    sed -i '/sys\.path\.insert(0,.*Path(__file__)\.parent/d' "$fpath"

    # Insert `import _path` after `from __future__` if not already present
    if ! grep -q "import _path" "$fpath"; then
        future_line=$(grep -n "^from __future__" "$fpath" | tail -1 | cut -d: -f1)
        if [[ -n "$future_line" ]]; then
            sed -i "${future_line}a import _path  # noqa: F401  — adds repo root to sys.path" "$fpath"
        else
            sed -i "1a import _path  # noqa: F401  — adds repo root to sys.path" "$fpath"
        fi
        echo "  PATCHED: $fname"
    else
        echo "  OK (already patched): $fname"
    fi
done

echo ""
echo "Done. Test with:"
echo "  cd $REPO_ROOT && python analyze/run_pipeline.py --help"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ANALYZE_DIR="$SCRIPT_DIR"

echo "Repo root  : $REPO_ROOT"
echo "Analyze dir: $ANALYZE_DIR"

# ── 1. Write _path.py ────────────────────────────────────────────────────────
cat > "$ANALYZE_DIR/_path.py" << 'PYEOF'
"""
_path.py — Repo-root bootstrap for analyze/ scripts.
Import as the first local import:  import _path  # noqa: F401
"""
from __future__ import annotations
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
PYEOF
echo "Written: $ANALYZE_DIR/_path.py"

# ── 2. Patch each relevant .py file ─────────────────────────────────────────
# Files that need the repo root on sys.path
TARGET_FILES=(
    "run_pipeline.py"
    "ppl_eval.py"
    "forgetting.py"
    "embedding_drift.py"
    "reading_time.py"
    "convergence.py"
    "visualise.py"
    "utils.py"
)

for fname in "${TARGET_FILES[@]}"; do
    fpath="$ANALYZE_DIR/$fname"
    [[ -f "$fpath" ]] || { echo "  SKIP (not found): $fname"; continue; }

    # a) Remove any existing wrong-directory sys.path.insert lines
    #    (matches both parent and parent.parent variants so re-running is safe)
    sed -i \
        '/sys\.path\.insert(0,.*Path(__file__)\.parent/d' \
        "$fpath"

    # b) If `import _path` is not already present, insert it after
    #    the last `from __future__` line (or after the module docstring
    #    if no __future__ import exists).
    if ! grep -q "import _path" "$fpath"; then
        # Find the line number of `from __future__ import annotations` if present
        future_line=$(grep -n "^from __future__" "$fpath" | tail -1 | cut -d: -f1)
        if [[ -n "$future_line" ]]; then
            sed -i "${future_line}a import _path  # noqa: F401  — adds repo root to sys.path" \
                "$fpath"
        else
            # Fallback: insert after the opening docstring block (first non-comment, non-blank line)
            sed -i "1a import _path  # noqa: F401  — adds repo root to sys.path" \
                "$fpath"
        fi
        echo "  PATCHED: $fname"
    else
        echo "  OK (already patched): $fname"
    fi
done

echo ""
echo "Done. Test with:"
echo "  cd $REPO_ROOT && python analyze/run_pipeline.py --help"