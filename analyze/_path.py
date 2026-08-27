"""
_path.py — Repo-root bootstrap for analyze/ scripts.
Import as the first local import:  import _path  # noqa: F401

Adds two directories to sys.path so analyze/ scripts work both as
`python analyze/foo.py` (script mode) and `python -m analyze.foo` (module
mode):
  * the repo root  — so `from models import ...` resolves
  * analyze/ itself — so sibling imports like `from ppl_utils import ...`
                       resolve under `-m` mode (script mode adds this
                       implicitly via sys.path[0]).
"""
from __future__ import annotations
import sys
from pathlib import Path

_ANALYZE_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _ANALYZE_DIR.parent
for _p in (_REPO_ROOT, _ANALYZE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
