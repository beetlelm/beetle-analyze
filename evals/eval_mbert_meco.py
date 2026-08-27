#!/usr/bin/env python3
"""
evals/eval_mbert_meco.py — mBERT pseudo-surprisal MECO L2 baseline.

Implements the pseudo-surprisal estimator of de Varda & Marelli (2022)
"The Effects of Surprisal across Languages: Results from Native and Non-native
Reading" using bert-base-multilingual-cased: for each target token in a MECO
L2 stimulus, mask the token, run mBERT once, and read off -log p(token | rest)
from the softmax over the MLM head.

Per-word pseudo-surprisal is the sum of pseudo-surprisals over the word's
sub-tokens (matching the convention used by ``meco_surprisal._get_surprisal``).

We then correlate model pseudo-surprisal with participant-level reading times
loaded via the same .rda machinery as
``eval_meco_english_l2_participants.py``, restricted to L1 in
{zho, nld, deu} per the project plan.

This produces a single Table-6 row labelled ``mBERT (pseudo-surprisal)``.

Usage
-----
    python evals/eval_mbert_meco.py \\
        --langs zho nld deu \\
        --output-root results/evals_paper/06_mbert

    python evals/eval_mbert_meco.py \\
        --model bert-base-multilingual-cased \\
        --rt-columns firstrun.dur gaze.dur totalobs.dur
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_ANALYZE_DIR = _REPO_ROOT / "analyze"
if str(_ANALYZE_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYZE_DIR))

from evals.eval_meco_english_l2_participants import (  # noqa: E402
    DEFAULT_RDA, DEFAULT_RDA_WAVE2, DEFAULT_MECO_L1_MAP,
    load_meco_rda,
)

logger = logging.getLogger(__name__)

DEFAULT_STIMULI = _REPO_ROOT / "data" / "meco_l2_stims.tsv"
DEFAULT_MODEL = "bert-base-multilingual-cased"
DEFAULT_LANGS = ("zho", "nld", "deu")
DEFAULT_RT_COLUMNS = ("firstrun.dur", "gaze.dur", "totalobs.dur", "dur")

# ---------------------------------------------------------------------------
# Eval ledger (best-effort)
# ---------------------------------------------------------------------------

try:
    from analyze.eval_ledger import is_done as _ledger_is_done
    from analyze.eval_ledger import record as _ledger_record
except Exception:
    def _ledger_is_done(*_a, **_kw) -> bool: return False
    def _ledger_record(*_a, **_kw) -> None: pass


# ---------------------------------------------------------------------------
# Pseudo-surprisal scorer
# ---------------------------------------------------------------------------

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_mbert(model_name: str):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    model.to(_device())
    if torch.cuda.is_available():
        model = model.to(torch.bfloat16)
    return model, tok


@torch.no_grad()
def _pseudo_surprisal_for_marked(
    model, tok, full_text: str, target_word: str
) -> float:
    """Mean pseudo-surprisal of all sub-tokens that belong to ``target_word``.

    de Varda & Marelli 2022 sum over sub-tokens; we follow the same convention
    used elsewhere in this repo (mean per-token NLL converted to bits at the
    word level — caller can scale). Returns nats.
    """
    enc = tok(full_text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"][0]
    attn = enc["attention_mask"][0]

    # Identify target sub-token span. We retokenise the target word (without
    # special tokens) and locate the contiguous span inside ``input_ids``.
    target_ids = tok(target_word, add_special_tokens=False)["input_ids"]
    if not target_ids:
        return float("nan")

    span_start = -1
    L = len(target_ids)
    for i in range(len(input_ids) - L + 1):
        if input_ids[i:i + L].tolist() == target_ids:
            span_start = i
            break
    if span_start < 0:
        # Fallback: BPE/wordpiece reassembly didn't match. Skip.
        return float("nan")

    mask_id = tok.mask_token_id
    surprisals: List[float] = []
    for off in range(L):
        masked = input_ids.clone()
        masked[span_start + off] = mask_id
        masked = masked.unsqueeze(0).to(_device())
        amask = attn.unsqueeze(0).to(_device())
        out = model(input_ids=masked, attention_mask=amask).logits
        logp = F.log_softmax(out[0, span_start + off].float(), dim=-1)
        gold = int(input_ids[span_start + off].item())
        surprisals.append(-float(logp[gold].item()))
    return float(np.sum(surprisals))


def score_stimuli(
    stimuli_path: Path, model_name: str, max_rows: int | None = None
) -> pd.DataFrame:
    df = pd.read_csv(stimuli_path, sep="\t")
    if max_rows:
        df = df.head(max_rows).copy()
    model, tok = _load_mbert(model_name)
    logger.info("[mbert-meco] scoring %d rows with %s", len(df), model_name)

    surps: List[float] = []
    for i, row in df.iterrows():
        # ``FullTextMarked`` wraps the target word in *asterisks*; we recover the
        # raw target by stripping them. ``FullText`` has the natural prefix.
        marked = str(row.get("FullTextMarked", ""))
        word = str(row.get("word", ""))
        full_text_natural = marked.replace("*", "")
        s = _pseudo_surprisal_for_marked(model, tok, full_text_natural, word)
        surps.append(s)
        if i % 200 == 0:
            logger.info("  row %d / %d", i, len(df))

    df = df.copy()
    df["pseudo_surprisal"] = surps
    df["model"] = model_name
    return df


# ---------------------------------------------------------------------------
# RT correlation per L1
# ---------------------------------------------------------------------------

def _participant_word_rt(rda_paths: Sequence[Path]) -> pd.DataFrame:
    """Load and lightly normalise the joint MECO L2 word-level RT frame."""
    df = load_meco_rda(list(rda_paths))
    # Standardise the participant L1 column name.
    if "lang" in df.columns and "L1" not in df.columns:
        df = df.rename(columns={"lang": "L1"})
    if "L1" not in df.columns:
        # Some exports use ``language`` for the L1 native code.
        for cand in ("language", "native_lang", "l1"):
            if cand in df.columns:
                df = df.rename(columns={cand: "L1"})
                break
    if "L1" not in df.columns:
        raise RuntimeError(
            "Could not locate L1 column in MECO RDA; "
            f"saw columns: {list(df.columns)[:30]}"
        )

    df["L1_iso"] = df["L1"].astype(str).str.lower().map(DEFAULT_MECO_L1_MAP)
    return df


def _correlate_with_rt(
    scored: pd.DataFrame, rt_df: pd.DataFrame, lang: str, rt_col: str
) -> dict:
    sub = rt_df[rt_df["L1_iso"] == lang].copy()
    if sub.empty or rt_col not in sub.columns:
        return {"lang": lang, "rt_col": rt_col, "n_pairs": 0,
                "spearman_rho": float("nan"), "spearman_p": float("nan"),
                "pearson_r": float("nan"), "pearson_p": float("nan")}

    # Group-mean RT by (itemid, wordnum) within this L1 cohort.
    keycols = [c for c in ("itemid", "trialid", "wordnum") if c in sub.columns]
    if "itemid" not in keycols and "trialid" in keycols:
        sub = sub.rename(columns={"trialid": "itemid"})
        keycols = ["itemid", "wordnum"]
    grp = (
        sub.groupby(["itemid", "wordnum"], dropna=True)[rt_col]
           .mean()
           .reset_index(name="mean_rt")
    )
    merged = scored.merge(grp, on=["itemid", "wordnum"], how="inner")
    merged = merged.dropna(subset=["pseudo_surprisal", "mean_rt"])
    if len(merged) < 10:
        return {"lang": lang, "rt_col": rt_col, "n_pairs": int(len(merged)),
                "spearman_rho": float("nan"), "spearman_p": float("nan"),
                "pearson_r": float("nan"), "pearson_p": float("nan")}

    rho, p_rho = stats.spearmanr(merged["pseudo_surprisal"], merged["mean_rt"])
    r, p_r = stats.pearsonr(merged["pseudo_surprisal"], merged["mean_rt"])
    return {"lang": lang, "rt_col": rt_col, "n_pairs": int(len(merged)),
            "spearman_rho": float(rho), "spearman_p": float(p_rho),
            "pearson_r": float(r), "pearson_p": float(p_r)}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--stimuli", type=Path, default=DEFAULT_STIMULI)
    p.add_argument("--rda", type=Path, nargs="+",
                   default=[DEFAULT_RDA, DEFAULT_RDA_WAVE2])
    p.add_argument("--langs", nargs="+", default=list(DEFAULT_LANGS))
    p.add_argument("--rt-columns", nargs="+", default=list(DEFAULT_RT_COLUMNS))
    p.add_argument("--output-root", type=Path,
                   default=_REPO_ROOT / "results" / "evals_paper" / "06_mbert")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Smoke-test cap on stimuli rows.")
    p.add_argument("--ledger-stage", default="mbert_meco")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    out_root: Path = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.force and _ledger_is_done(args.ledger_stage, args.model, "ALL"):
        logger.info("[mbert-meco] ledger says DONE for %s; pass --force to redo",
                    args.model)
        return

    scored_path = out_root / f"{args.model.replace('/', '_')}_surprisal.csv"
    if scored_path.exists() and not args.force:
        logger.info("[mbert-meco] reusing cached %s", scored_path)
        scored = pd.read_csv(scored_path)
    else:
        scored = score_stimuli(args.stimuli, args.model, max_rows=args.max_rows)
        scored.to_csv(scored_path, index=False)
        logger.info("[mbert-meco] wrote %s", scored_path)

    rt = _participant_word_rt(args.rda)
    logger.info("[mbert-meco] RT frame shape=%s L1s=%s",
                rt.shape, sorted(rt["L1_iso"].dropna().unique()))

    rows = []
    for lang in args.langs:
        for col in args.rt_columns:
            if col not in rt.columns:
                continue
            rows.append(_correlate_with_rt(scored, rt, lang, col))
    summary = pd.DataFrame(rows)
    summary_path = out_root / "summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("[mbert-meco] wrote %s\n%s", summary_path, summary.to_string())

    # JSON sidecar that the figure scripts pick up.
    (out_root / "summary.json").write_text(
        json.dumps({"model": args.model, "rows": rows}, indent=2)
    )

    try:
        _ledger_record(args.ledger_stage, args.model, "ALL", str(summary_path))
    except Exception:
        pass


if __name__ == "__main__":
    main()
