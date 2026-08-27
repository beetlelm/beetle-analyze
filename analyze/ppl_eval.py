"""
ppl_eval.py — FLORES Goldfish PPL evaluation.

For each language, scores every sentence in the FLORES devtest split under
every model in the corresponding MODEL_GROUPS entry.

Outputs
-------
results/ppl/
  ppl_<lang>_<model_slug>.csv   — per-sentence NLL + PPL
  ppl_summary.csv               — mean PPL per (model, language)
"""

from __future__ import annotations
from analyze import _path  # noqa: F401  — adds repo root to sys.path

from pathlib import Path

import pandas as pd

from ppl_utils import (
    GOLDFISH_TOKENS,
    LOAD_FAILURES,
    get_best_revision,
    load_flores_sentences,
    nll_to_ppl,
    score_sentences,
)
from models import MODEL_GROUPS
from model_naming import parse_repo

try:
    # convergence.list_checkpoints returns ["main"] + step_branches; used when
    # all_checkpoints=True so PPL iterates every step-N branch plus main.
    from convergence import list_checkpoints as _list_checkpoints
except Exception:  # pragma: no cover
    def _list_checkpoints(repo: str, hf_token=None) -> list[str]:  # type: ignore[override]
        return [get_best_revision(repo)]

try:
    from eval_ledger import record as _ledger_record
except Exception:  # pragma: no cover — ledger is best-effort
    def _ledger_record(*_a, **_kw) -> None:
        pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("results/ppl")


# ---------------------------------------------------------------------------
# Per-language evaluation
# ---------------------------------------------------------------------------

def evaluate_language(
    iso_code: str,
    repos: list[str],
    output_dir: Path = OUTPUT_DIR,
    goldfish_tokens: int | None = GOLDFISH_TOKENS,
    overwrite: bool = False,
    rank: int = 0,
    world_size: int = 1,
    output_root: "Path | str" = "results",
    all_checkpoints: bool = False,
) -> pd.DataFrame:
    """
    Score all FLORES sentences for *iso_code* under every repo in *repos*.

    Returns a DataFrame with columns
      [sentence_id, sentence, repo, revision, iso_code, nll, ppl].
    Skips repo/language combos whose per-revision CSV already exists unless
    overwrite=True.

    When ``all_checkpoints=True``, iterates every ``step-N`` branch plus
    ``main`` for each repo (via ``convergence.list_checkpoints``), writing
    one CSV per (repo, revision, iso_code). When False (legacy), scores
    only the highest ``step-N`` via :func:`get_best_revision`.

    Emits an ``eval_ledger`` record after each successful write so the
    ledger report reflects per-(language, checkpoint) coverage.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sentences = load_flores_sentences(iso_code)

    all_rows: list[dict] = []

    from tqdm.auto import tqdm
    # Pre-compute per-repo revision lists so the outer tqdm knows the total
    # (repo, revision) count and can emit an ETA instead of one bar per repo.
    work: list[tuple[str, str]] = []
    for repo in repos:
        if all_checkpoints:
            revisions = _list_checkpoints(repo)
        else:
            revisions = [get_best_revision(repo)]
        for rv in revisions:
            work.append((repo, rv))

    print(f"[ppl_eval] {iso_code}: {len(work)} (repo, revision) items to score "
          f"(rank={rank}/{world_size})")
    sharded = world_size > 1
    for repo, revision in tqdm(work, desc=f"ppl[{iso_code}]", unit="ckpt"):
        slug = repo.replace("/", "__")
        canonical_path = output_dir / f"ppl_{iso_code}_{slug}_{revision}.csv"
        # Per-rank shard path. When world_size==1 this collapses to the
        # canonical_path (no suffix) so existing downstream readers work
        # unchanged. When sharded, rank 0 reassembles after the barrier.
        shard_path = (
            output_dir
            / f"ppl_{iso_code}_{slug}_{revision}_rank{rank}of{world_size}.csv"
            if sharded else canonical_path
        )

        # Cache hit check: prefer the canonical reassembled CSV.
        if canonical_path.exists() and not overwrite:
            print(f"[ppl_eval] Skipping {repo}@{revision} ({iso_code}) — already exists.")
            all_rows.extend(pd.read_csv(canonical_path).to_dict("records"))
            _ledger_record(
                stage="ppl", repo=repo, checkpoint=revision,
                benchmark="flores", lang=iso_code, status="ok",
                extra={"cached": True},
                rank=rank, output_root=output_root,
            )
            continue
        # Under sharding, also skip if this rank's shard already exists
        # (resume support without redoing model load).
        if sharded and shard_path.exists() and not overwrite:
            print(f"[ppl_eval] rank={rank}: shard exists for "
                  f"{repo}@{revision} ({iso_code}) — reusing.")
            continue

        print(f"[ppl_eval] Scoring {iso_code} under {repo} @ {revision} "
              f"[rank {rank}/{world_size}] …")
        indexed = score_sentences(
            sentences, repo,
            goldfish_tokens=goldfish_tokens,
            revision=revision,
            rank=rank,
            world_size=world_size,
        )

        if indexed is None:
            print(f"[ppl_eval] SKIPPED {repo}@{revision} — model failed to load.")
            _ledger_record(
                stage="ppl", repo=repo, checkpoint=revision,
                benchmark="flores", lang=iso_code, status="load_failed",
                rank=rank, output_root=output_root,
            )
            continue

        rows = [
            {
                "sentence_id": i,
                "sentence":    sentences[i],
                "repo":        repo,
                "revision":    revision,
                "iso_code":    iso_code,
                "nll":         nll,
                "ppl":         nll_to_ppl(nll),
            }
            for (i, nll) in indexed
        ]
        pd.DataFrame(rows).to_csv(shard_path, index=False)
        all_rows.extend(rows)
        print(f"  → saved {shard_path}")
        _ledger_record(
            stage="ppl", repo=repo, checkpoint=revision,
            benchmark="flores", lang=iso_code, status="ok",
            rank=rank, output_root=output_root,
        )

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Summary across all languages
# ---------------------------------------------------------------------------

def _bucket_repos_by_lang(repos: list[str]) -> dict[str, list[str]]:
    """Group a list of repos by the languages their names declare (l1 and l2).

    A bilingual ``nld-eng`` repo lands under both ``'nld'`` and ``'eng'``;
    a monolingual ``eng`` repo lands only under ``'eng'``. Mirrors how
    ``MODEL_GROUPS`` indexes language codes but restricted to the already-
    filtered pool.
    """
    buckets: dict[str, list[str]] = {}
    for repo in repos:
        try:
            meta = parse_repo(repo)
        except Exception:
            continue
        for lg in (meta.l1, meta.l2):
            if lg:
                buckets.setdefault(lg, []).append(repo)
    return buckets


def run_all_languages(
    lang_codes: list[str] | None = None,
    overwrite: bool = False,
    rank: int = 0,
    world_size: int = 1,
    output_root: "Path | str" = "results",
    repos: list[str] | None = None,
    all_checkpoints: bool = False,
) -> pd.DataFrame:
    """
    Run PPL evaluation for all (or a subset of) languages.

    Parameters
    ----------
    lang_codes : optional subset of ISO-639-3 codes. Defaults to every key
        in ``MODEL_GROUPS`` (legacy behaviour) when ``repos`` is None.
    repos : optional pre-filtered list of repos (e.g. from
        ``analyze.run._resolve_repos(args)``). When provided, the repo pool
        for each language is derived from this list via ``parse_repo``
        rather than ``MODEL_GROUPS[lang]``. This is how ``--family`` /
        ``--orgs`` / strict ``--langs`` propagate into PPL.

    When ``world_size > 1``, the repo list for each language is sharded
    ``repos[rank::world_size]``. Each rank writes its own per-repo
    ``ppl_<lang>_<slug>.csv``; rank 0 builds ``ppl_summary.csv`` after a
    barrier by reading every per-repo CSV on disk.

    Returns summary DataFrame with columns
      [repo, revision, iso_code, mean_nll, mean_ppl, median_ppl, std_ppl].
    """
    from _rank_barrier import mark_done, reset, shard, wait_for_all

    # Resolve per-language buckets from either the caller-supplied pool or
    # the legacy MODEL_GROUPS merge.
    if repos is not None:
        buckets = _bucket_repos_by_lang(repos)
        if lang_codes is None:
            lang_codes = sorted(buckets.keys())
    else:
        if lang_codes is None:
            lang_codes = list(MODEL_GROUPS.keys())
        buckets = {code: list(MODEL_GROUPS.get(code, [])) for code in lang_codes}

    # Rank 0 clears old sentinels before the work begins.
    if rank == 0:
        reset("ppl", world_size, output_root)

    # Within-stage sharding policy:
    #   * When the per-lang repo pool is >= world_size, shard repos across
    #     ranks (each rank loads fewer models; sentences unsharded).
    #   * When the pool is smaller than world_size, have EVERY rank process
    #     every repo but shard sentences internally via score_sentences'
    #     rank/world_size args. This is the single-model-on-8-GPUs case.
    for code in lang_codes:
        repos_for_lang = buckets.get(code, [])
        if not repos_for_lang:
            continue
        if world_size <= 1 or len(repos_for_lang) >= world_size:
            # Classic repo-level sharding.
            my_repos = shard(repos_for_lang, rank, world_size)
            if not my_repos:
                print(f"[ppl_eval] rank={rank}: no repos for {code}; skipping.")
                continue
            evaluate_language(
                code, my_repos, overwrite=overwrite,
                rank=rank, world_size=1,  # classic mode writes canonical CSVs
                output_root=output_root,
                all_checkpoints=all_checkpoints,
            )
        else:
            # Within-stage sentence sharding across all ranks.
            print(f"[ppl_eval] rank={rank}: {code} has {len(repos_for_lang)} "
                  f"repo(s) < world_size={world_size}; using sentence-level "
                  f"sharding.")
            evaluate_language(
                code, repos_for_lang, overwrite=overwrite,
                rank=rank, world_size=world_size,
                output_root=output_root,
                all_checkpoints=all_checkpoints,
            )

    # Signal completion; rank 0 waits for all ranks before building summary.
    mark_done("ppl", rank, world_size, output_root)
    if rank != 0:
        return pd.DataFrame()
    if world_size > 1:
        wait_for_all("ppl", world_size, output_root)

    # Reassemble per-rank shard CSVs into canonical per-(repo, revision) CSVs.
    # Only needed when sentence-level sharding was used (rank-suffixed files
    # exist). Safe to run unconditionally: glob returns [] when none exist.
    for code in lang_codes:
        for repo in buckets.get(code, []):
            slug = repo.replace("/", "__")
            shard_glob = OUTPUT_DIR.glob(
                f"ppl_{code}_{slug}_*_rank*of{world_size}.csv"
            )
            # Group shards by canonical path (strip the _rank{r}of{W} suffix).
            from collections import defaultdict
            groups: dict[Path, list[Path]] = defaultdict(list)
            for sp in shard_glob:
                canon = OUTPUT_DIR / sp.name.replace(
                    f"_rank{sp.name.split('_rank')[1]}", ""
                )
                # cleaner: rebuild the canonical stem without the suffix
                stem = sp.stem
                # stem = ppl_{iso}_{slug}_{rev}_rank{r}of{W}
                base = "_".join(stem.split("_")[:-1])  # drop trailing _rankRofW
                canon = OUTPUT_DIR / f"{base}.csv"
                groups[canon].append(sp)
            for canon, shards in groups.items():
                if canon.exists() and not overwrite:
                    continue
                merged = (pd.concat([pd.read_csv(p) for p in shards],
                                    ignore_index=True)
                          .sort_values("sentence_id")
                          .reset_index(drop=True))
                merged.to_csv(canon, index=False)
                for p in shards:
                    p.unlink(missing_ok=True)

    # Build the summary by reading all per-(repo,revision) CSVs from disk
    # (safe across ranks because file names are slug-, iso-, and
    # revision-namespaced). We glob both the new revisioned pattern
    # (ppl_<iso>_<slug>_<revision>.csv) and the legacy unrevisioned one
    # (ppl_<iso>_<slug>.csv) for backwards compatibility with cached runs
    # predating CKPT_SCOPE.
    frames: list[pd.DataFrame] = []
    for code in lang_codes:
        for repo in buckets.get(code, []):
            slug = repo.replace("/", "__")
            legacy_path = OUTPUT_DIR / f"ppl_{code}_{slug}.csv"
            if legacy_path.exists():
                frames.append(pd.read_csv(legacy_path))
            for csv_path in OUTPUT_DIR.glob(f"ppl_{code}_{slug}_*.csv"):
                frames.append(pd.read_csv(csv_path))

    if not frames:
        return pd.DataFrame()

    full = pd.concat(frames, ignore_index=True)
    summary = (
        full
        .groupby(["repo", "revision", "iso_code"])
        .agg(
            mean_nll   = ("nll", "mean"),
            mean_ppl   = ("ppl", "mean"),
            median_ppl = ("ppl", "median"),
            std_ppl    = ("ppl", "std"),
            n_sentences= ("nll", "count"),
        )
        .reset_index()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "ppl_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n[ppl_eval] Summary saved → {summary_path}")

    if LOAD_FAILURES:
        print(f"\n[ppl_eval] {'='*56}")
        print(f"[ppl_eval] SKIPPED {len(LOAD_FAILURES)} model(s) due to load errors:")
        for repo, reason in LOAD_FAILURES.items():
            print(f"  ✗  {repo}")
            print(f"     {reason}")
        print(f"[ppl_eval] {'='*56}")

    return summary


# ---------------------------------------------------------------------------
# Convenience: score a single sentence interactively
# ---------------------------------------------------------------------------

def score_one(
    sentence: str,
    repo: str,
    goldfish_tokens: int | None = GOLDFISH_TOKENS,
) -> dict:
    """Quick helper for interactive / notebook use."""
    from ppl_utils import load_model_and_tokenizer, sentence_log_likelihood
    model, tokenizer = load_model_and_tokenizer(repo)
    nll = sentence_log_likelihood(sentence, model, tokenizer, goldfish_tokens)
    return {
        "repo":     repo,
        "revision": get_best_revision(repo),
        "sentence": sentence,
        "nll":      nll,
        "ppl":      nll_to_ppl(nll),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FLORES Goldfish PPL evaluation.")
    parser.add_argument("--langs", nargs="+", help="ISO codes to evaluate (default: all)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = run_all_languages(lang_codes=args.langs, overwrite=args.overwrite)
    print(summary.to_string())