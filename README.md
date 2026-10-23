# beetle-analyze

**Reproduce a headline BLiMP accuracy on CPU in under a second — then run the full interpretability and evaluation suite on your own Beetle checkpoints.**

`beetle-analyze` is the analysis component of the [Beetle](https://beetlelm.github.io)
project: the code used to study trained LM checkpoints. It covers BLiMP-style grammatical
minimal-pair scoring, learning-curve and phase-transition analysis, cross-lingual
representation analysis (CKA, probing, logit lens), sparse-autoencoder feature analysis,
reading-time / eye-tracking (MECO) and FLORES perplexity evaluation, functional
localization of syntactic units, and more.

It ships with a **minimal, self-contained quickstart** — a BLiMP-style analysis that
runs on CPU in under a second with no model downloads — so you can install the package
and reproduce a headline number immediately, then move on to the full research modules.

> This repository contains **code only**. No datasets, model checkpoints, eval outputs,
> figures, or paper artifacts are included. The full research modules expect you to
> supply your own checkpoints and data.

## Install

```bash
pip install -e .                    # full suite (pulls torch, transformers, datasets, ...)
pip install -e ".[dev]"             # + tests and the toy quickstart
pip install -e ".[all]"             # + every specialised extra
```

Requires Python 3.10–3.12. The full install is heavy (PyTorch + Hugging Face stack).
Specialised analyses are grouped as optional extras — `stats`, `eyetracking`, `sft`,
`saes`, `brain`, `viz`, `geometry`, `io`, `nlp` — install only what you need, e.g.
`pip install -e ".[stats,eyetracking]"`.

## Quickstart (no GPU, no downloads)

The bundled `beetle_analyze` package provides a headline BLiMP-style analysis over
*precomputed* sentence log-probabilities (i.e. eval output). It runs instantly on CPU:

```bash
beetle-analyze blimp
```

```
  scoring         : sum
  pairs           : 60
  overall accuracy: 0.6000  [95% CI 0.4667, 0.7167]
  per phenomenon  :
      anaphor_agreement               0.750  (n=12)
      determiner_noun_agreement       0.667  (n=12)
      ellipsis                        0.333  (n=12)
      npi_licensing                   0.750  (n=12)
      subject_verb_agreement          0.500  (n=12)
```

**The headline toy result is `overall accuracy = 0.6000`.** Expected metrics are
committed in [`baselines/`](baselines/) so you can diff your run.

### Python API

```python
from beetle_analyze import analyze_file, toy_data_path

result = analyze_file(toy_data_path(), scoring="sum")
print(result.overall_accuracy)      # 0.6
print(result.per_phenomenon)        # per-phenomenon DataFrame
```

### Change one variable and re-run

Raw `sum` scoring compares total sentence log-probabilities and carries a length bias.
Switch to per-token `mean` scoring to remove it:

```bash
beetle-analyze blimp --scoring mean     # accuracy 0.60 -> 0.80
```

Example configs for both variants live in
[`beetle_analyze/configs/`](beetle_analyze/configs/); a step-by-step walkthrough is in
[`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb).

### Bring your own eval output

Point the CLI at any JSONL file, one minimal pair per line:

```json
{"phenomenon": "subject_verb_agreement", "logprob_good": -21.3, "logprob_bad": -24.8, "n_tokens_good": 8, "n_tokens_bad": 8}
```

```bash
beetle-analyze blimp --input my_evals.jsonl --json-out result.json --plot-out fig.png
```

## The full research suite

The research modules operate on trained checkpoints and their eval outputs. They
generally require the full ML stack and a GPU, plus data you provide. Installed console
entry points include `beetle-cka`, `beetle-probe`, `beetle-baseline`,
`beetle-dev-trajectory`, `beetle-cognate-filter`, `beetle-byte-normalize`,
`beetle-units-localize`, `beetle-units-ablate`, `beetle-units-trajectory`. Most modules
are also runnable as `python -m analyze.<module>`, and the research runner drives a full
sweep:

```bash
python analyze/run.py --all --family <family> --langs <lang[,lang...]>
```

## Project layout

```
beetle_analyze/      # self-contained minimal quickstart: scoring, CLI, toy data, configs
analyze/             # main analysis package (BLiMP, MECO, convergence, phase transitions, ...)
  xling/             #   cross-lingual analysis (CKA, probing, logit lens, cognates)
  sae/               #   sparse-autoencoder feature analysis
  units/             #   functional localization + ablation of syntactic units
  brain/  causal/  lens/  forgetting_pkg/  meco_detailed/
evals/               # benchmark runners (BLiMP, MECO, FLORES, RACE, priming, UniversalCEFR)
finetune/            # instruction / LoRA fine-tuning pipeline
slt/                 # singular-learning-theory geometry analyses
scripts/             # orchestration / sweep run-scripts (.py and .sh)
legacy/              # earlier eval drivers kept for reference
models.py  utils.py  # shared model loading + evaluation helpers
tests/               # fast CPU smoke test for the quickstart
notebooks/           # quickstart.ipynb
baselines/           # expected metrics for the toy analysis
```

## Compute requirements

- **Quickstart:** CPU only, <1 s, no downloads. Just `numpy` + `pandas`.
- **Full suite:** the ML stack (PyTorch + Hugging Face) and, for most analyses, a
  CUDA GPU plus your own checkpoints and evaluation data.

## Reproduce / regenerate

```bash
pytest -q                            # runs the smoke tests, asserts the 0.60 headline
python scripts/make_toy_data.py      # regenerate the synthetic toy dataset (fixed seed)
```

The toy dataset is fully synthetic (fixed random seed) and contains no real model
outputs or research results.

### Continuous integration

A ready-to-use GitHub Actions workflow (install `.[dev]`, run `pytest`, smoke-run the
CLI on Python 3.10–3.12) ships as [`.github/ci.yml.disabled`](.github/ci.yml.disabled).
It was placed there because the automated push token lacked the `workflow` OAuth scope.
Enable it with:

```bash
git mv .github/ci.yml.disabled .github/workflows/ci.yml
git commit -m "Enable CI" && git push
```

## Learn more

- **Tutorial:** [beetle-analyze walkthrough](https://beetlelm.github.io/docs.html#beetle-analyze) on the project docs.
- **Research:** https://beetlelm.github.io/research.html
- **Project home:** https://beetlelm.github.io
- **Framework overview:** https://beetlelm.github.io/framework.html
- **Paper:** *Beetle: A Bilingual Model Suite for Modelling Second-Language Processing* (EMNLP 2026) — https://arxiv.org/abs/2609.22633

## Part of the Beetle ecosystem

- **[beetlelm](https://github.com/beetlelm/beetlelm)** — core training framework.
- **[beetle-data](https://github.com/beetlelm/beetle-data)** — bilingual data-preparation pipeline.
- **[beetle-analyze](https://github.com/beetlelm/beetle-analyze)** (this repo) — evaluation and interpretability analysis toolkit.
- **[beetle-explorer](https://github.com/beetlelm/beetle-explorer)** — no-code Colab model explorer.

## Citation / DOI

If you use this software, please cite it via [`CITATION.cff`](CITATION.cff).

A DOI can be minted through [Zenodo](https://zenodo.org): the maintainer enables the
`beetlelm/beetle-analyze` repository in Zenodo's GitHub integration, then publishing a
GitHub release (e.g. `v0.1.0`) archives it and issues a DOI. A
[`.zenodo.json`](.zenodo.json) is included for that step.

## License

[Apache-2.0](LICENSE) © Suchir Salhan.
