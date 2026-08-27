# beetle-analyze

A minimal, reproducible toolkit for analysing **language-model evaluation output**.

The headline analysis scores a [BLiMP](https://github.com/alexwarstadt/blimp)-style
set of grammatical *minimal pairs*: given the log-probabilities a model assigned to
a grammatical sentence and its near-identical ungrammatical counterpart, it reports
the fraction of pairs the model got right — overall and broken down by phenomenon,
with a bootstrap confidence interval.

It works on **precomputed eval output**, so it needs no GPU and downloads no models.
The bundled example runs on CPU in well under a second.

## Install

```bash
pip install beetle-analyze          # from PyPI (once published)
# or, from a clone:
pip install -e .                    # runtime only
pip install -e ".[dev]"             # + tests and plotting
```

Requires Python 3.10–3.12. The only runtime dependencies are `numpy` and `pandas`.

## Quickstart

Run the headline analysis on the bundled toy dataset:

```bash
beetle-analyze blimp
```

```
beetle-analyze BLiMP  (input=.../data/blimp_toy.jsonl)
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

**The headline toy result is `overall accuracy = 0.6000`.** A precomputed copy of
these metrics lives in [`baselines/blimp_toy_sum.json`](baselines/blimp_toy_sum.json)
so you can diff your run against the expected values.

### Python API

```python
from beetle_analyze import analyze_file, toy_data_path

result = analyze_file(toy_data_path(), scoring="sum")
print(result.overall_accuracy)      # 0.6
print(result.per_phenomenon)        # per-phenomenon DataFrame
```

## Change one variable and re-run

Raw `sum` scoring compares total sentence log-probabilities, which carries a length
bias — longer sentences score lower simply for having more tokens. Switch to per-token
`mean` scoring to remove it:

```bash
beetle-analyze blimp --scoring mean
```

Accuracy jumps from **0.60 → 0.80** as the length bias is removed (expected values in
[`baselines/blimp_toy_mean.json`](baselines/blimp_toy_mean.json)). This is the whole
loop: run → read the number → change a variable → re-run and see it move.

Example configs for both variants are bundled under
[`src/beetle_analyze/configs/`](src/beetle_analyze/configs/):

```bash
beetle-analyze blimp --config src/beetle_analyze/configs/toy_mean.json
```

## Bring your own eval output

Point the CLI (or `analyze_file`) at any JSONL file, one minimal pair per line:

```json
{"phenomenon": "subject_verb_agreement", "logprob_good": -21.3, "logprob_bad": -24.8, "n_tokens_good": 8, "n_tokens_bad": 8}
```

Required fields are `phenomenon`, `logprob_good`, `logprob_bad`; `n_tokens_good` /
`n_tokens_bad` are needed only for `mean` scoring. Produce such a file from your own
model by summing per-token log-probabilities over each sentence in a minimal-pair
benchmark.

```bash
beetle-analyze blimp --input my_evals.jsonl --json-out result.json --plot-out fig.png
```

`--plot-out` writes a per-phenomenon bar chart (needs the plotting extra:
`pip install "beetle-analyze[plots]"`).

## Compute requirements

None beyond a CPU. The analysis consumes an already-computed log-probability file, so
there is no model inference in this package. The bundled 60-pair example finishes in
a fraction of a second on a laptop. The 2000-resample bootstrap is the most expensive
step and is trivially fast.

## Project layout

```
src/beetle_analyze/
  scoring.py        # minimal-pair scoring: sum/mean, accuracy, bootstrap CI
  blimp.py          # load eval output -> per-phenomenon + overall accuracy
  plots.py          # optional bar chart (needs [plots] extra)
  cli.py            # `beetle-analyze` command
  data/             # bundled synthetic toy eval output
  configs/          # example JSON run configs
baselines/          # expected metrics for the toy analysis (diff target)
tests/              # fast CPU smoke tests
notebooks/          # quickstart.ipynb walkthrough
scripts/            # make_toy_data.py — regenerates the synthetic data
```

## Reproduce / regenerate

```bash
pytest -q                          # runs the smoke tests, asserts the 0.60 headline
python scripts/make_toy_data.py    # regenerate the synthetic toy dataset (fixed seed)
```

The toy dataset is fully synthetic and generated from a fixed random seed. It contains
no real model outputs or research results.

### Continuous integration

A ready-to-use GitHub Actions workflow (install `.[dev]`, run `pytest`, smoke-run the
CLI on Python 3.10–3.12) ships as [`.github/ci.yml.disabled`](.github/ci.yml.disabled).
It was placed there because the initial automated push lacked the `workflow` OAuth
scope. To enable it, move it into the workflows directory and push with a
`workflow`-scoped token:

```bash
git mv .github/ci.yml.disabled .github/workflows/ci.yml
git commit -m "Enable CI" && git push
```

## Citation / DOI

If you use this software, please cite it via [`CITATION.cff`](CITATION.cff).

A DOI can be minted through [Zenodo](https://zenodo.org): the maintainer enables the
`BeetleLM/beetle-analyze` repository in Zenodo's GitHub integration, then publishing a
GitHub release (e.g. `v0.1.0`) automatically archives it and issues a DOI. A
[`.zenodo.json`](.zenodo.json) with the metadata is included for that step. The DOI
badge will be added here once minted.

## License

[Apache-2.0](LICENSE) © Suchir Salhan.
