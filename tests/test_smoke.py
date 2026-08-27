"""Fast CPU smoke tests for the headline toy analysis."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from beetle_analyze import analyze_file, evaluate, load_pairs, toy_data_path
from beetle_analyze.cli import main

BASELINE_DIR = Path(__file__).resolve().parent.parent / "baselines"


def test_toy_data_present():
    p = toy_data_path()
    assert p.exists()
    df = load_pairs(p)
    assert len(df) == 60
    assert set(df["phenomenon"].unique()) == {
        "subject_verb_agreement",
        "determiner_noun_agreement",
        "anaphor_agreement",
        "npi_licensing",
        "ellipsis",
    }


def test_headline_sum_matches_baseline():
    result = analyze_file(toy_data_path(), scoring="sum")
    # Headline number: raw-sum BLiMP accuracy on the toy eval output.
    assert result.overall_accuracy == pytest.approx(0.60, abs=1e-6)
    baseline = json.loads((BASELINE_DIR / "blimp_toy_sum.json").read_text())
    assert result.to_dict()["overall_accuracy"] == pytest.approx(
        baseline["overall_accuracy"], abs=1e-6
    )


def test_mean_scoring_changes_result():
    # Changing the one variable (sum -> mean) removes the length bias and
    # raises accuracy. This is the "change a variable and re-run" story.
    sum_acc = analyze_file(toy_data_path(), scoring="sum").overall_accuracy
    mean_acc = analyze_file(toy_data_path(), scoring="mean").overall_accuracy
    assert mean_acc > sum_acc
    assert mean_acc == pytest.approx(0.80, abs=1e-6)


def test_accuracy_in_valid_range():
    result = analyze_file(toy_data_path())
    assert 0.0 <= result.overall_accuracy <= 1.0
    assert result.ci_low <= result.overall_accuracy <= result.ci_high


def test_cli_runs(capsys):
    rc = main(["blimp"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "overall accuracy: 0.6000" in out


def test_cli_json_out(tmp_path):
    out = tmp_path / "metrics.json"
    rc = main(["blimp", "--json-out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["n_pairs"] == 60
    assert data["overall_accuracy"] == pytest.approx(0.60, abs=1e-6)


def test_cli_plot_out(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "fig.png"
    rc = main(["blimp", "--plot-out", str(out)])
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0
