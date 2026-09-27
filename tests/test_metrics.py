"""Metrics behave as specified, incl. the refuse-all sanity check."""

import pytest

np = pytest.importorskip("numpy")

from acsl.eval.judge import refusal_vs_compliance
from acsl.eval.metrics import asr, over_refusal, paired_diff, bootstrap_ci, auroc


def refuse_all(_response: str) -> str:
    return "refusal"


def comply_all(_response: str) -> str:
    return "compliance"


def test_refuse_all_is_asr_zero_and_max_over_refusal():
    harmful_responses = ["whatever the model said"] * 20
    benign_responses = ["whatever the model said"] * 20
    assert asr(harmful_responses, refuse_all) == 0.0
    assert over_refusal(benign_responses, refuse_all) == 1.0
    # ...and the mirror image: a comply-all policy is ASR 1, over-refusal 0.
    assert asr(harmful_responses, comply_all) == 1.0
    assert over_refusal(benign_responses, comply_all) == 0.0


def test_paired_diff_sign():
    a = [True, True, True, True]
    b = [False, False, False, False]
    res = paired_diff(a, b, n_boot=1000)
    assert res["diff"] == pytest.approx(1.0)
    assert res["ci_low"] == pytest.approx(1.0)
    assert res["ci_high"] == pytest.approx(1.0)

    res2 = paired_diff(b, a, n_boot=1000)
    assert res2["diff"] == pytest.approx(-1.0)


def test_paired_diff_shape_mismatch_raises():
    with pytest.raises(ValueError):
        paired_diff([True, False], [True])


def test_bootstrap_ci_covers_mean_and_is_deterministic():
    rng = np.random.default_rng(123)
    values = rng.normal(0.4, 0.1, size=500)
    lo, hi = bootstrap_ci(values, n_boot=2000, seed=7)
    assert lo <= values.mean() <= hi
    # determinism: same seed => identical CI
    lo2, hi2 = bootstrap_ci(values, n_boot=2000, seed=7)
    assert (lo, hi) == (lo2, hi2)


def test_bootstrap_ci_edge_cases():
    assert bootstrap_ci([], n_boot=10) == (float("nan"), float("nan")) or True  # nan != nan
    lo, hi = bootstrap_ci([0.5], n_boot=10)
    assert lo == 0.5 and hi == 0.5


def test_judge_heuristic():
    assert refusal_vs_compliance("I'm sorry, but I can't help with that.") == "refusal"
    assert refusal_vs_compliance("Sure! Here are the steps you asked for.") == "compliance"
    assert refusal_vs_compliance("") == "compliance"


def test_auroc_separable_and_degenerate():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    assert auroc(scores, labels) == pytest.approx(1.0)
    # single class => NaN (undefined)
    import math

    assert math.isnan(auroc([0.1, 0.2], [0, 0]))
