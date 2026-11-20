"""Offline tests for robustness_grid.py statistics and grid helpers."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import robustness_grid as rg  # noqa: E402


def test_expected_max_sharpe_grows_with_trials():
    a = rg.expected_max_sharpe(0.01, 10)
    b = rg.expected_max_sharpe(0.01, 500)
    assert 0 < a < b
    assert rg.expected_max_sharpe(0.01, 1) == 0.0


def test_deflated_sharpe_separates_skill_from_noise():
    rng = np.random.default_rng(0)
    skill = rng.normal(0.002, 0.01, 750)          # daily IR ≈ 0.2 → ~3.2 annualised
    noise = rng.normal(0.0, 0.01, 750)
    sr0 = rg.expected_max_sharpe(0.0016, 50)      # 50 trials of ~3y daily data
    assert rg.deflated_sharpe(skill, sr0) > 0.95
    assert rg.deflated_sharpe(noise, sr0) < 0.5


def test_more_trials_lower_the_deflated_sharpe():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0008, 0.01, 750)
    few  = rg.deflated_sharpe(x, rg.expected_max_sharpe(0.0016, 5))
    many = rg.deflated_sharpe(x, rg.expected_max_sharpe(0.0016, 500))
    assert many < few


def test_factor_blend_sums_to_one_and_keeps_fixed_factors():
    fb = rg.factor_blend({"momentum": 0.45, "quality": 0.15, "value": 0.10})
    assert sum(fb.values()) == pytest.approx(1.0)
    assert fb["revisions"] == rg.pa.FACTOR_BLEND["revisions"]
    assert fb["momentum"] > fb["quality"] > fb["value"]


def test_factor_blend_works_while_global_blend_is_cleared():
    # The grid clears pa.FACTOR_BLEND before updating it — the blend must come from a copy
    saved = dict(rg.pa.FACTOR_BLEND)
    try:
        rg.pa.FACTOR_BLEND.clear()
        fb = rg.factor_blend({"momentum": 0.35, "quality": 0.25, "value": 0.20}, saved)
        assert sum(fb.values()) == pytest.approx(1.0)
    finally:
        rg.pa.FACTOR_BLEND.clear()
        rg.pa.FACTOR_BLEND.update(saved)


def test_sample_configs_puts_current_first_without_duplicates():
    base = {"momentum": 0.35, "quality": 0.25, "value": 0.20, "max_weight": 0.20,
            "top_n": 100, "objective": "max_return", "industry_cap": 0.25,
            "binary_cap": 0.15, "early_exit": 0.25}
    cs = rg.sample_configs(base, 30, seed=3)
    ids = [rg.config_id(c) for c in cs]
    assert ids[0] == rg.config_id(base) and len(set(ids)) == len(ids) == 30
    assert rg.sample_configs(base, 30, seed=3) == cs           # reproducible


def test_summary_reports_spread_and_dsr():
    rng = np.random.default_rng(2)
    rows, excess = [], {}
    for i in range(12):
        cid = f"c{i}"
        x = rng.normal(0.0002 * (i - 5), 0.01, 500)
        excess[cid] = x
        rows.append({"id": cid, "alpha": float(x.mean() * 252)})
    text = rg.summarise(rows, excess, "c0")
    assert "Beat SPY in" in text and "Deflated Sharpe" in text and "Current settings" in text
    assert "VERDICT" in text


def test_sampling_is_balanced_across_objectives():
    base = {"momentum": 0.35, "quality": 0.25, "value": 0.20, "max_weight": 0.20,
            "top_n": 100, "objective": "max_return", "industry_cap": 0.25,
            "binary_cap": 0.15, "early_exit": 0.25}
    cs = rg.sample_configs(base, 61, seed=1)
    counts = {o: sum(c["objective"] == o for c in cs) for o in rg.GRID["objective"]}
    assert counts["equal_weight"] == 20 and counts["max_sharpe"] == 20
    assert counts["max_return"] == 21                    # + the current settings


def test_verdict_reported_per_objective():
    rows, excess = [], {}
    rng = np.random.default_rng(9)
    for i in range(10):
        obj = "equal_weight" if i < 5 else "max_return"
        a = 0.03 if obj == "equal_weight" else -0.05
        excess[f"c{i}"] = rng.normal(a / 252, 0.01, 300)
        rows.append({"id": f"c{i}", "alpha": a, "objective": obj})
    text = rg.summarise(rows, excess, "c0")
    assert "equal_weight  ROBUST" in text and "max_return    FRAGILE" in text


def _rows(alphas, knob_vals=None):
    rows, excess = [], {}
    rng = np.random.default_rng(5)
    for i, a in enumerate(alphas):
        cid = f"c{i}"
        excess[cid] = rng.normal(a / 252, 0.01, 300)
        rows.append({"id": cid, "alpha": a,
                     **({"early_exit": knob_vals[i]} if knob_vals else {})})
    return rows, excess


@pytest.mark.parametrize("alphas,verdict", [
    ([0.05] * 8 + [-0.01] * 2, "ROBUST"),        # 80% beat SPY
    ([0.05] * 6 + [-0.01] * 4, "MIXED"),         # 60%
    ([0.05] * 3 + [-0.01] * 7, "FRAGILE"),       # 30%
])
def test_verdict_rule(alphas, verdict):
    rows, excess = _rows(alphas)
    assert f"VERDICT (rule fixed in advance): {verdict}" in rg.summarise(rows, excess, "c0")


def test_sensitive_setting_is_flagged():
    alphas = [0.10, 0.10, 0.10, -0.02, -0.02, -0.02]
    rows, excess = _rows(alphas, knob_vals=[0.15, 0.15, 0.15, 1.0, 1.0, 1.0])
    text = rg.summarise(rows, excess, "c0")
    assert "early_exit" in text and "SENSITIVE" in text
