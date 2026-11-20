"""Offline tests for the keep-what-I-have plan mode (profile["keep_existing"])."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import portfolio_analyzer_advanced as pa  # noqa: E402

PRICES = {"OLD": 100.0, "BIG": 100.0, "BAD": 100.0, "NEW1": 50.0, "NEW2": 25.0}
TARGET = {"NEW1": 0.5, "NEW2": 0.5}


def _plan(shares, capital=10_000, keep=True, sells=None, cap=0.20):
    profile = {"name": "Test", "existing_shares": shares, "capital": capital,
               "keep_existing": keep}
    return pa.build_personalized_plan(profile, TARGET, PRICES, sell_signals=sells,
                                      position_cap=cap)


def _row(plan, t):
    return next(p for p in plan["positions"] if p["Ticker"] == t)


def test_unpicked_holding_is_kept_not_sold():
    plan = _plan({"OLD": 10})
    assert _row(plan, "OLD")["Action"] == "KEEP"
    assert _row(plan, "OLD")["Tax Note"] == ""          # no sale → no tax event


def test_new_money_targets_exclude_kept_value():
    plan = _plan({"OLD": 10})                            # $1k kept, $10k new
    # Targets apply to the $10k left after the kept $1k, not to $11k
    assert _row(plan, "NEW1")["Target Now ($)"] == pytest.approx(5_000)
    deployed = sum(v for d in plan["deployment_now"] for k, v in d.items()
                   if k.startswith("Cost"))
    assert 9_000 < deployed <= 10_000                    # the new cash, all of it, nothing more


def test_exit_signal_still_suggests_sale():
    plan = _plan({"OLD": 10, "BAD": 10}, sells={"BAD": "FDA Rejection: CRL issued"})
    bad = _row(plan, "BAD")
    assert bad["Action"] == "EVALUATE SELL" and "CRL" in bad["Exit Reason"]
    assert _row(plan, "OLD")["Action"] == "KEEP"


def test_kept_holding_far_over_cap_is_trimmed_to_cap():
    # BIG = $8k of $12k total = 67% ≫ 20% cap × 1.25
    plan = _plan({"BIG": 80}, capital=4_000)
    big = _row(plan, "BIG")
    assert big["Action"] == "TRIM" and "cap" in big["Exit Reason"]
    assert big["Target Now ($)"] == pytest.approx(12_000 * 0.20)


def test_kept_holding_slightly_over_cap_is_left_alone():
    # $2.1k of $10k... = 21% vs 20% cap — inside the band, no taxable trim
    plan = _plan({"BIG": 21}, capital=7_900)
    assert _row(plan, "BIG")["Action"] == "KEEP"


def test_without_keep_mode_old_behaviour_is_unchanged():
    plan = _plan({"OLD": 10}, keep=False)
    assert _row(plan, "OLD")["Action"] == "EVALUATE SELL"
    assert _row(plan, "NEW1")["Target Now ($)"] == pytest.approx(11_000 * 0.5)


def test_sell_intent_overrides_too_new():
    profile = {"name": "Test", "existing_shares": {"OLD": 3}, "capital": 1_000,
               "keep_existing": True}
    plan = pa.build_personalized_plan(profile, TARGET, PRICES, too_new={"OLD"},
                                      sell_signals={"OLD": "your stated intent"})
    assert _row(plan, "OLD")["Action"] == "EVALUATE SELL"
    plan2 = pa.build_personalized_plan(profile, TARGET, PRICES, too_new={"OLD"})
    assert _row(plan2, "OLD")["Action"] == "HOLD — too new to model"


def test_exit_signals_thresholds():
    out = pa.exit_signals(
        ["A", "B", "C", "D"],
        catalyst_signals={"A": {"score": -1.0, "catalyst_type": "FDA Rejection", "headline": "CRL"},
                          "D": {"score": -0.5, "catalyst_type": "Analyst Downgrade"}},
        revision_signals={"B": {"score": -0.6}},
        structural={"C": {"severity": 0.7, "risk": "oversupply"}})
    assert set(out) == {"A", "B", "C"}                   # a single downgrade isn't enough
