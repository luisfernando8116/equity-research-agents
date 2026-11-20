"""Offline tests for the 1-year minimum holding period (backtest) and the
no-short-term-sales rule (live plan)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import portfolio_analyzer_advanced as pa  # noqa: E402


def _rotating(n=800, crash=False):
    """A leads until day 504 (first rebalance buys it), then B quadruples in the next
    quarter while A goes flat — or, with crash=True, falls 30%."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=n)
    a = np.concatenate([np.linspace(100, 200, 504),
                        np.linspace(200, 140 if crash else 200, 63),
                        np.full(n - 567, 140.0 if crash else 200.0)])
    b = np.concatenate([np.full(504, 100.0), np.linspace(100, 400, 63),
                        np.linspace(400, 450, n - 567)])
    px = pd.DataFrame({"A": a, "B": b}, index=idx)
    for i in range(4):
        px[f"N{i}"] = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    px["SPY"] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    return px


def _first_two_books(px, **kw):
    wf = pa.walk_forward_rolling(list(px.columns), px, "max_return", 1.0, 0.0, top_n=3,
                                 min_train=504, rebal_days=63, cost_bps=0, **kw)
    return wf, [w for _, _, w in wf["schedule"]]


def test_without_min_hold_the_book_rotates():
    _, books = _first_two_books(_rotating())
    assert books[0].get("A", 0) > 0.5                     # A bought at day 504
    assert books[1].get("A", 0) < books[0]["A"]           # sold within a quarter


def test_min_hold_keeps_position_for_a_year():
    wf, books = _first_two_books(_rotating(), min_hold_days=252)
    # A drifted down vs. B's run, but it can't be cut below its drifted weight
    assert books[1].get("A", 0) > 0.3
    assert wf["early_exits"] == 0


def test_large_drop_allows_early_exit():
    wf, books = _first_two_books(_rotating(crash=True), min_hold_days=252,
                                 early_exit_drawdown=0.25)
    assert wf["early_exits"] >= 1


def test_floor_weights_respected_by_optimizer():
    px = _rotating()
    w = pa.optimize_weights(["A", "B", "N0", "N1"], px.iloc[:700], "max_return", 1.0, 0.0,
                            floor_weights={"N0": 0.30})
    assert w.get("N0", 0) >= 0.30 - 1e-4


# ── live plan: no short-term sales ────────────────────────────────────────────

LOTS = {"BIG": {"shares": 80, "lt_shares": 10, "st_shares": 70, "lots": [
    {"date": "2025-01-02", "shares": 10, "price": 50.0, "cost": 500.0, "term": "LT",
     "lt_date": "2026-01-03"},
    {"date": "2026-09-01", "shares": 70, "price": 90.0, "cost": 6300.0, "term": "ST",
     "lt_date": "2027-09-02"}]}}


def _plan(no_st, sells=None):
    profile = {"name": "T", "existing_shares": {"BIG": 80}, "capital": 4_000,
               "keep_existing": True, "no_short_term_sales": no_st}
    return pa.build_personalized_plan(profile, {"NEW": 1.0}, {"BIG": 100.0, "NEW": 50.0},
                                      tax_lots=LOTS, sell_signals=sells, position_cap=0.20)


def _row(plan, t):
    return next(p for p in plan["positions"] if p["Ticker"] == t)


def test_trim_limited_to_long_term_shares():
    big = _row(_plan(no_st=True), "BIG")
    assert big["Action"] == "TRIM" and "long-term shares" in big["Exit Reason"]
    assert big["Target Now ($)"] == pytest.approx(8_000 - 10 * 100.0)   # only 10 LT shares


def test_without_rule_trim_goes_to_cap():
    big = _row(_plan(no_st=False), "BIG")
    assert big["Target Now ($)"] == pytest.approx(12_000 * 0.20)


def test_exit_signal_still_sells_short_term_shares():
    big = _row(_plan(no_st=True, sells={"BIG": "FDA Rejection: CRL"}), "BIG")
    assert big["Action"] == "EVALUATE SELL"                # "something happens"
