"""Offline tests for the data-quality gate and the decision log."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import portfolio_analyzer_advanced as pa  # noqa: E402
import decision_log as dl  # noqa: E402


def _ok_inputs():
    idx = pd.bdate_range("2021-01-04", periods=800)
    prices = pd.DataFrame({"SPY": np.linspace(100, 150, 800), "A": 50.0}, index=idx)
    results = {"S": {"n_years": 3.1, "weights": {"A": 1.0}, "cagr": 0.15,
                     "expected_return": 0.10}}
    wf = {"S": {"oos_annualized": 0.12}}
    return prices, results, wf


def test_clean_run_passes():
    prices, results, wf = _ok_inputs()
    issues = pa.data_quality_checks(prices, results, wf, {"A": 0.02},
                                    {"A": {"pe": 15}}, {"A": {"verdict": "FAIR VALUE"}},
                                    {"FDA": True})
    assert issues == []


def test_truncated_window_is_critical_and_blocks_outputs():
    prices, results, wf = _ok_inputs()
    results["S"]["n_years"] = 1.7          # the 20-month truncation bug
    issues = pa.data_quality_checks(prices, results, wf, {}, {}, {}, {})
    assert any(l == "CRITICAL" for l, _ in issues)
    with pytest.raises(pa.DataQualityError):
        pa.report_data_quality(issues)


def test_unit_bug_and_dead_layers_warn():
    prices, results, wf = _ok_inputs()
    wf["S"]["oos_annualized"] = None
    issues = pa.data_quality_checks(prices, results, wf, {"JNJ": 2.05}, {}, {},
                                    {"FDA / ClinicalTrials": False})
    msgs = " ".join(m for _, m in issues)
    assert "dividend" in msgs and "FDA" in msgs and "walk-forward" in msgs
    assert all(l == "WARN" for l, _ in issues)


def _plan(existing_shares, target, prices, capital=0, lots=None):
    profile = {"name": "T", "capital": capital, "existing_shares": existing_shares}
    plan = pa.build_personalized_plan(profile, target, prices, tax_lots=lots)
    return {p["Ticker"]: p for p in plan["positions"]}


def test_drift_band_avoids_needless_trim():
    # A worth $1,100 vs $1,000 target (+10%) — inside the 25% band → HOLD, no tax event
    rows = _plan({"A": 11, "B": 9}, {"A": 0.5, "B": 0.5}, {"A": 100.0, "B": 100.0})
    assert rows["A"]["Action"] == "HOLD"


def test_trim_outside_band_carries_short_term_tax_note():
    from datetime import datetime, timedelta
    bought = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")
    lots = {"A": {"shares": 30, "cost_basis": 1500.0, "first_bought": bought, "days_held": 100}}
    rows = _plan({"A": 30, "B": 10}, {"A": 0.5, "B": 0.5}, {"A": 100.0, "B": 100.0}, lots=lots)
    assert rows["A"]["Action"] == "TRIM"
    assert "SHORT-TERM" in rows["A"]["Tax Note"]


def test_sell_without_logged_basis_says_so_and_tiny_buys_skipped():
    rows = _plan({"OLD": 5}, {"A": 0.99, "B": 0.01}, {"OLD": 100.0, "A": 100.0, "B": 10.0},
                 capital=5000)
    assert rows["OLD"]["Action"] == "EVALUATE SELL"
    assert "Cost basis unknown" in rows["OLD"]["Tax Note"]
    assert rows["B"]["Action"] == "SKIP (< $100)"           # 1% of $5.5k = $55


def test_decision_log_roundtrip_and_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "LOG_PATH", tmp_path / "log.jsonl")
    dl.record_run("r1", "S", {"S": {"A": 0.5, "B": 0.5}}, {"A": 100.0, "B": 50.0}, 400.0,
                  bl_views=[{"ticker": "A", "expected_return": 0.4, "confidence": 0.4}])
    entry = dl.load_log()[0]
    assert entry["recommended"] == "S" and entry["entry_prices"]["A"] == 100.0

    run_date = pd.Timestamp(entry["date"])
    idx = pd.bdate_range(run_date + pd.Timedelta(days=1), periods=130)
    closes = pd.DataFrame({"A": 110.0, "B": 45.0, "SPY": 420.0}, index=idx)
    scored = dl.score_entry(entry, closes)["S"]
    port, spy = scored["3m"]
    assert port == pytest.approx(0.5 * 0.10 + 0.5 * -0.10)   # +10% and -10% → 0%
    assert spy == pytest.approx(0.05)
    assert "6m" in scored and "12m" not in scored            # 12m not reached yet
