"""
Offline tests for the backtest-integrity fixes: point-in-time fundamentals,
after-tax replay, group caps (industry / binary-event), 1/N benchmark, and the
plan extras (fractional shares, monthly routing, earnings note, binary exposure).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import point_in_time as pit_mod  # noqa: E402
import portfolio_analyzer_advanced as pa  # noqa: E402


def _prices(n_days=900, n_tickers=8, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, (n_days, n_tickers)),
                                             axis=0)),
                      index=idx, columns=[f"T{i}" for i in range(n_tickers)])
    px["SPY"] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n_days)))
    return px


# ── Point-in-time fundamentals ────────────────────────────────────────────────

FACTS = {"facts": {"us-gaap": {
    "EarningsPerShareDiluted": {"units": {"USD/shares": [
        # FY2021 as originally filed, then restated a year later (restatement must lose)
        {"form": "10-K", "fp": "FY", "start": "2021-01-01", "end": "2021-12-31",
         "val": 4.0, "filed": "2022-02-15"},
        {"form": "10-K", "fp": "FY", "start": "2021-01-01", "end": "2021-12-31",
         "val": 1.0, "filed": "2023-02-15"},
        {"form": "10-K", "fp": "FY", "start": "2022-01-01", "end": "2022-12-31",
         "val": 5.0, "filed": "2023-02-15"},
        {"form": "10-Q", "fp": "Q1", "start": "2022-01-01", "end": "2022-03-31",
         "val": 9.9, "filed": "2022-05-01"},           # quarterly: ignored
    ]}},
    "NetIncomeLoss": {"units": {"USD": [
        {"form": "10-K", "fp": "FY", "start": "2021-01-01", "end": "2021-12-31",
         "val": 400.0, "filed": "2022-02-15"},
        {"form": "10-K", "fp": "FY", "start": "2022-01-01", "end": "2022-12-31",
         "val": 500.0, "filed": "2023-02-15"},
    ]}},
    "StockholdersEquity": {"units": {"USD": [
        {"form": "10-K", "fp": "FY", "end": "2021-12-31", "val": 2000.0, "filed": "2022-02-15"},
        {"form": "10-K", "fp": "FY", "end": "2022-12-31", "val": 2500.0, "filed": "2023-02-15"},
    ]}},
}}}


def test_original_filing_wins_over_restatement():
    recs = pit_mod.extract_records(FACTS)
    fy21 = next(r for r in recs if r["end"] == "2021-12-31")
    assert fy21["eps"] == 4.0 and fy21["filed"] == "2022-02-15"
    assert len(recs) == 2                                      # 10-Q ignored


def _pit(splits=None):
    idx = pd.bdate_range("2021-06-01", "2024-06-28")
    px = pd.DataFrame({"X": np.full(len(idx), 40.0)}, index=idx)
    return pit_mod.PointInTimeFundamentals({"X": pit_mod.extract_records(FACTS)}, px, splits)


def test_value_is_only_known_after_filing_date():
    p = _pit()
    assert p.at("X", pd.Timestamp("2022-02-14")) is None       # FY2021 not filed yet
    v = p.at("X", pd.Timestamp("2022-03-01"))
    assert v["pe"] == pytest.approx(40 / 4.0) and v["roe"] == pytest.approx(0.20)
    assert p.at("X", pd.Timestamp("2023-03-01"))["pe"] == pytest.approx(40 / 5.0)


def test_stale_filing_is_dropped():
    # last 10-K filed 2023-02-15 → usable until +550 days (2024-08-18), not after
    assert _pit().at("X", pd.Timestamp("2024-08-01")) is not None
    assert _pit().at("X", pd.Timestamp("2024-09-30")) is None


def test_eps_put_on_split_adjusted_basis():
    splits = pd.DataFrame({"X": [4.0]}, index=[pd.Timestamp("2022-06-01")])
    v = _pit(splits).at("X", pd.Timestamp("2022-03-01"))
    # filed EPS 4.0 pre-split → 1.0 on today's share basis; adjusted price 40 → P/E 40
    assert v["pe"] == pytest.approx(40.0)


def test_walk_forward_uses_point_in_time_callable():
    px = _prices()
    seen = []

    def fundamentals_at(d):
        seen.append(d)
        return {}
    pa.walk_forward_rolling(list(px.columns), px, "max_sharpe", 0.3, 0.0, top_n=5,
                            min_train=504, rebal_days=63, cost_bps=0,
                            fundamentals_at=fundamentals_at)
    assert seen and all(d < px.index[504 + 63 * i] for i, d in enumerate(seen))


# ── After-tax replay ──────────────────────────────────────────────────────────

def test_after_tax_replay_zero_tax_matches_walk_forward():
    px = _prices(seed=4)
    wf = pa.walk_forward_rolling(list(px.columns), px, "max_sharpe", 0.3, 0.0, top_n=5,
                                 min_train=504, rebal_days=63, cost_bps=10)
    at = pa.after_tax_replay(wf["schedule"], px, st_rate=0.0, lt_rate=0.0, cost_bps=10)
    assert at["pre_tax_annualized"] == pytest.approx(wf["oos_annualized"], abs=0.01)
    assert at["after_tax_annualized"] == pytest.approx(at["pre_tax_annualized"])


def test_taxes_reduce_return_and_short_term_costs_more():
    idx = pd.bdate_range("2020-01-01", periods=700)
    up = pd.Series(np.linspace(100, 300, 700), index=idx)
    px = pd.DataFrame({"A": up, "B": up, "SPY": up})
    # Swap A → B every quarter: every gain is realised short-term
    sched = [(s, min(s + 63, 700), {"A": 1.0} if k % 2 == 0 else {"B": 1.0})
             for k, s in enumerate(range(100, 700, 63))]
    st = pa.after_tax_replay(sched, px, st_rate=0.37, lt_rate=0.15, cost_bps=0)
    lt_only = pa.after_tax_replay(sched, px, st_rate=0.15, lt_rate=0.15, cost_bps=0)
    assert st["after_tax_annualized"] < st["pre_tax_annualized"]
    assert st["short_term_share"] == pytest.approx(1.0)
    assert st["tax_drag"] > lt_only["tax_drag"]


def test_buy_and_hold_defers_tax():
    idx = pd.bdate_range("2020-01-01", periods=700)
    px = pd.DataFrame({"A": np.linspace(100, 300, 700), "SPY": np.linspace(100, 200, 700)},
                      index=idx)
    sched = [(s, min(s + 63, 700), {"A": 1.0}) for s in range(100, 700, 63)]
    at = pa.after_tax_replay(sched, px, st_rate=0.37, lt_rate=0.2, cost_bps=0)
    assert at["tax_drag"] == pytest.approx(0.0, abs=1e-9)       # never sold → no tax


# ── Group caps, binary-event set, 1/N ─────────────────────────────────────────

def test_industry_group_cap_binds():
    px = _prices(n_tickers=6, seed=9)
    for t in ("T0", "T1", "T2"):                              # three strong "refiners"
        px[t] = 100 * np.exp(np.linspace(0, 1.2, len(px)))
    groups = [({"T0", "T1", "T2"}, 0.25)]
    w = pa.optimize_weights(list(px.columns), px, "max_return", 0.20, 0.0, group_caps=groups)
    assert sum(w.get(t, 0) for t in ("T0", "T1", "T2")) <= 0.25 + 1e-4


def test_binary_set_excludes_megacaps():
    infos = {"RVMD": {"industry": "Biotechnology", "marketCap": 40e9},
             "VRTX": {"industry": "Biotechnology", "marketCap": 120e9},
             "RHHBY": {"industry": "Drug Manufacturers", "marketCap": 250e9},
             "SMALL": {"industry": "Medical Devices", "marketCap": 5e9}}
    s = pa.binary_event_tickers(infos, fda_tickers={"RHHBY", "SMALL"})
    assert s == {"RVMD", "SMALL"}


def test_build_group_caps_only_multi_name_industries():
    infos = {"A": {"industry": "Refining"}, "B": {"industry": "Refining"},
             "C": {"industry": "Software"}}
    groups = pa.build_group_caps(infos)
    assert groups == [({"A", "B"}, pa.INDUSTRY_MAX_WEIGHT)]


def test_equal_weight_objective():
    px = _prices(n_tickers=4)
    w = pa.optimize_weights(["T0", "T1", "T2", "T3"], px, "equal_weight", 0.2, 0.0)
    assert all(v == pytest.approx(0.25) for v in w.values())


def test_equal_weight_capped_respects_industry_and_binary_caps():
    ranked = ["R1", "R2", "R3", "R4", "R5", "B1", "B2", "B3", "X1", "X2", "X3", "X4",
              "X5", "X6", "X7", "X8", "X9", "X10"]
    groups = [({"R1", "R2", "R3", "R4", "R5"}, 0.25),          # refiners
              ({"B1", "B2", "B3"}, 0.15)]                      # binary-event names
    w = pa.equal_weight_capped(ranked, 15, groups, {"B1": 0.06, "B2": 0.06, "B3": 0.06})
    assert len(w) == 15 and sum(w.values()) == pytest.approx(1.0)
    assert sum(w.get(t, 0) for t in groups[0][0]) <= 0.25 + 1e-9
    assert sum(w.get(t, 0) for t in groups[1][0]) <= 0.15 + 1e-9
    assert all(w.get(t, 0) <= 0.06 + 1e-9 for t in ("B1", "B2", "B3"))
    # 4th refiner (26.7% > 25%) and 3rd binary name (18% > 15%) skipped; next-ranked fill in
    assert {"R4", "R5", "B3"}.isdisjoint(w) and "X10" in w


def test_equal_weight_capped_keeps_locked_positions():
    w = pa.equal_weight_capped(["A", "B", "C", "D"], 3, locked={"Z": 0.40})
    assert w["Z"] == pytest.approx(0.40)
    assert w["A"] == pytest.approx(0.30) and w["B"] == pytest.approx(0.30) and "C" not in w


def test_turnover_penalty_keeps_previous_book():
    px = _prices(n_tickers=6, seed=12)
    free = pa.optimize_weights(list(px.columns), px, "max_sharpe", 0.4, 0.0)
    prev = {t: 1 / 6 for t in px.columns if t != "SPY"}
    prev["SPY"] = 1 / 6
    sticky = pa.optimize_weights(list(px.columns), px, "max_sharpe", 0.4, 0.0,
                                 prev_weights=prev, turnover_penalty=5.0)
    to = lambda w: sum(abs(w.get(t, 0) - prev.get(t, 0)) for t in set(w) | set(prev))
    assert to(sticky) < to(free)


# ── Plan extras ───────────────────────────────────────────────────────────────

PRICES = {"NEW1": 33.0, "NEW2": 70.0, "BIO": 50.0}
TARGET = {"NEW1": 0.4, "NEW2": 0.4, "BIO": 0.2}


def _plan(**kw):
    profile = {"name": "T", "existing_shares": {}, "capital": 10_000,
               "monthly_contribution": 500, **kw.pop("profile", {})}
    return pa.build_personalized_plan(profile, TARGET, PRICES, **kw)


def test_fractional_shares_deploy_almost_everything():
    whole = _plan()
    frac = _plan(profile={"fractional_shares": True})
    assert frac["cash_leftover_now"] < whole["cash_leftover_now"]
    assert frac["cash_leftover_now"] < 5
    assert any(isinstance(d["Shares"], float) and d["Shares"] % 1 for d in frac["deployment_now"])


def test_monthly_contribution_goes_to_most_underweight():
    plan = _plan(profile={"capital": 2_000})          # leaves big gaps to fill monthly
    first = plan["contribution_schedule"][0]["Suggestion"]
    assert "$" in first and "most underweight" in first


def test_earnings_note_on_imminent_report():
    plan = _plan(earnings_days={"NEW1": 5, "NEW2": 40})
    notes = {d["Ticker"]: d.get("Note", "") for d in plan["deployment_now"]}
    assert "earnings in 5d" in notes["NEW1"] and notes.get("NEW2", "") == ""


def test_binary_exposure_reported():
    plan = _plan(binary_set={"BIO"})
    be = plan["binary_exposure"]
    assert be["names"] == ["BIO"] and 0.15 < be["pct"] < 0.25
