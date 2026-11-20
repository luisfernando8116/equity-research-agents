"""
Offline regression tests for portfolio_analyzer_advanced.py core logic.
Synthetic data only — no network. Run:  python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import portfolio_analyzer_advanced as pa  # noqa: E402


def _prices(n_days=756, n_tickers=6, seed=0):
    rng   = np.random.default_rng(seed)
    idx   = pd.bdate_range("2022-01-03", periods=n_days)
    rets  = rng.normal(0.0004, 0.015, (n_days, n_tickers))
    px    = 100 * np.exp(np.cumsum(rets, axis=0))
    cols  = [f"T{i}" for i in range(n_tickers)]
    df    = pd.DataFrame(px, index=idx, columns=cols)
    spy   = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n_days)))
    df["SPY"] = spy
    return df


# ── dividend yield units ──────────────────────────────────────────────────────

def test_dividend_yield_prefers_rate_over_price():
    info = {"dividendRate": 5.36, "currentPrice": 261.44, "dividendYield": 2.05}
    assert pa._dividend_yield_fraction(info) == pytest.approx(0.0205, abs=1e-4)


def test_dividend_yield_percent_field_is_divided():
    assert pa._dividend_yield_fraction({"dividendYield": 2.05}) == pytest.approx(0.0205)


def test_dividend_yield_missing_is_zero():
    assert pa._dividend_yield_fraction({}) == 0.0


# ── short-history ticker must not truncate the window ─────────────────────────

def test_aligned_window_drops_short_history_ticker():
    px = _prices()
    px["IPO"] = np.nan
    px.iloc[-90:, px.columns.get_loc("IPO")] = 50.0   # only last 90 days
    keep, sub = pa._aligned_window(list(px.columns), px)
    assert "IPO" not in keep
    assert len(sub) == len(px)                         # full history retained


def test_optimizer_survives_ticker_with_no_data_in_window():
    px = _prices()
    px["IPO"] = np.nan                                  # no data at all
    w = pa.optimize_weights(list(px.columns), px, "max_sharpe", 0.40, 0.0)
    assert w, "optimizer must still return holdings"
    assert "IPO" not in w
    assert sum(w.values()) == pytest.approx(1.0)


def test_walk_forward_never_reports_fake_zero():
    px = _prices()
    px["IPO"] = np.nan
    px.iloc[-90:, px.columns.get_loc("IPO")] = 50.0
    wf = pa.walk_forward_backtest(list(px.columns), px, "max_sharpe", 0.40, 0.0)
    assert wf["oos_annualized"] is not None
    assert wf["oos_annualized"] != 0.0
    assert wf["oos_sharpe"] is not None


def test_walk_forward_screen_has_no_test_period_lookahead():
    px = _prices(n_tickers=30, seed=3)
    split = int(len(px) * 0.67)
    leak = np.full(len(px), 100.0)
    leak[split:] = np.linspace(100, 400, len(px) - split)   # flat in train, +300% in test
    px["LEAK"] = leak
    today_top = pa.multi_factor_screen(px, {}, top_n=5)
    assert "LEAK" in today_top                   # full-history screen would pick it
    wf = pa.walk_forward_backtest(today_top, px, "max_sharpe", 0.40, 0.0,
                                  screen_universe=list(px.columns), fundamentals={}, top_n=5)
    assert "LEAK" not in wf["holdings"]          # train-only screen must not


def test_rolling_walk_forward_folds_and_costs():
    px = _prices(n_days=900, n_tickers=12, seed=5)
    universe = [c for c in px.columns]
    wf  = pa.walk_forward_rolling(universe, px, "max_sharpe", 0.30, 0.0, top_n=8,
                                  min_train=504, rebal_days=63, cost_bps=0)
    wf2 = pa.walk_forward_rolling(universe, px, "max_sharpe", 0.30, 0.0, top_n=8,
                                  min_train=504, rebal_days=63, cost_bps=200)
    assert wf["n_folds"] == int(np.ceil((900 - 504) / 63))
    assert 0.0 <= wf["hit_rate"] <= 1.0
    assert wf2["oos_annualized"] < wf["oos_annualized"]      # costs must bite
    assert wf["oos_max_dd"] <= 0


def test_rolling_walk_forward_never_holds_future_winner_early():
    px = _prices(n_days=900, n_tickers=12, seed=6)
    late = np.full(900, 100.0)
    late[885:] = np.linspace(100, 500, 15)                  # takes off at day 885
    px["LATE"] = late
    wf = pa.walk_forward_rolling(list(px.columns), px, "max_sharpe", 0.30, 0.0, top_n=5,
                                 min_train=504, rebal_days=63, cost_bps=0)
    # Rebalances at 504, 567 … 882 — the last one only sees data through day 881,
    # before the take-off, so LATE can't be in the final book.
    assert "LATE" not in wf["holdings"]


def test_optimizer_respects_ticker_caps():
    px = _prices(n_tickers=6, seed=9)
    px["T0"] = 100 * np.exp(np.linspace(0, 1.5, len(px)))   # dominant winner
    w = pa.optimize_weights(list(px.columns), px, "max_return", 0.25, 0.0,
                            ticker_caps={"T0": 0.06})
    assert w.get("T0", 0) <= 0.06 + 1e-6


def test_trim_keeps_caps_after_renormalising():
    w = {f"S{i}": 0.05 for i in range(20)}                   # 20 × 5%
    w["BIO"] = 0.06
    out = pa.trim_to_max_holdings(w, 15, max_weight=0.15, caps={"BIO": 0.06})
    assert len(out) == 15
    assert sum(out.values()) == pytest.approx(1.0)
    assert out["BIO"] <= 0.06 + 1e-9                        # plain rescale gave 6.7%+
    assert max(out.values()) <= 0.15 + 1e-9


def test_walk_forward_point_in_time_membership_gate():
    px = _prices(n_days=900, n_tickers=8, seed=11)
    star = 100 * np.exp(np.linspace(0, 2.0, 900))           # best performer throughout
    px["STAR"] = star
    joined = px.index[850]                                   # joined the index late
    members = lambda d: {c for c in px.columns if c != "STAR"} | ({"STAR"} if d >= joined else set())
    wf = pa.walk_forward_rolling(list(px.columns), px, "max_return", 0.30, 0.0, top_n=4,
                                 min_train=504, rebal_days=63, cost_bps=0,
                                 membership=members, sp_ever=set(px.columns))
    no_gate = pa.walk_forward_rolling(list(px.columns), px, "max_return", 0.30, 0.0, top_n=4,
                                      min_train=504, rebal_days=63, cost_bps=0)
    # Gated: STAR only eligible at the 882 rebalance → backtest can't ride its whole run
    assert wf["oos_annualized"] < no_gate["oos_annualized"]


# ── projections must not compound in-sample CAGR ──────────────────────────────

def test_forward_expected_return_is_capm_bounded():
    px  = _prices()
    w   = {c: 1 / 6 for c in px.columns if c != "SPY"}
    res = pa.compute_metrics(px, w, "test")
    er, beta = pa.forward_expected_return(res["port_ret"], px)
    assert 0.3 <= beta <= 2.5
    assert er == pytest.approx(pa.RISK_FREE_RATE + beta * pa.SV_EQUITY_RISK_PREMIUM)


def test_monte_carlo_median_is_sane():
    mc = pa.monte_carlo(0.11, 0.20)
    p50 = float(np.median(mc[:, -1]))
    # $15k at ~9% log drift for 15 years ≈ $57k; must be nowhere near $10M+
    assert 30_000 < p50 < 120_000


# ── DCF / valuation ───────────────────────────────────────────────────────────

_VLO_LIKE = {   # refining-cycle rebound: earningsGrowth +453% (real Oct-2026 values)
    "industry": "Oil & Gas Refining & Marketing", "sector": "Energy", "beta": 0.592,
    "revenueGrowth": 0.517, "earningsGrowth": 4.535, "forwardEps": 43.14, "trailingEps": 23.99,
    "freeCashflow": 8.416e9, "totalDebt": 1.135e10, "totalCash": 7.874e9,
    "sharesOutstanding": 2.879e8, "currentPrice": 433.75,
}


def test_cyclical_rebound_does_not_extrapolate():
    assert pa._sv_growth_rate(_VLO_LIKE) == pytest.approx(pa.SV_CYCLICAL_GROWTH)
    dcf = pa._sv_dcf(_VLO_LIKE, pa._sv_growth_rate(_VLO_LIKE))
    assert dcf["upside_pct"] < 100          # was +869% before the fix


def test_gaap_vs_adjusted_eps_gap_is_not_growth():
    info = {"revenueGrowth": 0.058, "earningsGrowth": 0.688,
            "forwardEps": 14.06, "trailingEps": 7.23, "industry": "Medical Distribution"}
    assert pa._sv_growth_rate(info) == pytest.approx(0.058)   # capped by revenue


def test_wacc_floor_and_blume():
    assert pa._sv_wacc({"beta": 0.2}) >= pa.SV_WACC_FLOOR     # JNJ-like beta can't imply ~6%
    assert pa._sv_wacc({"beta": 2.2}) < pa.RISK_FREE_RATE + 2.2 * pa.SV_EQUITY_RISK_PREMIUM


def test_financials_skip_dcf():
    v = pa.valuation_signal("JPM", {**_VLO_LIKE, "sector": "Financial Services",
                                     "industry": "Banks - Diversified"})
    assert v["dcf_upside"] is None and v["buy_low"] is None


def test_valuation_tilt_direction_with_negative_mu():
    out = pa.apply_valuation_tilt({"A": -0.10, "B": -0.10},
                                  {"A": {"verdict": "UNDERVALUED"},
                                   "B": {"verdict": "OVERVALUED"}})
    assert out["A"] > -0.10 > out["B"]


def test_relative_verdicts_split_into_thirds_even_when_all_dcf_negative():
    sigs = {f"S{i}": {"verdict": "OVERVALUED", "dcf_upside": -90 + 10 * i, "multiples": 0.0}
            for i in range(9)}
    sigs["ETF"] = {"verdict": "—", "dcf_upside": None}
    out = pa.rerank_valuation_verdicts(sigs)
    verdicts = [out[f"S{i}"]["verdict"] for i in range(9)]
    assert verdicts.count("UNDERVALUED") == 3 and verdicts.count("OVERVALUED") == 3
    assert out["S8"]["verdict"] == "UNDERVALUED"     # highest DCF upside = cheapest
    assert out["ETF"]["verdict"] == "—"


def test_yfinance_sector_names_resolve():
    for s in ("Consumer Cyclical", "Consumer Defensive", "Financial Services"):
        assert s in pa.SV_SECTOR_BENCHMARKS


# ── recommender treats missing OOS as no evidence ─────────────────────────────

def test_recommender_penalises_missing_walk_forward():
    results = {k: {"sharpe": 5.0, "max_dd": -0.1, "cagr": 1.0} for k in pa.PORTFOLIOS}
    good = {"alpha": 0.05, "oos_sharpe": 1.2, "oos_max_dd": -0.15}
    none = {"alpha": None, "oos_sharpe": None, "oos_max_dd": None}
    keys = list(pa.PORTFOLIOS)
    wf = {keys[0]: good, keys[1]: none, keys[2]: none}
    rec = pa.recommend_scenario(results, wf, {"regime": "Neutral"})
    assert rec["recommended_key"] == keys[0]
    assert rec["scores"][keys[1]]["alpha_missing"] is True
