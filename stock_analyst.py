"""
Stock Analyst — Interactive Single-Stock Deep-Dive
===================================================
Fetches fundamentals, technicals, DCF valuation, analyst consensus,
insider signals, and short interest for any ticker, then produces a
scored MODEL OPINION: BUY / ACCUMULATE / HOLD / REDUCE / AVOID.

Usage
-----
  python stock_analyst.py NVDA
  python stock_analyst.py NVDA AAPL MSFT        ← side-by-side comparison
  python stock_analyst.py                        ← interactive prompt

Output
------
  Console report  +  stock_analysis_<TICKER>_<ts>.html  in Financial_analysis/
"""

import sys
import subprocess

REQUIRED = {"yfinance": "yfinance", "pandas": "pandas", "numpy": "numpy"}
for mod, pkg in REQUIRED.items():
    try:
        __import__(mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

# ─── SECTOR BENCHMARKS (for relative valuation) ───────────────────────────────

SECTOR_BENCHMARKS = {
    "Technology":             {"fwd_pe": 27, "ev_ebitda": 22, "pb": 7.0, "roe": 0.25},
    "Communication Services": {"fwd_pe": 19, "ev_ebitda": 14, "pb": 3.5, "roe": 0.18},
    "Consumer Discretionary": {"fwd_pe": 22, "ev_ebitda": 16, "pb": 6.0, "roe": 0.20},
    "Consumer Staples":       {"fwd_pe": 20, "ev_ebitda": 15, "pb": 5.0, "roe": 0.22},
    "Energy":                 {"fwd_pe": 12, "ev_ebitda":  7, "pb": 2.0, "roe": 0.15},
    "Financials":             {"fwd_pe": 13, "ev_ebitda": 11, "pb": 1.5, "roe": 0.12},
    "Healthcare":             {"fwd_pe": 18, "ev_ebitda": 14, "pb": 3.0, "roe": 0.17},
    "Industrials":            {"fwd_pe": 19, "ev_ebitda": 14, "pb": 4.0, "roe": 0.18},
    "Basic Materials":        {"fwd_pe": 15, "ev_ebitda": 10, "pb": 2.0, "roe": 0.14},
    "Real Estate":            {"fwd_pe": 30, "ev_ebitda": 20, "pb": 2.0, "roe": 0.08},
    "Utilities":              {"fwd_pe": 18, "ev_ebitda": 11, "pb": 1.8, "roe": 0.10},
    "Default":                {"fwd_pe": 19, "ev_ebitda": 14, "pb": 3.0, "roe": 0.16},
}
# yfinance uses Morningstar sector names, not GICS — alias so they don't fall to Default
SECTOR_BENCHMARKS["Consumer Cyclical"]  = SECTOR_BENCHMARKS["Consumer Discretionary"]
SECTOR_BENCHMARKS["Consumer Defensive"] = SECTOR_BENCHMARKS["Consumer Staples"]
SECTOR_BENCHMARKS["Financial Services"] = SECTOR_BENCHMARKS["Financials"]

RISK_FREE_RATE      = 0.05
EQUITY_RISK_PREMIUM = 0.055
TERMINAL_GROWTH     = 0.025


# ─── DATA FETCHING ─────────────────────────────────────────────────────────────

def fetch_stock_data(ticker: str) -> dict:
    import portfolio_analyzer_advanced as _pa   # shared valuation + currency logic
    print(f"  {_pa.refresh_market_rates()}")
    t = yf.Ticker(ticker)
    info  = dict(t.info or {})
    # Statement FCF (3-yr avg) — Yahoo's info["freeCashflow"] is a levered estimate
    try:
        cf = t.cashflow
        if cf is not None and "Free Cash Flow" in cf.index:
            vals = cf.loc["Free Cash Flow"].dropna().iloc[:3]
            if len(vals):
                info["fcfStatementAvg"] = float(vals.mean())
    except Exception:
        pass
    try:
        info["revenueGrowthFwd"] = _pa.forward_revenue_growth(ticker, t.revenue_estimate)
    except Exception:
        info["revenueGrowthFwd"] = None
    # Foreign filers (e.g. Roche RHHBY reports in CHF, trades in USD) → one currency
    info = _pa.normalize_currency(info)
    hist  = t.history(period="2y", auto_adjust=True)
    hist1y = t.history(period="1y", auto_adjust=True)

    # Analyst recommendations
    try:
        recs = t.recommendations
    except Exception:
        recs = None

    # Insider transactions
    try:
        ins = t.insider_transactions
    except Exception:
        ins = None

    # Earnings calendar
    try:
        cal = t.calendar
    except Exception:
        cal = None

    # SPY for relative performance
    try:
        spy = yf.Ticker("SPY").history(period="1y", auto_adjust=True)["Close"]
    except Exception:
        spy = pd.Series(dtype=float)

    return {
        "ticker": ticker.upper(),
        "info":   info,
        "hist":   hist,
        "hist1y": hist1y,
        "recs":   recs,
        "ins":    ins,
        "cal":    cal,
        "spy":    spy,
    }


# ─── ANALYSIS MODULES ──────────────────────────────────────────────────────────

def _price_performance(hist: pd.DataFrame, spy: pd.Series) -> dict:
    if hist.empty:
        return {}
    close = hist["Close"].dropna()
    perf  = {}
    today = close.index[-1]
    for label, days in [("1 Month", 21), ("3 Months", 63),
                        ("6 Months", 126), ("1 Year", 252)]:
        if len(close) >= days:
            ret = (close.iloc[-1] / close.iloc[-days]) - 1
            perf[label] = ret
        else:
            perf[label] = None

    # vs SPY
    spy_perf = {}
    for label, days in [("1 Month", 21), ("3 Months", 63),
                        ("6 Months", 126), ("1 Year", 252)]:
        if not spy.empty and len(spy) >= days:
            spy_perf[label] = (spy.iloc[-1] / spy.iloc[-days]) - 1
        else:
            spy_perf[label] = None

    # 52-week range
    hi52 = float(close.iloc[-252:].max()) if len(close) >= 252 else float(close.max())
    lo52 = float(close.iloc[-252:].min()) if len(close) >= 252 else float(close.min())
    cur  = float(close.iloc[-1])
    pct_from_hi = (cur - hi52) / hi52
    pct_from_lo = (cur - lo52) / lo52

    # Momentum score (risk-adjusted 12m return)
    if len(close) >= 252:
        rets  = close.pct_change().dropna().iloc[-252:]
        ann_r = rets.mean() * 252
        ann_v = rets.std() * np.sqrt(252)
        mom_score = ann_r / ann_v if ann_v > 0 else 0.0
    else:
        mom_score = 0.0

    return {
        "current_price": cur,
        "52w_high":      round(hi52, 2),
        "52w_low":       round(lo52, 2),
        "pct_from_52w_high": pct_from_hi,
        "pct_from_52w_low":  pct_from_lo,
        "performance":   perf,
        "spy_perf":      spy_perf,
        "momentum_score": round(float(mom_score), 3),
    }


def _dcf_analysis(info: dict) -> dict:
    """
    Three-scenario DCF using the SAME engine as portfolio_analyzer_advanced.py
    (one DCF everywhere): statement FCF, growth = min(revenue, earnings) capped
    20% (3% for cyclicals), fading to terminal, Blume-adjusted WACC with a 7%
    floor, no FCF DCF for banks/insurers/REITs. `info` is already currency-normalised.
    """
    import portfolio_analyzer_advanced as _pa

    g     = _pa._sv_growth_rate(info)
    g_con = max(0.0, g - 0.03)                       # additive spreads, as in the analyzer
    g_opt = min(_pa.SV_MAX_GROWTH * 1.25, g + 0.04)
    if info.get("sector") in _pa.SV_NO_DCF_SECTORS:
        iv_base = iv_con = iv_opt = None
    else:
        iv_base, iv_con, iv_opt = ((_pa._sv_dcf(info, x) or {}).get("intrinsic")
                                   for x in (g, g_con, g_opt))

    px  = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    wacc = _pa._sv_wacc(info)

    result = {
        "base_growth":         round(g, 3),
        "conservative_growth": round(g_con, 3),
        "optimistic_growth":   round(g_opt, 3),
        "wacc":                round(wacc, 3),
        "intrinsic_base":      iv_base,
        "intrinsic_conservative": iv_con,
        "intrinsic_optimistic":   iv_opt,
        "current_price":       px if px > 0 else None,
        "dcf_signal":          "N/A",
        "upside_base":         None,
    }

    if iv_base and px and px > 0:
        upside = (iv_base - px) / px
        result["upside_base"] = round(upside * 100, 1)
        result["dcf_signal"]  = (
            "UNDERVALUED" if upside >  0.15 else
            "OVERVALUED"  if upside < -0.15 else
            "FAIR VALUE"
        )

    return result


def _catalysts(ticker: str) -> dict:
    """Dated FDA decisions, recent approvals/trials, scored news and structural
    risks — the same intelligence layers the portfolio analyzer uses."""
    import portfolio_analyzer_advanced as _pa
    out: dict = {"fda_decision": None, "fda": None, "news_signal": None,
                 "headlines": [], "structural": None}
    try:
        sigs, _ = _pa.upcoming_fda_signals()
        out["fda_decision"] = sigs.get(ticker)
    except Exception:
        pass
    try:
        out["fda"] = _pa.fetch_fda_pipeline([ticker], sector_map=None).get(ticker)
    except Exception:
        pass
    try:
        out["news_signal"] = _pa.fetch_catalyst_signals([ticker]).get(ticker)
    except Exception:
        pass
    try:
        cut = datetime.now() - timedelta(days=45)
        out["headlines"] = sorted([(d, h) for h, _, d in _pa.news_items(ticker) if d >= cut],
                                  reverse=True)[:6]
    except Exception:
        pass
    try:
        out["structural"] = _pa.assess_structural_risks([ticker]).get(ticker)
    except Exception:
        pass
    try:
        import sec_edgar
        out["sec_enabled"] = sec_edgar.enabled()
        out["sec_8k"] = sec_edgar.recent_8k(ticker, days=90) if sec_edgar.enabled() else []
    except Exception:
        out["sec_8k"] = []
    try:
        import analyst_signals as _as
        out["analyst"] = _as.fetch_revision_signals(
            [ticker], _pa._safe_attr, _pa._safe_info).get(ticker)
    except Exception:
        out["analyst"] = None
    return out


def _analyst_lines(a: dict | None) -> list[str]:
    """Plain-text analyst block: revisions (a screen factor) + targets/ratings (context only)."""
    if not a:
        return []
    m, c = a["metrics"], a.get("context") or {}
    rev = [f"revision score {a['score']:+.2f}"]
    if m.get("trend_90d") is not None:
        rev.append(f"{m.get('trend_period', '+1y')} EPS est. {m['trend_90d']:+.1%} in 90d")
    if m.get("breadth") is not None:
        rev.append(f"breadth {m['breadth']:+.2f}")
    if m.get("surprise_hit") is not None:
        rev.append(f"beat {m['surprise_hit']:.0%} of last {m['n_quarters']}q")
    lines = ["Revisions:      " + "  ·  ".join(rev)]
    if c.get("target_mean"):
        lines.append(f"Targets/rating: mean ${c['target_mean']:,.2f}"
                     + (f" ({c['upside']:+.0%})" if c.get("upside") is not None else "")
                     + f"  range ${c.get('target_low') or 0:,.0f}–${c.get('target_high') or 0:,.0f}"
                     + f"  ·  {c.get('rating_key') or '—'} ({c.get('n_analysts', 0)} analysts)"
                     + "  — context only, not scored")
    lines += [f"⚑ {f}" for f in c.get("flags", [])]
    return lines


def _valuation_multiples(info: dict) -> dict:
    sector = info.get("sector", "Default")
    bench  = SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS["Default"])

    def _rel(val, bm_key):
        v = info.get(val)
        b = bench.get(bm_key)
        if v and b and float(v) > 0:
            r = float(v) / b
            label = ("CHEAP" if r < 0.75 else
                     "EXPENSIVE" if r > 1.25 else "FAIR")
            return {"value": round(float(v), 1), "vs_sector": label,
                    "ratio": round(r, 2)}
        return {"value": None, "vs_sector": "N/A", "ratio": None}

    return {
        "Forward P/E":    _rel("forwardPE",          "fwd_pe"),
        "EV/EBITDA":      _rel("enterpriseToEbitda", "ev_ebitda"),
        "Price/Book":     _rel("priceToBook",        "pb"),
        "Trailing P/E":   {"value": round(float(info["trailingPE"]), 1)
                           if info.get("trailingPE") else None,
                           "vs_sector": "—", "ratio": None},
        "PEG Ratio":      {"value": round(float(info["trailingPegRatio"]), 2)
                           if info.get("trailingPegRatio") else None,
                           "vs_sector": ("CHEAP" if (info.get("trailingPegRatio") or 99) < 1
                                         else "FAIR" if (info.get("trailingPegRatio") or 99) < 2
                                         else "EXPENSIVE"),
                           "ratio": None},
        "sector":         sector,
        "sector_benchmarks": bench,
    }


def _financial_health(info: dict) -> dict:
    def _pct(key):
        v = info.get(key)
        return round(float(v) * 100, 1) if v is not None else None

    def _val(key):
        v = info.get(key)
        return round(float(v), 3) if v is not None else None

    mc = info.get("marketCap")
    mc_label = ("Mega-cap (>$200B)" if mc and mc > 200e9 else
                "Large-cap ($10B–$200B)" if mc and mc > 10e9 else
                "Mid-cap ($2B–$10B)"  if mc and mc >  2e9 else
                "Small-cap (<$2B)"    if mc else "N/A")

    return {
        "Revenue Growth (YoY)":   _pct("revenueGrowth"),
        "Earnings Growth (YoY)":  _pct("earningsGrowth"),
        "Gross Margin":           _pct("grossMargins"),
        "Operating Margin":       _pct("operatingMargins"),
        "Net Margin":             _pct("profitMargins"),
        "ROE":                    _pct("returnOnEquity"),
        "ROA":                    _pct("returnOnAssets"),
        # yfinance reports debtToEquity ×100 (69.5 → 0.695×)
        "Debt/Equity":            (round(float(info["debtToEquity"]) / 100, 2)
                                   if info.get("debtToEquity") is not None else None),
        "Current Ratio":          _val("currentRatio"),
        "Free Cash Flow ($M)":    (round(float(info["freeCashflow"]) / 1e6, 1)
                                   if info.get("freeCashflow") else None),
        "Market Cap":             mc_label,
        # yfinance reports dividendYield already in percent (2.05 = 2.05%)
        "Dividend Yield":         (round(float(info["dividendYield"]), 2)
                                   if info.get("dividendYield") is not None else None),
        "Beta":                   _val("beta"),
    }


def _analyst_consensus(recs) -> dict:
    if recs is None or (hasattr(recs, "empty") and recs.empty):
        return {"rating": "N/A", "buy_pct": None, "detail": {}}
    try:
        # Recent 3 months only
        if hasattr(recs, "index") and hasattr(recs.index, "tz_localize"):
            pass
        recent = recs.tail(4) if len(recs) >= 4 else recs
        cols   = {c.lower(): c for c in recent.columns}
        buy_c  = cols.get("strongbuy") or cols.get("strong_buy", "")
        abuy_c = cols.get("buy", "")
        hold_c = cols.get("hold", "")
        sell_c = cols.get("sell", "")
        ssell_c= cols.get("strongsell") or cols.get("strong_sell", "")

        def _s(col):
            return int(recent[col].iloc[-1]) if col and col in recent.columns else 0

        strong_buy = _s(buy_c)
        buy        = _s(abuy_c)
        hold       = _s(hold_c)
        sell       = _s(sell_c)
        strong_sell= _s(ssell_c)
        total      = strong_buy + buy + hold + sell + strong_sell
        if total == 0:
            return {"rating": "N/A", "buy_pct": None, "detail": {}}
        buy_pct = (strong_buy + buy) / total
        rating  = ("STRONG BUY"  if buy_pct >= 0.70 else
                   "BUY"         if buy_pct >= 0.55 else
                   "HOLD"        if buy_pct >= 0.35 else
                   "SELL")
        return {
            "rating":   rating,
            "buy_pct":  round(buy_pct * 100, 1),
            "total":    total,
            "detail":   {
                "Strong Buy": strong_buy, "Buy": buy, "Hold": hold,
                "Sell": sell, "Strong Sell": strong_sell,
            },
        }
    except Exception:
        return {"rating": "N/A", "buy_pct": None, "detail": {}}


def _insider_activity(ins) -> dict:
    if ins is None or (hasattr(ins, "empty") and ins.empty):
        return {"signal": "N/A", "net_buys": None, "detail": "No data"}
    try:
        cutoff = datetime.now() - timedelta(days=90)
        cols   = {c.lower().replace(" ", "_"): c for c in ins.columns}
        date_c = cols.get("start_date") or cols.get("date") or ins.columns[0]
        trans_c= cols.get("transaction") or cols.get("text", "")
        ins["_date"] = pd.to_datetime(ins[date_c], errors="coerce")
        recent = ins[ins["_date"] >= cutoff]
        if recent.empty:
            return {"signal": "NEUTRAL", "net_buys": 0,
                    "detail": "No insider activity in last 90 days"}
        buys  = sells = 0
        if trans_c and trans_c in recent.columns:
            buys  = recent[trans_c].str.contains("Buy|Purchase", case=False, na=False).sum()
            sells = recent[trans_c].str.contains("Sell|Sale",    case=False, na=False).sum()
        net    = int(buys) - int(sells)
        signal = "BULLISH" if net > 0 else ("BEARISH" if net < 0 else "NEUTRAL")
        return {
            "signal":   signal,
            "net_buys": net,
            "buys":     int(buys),
            "sells":    int(sells),
            "detail":   f"{int(buys)} buy(s), {int(sells)} sell(s) in last 90 days",
        }
    except Exception:
        return {"signal": "N/A", "net_buys": None, "detail": "Parse error"}


def _short_interest(info: dict) -> dict:
    sp = info.get("shortPercentOfFloat")
    sr = info.get("shortRatio")
    if sp is None:
        return {"signal": "N/A", "pct_float": None, "days_to_cover": None}
    sp = float(sp)
    signal = ("HIGH" if sp >= 0.15 else "ELEVATED" if sp >= 0.10 else "NORMAL")
    return {
        "signal":        signal,
        "pct_float":     round(sp * 100, 1),
        "days_to_cover": round(float(sr), 1) if sr else None,
    }


def _earnings_calendar(cal) -> dict:
    if cal is None:
        return {"date": None, "days_away": None}
    try:
        today = datetime.now().date()
        if isinstance(cal, dict):
            date_val = cal.get("Earnings Date")
            if isinstance(date_val, (list, tuple)):
                date_val = date_val[0]
        elif hasattr(cal, "iloc") and not cal.empty:
            date_val = cal.iloc[0, 0]
        else:
            return {"date": None, "days_away": None}
        if date_val is None:
            return {"date": None, "days_away": None}
        ed        = pd.Timestamp(date_val).date()
        days_away = (ed - today).days
        return {"date": str(ed), "days_away": days_away}
    except Exception:
        return {"date": None, "days_away": None}


# ─── MODEL OPINION SCORING ────────────────────────────────────────────────────

def compute_opinion(dcf: dict, multiples: dict, health: dict,
                    price_data: dict, analyst: dict, insider: dict,
                    short: dict) -> dict:
    """
    Weighted composite score → BUY / ACCUMULATE / HOLD / REDUCE / AVOID

    Weights:
      DCF valuation      35%
      Momentum           20%
      Financial quality  20%
      Analyst consensus  15%
      Insider signal     10%
    """
    scores = {}

    # ── DCF (35%) ─────────────────────────────────────────────────────────────
    sig = dcf.get("dcf_signal", "N/A")
    if sig == "UNDERVALUED":
        scores["dcf"] = 1.0
    elif sig == "FAIR VALUE":
        scores["dcf"] = 0.5
    elif sig == "OVERVALUED":
        scores["dcf"] = 0.1
    else:
        scores["dcf"] = 0.5   # no data → neutral

    # Boost/penalise by magnitude of upside
    upside = dcf.get("upside_base")
    if upside is not None:
        scores["dcf"] = max(0.0, min(1.0, 0.5 + upside / 100))

    # ── Momentum (20%) ────────────────────────────────────────────────────────
    mom = price_data.get("momentum_score", 0.0)
    # Normalise: Sharpe-like score of 2.0 = top, -1.0 = bottom
    scores["momentum"] = max(0.0, min(1.0, (mom + 1.0) / 3.0))

    # ── Financial quality (20%) ───────────────────────────────────────────────
    q_scores = []
    rev_g = health.get("Revenue Growth (YoY)")
    if rev_g is not None:
        q_scores.append(max(0.0, min(1.0, (rev_g + 10) / 50)))  # -10%→0, +40%→1
    roe = health.get("ROE")
    if roe is not None:
        q_scores.append(max(0.0, min(1.0, (roe + 5) / 35)))     # -5%→0, +30%→1
    net_m = health.get("Net Margin")
    if net_m is not None:
        q_scores.append(max(0.0, min(1.0, (net_m + 5) / 35)))
    scores["quality"] = float(np.mean(q_scores)) if q_scores else 0.5

    # Penalise high debt
    de = health.get("Debt/Equity")
    if de and de > 2.0:
        scores["quality"] *= 0.85

    # ── Analyst consensus (15%) ───────────────────────────────────────────────
    bp = analyst.get("buy_pct")
    scores["analyst"] = (bp / 100) if bp is not None else 0.5

    # ── Insider signal (10%) ─────────────────────────────────────────────────
    isig = insider.get("signal", "N/A")
    scores["insider"] = (1.0 if isig == "BULLISH" else
                         0.0 if isig == "BEARISH" else 0.5)

    # ── Composite ─────────────────────────────────────────────────────────────
    weights = {"dcf": 0.35, "momentum": 0.20, "quality": 0.20,
               "analyst": 0.15, "insider": 0.10}
    composite = sum(scores[k] * weights[k] for k in weights)

    # ── Short interest flag (informational, not in score) ─────────────────────
    short_flag = short.get("signal", "N/A")

    if composite >= 0.70:
        opinion = "BUY"
        color   = "#4caf50"
        stars   = "★★★★★"
    elif composite >= 0.57:
        opinion = "ACCUMULATE"
        color   = "#8bc34a"
        stars   = "★★★★☆"
    elif composite >= 0.43:
        opinion = "HOLD"
        color   = "#ff9800"
        stars   = "★★★☆☆"
    elif composite >= 0.30:
        opinion = "REDUCE"
        color   = "#f44336"
        stars   = "★★☆☆☆"
    else:
        opinion = "AVOID"
        color   = "#b71c1c"
        stars   = "★☆☆☆☆"

    # Build reasoning bullets
    reasons = []
    if scores["dcf"] >= 0.65:
        reasons.append(f"DCF suggests stock is <strong>undervalued</strong> "
                       f"(est. {dcf.get('upside_base', 0):+.1f}% upside to intrinsic value)")
    elif scores["dcf"] <= 0.35:
        reasons.append(f"DCF suggests stock may be <strong>overvalued</strong> "
                       f"(est. {dcf.get('upside_base', 0):+.1f}% vs intrinsic value)")
    else:
        reasons.append("DCF valuation is near fair value")

    if scores["momentum"] >= 0.65:
        reasons.append("Strong price momentum — recent outperformer")
    elif scores["momentum"] <= 0.35:
        reasons.append("Weak price momentum — recent underperformer")

    if scores["quality"] >= 0.65:
        reasons.append("Healthy financials: solid revenue growth, margins, and ROE")
    elif scores["quality"] <= 0.35:
        reasons.append("Financial quality concerns: low growth, thin margins, or weak ROE")

    if scores["analyst"] >= 0.65:
        reasons.append(f"Analyst consensus is <strong>bullish</strong> "
                       f"({analyst.get('buy_pct','—')}% buy ratings)")
    elif scores["analyst"] <= 0.35:
        reasons.append(f"Analyst consensus is <strong>cautious</strong> "
                       f"({analyst.get('buy_pct','—')}% buy ratings)")

    if isig == "BULLISH":
        reasons.append("Insiders have been <strong>net buyers</strong> in the last 90 days — "
                       "positive signal from those who know the business best")
    elif isig == "BEARISH":
        reasons.append("Insiders have been <strong>net sellers</strong> in the last 90 days")

    if short_flag == "HIGH":
        reasons.append(f"⚠ High short interest ({short.get('pct_float','—')}% of float) — "
                       "elevated risk but potential short-squeeze catalyst")
    elif short_flag == "ELEVATED":
        reasons.append(f"Elevated short interest ({short.get('pct_float','—')}% of float)")

    return {
        "opinion":   opinion,
        "color":     color,
        "stars":     stars,
        "composite": round(composite, 3),
        "scores":    {k: round(v, 3) for k, v in scores.items()},
        "weights":   weights,
        "reasons":   reasons,
    }


# ─── CONSOLE REPORT ───────────────────────────────────────────────────────────

def print_report(ticker: str, data: dict, analysis: dict) -> None:
    info      = data["info"]
    name      = info.get("longName") or info.get("shortName") or ticker
    sector    = info.get("sector", "Unknown")
    industry  = info.get("industry", "Unknown")
    price     = analysis["price"]["current_price"]
    opinion   = analysis["opinion"]
    now       = datetime.now().strftime("%Y-%m-%d %H:%M")

    W = 72
    print("\n" + "═" * W)
    print(f"  STOCK ANALYSIS  —  {ticker}  |  {name}")
    print(f"  {sector}  ·  {industry}  |  {now}")
    print("═" * W)

    # Opinion banner
    stars = opinion["stars"]
    op    = opinion["opinion"]
    score = opinion["composite"]
    print(f"\n  MODEL OPINION:  {op}  {stars}  (score: {score:.3f} / 1.000)")
    print(f"  {'─'*60}")
    for r in opinion["reasons"]:
        clean = r.replace("<strong>", "").replace("</strong>", "")
        print(f"  • {clean}")

    # Price snapshot
    pd_ = analysis["price"]
    print(f"\n  ── PRICE SNAPSHOT")
    print(f"     Current:   ${price:,.2f}"
          f"   52w High: ${pd_['52w_high']:,.2f} ({pd_['pct_from_52w_high']:+.1%})"
          f"   52w Low:  ${pd_['52w_low']:,.2f} ({pd_['pct_from_52w_low']:+.1%})")
    print(f"     {'Period':<12}  {'Stock':>8}  {'SPY':>8}  {'vs SPY':>8}")
    for period in ["1 Month", "3 Months", "6 Months", "1 Year"]:
        st = pd_["performance"].get(period)
        sp = pd_["spy_perf"].get(period)
        vs = (st - sp) if st is not None and sp is not None else None
        st_s = f"{st:+.1%}" if st is not None else "  N/A"
        sp_s = f"{sp:+.1%}" if sp is not None else "  N/A"
        vs_s = f"{vs:+.1%}" if vs is not None else "  N/A"
        print(f"     {period:<12}  {st_s:>8}  {sp_s:>8}  {vs_s:>8}")

    # DCF
    dcf = analysis["dcf"]
    print(f"\n  ── DCF VALUATION  (WACC: {dcf['wacc']:.1%}  |  Base growth: {dcf['base_growth']:.1%})")
    for label, key in [("Conservative", "intrinsic_conservative"),
                       ("Base case",    "intrinsic_base"),
                       ("Optimistic",   "intrinsic_optimistic")]:
        iv = dcf.get(key)
        if iv and price and price > 0:
            upside = (iv - price) / price
            marker = " ← BASE" if "base" in key.lower() else ""
            print(f"     {label:<14}  ${iv:>8,.2f}  ({upside:+.1%} vs today){marker}")
        else:
            print(f"     {label:<14}  N/A")
    print(f"     Signal:        {dcf['dcf_signal']}")

    # Multiples
    mult = analysis["multiples"]
    print(f"\n  ── VALUATION MULTIPLES  (sector: {mult['sector']})")
    for name2, m in mult.items():
        if isinstance(m, dict) and m.get("value") is not None:
            val_s = str(m["value"])
            vs_s  = m.get("vs_sector", "—")
            print(f"     {name2:<20}  {val_s:>8}  [{vs_s}]")

    # Financial health
    hlth = analysis["health"]
    print(f"\n  ── FINANCIAL HEALTH")
    for k, v in hlth.items():
        if v is not None:
            if "Margin" in k or "Growth" in k or "ROE" in k or "ROA" in k or "Yield" in k:
                val_s = f"{v}%"
            elif k == "Debt/Equity":
                val_s = f"{v}×"
            elif "($M)" in k:
                val_s = f"${v:,.1f}M"
            else:
                val_s = str(v)
            print(f"     {k:<30}  {val_s}")

    # Analyst
    an = analysis["analyst"]
    if an.get("rating") != "N/A":
        print(f"\n  ── ANALYST CONSENSUS  ({an.get('total','—')} analysts)")
        print(f"     Rating:  {an['rating']}  ({an.get('buy_pct','—')}% buy)")
        for k, v in an.get("detail", {}).items():
            bar = "█" * v
            print(f"     {k:<14}  {v:>3}  {bar}")

    # Insider + Short
    ins  = analysis["insider"]
    shrt = analysis["short"]
    print(f"\n  ── SIGNALS")
    print(f"     Insider (90d):  {ins['signal']}  —  {ins.get('detail','')}")
    if shrt.get("pct_float"):
        print(f"     Short interest: {shrt['signal']}  "
              f"({shrt['pct_float']}% of float  |  "
              f"{shrt.get('days_to_cover','—')} days to cover)")

    # Earnings
    earn = analysis["earnings"]
    if earn.get("date"):
        days = earn["days_away"]
        warn = "  ⚠ UPCOMING" if days is not None and 0 <= days <= 30 else ""
        print(f"     Next earnings:  {earn['date']}  ({days}d away){warn}")

    cat = analysis.get("catalysts") or {}
    print(f"\n  ── CATALYSTS & RISKS")
    if data["info"].get("fx_converted"):
        print(f"     Currency:       financials converted {data['info']['fx_converted']}")
    if data["info"].get("fx_warning"):
        print(f"     ⚠ {data['info']['fx_warning']}")
    if cat.get("fda_decision"):
        print(f"     📅 {cat['fda_decision']['headline']}")
    if cat.get("fda"):
        print(f"     FDA / trials:   {cat['fda']['fda_summary'][:95]}")
    ns = cat.get("news_signal")
    if ns:
        arrow = "▲" if ns.get("signal") == "POSITIVE" else "▼"
        print(f"     {arrow} {ns.get('catalyst_type','')}: {str(ns.get('headline',''))[:90]}")
    if cat.get("structural"):
        s = cat["structural"]
        print(f"     ⚠ Structural risk: {s['risk']} (severity {s['severity']}) — {s['note'][:70]}")
    if cat.get("headlines"):
        print(f"     Recent headlines:")
        for d, h in cat["headlines"]:
            print(f"       {d:%Y-%m-%d}  {h[:95]}")
    if cat.get("sec_8k"):
        print(f"     SEC 8-K filings (90d):")
        for f in cat["sec_8k"][:6]:
            print(f"       {f['date']}  {'; '.join(f['meaning']) or 'exhibits only'}")
    elif cat.get("sec_enabled") is False:
        print(f"     SEC EDGAR off — set SEC_USER_AGENT to list 8-K filings")
    for line in _analyst_lines(cat.get("analyst")):
        print(f"     {line}")
    if not any(cat.get(k) for k in ("fda_decision", "fda", "news_signal", "structural",
                                     "headlines", "sec_8k", "analyst")):
        print("     None found (news feeds empty and no listed FDA dates)")

    print("\n" + "═" * W)
    print("  ⚠  This analysis is algorithmic and educational only.")
    print("     It does not constitute investment advice.")
    print("═" * W + "\n")


# ─── HTML REPORT ──────────────────────────────────────────────────────────────

def generate_html(ticker: str, data: dict, analysis: dict) -> str:
    info    = data["info"]
    name    = info.get("longName") or info.get("shortName") or ticker
    sector  = info.get("sector", "Unknown")
    industry= info.get("industry", "Unknown")
    price   = analysis["price"]["current_price"]
    opinion = analysis["opinion"]
    now     = datetime.now().strftime("%Y-%m-%d %H:%M")
    desc    = info.get("longBusinessSummary", "")
    if desc and len(desc) > 500:
        desc = desc[:500] + "…"

    def esc(v):
        return str(v).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

    def pct(v):
        return f"{v:+.1%}" if v is not None else "N/A"

    def dollar(v):
        try:
            return f"${float(v):,.2f}"
        except Exception:
            return str(v)

    op    = opinion["opinion"]
    color = opinion["color"]
    score = opinion["composite"]
    stars = opinion["stars"]
    sc    = opinion["scores"]
    wt    = opinion["weights"]

    dcf  = analysis["dcf"]
    mult = analysis["multiples"]
    hlth = analysis["health"]
    pd_  = analysis["price"]
    an   = analysis["analyst"]
    ins  = analysis["insider"]
    shrt = analysis["short"]
    earn = analysis["earnings"]

    CSS = f"""
:root{{--bg:#0f1117;--panel:#1a1d2e;--grid:#2a2d3e;--text:#e0e0e0;
      --muted:#8888aa;--accent:{color};--green:#4caf50;--red:#f44336;--orange:#ff9800;}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',sans-serif;
     font-size:14px;padding:32px;max-width:1100px;margin:0 auto;}}
h2{{font-size:1.1rem;margin:28px 0 10px;color:var(--text);
   border-bottom:1px solid var(--grid);padding-bottom:6px;}}
p{{color:var(--muted);font-size:0.85rem;margin-bottom:10px;line-height:1.6;}}
table{{width:100%;border-collapse:collapse;margin-bottom:16px;font-size:0.83rem;}}
th{{background:var(--panel);color:var(--muted);padding:8px 10px;text-align:left;
    border-bottom:1px solid var(--grid);}}
td{{padding:7px 10px;border-bottom:1px solid var(--grid);}}
tr:hover td{{background:rgba(255,255,255,0.02);}}
.num{{text-align:right;}}
.good{{color:var(--green);}} .bad{{color:var(--red);}} .neutral{{color:var(--muted);}}
.hero{{background:var(--panel);border-radius:12px;padding:28px;margin-bottom:28px;
       border-left:5px solid {color};}}
.opinion{{font-size:2.2rem;font-weight:900;color:{color};letter-spacing:2px;}}
.stars{{font-size:1.4rem;margin-left:12px;color:{color};}}
.score-bar{{height:8px;border-radius:4px;background:var(--grid);margin-top:4px;}}
.score-fill{{height:8px;border-radius:4px;background:{color};}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;}}
.card{{background:var(--panel);border-radius:8px;padding:16px;}}
.card h3{{font-size:0.85rem;color:var(--muted);margin-bottom:10px;text-transform:uppercase;
          letter-spacing:1px;}}
.reasons li{{margin:6px 0;font-size:0.85rem;line-height:1.5;}}
.disclaimer{{margin-top:36px;padding:14px;background:var(--panel);border-radius:8px;
             color:var(--muted);font-size:0.78rem;line-height:1.6;}}
.tag{{display:inline-block;padding:3px 10px;border-radius:20px;font-size:0.78rem;
      font-weight:600;}}
"""

    parts = [
        "<!DOCTYPE html><html lang='en'>",
        f"<head><meta charset='UTF-8'><title>{esc(ticker)} — Stock Analysis</title>",
        f"<style>{CSS}</style></head><body>",
    ]

    # Hero
    parts.append(f"""
<div class="hero">
  <div style="display:flex;align-items:baseline;gap:16px;flex-wrap:wrap">
    <span class="opinion">{esc(op)}</span>
    <span class="stars">{esc(stars)}</span>
    <span style="color:var(--muted);font-size:1rem">Score: {score:.3f} / 1.000</span>
  </div>
  <div style="margin-top:10px;font-size:1.5rem;font-weight:700">{esc(ticker)}
    <span style="color:var(--muted);font-size:1rem;font-weight:400"> — {esc(name)}</span>
  </div>
  <div style="color:var(--muted);margin-top:4px;font-size:0.85rem">
    {esc(sector)} · {esc(industry)} &nbsp;|&nbsp;
    Current price: <strong style="color:var(--text)">{dollar(price)}</strong>
    &nbsp;|&nbsp; Generated: {esc(now)}
  </div>
  {f'<p style="margin-top:12px;font-size:0.83rem">{esc(desc)}</p>' if desc else ""}
</div>""")

    # Reasons
    parts.append("<h2>Why this opinion?</h2><ul class='reasons'>")
    for r in opinion["reasons"]:
        parts.append(f"<li>{r}</li>")
    parts.append("</ul>")

    # Score breakdown
    dim_labels = {
        "dcf":      ("DCF Valuation",       "35%"),
        "momentum": ("Price Momentum",       "20%"),
        "quality":  ("Financial Quality",    "20%"),
        "analyst":  ("Analyst Consensus",    "15%"),
        "insider":  ("Insider Activity",     "10%"),
    }
    parts.append("<h2>Score Breakdown</h2>")
    parts.append('<table><thead><tr><th>Dimension</th><th>Weight</th>'
                 '<th class="num">Score</th><th>Bar</th></tr></thead><tbody>')
    for k, (label, weight) in dim_labels.items():
        v   = sc.get(k, 0)
        bar = f'<div class="score-bar"><div class="score-fill" style="width:{v*100:.0f}%"></div></div>'
        c   = "good" if v >= 0.65 else ("bad" if v <= 0.35 else "neutral")
        parts.append(f'<tr><td><strong>{esc(label)}</strong></td><td class="num neutral">{esc(weight)}</td>'
                     f'<td class="num {c}">{v:.3f}</td><td style="min-width:120px">{bar}</td></tr>')
    parts.append("</tbody></table>")

    # Two-column grid: price + DCF
    parts.append('<div class="grid2">')

    # Price performance
    parts.append(f"""<div class="card"><h3>Price Performance</h3>
<table><thead><tr><th>Period</th><th class="num">Stock</th>
<th class="num">SPY</th><th class="num">vs SPY</th></tr></thead><tbody>""")
    for period in ["1 Month", "3 Months", "6 Months", "1 Year"]:
        st = pd_["performance"].get(period)
        sp = pd_["spy_perf"].get(period)
        vs = (st - sp) if st is not None and sp is not None else None
        sc2 = "good" if (vs or 0) >= 0 else "bad"
        parts.append(f'<tr><td>{esc(period)}</td>'
                     f'<td class="num">{pct(st)}</td>'
                     f'<td class="num neutral">{pct(sp)}</td>'
                     f'<td class="num {sc2}">{pct(vs)}</td></tr>')
    parts.append(f"""</tbody></table>
<div style="font-size:0.8rem;color:var(--muted);margin-top:8px">
  52-week range: ${pd_['52w_low']:,.2f} — ${pd_['52w_high']:,.2f}<br>
  Current: {dollar(price)}
  ({pct(pd_['pct_from_52w_high'])} from 52w high)
</div></div>""")

    # DCF
    parts.append(f"""<div class="card"><h3>DCF Valuation</h3>
<p style="margin-bottom:8px">WACC: {dcf['wacc']:.1%} &nbsp;|&nbsp;
Base growth rate: {dcf['base_growth']:.1%}</p>
<table><thead><tr><th>Scenario</th><th class="num">Intrinsic Value</th>
<th class="num">vs Today</th></tr></thead><tbody>""")
    for label, key in [("Conservative", "intrinsic_conservative"),
                       ("Base case",    "intrinsic_base"),
                       ("Optimistic",   "intrinsic_optimistic")]:
        iv = dcf.get(key)
        if iv and price and price > 0:
            up  = (iv - price) / price
            c   = "good" if up > 0 else "bad"
            bold = " style='font-weight:700'" if "base" in key else ""
            parts.append(f'<tr{bold}><td>{esc(label)}</td>'
                         f'<td class="num">{dollar(iv)}</td>'
                         f'<td class="num {c}">{pct(up)}</td></tr>')
    sig_c = ("good" if dcf["dcf_signal"] == "UNDERVALUED" else
             "bad"  if dcf["dcf_signal"] == "OVERVALUED"  else "neutral")
    parts.append(f"""</tbody></table>
<div style="margin-top:8px">Signal: <strong class="{sig_c}">{esc(dcf["dcf_signal"])}</strong>
</div></div>""")

    parts.append("</div>")   # close grid2

    # Valuation multiples
    parts.append(f"<h2>Valuation Multiples — vs {esc(mult['sector'])} sector</h2>")
    parts.append('<table><thead><tr><th>Metric</th><th class="num">Value</th>'
                 '<th>vs Sector</th></tr></thead><tbody>')
    for mname, m in mult.items():
        if isinstance(m, dict) and m.get("value") is not None:
            vs   = m.get("vs_sector", "—")
            mc   = ("good" if vs == "CHEAP" else "bad" if vs == "EXPENSIVE" else "neutral")
            parts.append(f'<tr><td>{esc(mname)}</td>'
                         f'<td class="num">{esc(str(m["value"]))}</td>'
                         f'<td class="{mc}">{esc(vs)}</td></tr>')
    parts.append("</tbody></table>")

    # Financial health
    parts.append("<h2>Financial Health</h2>")
    parts.append('<table><thead><tr><th>Metric</th><th class="num">Value</th>'
                 '</tr></thead><tbody>')
    for k, v in hlth.items():
        if v is not None:
            val_s = f"{v}%" if any(x in k for x in ["Margin","Growth","ROE","ROA","Yield"]) else str(v)
            c = ""
            if any(x in k for x in ["Growth","ROE","ROA","Margin","Free Cash"]):
                try:
                    c = "good" if float(str(v).replace("%","")) > 0 else "bad"
                except Exception:
                    pass
            parts.append(f'<tr><td>{esc(k)}</td>'
                         f'<td class="num {c}">{esc(val_s)}</td></tr>')
    parts.append("</tbody></table>")

    # Signals row
    parts.append('<div class="grid2">')

    # Analyst
    parts.append('<div class="card"><h3>Analyst Consensus</h3>')
    if an.get("rating") != "N/A":
        an_c = ("good" if "BUY" in an["rating"] else
                "bad"  if "SELL" in an["rating"] else "neutral")
        parts.append(f'<div style="font-size:1.3rem;font-weight:700" class="{an_c}">'
                     f'{esc(an["rating"])}</div>')
        parts.append(f'<p>{an.get("buy_pct","—")}% buy ratings across '
                     f'{an.get("total","—")} analysts</p>')
        parts.append('<table><tbody>')
        for k, v in an.get("detail", {}).items():
            w = max(1, v * 10)
            bar_c = "#4caf50" if "Buy" in k else ("#f44336" if "Sell" in k else "#888")
            parts.append(f'<tr><td style="width:120px">{esc(k)}</td>'
                         f'<td style="width:40px" class="num">{v}</td>'
                         f'<td><div style="height:6px;border-radius:3px;'
                         f'background:{bar_c};width:{w}px"></div></td></tr>')
        parts.append("</tbody></table>")
    else:
        parts.append('<p>No analyst data available.</p>')
    parts.append("</div>")

    # Insider + Short + Earnings
    parts.append('<div class="card"><h3>Signals</h3>')
    ins_c = ("good" if ins["signal"] == "BULLISH" else
             "bad"  if ins["signal"] == "BEARISH"  else "neutral")
    parts.append(f'<p><strong>Insider activity (90d):</strong> '
                 f'<span class="{ins_c}">{esc(ins["signal"])}</span> '
                 f'— {esc(ins.get("detail",""))}</p>')

    if shrt.get("pct_float"):
        sh_c = ("bad" if shrt["signal"] == "HIGH" else
                "neutral" if shrt["signal"] == "ELEVATED" else "good")
        dtc = f', {shrt["days_to_cover"]}d to cover' if shrt.get("days_to_cover") else ""
        parts.append(f'<p><strong>Short interest:</strong> '
                     f'<span class="{sh_c}">{esc(shrt["signal"])}</span> '
                     f'— {shrt["pct_float"]}% of float{dtc}'
                     f'</p>')

    if earn.get("date"):
        days = earn["days_away"]
        warn_c = "bad" if days is not None and 0 <= days <= 30 else "neutral"
        parts.append(f'<p><strong>Next earnings:</strong> '
                     f'<span class="{warn_c}">{esc(earn["date"])} '
                     f'({days}d away)'
                     f'{"  ⚠ Upcoming" if 0 <= (days or 99) <= 30 else ""}'
                     f'</span></p>')
    parts.append("</div></div>")   # close card + grid2

    # Catalysts & risks
    cat = analysis.get("catalysts") or {}
    items = []
    if data["info"].get("fx_converted"):
        items.append(f'Currency: financials converted {esc(data["info"]["fx_converted"])}')
    if cat.get("fda_decision"):
        items.append(f'<strong>📅 {esc(cat["fda_decision"]["headline"])}</strong>')
    if cat.get("fda"):
        items.append(f'FDA / trials: {esc(cat["fda"]["fda_summary"])}')
    if cat.get("news_signal"):
        ns = cat["news_signal"]
        items.append(f'{esc(ns.get("catalyst_type",""))}: {esc(str(ns.get("headline",""))[:140])}')
    if cat.get("structural"):
        s = cat["structural"]
        items.append(f'<span class="bad">Structural risk — {esc(s["risk"])} '
                     f'(severity {s["severity"]}): {esc(s["note"])}</span>')
    for d, h in cat.get("headlines", []):
        items.append(f'{d:%Y-%m-%d} — {esc(h[:140])}')
    items += [esc(line) for line in _analyst_lines(cat.get("analyst"))]
    parts.append('<div class="card"><h3>Catalysts &amp; Risks</h3>'
                 + ("".join(f"<p>{i}</p>" for i in items) if items
                    else "<p>None found (news feeds empty and no listed FDA dates).</p>")
                 + "</div>")

    # Disclaimer
    parts.append(f"""
<div class="disclaimer">
  <strong>Disclaimer:</strong> This report is generated by an algorithmic stock analyzer
  for educational purposes only. It does not constitute personalised financial or investment
  advice. The MODEL OPINION is a rule-based composite score — not a guarantee of future
  performance. Always conduct your own research and consult a licensed financial advisor
  before investing. &nbsp;·&nbsp; Generated: {esc(now)}
</div>
</body></html>""")

    return "\n".join(parts)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def analyse(ticker: str) -> None:
    ticker = ticker.upper().strip()
    print(f"\n  Fetching data for {ticker}...")

    data = fetch_stock_data(ticker)
    info = data["info"]

    if not info or not info.get("regularMarketPrice") and not info.get("currentPrice"):
        print(f"  ✗ Could not fetch data for '{ticker}'. "
              f"Check the ticker symbol and try again.")
        return

    # Run all modules
    price_data = _price_performance(data["hist"], data["spy"])
    dcf        = _dcf_analysis(info)
    multiples  = _valuation_multiples(info)
    health     = _financial_health(info)
    analyst    = _analyst_consensus(data["recs"])
    insider    = _insider_activity(data["ins"])
    short      = _short_interest(info)
    earnings   = _earnings_calendar(data["cal"])
    opinion    = compute_opinion(dcf, multiples, health, price_data,
                                 analyst, insider, short)
    catalysts  = _catalysts(ticker)

    analysis = {
        "catalysts": catalysts,
        "price":    price_data,
        "dcf":      dcf,
        "multiples":multiples,
        "health":   health,
        "analyst":  analyst,
        "insider":  insider,
        "short":    short,
        "earnings": earnings,
        "opinion":  opinion,
    }

    # Console
    print_report(ticker, data, analysis)

    # HTML
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"stock_analysis_{ticker}_{ts}.html"
    html     = generate_html(ticker, data, analysis)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✓ HTML report saved → {out_path}\n")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    if args:
        for ticker in args:
            analyse(ticker)
        return

    # Interactive mode
    print("\n  Stock Analyst — enter a ticker to analyse (or 'quit' to exit)")
    while True:
        try:
            raw = input("  Ticker: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw or raw.lower() in ("quit", "exit", "q"):
            break
        for t in raw.replace(",", " ").split():
            analyse(t)


if __name__ == "__main__":
    main()
