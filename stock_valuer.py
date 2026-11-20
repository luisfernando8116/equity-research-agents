"""
Stock Valuation Analyzer
========================
Answers: Is this stock overpriced, fairly priced, or underpriced?

Methods used (institutional best practice):
  1. Discounted Cash Flow (DCF) — 3 scenarios: conservative / base / optimistic
  2. Multi-metric scoring — P/E, Forward P/E, PEG, EV/EBITDA, P/FCF, P/Book
  3. Graham Number — Benjamin Graham's intrinsic value floor
  4. Earnings Yield vs Risk-Free Rate — relative to bonds
  5. Cyclical detection — special handling for memory, energy, mining, etc.

Usage:
  python stock_valuer.py               → analyze default watchlist
  python stock_valuer.py MU            → analyze single ticker
  python stock_valuer.py MU NVDA AAPL  → analyze multiple tickers
"""

import sys
import subprocess

if sys.version_info < (3, 9):
    sys.exit(f"Python 3.9+ required. Running {sys.version}.")

REQUIRED = {"yfinance": "yfinance", "pandas": "pandas",
            "numpy": "numpy", "openpyxl": "openpyxl"}
missing = []
for mod, pkg in REQUIRED.items():
    try:
        __import__(mod)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f"Installing: {', '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

RISK_FREE_RATE      = 0.045   # 10-year US Treasury yield (update periodically)
EQUITY_RISK_PREMIUM = 0.055   # long-run US equity risk premium
TERMINAL_GROWTH     = 0.025   # long-run terminal growth (below GDP)

# Default tickers to analyze when none are passed as arguments
DEFAULT_TICKERS = ["MU", "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "GH"]

# Sector median valuation benchmarks (approximate current market levels)
# Update these periodically as market valuations shift
SECTOR_BENCHMARKS = {
    "Technology":             {"pe": 32, "fwd_pe": 27, "peg": 1.8, "ev_ebitda": 22, "pfcf": 28, "pb": 7.0},
    "Communication Services": {"pe": 22, "fwd_pe": 19, "peg": 1.5, "ev_ebitda": 14, "pfcf": 20, "pb": 3.5},
    "Consumer Discretionary": {"pe": 25, "fwd_pe": 22, "peg": 1.9, "ev_ebitda": 16, "pfcf": 22, "pb": 6.0},
    "Consumer Staples":       {"pe": 22, "fwd_pe": 20, "peg": 2.5, "ev_ebitda": 15, "pfcf": 20, "pb": 5.0},
    "Energy":                 {"pe": 13, "fwd_pe": 12, "peg": 1.0, "ev_ebitda":  7, "pfcf": 11, "pb": 2.0},
    "Financials":             {"pe": 15, "fwd_pe": 13, "peg": 1.5, "ev_ebitda": 11, "pfcf": 14, "pb": 1.5},
    "Healthcare":             {"pe": 22, "fwd_pe": 18, "peg": 1.6, "ev_ebitda": 14, "pfcf": 18, "pb": 3.0},
    "Industrials":            {"pe": 22, "fwd_pe": 19, "peg": 1.8, "ev_ebitda": 14, "pfcf": 19, "pb": 4.0},
    "Basic Materials":        {"pe": 18, "fwd_pe": 15, "peg": 1.5, "ev_ebitda": 10, "pfcf": 16, "pb": 2.0},
    "Real Estate":            {"pe": 35, "fwd_pe": 30, "peg": 2.5, "ev_ebitda": 20, "pfcf": 25, "pb": 2.0},
    "Utilities":              {"pe": 20, "fwd_pe": 18, "peg": 2.2, "ev_ebitda": 11, "pfcf": 17, "pb": 1.8},
    "Default":                {"pe": 22, "fwd_pe": 19, "peg": 1.8, "ev_ebitda": 14, "pfcf": 20, "pb": 3.0},
}

# Industries where earnings cycle wildly — P/E alone is misleading
CYCLICAL_INDUSTRIES = {
    "Semiconductors", "Semiconductor Memory", "Semiconductor Equipment & Materials",
    "Oil & Gas Integrated", "Oil & Gas E&P", "Oil & Gas Refining & Marketing",
    "Steel", "Aluminum", "Copper", "Gold", "Silver", "Coal",
    "Chemicals", "Specialty Chemicals", "Agricultural Chemicals",
    "Auto Manufacturers", "Auto Parts", "Airlines", "Hotels & Motels", "Resorts & Casinos",
    "Paper & Paper Products", "Lumber & Wood Production",
}

# Growth stocks — P/E less useful; FCF growth and TAM matter more
GROWTH_INDUSTRIES = {
    "Software—Application", "Software—Infrastructure", "Internet Content & Information",
    "Health Information Services", "Biotechnology", "Diagnostics & Research",
    "Electronic Components", "Computer Hardware",
}

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_EXCEL = f"valuation_report_{ts}.xlsx"


# ─── DATA FETCHING ────────────────────────────────────────────────────────────

def fetch_ticker_data(ticker: str) -> dict:
    """Fetch comprehensive fundamental data from yfinance."""
    t = yf.Ticker(ticker)
    info = t.info

    # Also try to get trailing FCF from cash flow statement for robustness
    try:
        cf = t.cashflow
        if cf is not None and not cf.empty:
            fcf_row = None
            for row_name in ["Free Cash Flow", "freeCashFlow"]:
                if row_name in cf.index:
                    fcf_row = cf.loc[row_name]
                    break
            if fcf_row is not None:
                # Use most recent annual FCF
                fcf_from_stmt = float(fcf_row.iloc[0]) if not pd.isna(fcf_row.iloc[0]) else None
                if fcf_from_stmt and "freeCashflow" not in info:
                    info["freeCashflow_stmt"] = fcf_from_stmt
    except Exception:
        pass

    return info


# ─── DCF ENGINE ───────────────────────────────────────────────────────────────

def estimate_wacc(info: dict) -> float:
    """Estimate WACC using CAPM: Rf + Beta × ERP."""
    beta = info.get("beta")
    if beta is None or beta <= 0:
        beta = 1.2   # market average fallback
    beta = min(beta, 3.0)   # cap extreme betas
    return RISK_FREE_RATE + beta * EQUITY_RISK_PREMIUM


def run_dcf(info: dict, growth_rate: float) -> dict | None:
    """
    Two-stage DCF model:
      Phase 1 (Yr 1-5):  grow FCF at growth_rate
      Phase 2 (Yr 6-10): grow FCF at growth_rate × 0.5 (slowdown)
      Terminal:          grow FCF at TERMINAL_GROWTH in perpetuity

    Returns intrinsic value per share and upside/downside vs current price.
    """
    # FCF — prefer direct field, fall back to cash flow statement
    fcf = info.get("freeCashflow") or info.get("freeCashflow_stmt")
    if not fcf or fcf <= 0:
        # Last resort: use net income (lower quality but better than nothing)
        fcf = info.get("netIncomeToCommon")
    if not fcf or fcf <= 0:
        return None   # cannot run DCF without positive cash flow

    wacc = estimate_wacc(info)
    if wacc <= TERMINAL_GROWTH:
        wacc = TERMINAL_GROWTH + 0.03   # prevent division by zero

    pv = 0.0
    current_fcf = float(fcf)
    g1 = growth_rate
    g2 = growth_rate * 0.5

    # Phase 1: years 1–5
    for year in range(1, 6):
        current_fcf *= (1 + g1)
        pv += current_fcf / (1 + wacc) ** year

    # Phase 2: years 6–10
    for year in range(6, 11):
        current_fcf *= (1 + g2)
        pv += current_fcf / (1 + wacc) ** year

    # Terminal value (Gordon Growth Model)
    terminal_fcf = current_fcf * (1 + TERMINAL_GROWTH)
    terminal_value = terminal_fcf / (wacc - TERMINAL_GROWTH)
    pv_terminal = terminal_value / (1 + wacc) ** 10

    enterprise_value = pv + pv_terminal

    # Bridge to equity value: EV − Net Debt
    total_debt = info.get("totalDebt") or 0.0
    total_cash = info.get("totalCash") or 0.0
    equity_value = enterprise_value - float(total_debt) + float(total_cash)

    if equity_value <= 0:
        return None

    shares = (info.get("sharesOutstanding")
              or info.get("impliedSharesOutstanding"))
    if not shares or float(shares) <= 0:
        return None

    intrinsic = equity_value / float(shares)

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if not price:
        return None

    upside = (intrinsic - float(price)) / float(price)

    return {
        "intrinsic_value": round(intrinsic, 2),
        "current_price":   round(float(price), 2),
        "upside_pct":      round(upside * 100, 1),
        "growth_rate":     round(growth_rate * 100, 1),
        "wacc_pct":        round(wacc * 100, 1),
        "signal":          ("UNDERVALUED" if upside > 0.15
                            else "OVERVALUED"  if upside < -0.15
                            else "FAIR VALUE"),
    }


def build_growth_rate(info: dict) -> float:
    """
    Estimate base-case FCF growth rate from available data.
    Uses a blend of revenue growth and earnings growth, capped reasonably.
    """
    rev_g  = info.get("revenueGrowth")  or 0.0
    earn_g = info.get("earningsGrowth") or 0.0
    eps_fwd = info.get("forwardEps")
    eps_ttm = info.get("trailingEps")

    implied_g = 0.0
    if eps_fwd and eps_ttm and float(eps_ttm) > 0:
        implied_g = (float(eps_fwd) / float(eps_ttm)) - 1.0

    rates = [r for r in [rev_g, earn_g, implied_g] if r != 0.0]
    g_base = float(np.mean(rates)) if rates else 0.08   # default 8%

    # Sanity bounds: 0% to 40%
    return max(0.0, min(0.40, g_base))


# ─── MULTIPLES SCORING ────────────────────────────────────────────────────────

def _score(value, benchmark, lower_is_better: bool = True) -> tuple[str, str]:
    """
    Returns (signal, explanation) for one metric.
    signal: ATTRACTIVE / NEUTRAL / EXPENSIVE
    """
    if value is None:
        return "N/A", "no data"

    ratio = value / benchmark if benchmark else 1.0

    if lower_is_better:
        if ratio < 0.75:
            return "ATTRACTIVE", f"{value:.1f}x vs sector {benchmark:.1f}x  (-{(1-ratio)*100:.0f}%)"
        elif ratio < 1.15:
            return "NEUTRAL",    f"{value:.1f}x vs sector {benchmark:.1f}x  (±fair)"
        else:
            return "EXPENSIVE",  f"{value:.1f}x vs sector {benchmark:.1f}x  (+{(ratio-1)*100:.0f}%)"
    else:
        # Higher is better (e.g., earnings yield)
        if ratio > 1.25:
            return "ATTRACTIVE", f"{value:.2%} vs benchmark {benchmark:.2%}"
        elif ratio > 0.85:
            return "NEUTRAL",    f"{value:.2%} vs benchmark {benchmark:.2%}"
        else:
            return "EXPENSIVE",  f"{value:.2%} vs benchmark {benchmark:.2%}"


def score_multiples(info: dict) -> dict:
    """Score each valuation multiple vs sector benchmark."""
    sector = info.get("sector", "Default")
    bench  = SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS["Default"])

    market_cap = info.get("marketCap")
    fcf        = info.get("freeCashflow") or info.get("freeCashflow_stmt")
    pfcf       = (float(market_cap) / float(fcf)
                  if market_cap and fcf and float(fcf) > 0 else None)

    pe_t  = info.get("trailingPE")
    pe_f  = info.get("forwardPE")
    peg   = info.get("pegRatio")
    ev_eb = info.get("enterpriseToEbitda")
    pb    = info.get("priceToBook")

    # Clamp obviously bad data (negative P/E due to loss, extreme outliers)
    pe_t  = pe_t  if pe_t  and 0 < pe_t  < 500 else None
    pe_f  = pe_f  if pe_f  and 0 < pe_f  < 500 else None
    peg   = peg   if peg   and 0 < peg   < 20  else None
    ev_eb = ev_eb if ev_eb and 0 < ev_eb < 200 else None
    pb    = pb    if pb    and 0 < pb    < 100 else None
    pfcf  = pfcf  if pfcf  and 0 < pfcf  < 500 else None

    # Earnings yield vs risk-free rate
    ey = (1.0 / pe_t) if pe_t else None
    ey_signal, ey_note = _score(ey, RISK_FREE_RATE, lower_is_better=False) if ey else ("N/A", "no data")

    scores = {
        "Trailing P/E":       _score(pe_t,  bench["pe"],       True),
        "Forward P/E":        _score(pe_f,  bench["fwd_pe"],   True),
        "PEG Ratio":          _score(peg,   bench["peg"],      True),
        "EV/EBITDA":          _score(ev_eb, bench["ev_ebitda"],True),
        "Price/Free CF":      _score(pfcf,  bench["pfcf"],     True),
        "Price/Book":         _score(pb,    bench["pb"],       True),
        "Earnings Yield vs RF": (ey_signal, ey_note),
    }
    return scores


def graham_number(info: dict) -> dict | None:
    """
    Graham Number = √(22.5 × EPS × Book Value Per Share)
    Ben Graham's maximum price for a defensive stock.
    Designed for stable, asset-backed value stocks — use as reference only for growth/tech.
    """
    eps  = info.get("trailingEps")
    bvps = info.get("bookValuePerShare")
    if not eps or not bvps or float(eps) <= 0 or float(bvps) <= 0:
        return None

    gn    = (22.5 * float(eps) * float(bvps)) ** 0.5
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if not price:
        return None

    premium = (float(price) - gn) / gn
    return {
        "graham_number": round(gn, 2),
        "current_price": round(float(price), 2),
        "premium_to_graham": round(premium * 100, 1),
        "signal": ("BELOW GRAHAM (attractive)" if premium < -0.15
                   else "ABOVE GRAHAM (expensive by value standards)" if premium > 0.15
                   else "NEAR GRAHAM"),
    }


# ─── STOCK CLASSIFIER ─────────────────────────────────────────────────────────

def classify_stock(info: dict) -> str:
    """Categorize stock as Cyclical / Growth / Dividend / Value / Blend."""
    industry  = info.get("industry", "")
    dy        = (info.get("dividendYield") or 0.0) / 100   # yfinance reports percent
    pe_f      = info.get("forwardPE")
    earn_g    = info.get("earningsGrowth") or 0.0

    if industry in CYCLICAL_INDUSTRIES:
        return "Cyclical"
    if float(dy) > 0.035:
        return "Dividend Income"
    if float(earn_g) > 0.20 or (pe_f and float(pe_f) > 35):
        return "High Growth"
    if pe_f and float(pe_f) < 14:
        return "Value"
    if industry in GROWTH_INDUSTRIES:
        return "Growth"
    return "Blend"


CYCLICAL_ADVICE = {
    "Semiconductor Memory": (
        "DRAM/NAND memory is one of the most cyclical businesses in tech. "
        "Earnings swing from massive losses to record profits across the supply/demand cycle. "
        "AVOID over-relying on trailing P/E — it reflects cycle position, not business quality. "
        "PREFER: EV/EBITDA, Price/Book, and FCF yield. "
        "Current tailwind: AI workloads (especially HBM) are driving memory demand. "
        "Key risk: oversupply cycles return quickly when manufacturers expand capacity."
    ),
    "Semiconductors": (
        "Semiconductor equipment and chips are cyclical with 3–5 year cycles tied to capex spending. "
        "PREFER: EV/EBITDA and Price/Book over trailing P/E. "
        "AI buildout is a structural tailwind, but cyclical downturns still happen."
    ),
    "Oil & Gas E&P": (
        "Oil & gas earnings depend heavily on commodity prices. P/E at cycle peaks looks cheap; "
        "at troughs looks terrible — neither reflects true value. "
        "PREFER: EV/EBITDA, FCF yield, and NAV (Net Asset Value) per share."
    ),
    "Default": (
        "This industry shows earnings cyclicality. Trailing P/E can mislead — "
        "favor EV/EBITDA and Price/Book as more stable metrics across the cycle."
    ),
}


# ─── MASTER VALUATION FUNCTION ────────────────────────────────────────────────

def full_valuation(ticker: str) -> dict:
    """Run complete valuation analysis on a single ticker."""
    print(f"  Analyzing {ticker}...", end=" ", flush=True)

    try:
        info = fetch_ticker_data(ticker)
    except Exception as e:
        print(f"ERROR ({e})")
        return {"ticker": ticker, "error": str(e)}

    name    = info.get("shortName") or info.get("longName") or ticker
    price   = info.get("currentPrice") or info.get("regularMarketPrice")
    mcap    = info.get("marketCap")
    sector  = info.get("sector", "Unknown")
    industry = info.get("industry", "Unknown")

    if not price:
        print("no price data")
        return {"ticker": ticker, "error": "No price data available"}

    stock_type  = classify_stock(info)
    is_cyclical = industry in CYCLICAL_INDUSTRIES

    # DCF — three scenarios
    g_base = build_growth_rate(info)
    dcf_conservative = run_dcf(info, max(0.01, g_base * 0.5))
    dcf_base         = run_dcf(info, g_base)
    dcf_optimistic   = run_dcf(info, min(0.40, g_base * 1.5))

    # Multiples scoring
    multiples = score_multiples(info)

    # Graham Number
    gn = graham_number(info)

    # Overall verdict
    # Collect all definitive signals
    signals = []
    for _, (sig, _) in multiples.items():
        if sig == "ATTRACTIVE":
            signals.append(1)
        elif sig == "EXPENSIVE":
            signals.append(-1)
        elif sig == "NEUTRAL":
            signals.append(0)

    if dcf_base:
        if dcf_base["signal"] == "UNDERVALUED":
            signals.extend([1, 1])   # weight DCF more heavily
        elif dcf_base["signal"] == "OVERVALUED":
            signals.extend([-1, -1])
        else:
            signals.append(0)

    composite = float(np.mean(signals)) if signals else 0.0

    if composite > 0.20:
        verdict = "UNDERVALUED"
    elif composite < -0.20:
        verdict = "OVERVALUED"
    else:
        verdict = "FAIR VALUE"

    # Extra context metrics
    roe   = info.get("returnOnEquity")
    roa   = info.get("returnOnAssets")
    gm    = info.get("grossMargins")
    om    = info.get("operatingMargins")
    de    = info.get("debtToEquity")
    de    = de / 100 if de is not None else None                # yfinance reports ×100
    cr    = info.get("currentRatio")
    dy    = (info.get("dividendYield") or 0.0) / 100           # yfinance reports percent
    rev_g = info.get("revenueGrowth")
    earn_g = info.get("earningsGrowth")

    # Buy price range derived from 3-scenario DCF
    buy_below   = dcf_conservative["intrinsic_value"] if dcf_conservative else None
    fair_low    = dcf_conservative["intrinsic_value"] if dcf_conservative else None
    fair_high   = dcf_base["intrinsic_value"]         if dcf_base         else None
    target_opti = dcf_optimistic["intrinsic_value"]   if dcf_optimistic   else None

    print(f"done → {verdict}")

    return {
        "ticker":       ticker,
        "name":         name,
        "price":        float(price),
        "market_cap":   float(mcap) if mcap else None,
        "sector":       sector,
        "industry":     industry,
        "stock_type":   stock_type,
        "is_cyclical":  is_cyclical,
        "dcf": {
            "conservative": dcf_conservative,
            "base":         dcf_base,
            "optimistic":   dcf_optimistic,
            "g_base_pct":   round(g_base * 100, 1),
            "wacc_pct":     round(estimate_wacc(info) * 100, 1),
        },
        "buy_range": {
            "strong_buy_below":  buy_below,
            "fair_entry_low":    fair_low,
            "fair_entry_high":   fair_high,
            "optimistic_target": target_opti,
        },
        "multiples":    multiples,
        "composite_score": round(composite, 3),
        "graham":       gn,
        "verdict":      verdict,
        "fundamentals": {
            "ROE":             f"{roe:.1%}"   if roe   else "N/A",
            "ROA":             f"{roa:.1%}"   if roa   else "N/A",
            "Gross Margin":    f"{gm:.1%}"    if gm    else "N/A",
            "Operating Margin":f"{om:.1%}"    if om    else "N/A",
            "Debt/Equity":     f"{de:.1f}x"   if de    else "N/A",
            "Current Ratio":   f"{cr:.2f}"    if cr    else "N/A",
            "Dividend Yield":  f"{dy:.2%}"    if dy    else "—",
            "Revenue Growth":  f"{rev_g:.1%}" if rev_g else "N/A",
            "Earnings Growth": f"{earn_g:.1%}"if earn_g else "N/A",
        },
    }


# ─── REPORTING ────────────────────────────────────────────────────────────────

VERDICT_ICONS = {
    "UNDERVALUED": "★  UNDERVALUED",
    "OVERVALUED":  "⚠  OVERVALUED",
    "FAIR VALUE":  "◆  FAIR VALUE",
}

SIGNAL_ICONS = {
    "ATTRACTIVE": "✓",
    "NEUTRAL":    "–",
    "EXPENSIVE":  "✗",
    "N/A":        "?",
}


def fmt_mcap(v):
    if v is None:
        return "N/A"
    if v >= 1e12:
        return f"${v/1e12:.2f}T"
    if v >= 1e9:
        return f"${v/1e9:.1f}B"
    return f"${v/1e6:.0f}M"


def print_valuation_report(result: dict):
    if "error" in result:
        print(f"\n  {result['ticker']}: ERROR — {result['error']}")
        return

    t    = result["ticker"]
    name = result["name"]
    px   = result["price"]
    mcap = result["market_cap"]
    verdict = result["verdict"]

    bar = "═" * 70
    print(f"\n{bar}")
    print(f"  {t}  —  {name}")
    print(f"  {result['sector']}  |  {result['industry']}  |  Type: {result['stock_type']}")
    print(f"  Current Price: ${px:,.2f}   Market Cap: {fmt_mcap(mcap)}")
    print(bar)

    # ── DCF ──────────────────────────────────────────────────────────────────
    dcf = result["dcf"]
    print(f"\n  ── INTRINSIC VALUE  (Discounted Cash Flow)")
    print(f"     Base growth rate: {dcf['g_base_pct']}%/yr   |   WACC: {dcf['wacc_pct']}%   "
          f"|   Terminal growth: {TERMINAL_GROWTH:.1%}")
    print()

    scenario_order = [("conservative", "Conservative"), ("base", "Base Case "),
                      ("optimistic",   "Optimistic ")]
    for key, label in scenario_order:
        s = dcf[key]
        if s:
            arrow = "▲" if s["upside_pct"] > 0 else "▼"
            print(f"     {label}  (FCF growth {s['growth_rate']:4.1f}%/yr)"
                  f"  →  Intrinsic ${s['intrinsic_value']:>8,.2f}  "
                  f"[{arrow} {s['upside_pct']:+.1f}%]   {s['signal']}")
        else:
            print(f"     {label}  →  N/A (insufficient cash flow data)")

    # ── Multiples ─────────────────────────────────────────────────────────────
    print(f"\n  ── MULTIPLE SCORING  (vs {result['sector']} sector median)")
    print(f"     {'Metric':<22}  {'Signal':<11}  Detail")
    print(f"     {'─'*20}  {'─'*10}  {'─'*34}")
    attract, expensive, neutral, na = 0, 0, 0, 0
    for metric, (sig, note) in result["multiples"].items():
        icon = SIGNAL_ICONS.get(sig, "?")
        print(f"     {metric:<22}  {icon} {sig:<9}  {note}")
        if sig == "ATTRACTIVE": attract += 1
        elif sig == "EXPENSIVE": expensive += 1
        elif sig == "NEUTRAL":   neutral += 1
        else:                    na += 1

    total_scored = attract + expensive + neutral
    if total_scored > 0:
        print(f"\n     Multiples summary: "
              f"{attract} attractive  |  {neutral} neutral  |  {expensive} expensive"
              f"  ({na} missing data)")

    # ── Graham Number ─────────────────────────────────────────────────────────
    gn = result.get("graham")
    if gn:
        print(f"\n  ── GRAHAM NUMBER  (Benjamin Graham's intrinsic floor)")
        print(f"     Graham Number: ${gn['graham_number']:,.2f}   "
              f"Current: ${gn['current_price']:,.2f}   "
              f"Premium: {gn['premium_to_graham']:+.1f}%   →  {gn['signal']}")
        if result["stock_type"] in ("High Growth", "Growth"):
            print(f"     ⚠  Graham Number applies to stable value stocks — "
                  f"use as a floor reference only for {result['stock_type']} companies")

    # ── Fundamentals ──────────────────────────────────────────────────────────
    print(f"\n  ── BUSINESS QUALITY")
    fd = result["fundamentals"]
    row1 = (f"     ROE {fd['ROE']:<8}  ROA {fd['ROA']:<8}  "
            f"Gross Margin {fd['Gross Margin']:<8}  Op. Margin {fd['Operating Margin']}")
    row2 = (f"     Debt/Equity {fd['Debt/Equity']:<8}  "
            f"Current Ratio {fd['Current Ratio']:<6}  "
            f"Rev Growth {fd['Revenue Growth']:<8}  "
            f"Earnings Growth {fd['Earnings Growth']}")
    print(row1)
    print(row2)

    # ── Cyclical / Growth Context ─────────────────────────────────────────────
    if result["is_cyclical"]:
        advice = (CYCLICAL_ADVICE.get(result["industry"])
                  or CYCLICAL_ADVICE["Default"])
        print(f"\n  ── CYCLICAL STOCK ALERT  ⚡")
        # Word-wrap to 66 chars
        words = advice.split()
        line, lines = "     ", []
        for word in words:
            if len(line) + len(word) + 1 > 70:
                lines.append(line)
                line = "     " + word
            else:
                line += (" " if line.strip() else "") + word
        if line.strip():
            lines.append(line)
        for l in lines:
            print(l)

    # ── Overall Verdict ───────────────────────────────────────────────────────
    icon_label = VERDICT_ICONS.get(verdict, verdict)
    print(f"\n  ── OVERALL VERDICT")
    print(f"     {icon_label}   (composite score: {result['composite_score']:+.2f})")
    dcf_b = dcf["base"]
    if dcf_b:
        print(f"     DCF base case: ${dcf_b['intrinsic_value']:,.2f} intrinsic  "
              f"vs ${dcf_b['current_price']:,.2f} current  "
              f"→  {dcf_b['upside_pct']:+.1f}% upside")

    if verdict == "OVERVALUED":
        print(f"     Interpretation: Current price appears to price in "
              f"very optimistic assumptions.")
        print(f"     Consider: wait for a pullback, or review the optimistic DCF scenario.")
    elif verdict == "UNDERVALUED":
        print(f"     Interpretation: Multiple metrics suggest price is below fair value.")
        print(f"     Consider: current price may offer a margin of safety.")
    else:
        print(f"     Interpretation: Price appears in line with fundamentals.")
        print(f"     Consider: monitor for a better entry point or catalyst.")

    # ── Price Target / Buy Range ──────────────────────────────────────────────
    br = result.get("buy_range", {})
    fl  = br.get("fair_entry_low")
    fh  = br.get("fair_entry_high")
    ot  = br.get("optimistic_target")
    if fl or fh or ot:
        print(f"\n  ── PRICE RECOMMENDATION  (based on DCF scenarios vs ${px:,.2f} today)")
        if fl and fh:
            status = ("BELOW fair range — looks attractive" if px < fl else
                      "ABOVE fair range — expensive relative to base DCF" if px > fh else
                      "WITHIN fair range")
            print(f"     Fair value range:   ${fl:>9,.2f}  —  ${fh:>9,.2f}   ← {status}")
        if ot:
            upside_pct = (ot - px) / px * 100
            print(f"     Optimistic target:  ${ot:>9,.2f}   ({upside_pct:+.1f}% upside from today)")
        if fl:
            pct_above = (px - fl) / fl * 100
            if pct_above > 0:
                print(f"     Entry note: trading {pct_above:.1f}% above conservative DCF — "
                      f"limited margin of safety at current price")
            else:
                print(f"     Entry note: trading {abs(pct_above):.1f}% below conservative DCF — "
                      f"strong margin of safety")

    print(f"{bar}\n")


# ─── EXCEL EXPORT ─────────────────────────────────────────────────────────────

def export_excel(results: list[dict]):
    """Export all valuation results to a structured Excel workbook."""
    from openpyxl.styles import Font, PatternFill, Alignment

    rows = []
    dcf_rows = []

    for r in results:
        if "error" in r:
            continue

        multiples = r["multiples"]

        def mv(key, idx=0):
            """Get metric value string from multiples dict."""
            return multiples.get(key, ("N/A", ""))[idx]

        rows.append({
            "Ticker":          r["ticker"],
            "Company":         r["name"],
            "Price ($)":       r["price"],
            "Market Cap":      fmt_mcap(r["market_cap"]),
            "Sector":          r["sector"],
            "Industry":        r["industry"],
            "Type":            r["stock_type"],
            "VERDICT":         r["verdict"],
            "Composite Score": r["composite_score"],
            "DCF Base ($)":         (r["dcf"]["base"]["intrinsic_value"]
                                     if r["dcf"]["base"] else "N/A"),
            "DCF Upside %":         (r["dcf"]["base"]["upside_pct"]
                                     if r["dcf"]["base"] else "N/A"),
            "Fair Range Low ($)":   (r["buy_range"]["fair_entry_low"]
                                     if r.get("buy_range", {}).get("fair_entry_low") else "N/A"),
            "Fair Range High ($)":  (r["buy_range"]["fair_entry_high"]
                                     if r.get("buy_range", {}).get("fair_entry_high") else "N/A"),
            "Optimistic Target ($)":(r["buy_range"]["optimistic_target"]
                                     if r.get("buy_range", {}).get("optimistic_target") else "N/A"),
            "FCF Growth %":    r["dcf"]["g_base_pct"],
            "WACC %":          r["dcf"]["wacc_pct"],
            "Trailing P/E":    mv("Trailing P/E"),
            "Forward P/E":     mv("Forward P/E"),
            "PEG Ratio":       mv("PEG Ratio"),
            "EV/EBITDA":       mv("EV/EBITDA"),
            "P/FCF":           mv("Price/Free CF"),
            "P/Book":          mv("Price/Book"),
            "Graham #":        (r["graham"]["graham_number"]
                                if r.get("graham") else "N/A"),
            "Graham Premium %": (r["graham"]["premium_to_graham"]
                                 if r.get("graham") else "N/A"),
            "ROE":             r["fundamentals"]["ROE"],
            "Gross Margin":    r["fundamentals"]["Gross Margin"],
            "Rev Growth":      r["fundamentals"]["Revenue Growth"],
            "Earnings Growth": r["fundamentals"]["Earnings Growth"],
            "Debt/Equity":     r["fundamentals"]["Debt/Equity"],
            "Dividend Yield":  r["fundamentals"]["Dividend Yield"],
        })

        for scenario, label in [("conservative", "Conservative"),
                                 ("base",         "Base Case"),
                                 ("optimistic",   "Optimistic")]:
            s = r["dcf"][scenario]
            if s:
                dcf_rows.append({
                    "Ticker":          r["ticker"],
                    "Company":         r["name"],
                    "Scenario":        label,
                    "FCF Growth %":    s["growth_rate"],
                    "WACC %":          s["wacc_pct"],
                    "Intrinsic ($)":   s["intrinsic_value"],
                    "Current Price ($)": s["current_price"],
                    "Upside %":        s["upside_pct"],
                    "Signal":          s["signal"],
                })

    if not rows:
        print("  No valid results to export.")
        return

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:

        # Sheet 1: Summary
        df_summary = pd.DataFrame(rows)
        df_summary.to_excel(writer, sheet_name="Valuation_Summary", index=False)

        ws = writer.sheets["Valuation_Summary"]
        verdict_colors = {
            "UNDERVALUED": "1B5E20",   # dark green
            "OVERVALUED":  "B71C1C",   # dark red
            "FAIR VALUE":  "1A237E",   # dark blue
        }
        for row_idx in range(2, ws.max_row + 1):
            verdict_cell = ws.cell(row=row_idx, column=8)
            verdict = verdict_cell.value
            color   = verdict_colors.get(verdict, "2A2D3E")
            verdict_cell.fill = PatternFill("solid", fgColor=color)
            verdict_cell.font = Font(bold=True, color="FFFFFF")

        # Sheet 2: DCF Scenarios
        if dcf_rows:
            pd.DataFrame(dcf_rows).to_excel(writer, sheet_name="DCF_Scenarios", index=False)

        # Sheet 3: Multiple Signals Detail
        sig_rows = []
        for r in results:
            if "error" in r:
                continue
            for metric, (sig, note) in r["multiples"].items():
                sig_rows.append({
                    "Ticker":  r["ticker"],
                    "Company": r["name"],
                    "Metric":  metric,
                    "Signal":  sig,
                    "Detail":  note,
                })
        if sig_rows:
            pd.DataFrame(sig_rows).to_excel(writer, sheet_name="Multiple_Signals", index=False)

    print(f"\n  ✓ Excel saved: {OUTPUT_EXCEL}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # Parse tickers from command line or use defaults
    tickers = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TICKERS
    tickers = [t.upper().strip() for t in tickers]

    print("=" * 70)
    print("STOCK VALUATION ANALYZER")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Method:  DCF (3 scenarios) + 7 multiples + Graham Number")
    print(f"RF Rate: {RISK_FREE_RATE:.1%}  |  ERP: {EQUITY_RISK_PREMIUM:.1%}  "
          f"|  Terminal Growth: {TERMINAL_GROWTH:.1%}")
    print("=" * 70)
    print()

    results = []
    for ticker in tickers:
        result = full_valuation(ticker)
        results.append(result)

    print("\n" + "=" * 70)
    print("VALUATION REPORTS")
    print("=" * 70)

    for result in results:
        print_valuation_report(result)

    # Summary table
    valid = [r for r in results if "error" not in r]
    if len(valid) > 1:
        print("=" * 70)
        print("SUMMARY TABLE")
        print("=" * 70)
        print(f"  {'Ticker':<7}  {'Price':>8}  {'DCF Base':>9}  "
              f"{'Upside':>8}  {'Multiples':>16}  {'Verdict'}")
        print(f"  {'─'*6}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*16}  {'─'*14}")
        for r in valid:
            dcf_b = r["dcf"]["base"]
            dcf_str = (f"${dcf_b['intrinsic_value']:>8,.2f}  {dcf_b['upside_pct']:>+7.1f}%"
                       if dcf_b else f"{'N/A':>9}  {'N/A':>8}")
            attract = sum(1 for _, (s, _) in r["multiples"].items() if s == "ATTRACTIVE")
            expensive = sum(1 for _, (s, _) in r["multiples"].items() if s == "EXPENSIVE")
            mult_str = f"{attract} attract / {expensive} expensive"
            print(f"  {r['ticker']:<7}  ${r['price']:>7,.2f}  {dcf_str}  "
                  f"{mult_str:>16}  {r['verdict']}")

    export_excel(results)

    print("\n⚠  Disclaimer: Valuations are model-based estimates, not financial advice.")
    print("   DCF is highly sensitive to growth rate assumptions — treat as a range.")
    print("   For cyclicals (MU, energy, mining): favor EV/EBITDA and P/Book over P/E.")


if __name__ == "__main__":
    main()
