"""
Financial Portfolio Analyzer — Advanced Edition
================================================
Institutional-grade features on top of mean-variance optimization:
  • Black-Litterman expected return stabilization (no-views equilibrium blended with sample)
  • Multi-factor pre-screening: momentum 40% + quality (ROE) 25% + value (E/P) 20% + catalyst 15%
  • International diversification — Global Growth enforces ≥15% in international ETFs
  • CVaR (Expected Shortfall) and Sortino ratio alongside Sharpe
  • True walk-forward backtest: train on first 2yr, evaluate on last 1yr vs SPY
  • Historical stress tests: 2008 / 2020 COVID / 2022 rate shock / 1987 — beta-adjusted
  • Market regime detection via SPY 200-day moving average
  • Rebalancing drift alerts: flags positions that drifted ≥4% from target
  • Factor exposure report: weighted P/E, ROE, momentum tilt per portfolio

All three scenarios maintain an aggressive, high-upside orientation.
"""

import sys
import subprocess

if sys.version_info < (3, 9):
    sys.exit(f"Python 3.9 or higher required. Running {sys.version}.")

# Force UTF-8 output on Windows so Unicode symbols (★ ⚠ ≥ → ✓) don't crash cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REQUIRED = {
    "yfinance": "yfinance", "pandas": "pandas", "numpy": "numpy",
    "matplotlib": "matplotlib", "scipy": "scipy", "openpyxl": "openpyxl",
    "requests": "requests", "bs4": "beautifulsoup4", "networkx": "networkx",
}
missing = []
for module, package in REQUIRED.items():
    try:
        __import__(module)
    except ImportError:
        missing.append(package)
if missing:
    print(f"Installing: {', '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    print("Done.\n")

import math
import os
import warnings
warnings.filterwarnings("ignore")

# Knowledge Graph — imported after REQUIRED check so networkx is guaranteed available
try:
    from knowledge_graph import FinancialKnowledgeGraph as _KGClass
    _KG_AVAILABLE = True
except ImportError as _kg_err:
    _KG_AVAILABLE = False

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

INITIAL_INVESTMENT = 15_000
START_DATE = (datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
END_DATE   = datetime.now().strftime("%Y-%m-%d")
RISK_FREE_RATE   = 0.05   # 10-yr Treasury — DCF discount rate & CAPM. Refreshed from FRED each run.
CASH_RATE        = 0.045  # 3-month T-bill — Sharpe / Sortino hurdle.  Refreshed from FRED each run.
PROJECTION_YEARS = 15
MC_SIMULATIONS   = 1_000
TOP_N_FACTORS    = 100    # default tickers to carry into optimization
API_TIER_N       = 250    # names (by factor composite) that get news/FDA/SEC/valuation checks
MAX_HOLDINGS     = 15     # max positions in the final recommended portfolio (post-optimization trim)
# ── Structural / competitive risks ───────────────────────────────────────────
# External threats a price-and-fundamentals model can't see (new state-backed
# competitors, regulation, technology shifts). `severity` (0–1) is YOUR judgment,
# like a BL view's confidence. It is escalated automatically (+0.1 per recent
# headline mentioning a watch term, max +0.3) and then:
#   • lowers the factor-screen score by up to 0.15 × severity
#   • lowers BL expected return by STRUCTURAL_MU_HAIRCUT × severity
#   • shows as the risk note on the position's rationale card
STRUCTURAL_RISKS: list[dict] = [
    {
        "name":        "Chinese memory oversupply",
        "tickers":     {"SNDK", "MU"},
        "severity":    0.5,    # STARTING VALUE — raised by the investor (Sep 2026); confirm/adjust
        "watch_terms": ["CXMT", "ChangXin", "YMTC", "Yangtze Memory"],
        "note":        "State-backed CXMT (post-IPO) and YMTC adding DRAM/NAND capacity — "
                       "pricing pressure on memory makers after the cycle peak",
    },
]
STRUCTURAL_MU_HAIRCUT = 0.05   # at severity 1.0, −5%/yr on BL expected return

# ── Values-based exclusions (investor preference) ────────────────────────────
# Never recommended or bought; existing holdings are not affected. Defined per
# investor in profiles_local.py (VALUE_EXCLUSIONS); see profiles_example.py.
try:
    from profiles_local import VALUE_EXCLUSIONS as _VX
except ImportError:
    try:
        from profiles_example import VALUE_EXCLUSIONS as _VX
    except ImportError:
        _VX = {}
EXCLUDED_INDUSTRIES = set(_VX.get("industries", ()))   # Yahoo industry names
EXCLUDED_TICKERS    = set(_VX.get("tickers", ()))      # named companies


def excluded_by_preference(ticker: str, info: dict | None = None) -> str | None:
    """Reason a ticker is excluded by the investor's values screen, else None."""
    if ticker in EXCLUDED_TICKERS:
        return "weapons-focused"
    ind = (info or {}).get("industry")
    if ind in EXCLUDED_INDUSTRIES:
        return ind
    return None

# ── Upcoming FDA decisions (PDUFA dates) ─────────────────────────────────────
# Dated regulatory catalysts. There is no free official PDUFA API, so this list
# is maintained by hand — the run prints news headlines mentioning PDUFA /
# priority review as candidates to add. Verify every date against a company
# press release or FDA notice before adding. After the date passes, set
# "outcome" ("approved" / "CRL" / "delayed") — the quality gate reminds you.
UPCOMING_FDA_DECISIONS: list[dict] = [
    {
        "ticker":     "RHHBY",
        "drug":       "giredestrant",
        "indication": "Adjuvant ER+/HER2- early breast cancer (stage I–III); lidERA Ph3: "
                      "−30% risk of invasive recurrence or death",
        "date":       "2026-11-30",
        "review":     "Priority Review",
        "source":     "Roche / FDA NDA acceptance, Jun 2026 (pharmexec.com, rttnews.com)",
        "outcome":    None,
    },
    # RHHBY giredestrant + everolimus (ESR1-mut advanced BC): decision "December 2026" —
    # add once Roche publishes the exact date.
]
FDA_DECISION_WINDOW = 120   # days ahead within which a decision boosts the catalyst score

BIOTECH_MAX_WEIGHT = 0.06 # single-name cap for binary-event biotechs (one trial can halve them);
                          # a ticker with an investor BL view keeps the scenario cap in that
                          # investor's personal re-optimisation
SECTOR_CAP       = 0.35
MIN_HISTORY_DAYS = 200

# Black-Litterman parameters
BL_DELTA = 2.5    # market risk aversion coefficient
BL_ALPHA = 0.40   # weight toward sample returns (0 = pure equilibrium, 1 = pure sample)

# Valuation tilt — DCF/multiples verdict adjusts BL expected returns before optimization
VALUATION_TILT     = True   # set False to skip valuation analysis entirely
VALUATION_TILT_MAG = 0.10   # ±10% mu adjustment for UNDERVALUED / OVERVALUED stocks

SV_EQUITY_RISK_PREMIUM = 0.055
SV_TERMINAL_GROWTH     = 0.025

SV_SECTOR_BENCHMARKS = {
    "Technology":             {"fwd_pe": 27, "ev_ebitda": 22, "pb": 7.0},
    "Communication Services": {"fwd_pe": 19, "ev_ebitda": 14, "pb": 3.5},
    "Consumer Discretionary": {"fwd_pe": 22, "ev_ebitda": 16, "pb": 6.0},
    "Consumer Staples":       {"fwd_pe": 20, "ev_ebitda": 15, "pb": 5.0},
    "Energy":                 {"fwd_pe": 12, "ev_ebitda":  7, "pb": 2.0},
    "Financials":             {"fwd_pe": 13, "ev_ebitda": 11, "pb": 1.5},
    "Healthcare":             {"fwd_pe": 18, "ev_ebitda": 14, "pb": 3.0},
    "Industrials":            {"fwd_pe": 19, "ev_ebitda": 14, "pb": 4.0},
    "Basic Materials":        {"fwd_pe": 15, "ev_ebitda": 10, "pb": 2.0},
    "Real Estate":            {"fwd_pe": 30, "ev_ebitda": 20, "pb": 2.0},
    "Utilities":              {"fwd_pe": 18, "ev_ebitda": 11, "pb": 1.8},
    "Default":                {"fwd_pe": 19, "ev_ebitda": 14, "pb": 3.0},
}
# yfinance uses Morningstar sector names, not GICS — alias so they don't fall to Default
SV_SECTOR_BENCHMARKS["Consumer Cyclical"]  = SV_SECTOR_BENCHMARKS["Consumer Discretionary"]
SV_SECTOR_BENCHMARKS["Consumer Defensive"] = SV_SECTOR_BENCHMARKS["Consumer Staples"]
SV_SECTOR_BENCHMARKS["Financial Services"] = SV_SECTOR_BENCHMARKS["Financials"]

# Boom-bust industries where current cash flow is a poor guide to the next decade.
# Broad "Semiconductors" (NVDA, AVGO) is deliberately excluded — memory and
# equipment are the genuinely cyclical parts of the chip industry.
SV_CYCLICAL_INDUSTRIES = {
    "Semiconductor Memory", "Semiconductor Equipment & Materials",
    "Oil & Gas E&P", "Oil & Gas Integrated", "Oil & Gas Refining & Marketing",
    "Steel", "Aluminum", "Copper", "Auto Manufacturers", "Airlines",
}

# Factor screen weights — actual blend used in multi_factor_screen()
FACTOR_BLEND = {"momentum": 0.35, "quality": 0.25, "value": 0.20,
                "revisions": 0.10, "catalyst": 0.10}   # revisions: see analyst_signals.py

# Momentum Freshness Filter — penalises late-cycle momentum, rewards early-cycle
# Stocks with extreme runs near their 52-week high get a reduced momentum score;
# stocks with recent acceleration from a low base get a bonus.
MF_RUN_THRESHOLD  = 1.50   # 12m raw return above which late-cycle penalty starts (150%)
MF_RUN_PENALTY    = 0.10   # score reduction per 100% excess return above threshold
MF_RANGE_THRESHOLD = 0.75  # 52-week range position (0=low,1=high) above which range penalty starts
MF_RANGE_PENALTY  = 0.40   # penalty magnitude when stock is at top of its 52-week range
MF_ACCEL_CAP      = 0.20   # max momentum acceleration bonus/penalty (±20%)

# Time-decay tiers: (days_from_today_start, days_from_today_end, multiplier)
# Smooth 5-year decay — recent data stays 10× more influential than oldest
DECAY_TIERS = [
    (   0,  252, 5.00),   # last 12 months      (most recent)
    ( 252,  504, 3.00),   # 12–24 months
    ( 504,  756, 1.50),   # 24–36 months        (COVID crash enters here)
    ( 756, 1008, 0.75),   # 36–48 months
    (1008, 1260, 0.50),   # 48–60 months        (oldest, lowest weight)
]

CUSTOM_TICKERS = ["GH", "RVMD", "SPY"]   # SPY required for regime & stress benchmark
# RVMD: Revolution Medicines — daraxonrasib FDA approval for pancreatic cancer (Sep 2026)
#   Pan-RAS(ON) inhibitor; survival doubling in PDAC; platform extends to NSCLC, CRC

INTERNATIONAL_TICKERS = {
    "VEA", "VWO", "EFA", "EEM", "IEMG", "VT", "ACWI",
    "INDA", "FXI", "EWJ", "EWZ", "KWEB", "MCHI",
}

INDEX_TICKERS = {
    "SPY", "QQQ", "VTI", "IWM", "VYM", "SCHD", "DVY", "HDV", "DGRO",
    "XLK", "XLV", "XLF", "XLE", "XLU", "XLI", "XLP", "XLY", "SOXX", "IBB",
    "GLD", "VNQ", "GDX", "JEPI",
}

STRESS_SCENARIOS = [
    ("2008 Financial Crisis", -0.565, 18),
    ("2020 COVID Crash",       -0.340,  1),
    ("2022 Rate Shock",        -0.255, 10),
    ("1987 Black Monday",      -0.226,  1),
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CHART = os.path.join(OUTPUT_DIR, f"advanced_charts_{ts}.png")
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, f"advanced_portfolio_{ts}.xlsx")
OUTPUT_HTML  = os.path.join(OUTPUT_DIR, f"advanced_report_{ts}.html")

# ── Biotech Pipeline Intelligence ─────────────────────────────────────────────
# When True: scores all companies in biotech_pipeline.py BIOTECH_UNIVERSE,
# auto-adds high-scoring tickers to the screening universe, and injects
# pipeline scores into the catalyst signal component.
RUN_BIOTECH_PIPELINE    = True
PIPELINE_AUTO_ADD_SCORE = 0.72   # tickers scoring above this enter the optimizer universe
                                 # (kept equal to biotech_pipeline.PIPELINE_SCORE_THRESHOLD)

# ── Current holdings — fill in to get rebalance-to-target plan ───────────────
# Format: {"TICKER": dollar_value_held}  e.g. {"AAPL": 2000, "MSFT": 1500}
CURRENT_HOLDINGS: dict = {}

# Dollar-Cost Averaging schedule config
DCA_TRANCHES     = 3    # split new capital across this many purchases
DCA_INTERVAL     = 30   # calendar days between tranches

# Earnings proximity warning window (days)
EARNINGS_WARN_DAYS = 30

# Thematic clusters — used to detect concentration beyond GICS sectors
THEMES: dict = {
    "AI & Semiconductors": {
        "NVDA", "AMD", "INTC", "AVGO", "QCOM", "AMAT", "LRCX", "KLAC",
        "MRVL", "MU", "SNPS", "CDNS", "ARM", "SMCI",
    },
    "Cloud & SaaS": {
        "MSFT", "AMZN", "GOOGL", "CRM", "NOW", "SNOW", "DDOG", "HUBS",
        "WDAY", "VEEV", "ZS", "OKTA", "TEAM", "MDB",
    },
    "Cybersecurity": {
        "CRWD", "PANW", "FTNT", "ZS", "OKTA", "S", "CYBR",
    },
    "Biotech & MedTech": {
        "LLY", "ABBV", "MRNA", "BNTX", "REGN", "VRTX", "ISRG", "DXCM",
        "ILMN", "IDXX", "TMO", "DHR",
    },
    "Clean Energy & EV": {
        "TSLA", "ENPH", "SEDG", "FSLR", "NEE", "BEP", "RIVN", "LCID",
        "ON", "PLUG",
    },
    "Consumer Internet": {
        "META", "NFLX", "SNAP", "PINS", "SPOT", "UBER", "LYFT", "ABNB",
        "DASH", "TTD",
    },
    "Financials & Fintech": {
        "JPM", "BAC", "GS", "MS", "V", "MA", "PYPL", "SQ", "AFRM",
        "SOFI", "COIN",
    },
    # ── Emerging / High-Speculation Themes ────────────────────────────────────
    "Space & Defense Tech": {
        "RKLB", "ASTS", "LUNR", "LMT", "NOC", "RTX", "GD", "BA",
        "HWM", "TDG", "KTOS", "RCAT",
    },
    "Nuclear & Next-Gen Power": {
        "CEG", "VST", "NNE", "SMR", "OKLO", "BWXT", "CCJ", "UEC",
        "DNN", "LEU",
    },
    "Quantum Computing": {
        "IONQ", "RGTI", "QUBT", "QBTS", "IBM", "GOOGL", "MSFT",
    },
    "Longevity & Gene Editing": {
        "BEAM", "NTLA", "CRSP", "EDIT", "FATE", "BLUE", "VERV",
        "ARKG",
    },
    "Robotics & Automation": {
        "ISRG", "ABB", "TER", "BRKS", "ONTO", "NNDM", "RBOT",
        "KVYO", "PATH",
    },
    "Digital Assets & Blockchain": {
        "COIN", "MSTR", "MARA", "RIOT", "CLSK", "BTBT", "HUT",
        "HOOD",
    },
}

# ── Universe membership ──────────────────────────────────────────────────────
# Rules-based: S&P 500 + NASDAQ-100 + S&P MidCap 400 (≈950 names). Mid-caps are
# where most "next RVMD/PRAX" names sit before promotion to the S&P 500.
INCLUDE_MIDCAP_400 = True
# The hand-picked thematic list below is a theme bet, not a method — OFF by default.
INCLUDE_EMERGING   = False

# ── Emerging tickers — added directly to universe, bypassing index membership
# These are smaller / newer names not yet in S&P 500 or NASDAQ-100.
# They need ≥ EMERGING_MIN_HISTORY_DAYS of price data to pass the price screen.
# Only used when INCLUDE_EMERGING = True.  VERV removed (acquired, no longer trades).
EMERGING_TICKERS: set = {
    # Space & launch
    "RKLB", "ASTS", "LUNR", "KTOS", "RCAT",
    # Nuclear microreactors / uranium
    "NNE", "SMR", "OKLO", "UEC", "DNN",
    # Quantum computing
    "IONQ", "RGTI", "QUBT", "QBTS",
    # Gene editing / longevity
    "BEAM", "NTLA", "CRSP", "EDIT",
    # Robotics
    "NNDM", "PATH",
    # Digital assets
    "MARA", "RIOT", "CLSK", "HUT",
}

# Emerging tickers use a shorter minimum history requirement (they may be newer)
EMERGING_MIN_HISTORY_DAYS = 90

# ── Investor Profiles ─────────────────────────────────────────────────────────
# Personal profiles and tax settings live in profiles_local.py (git-ignored, never
# uploaded). Without it the analyzer runs on profiles_example.py's fictional data.
# Cloud agents receive the real values through their routine instructions.
try:
    from profiles_local import INVESTOR_PROFILES, TAX_OVERRIDES
    PROFILES_SOURCE = "profiles_local.py"
except ImportError:
    from profiles_example import INVESTOR_PROFILES, TAX_OVERRIDES
    PROFILES_SOURCE = "profiles_example.py (demo data)"

# Three scenarios — all aggressive-oriented
PORTFOLIOS = {
    "GlobalGrowth": {
        "label":        "Global Growth (BL-Enhanced)",
        "description":  "BL-stabilized Sharpe optimization with ≥15% international ETFs",
        "objective":    "max_sharpe",
        "max_weight":   0.15,
        "min_weight":   0.00,
        "int_floor":    0.15,       # minimum 15% in INTERNATIONAL_TICKERS
        "screen_top_n": TOP_N_FACTORS,
        "use_bl":       True,
        "holdings":     {},
        "color":        "#00BCD4",
    },
    "AggressiveGrowth": {
        "label":        "Aggressive Growth",
        "description":  "Max return with momentum + quality + value factor tilt",
        "objective":    "max_return",
        "max_weight":   0.20,
        "min_weight":   0.00,
        "int_floor":    0.00,
        "screen_top_n": TOP_N_FACTORS,
        "use_bl":       False,
        "holdings":     {},
        "color":        "#FF9800",
    },
    "HighConviction": {
        "label":        "High Conviction",
        "description":  "Concentrated max-return portfolio from top-50 factor-ranked names",
        "objective":    "max_return",
        "max_weight":   0.25,
        "min_weight":   0.02,
        "int_floor":    0.00,
        "screen_top_n": 50,         # tighter universe for maximum concentration
        "use_bl":       False,
        "holdings":     {},
        "color":        "#F44336",
    },
}

# Minimal fallback if Wikipedia fetch fails
FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "JPM", "V",
    "MA", "UNH", "XOM", "LLY", "JNJ", "HD", "PG", "COST", "ABBV", "MRK",
    "AMD", "CRM", "ORCL", "NFLX", "ADBE", "NOW", "INTU", "PANW", "AMAT", "MU",
    "QCOM", "LRCX", "KLAC", "MRVL", "SNPS", "CDNS", "TTD", "DDOG", "SNOW", "CRWD",
    "SPY", "QQQ", "VTI", "VEA", "VWO", "EFA", "EEM", "IEMG", "VT", "ACWI", "GH",
]

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def pct_or_na(v, spec: str = "+.1%") -> str:
    """Format a fraction as a percent, or 'N/A' when the value is missing."""
    return "N/A" if v is None else format(v, spec)


def fmt_dollars(x, pos=None):
    if abs(x) >= 1_000_000:
        return f"${x / 1e6:.1f}M"
    if abs(x) >= 1_000:
        return f"${x / 1e3:.0f}K"
    return f"${x:.0f}"


# ─── DATA FETCHING ─────────────────────────────────────────────────────────────

def fetch_universe() -> list[str]:
    import io as _io
    import requests as _req

    tickers = set(FALLBACK_UNIVERSE) | INTERNATIONAL_TICKERS | set(CUSTOM_TICKERS)
    if INCLUDE_EMERGING:
        tickers |= EMERGING_TICKERS
    # Wikipedia blocks the default pandas/lxml user-agent — send a browser UA
    _headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    sp_t, nq_t = [], []
    try:
        resp = _req.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=_headers, timeout=20
        )
        resp.raise_for_status()
        sp500 = pd.read_html(_io.StringIO(resp.text))[0]
        sp_t  = sp500["Symbol"].str.replace(".", "-", regex=False).tolist()
        tickers |= set(sp_t)
    except Exception as e:
        print(f"      S&P 500 fetch failed ({e})")
    # NASDAQ-100 — Nasdaq's own constituent list first. Wikipedia removed the
    # components table from its Nasdaq-100 page, so that scrape silently returned
    # 0 tickers in every run until Oct 2026 (ASML, ARM, MELI… were missing).
    try:
        resp = _req.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100",
                        headers={**_headers, "Accept": "application/json"}, timeout=20)
        resp.raise_for_status()
        rows = ((resp.json().get("data") or {}).get("data") or {}).get("rows") or []
        nq_t = [r["symbol"].replace(".", "-") for r in rows if r.get("symbol")]
    except Exception as e:
        print(f"      Nasdaq.com NASDAQ-100 list failed ({type(e).__name__}) — trying Wikipedia")
    if len(nq_t) < 50:
        try:
            resp = _req.get("https://en.wikipedia.org/wiki/Nasdaq-100",
                            headers=_headers, timeout=20)
            resp.raise_for_status()
            for tbl in pd.read_html(_io.StringIO(resp.text)):
                col = next((c for c in tbl.columns
                            if "ticker" in str(c).lower() or "symbol" in str(c).lower()), None)
                if col is not None and len(tbl) > 50:
                    nq_t = tbl[col].dropna().astype(str).str.replace(".", "-", regex=False).tolist()
                    break
        except Exception as e:
            print(f"      NASDAQ-100 Wikipedia fallback failed ({type(e).__name__})")
    if len(nq_t) < 50:
        print(f"      ⚠ NASDAQ-100 list unavailable — universe is S&P 500 + extras only")
    tickers |= set(nq_t)

    # S&P MidCap 400 — rules-based mid-cap coverage
    mc_t = []
    if INCLUDE_MIDCAP_400:
        try:
            resp = _req.get("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
                            headers=_headers, timeout=20)
            resp.raise_for_status()
            for tbl in pd.read_html(_io.StringIO(resp.text)):
                if "Symbol" in tbl.columns and len(tbl) > 300:
                    mc_t = tbl["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
                    break
        except Exception as e:
            print(f"      S&P 400 fetch failed ({type(e).__name__})")
        if len(mc_t) < 300:
            print(f"      ⚠ S&P MidCap 400 list unavailable — mid-caps not covered this run")
        tickers |= set(mc_t)

    if sp_t or nq_t:
        emg = f" + {len(EMERGING_TICKERS)} emerging" if INCLUDE_EMERGING else ""
        print(f"      Universe: {len(sp_t)} S&P 500 (Wikipedia) + {len(nq_t)} NASDAQ-100 (Nasdaq.com) + "
              f"{len(mc_t)} S&P 400 mid-caps (Wikipedia){emg} → {len(tickers)} unique")
    else:
        print(f"      Wikipedia unavailable -- using fallback universe ({len(tickers)} tickers)")
    return sorted(tickers)


def fetch_prices(tickers: list[str]) -> pd.DataFrame:
    raw = yf.download(tickers, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        return raw["Close"].dropna(how="all")
    return raw.dropna(how="all")


def fetch_backtest_prices(tickers: list[str], years: int = 10):
    """Long price history for the walk-forward only. One batched download returning
      prices      — total-return (split + dividend adjusted): backtest returns
      split_close — split-adjusted only: P/E (dividend adjustment deflates old prices,
                    JPM's 2020 P/E read 7.4 instead of ~8.8)
      splits      — split ratios, to put SEC-filed EPS on the same share basis."""
    start = (pd.Timestamp(END_DATE) - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    raw = yf.download(sorted(set(tickers)), start=start, end=END_DATE,
                      auto_adjust=False, actions=True, progress=False)
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw.dropna(how="all"), raw.dropna(how="all"), pd.DataFrame()
    lvl = raw.columns.get_level_values(0)
    prices = raw["Adj Close" if "Adj Close" in lvl else "Close"].dropna(how="all")
    split_close = raw["Close"].reindex(prices.index)
    splits = raw["Stock Splits"].replace(0, np.nan) if "Stock Splits" in lvl else pd.DataFrame()
    return prices, split_close, splits


def fetch_sector_map(tickers: list[str]) -> dict:
    def _get(t):
        return t, _safe_info(t).get("sector", "Unknown") or "Unknown"
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(pool.map(_get, tickers))


def fetch_current_prices(tickers: list[str]) -> dict:
    def _get(t):
        fi = _safe_attr(t, "fast_info")
        try:
            return t, fi.last_price if fi is not None else None
        except Exception:
            return t, None
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(pool.map(_get, tickers))


def _dividend_yield_fraction(info: dict) -> float:
    """Annual dividend yield as a fraction (0.0205 = 2.05%).
    yfinance >= 0.2.5x reports `dividendYield` in percent (2.05), not a fraction,
    so derive it from dividendRate / price and fall back to the trailing field."""
    rate  = info.get("dividendRate")
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if rate and price:
        return float(rate) / float(price)
    trailing = info.get("trailingAnnualDividendYield")
    if trailing:
        return float(trailing)
    dy = info.get("dividendYield")
    return float(dy) / 100.0 if dy else 0.0


def refresh_market_rates() -> str:
    """
    Pull today's Treasury yields from FRED (keyless CSV endpoint):
      DGS10  → RISK_FREE_RATE (DCF discount rate, CAPM expected return)
      DGS3MO → CASH_RATE      (Sharpe / Sortino hurdle)
    Keeps the hard-coded defaults if FRED is unreachable. Returns a log line.
    """
    import io
    import requests
    global RISK_FREE_RATE, CASH_RATE
    got = {}
    for sid in ("DGS10", "DGS3MO"):
        try:
            r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                             params={"id": sid}, timeout=15)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            vals = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna()
            if len(vals):
                got[sid] = (float(vals.iloc[-1]) / 100, str(df.iloc[vals.index[-1], 0]))
        except Exception:
            pass
    if "DGS10" in got:
        RISK_FREE_RATE = got["DGS10"][0]
    if "DGS3MO" in got:
        CASH_RATE = got["DGS3MO"][0]
    if got:
        asof = max(d for _, d in got.values())
        return (f"FRED rates (as of {asof}): 10-yr {RISK_FREE_RATE:.2%} (DCF/CAPM)  ·  "
                f"3-mo {CASH_RATE:.2%} (Sharpe)")
    return (f"⚠ FRED unreachable — using defaults 10-yr {RISK_FREE_RATE:.2%}, "
            f"3-mo {CASH_RATE:.2%}")


def forward_revenue_growth(ticker: str, ests=None) -> float | None:
    """Analysts' next-fiscal-year revenue growth (yfinance revenue_estimate '+1y',
    else '0y'), with the horizon optimism haircut applied (analyst_signals.debias_growth).
    None when there is no coverage."""
    from analyst_signals import debias_growth
    try:
        df = ests if ests is not None else _safe_attr(ticker, "revenue_estimate")
        if df is None or getattr(df, "empty", True) or "growth" not in df.columns:
            return None
        for period in ("+1y", "0y"):
            if period in df.index and pd.notna(df.loc[period, "growth"]):
                return debias_growth(float(df.loc[period, "growth"]), period)
    except Exception:
        return None
    return None


_FX_CACHE: dict = {}


def fx_rate(from_ccy: str, to_ccy: str) -> float | None:
    """Spot FX rate from Yahoo (e.g. CHFUSD=X). Cached per run."""
    if not from_ccy or not to_ccy or from_ccy == to_ccy:
        return 1.0
    key = f"{from_ccy}{to_ccy}"
    if key not in _FX_CACHE:
        try:
            fi = _call_with_timeout(lambda: yf.Ticker(f"{key}=X").fast_info)
            px = float(fi.last_price) if fi is not None and fi.last_price else None
        except Exception:
            px = None
        _FX_CACHE[key] = px
    return _FX_CACHE[key]


# Company totals — always in the reporting currency on Yahoo
_FX_TOTAL_FIELDS = ("freeCashflow", "operatingCashflow", "totalDebt", "totalCash",
                    "netIncomeToCommon", "totalRevenue", "ebitda", "fcfStatementAvg")
# Per-share fields — Yahoo is INCONSISTENT: Roche's EPS is in CHF, but TSM's
# (TWD filer) is already in USD. Converted only when they match the reporting currency.
_FX_PER_SHARE_FIELDS = ("trailingEps", "forwardEps", "bookValue")


def _per_share_in_reporting_ccy(info: dict, rate: float) -> bool:
    """Decide which currency Yahoo used for per-share fields by comparing EPS with
    net income ÷ shares in each currency (absolute values, so losses work too).
    TSM: 15.43 vs 487 TWD / 14.96 USD → USD.  Roche: 2.41 vs 2.24 CHF / 2.71 USD → CHF.
    Without the data, assume already converted: a missed conversion is ~20% off,
    a double conversion is ~30× off (TSM's P/E went 27 → 883)."""
    import math
    ni, sh, eps = info.get("netIncomeToCommon"), info.get("sharesOutstanding"), info.get("trailingEps")
    try:
        imp = abs(float(ni)) / float(sh)
        e   = abs(float(eps))
        if imp <= 0 or e <= 0:
            return False
        return abs(math.log(e / imp)) < abs(math.log(e / (imp * rate)))
    except (TypeError, ValueError, ZeroDivisionError):
        return False


def normalize_currency(info: dict) -> dict:
    """
    Foreign ADRs (Roche RHHBY, Novartis, Toyota…): Yahoo reports fundamentals in
    the REPORTING currency (financialCurrency, e.g. CHF) but the price in the
    TRADING currency (USD), and its ratios mix the two — Roche's P/E showed 25.4
    instead of 20.9 (CHF→USD 1.21). Returns a copy with money fields converted
    to the trading currency and P/E, P/B, EV/EBITDA recomputed.
    """
    fin, trd = info.get("financialCurrency"), info.get("currency")
    if not fin or not trd or fin == trd:
        return info
    rate = fx_rate(fin, trd)
    if not rate:
        out = dict(info)
        out["fx_warning"] = f"{fin}→{trd} rate unavailable — ratios may mix currencies"
        return out
    out = dict(info)
    per_share = _per_share_in_reporting_ccy(info, rate)   # decide BEFORE converting totals
    fields = _FX_TOTAL_FIELDS + (_FX_PER_SHARE_FIELDS if per_share else ())
    for f in fields:
        if out.get(f) is not None:
            out[f] = float(out[f]) * rate
    px = out.get("currentPrice") or out.get("regularMarketPrice")
    if px:
        if out.get("trailingEps") and out["trailingEps"] > 0:
            out["trailingPE"] = px / out["trailingEps"]
        if out.get("forwardEps") and out["forwardEps"] > 0:
            out["forwardPE"] = px / out["forwardEps"]
        if out.get("bookValue") and out["bookValue"] > 0:
            out["priceToBook"] = px / out["bookValue"]
        sh = out.get("sharesOutstanding")
        if sh and out.get("ebitda") and out["ebitda"] > 0:
            ev = px * float(sh) + float(out.get("totalDebt") or 0) - float(out.get("totalCash") or 0)
            out["enterpriseToEbitda"] = ev / out["ebitda"]
    out["fx_converted"] = (f"{fin}→{trd} @ {rate:.4f}"
                           + ("" if per_share else " (per-share data already in "
                                                   f"{trd})"))
    return out


def fetch_dividend_yields(tickers: list[str]) -> dict:
    def _get(t):
        return t, _dividend_yield_fraction(_safe_info(t))
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(pool.map(_get, tickers))


def fetch_market_caps(tickers: list[str]) -> dict:
    def _get(t):
        fi = _safe_attr(t, "fast_info")
        try:
            mc = fi.market_cap if fi is not None else None
            return t, float(mc) if mc and mc > 0 else None
        except Exception:
            return t, None
    with ThreadPoolExecutor(max_workers=8) as pool:
        raw = dict(pool.map(_get, tickers))
    valid  = [v for v in raw.values() if v is not None]
    median = float(np.median(valid)) if valid else 1e10
    return {t: (v if v is not None else median) for t, v in raw.items()}


import json as _json
import os as _os
import threading as _threading
import time as _time_mod

# ── yfinance rate-limit / cache layer ────────────────────────────────────────
# Yahoo Finance free tier aggressively rate-limits in 2025 with 401 "Invalid Crumb".
# Root cause: many threads each refreshing their own crumb invalidate each other.
#
# Fix strategy:
#   1. Single global session — one crumb, no inter-thread races
#   2. Semaphore(1) — fully serialized live requests (slow but never 401)
#   3. Disk cache for .info (23 h TTL) AND all attrs (4 h TTL)
#      → first run is slow; every subsequent run within the TTL is instant
#   4. Pre-screen to top-200 tickers before API-heavy steps (see main())

import pickle as _pickle

_YF_CACHE_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".yf_cache")
_YF_ATTR_DIR  = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".yf_attr_cache")
_YF_CACHE_TTL = 23 * 3600   # 23 h for .info
_YF_ATTR_TTL  =  4 * 3600   #  4 h for news / calendar / fast_info
_YF_SEM       = _threading.Semaphore(1)   # fully serialized — one crumb, no races
_YF_LOCK      = _threading.Lock()
_YF_SESSION   = None         # single global session shared by all threads
_YF_SENTINEL  = object()     # sentinel for "not in cache"


def _yf_get_session():
    global _YF_SESSION
    with _YF_LOCK:
        if _YF_SESSION is None:
            try:
                from curl_cffi import requests as _cf
                _YF_SESSION = _cf.Session(impersonate="chrome")
            except Exception:
                import requests as _rq
                s = _rq.Session()
                s.headers["User-Agent"] = (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                )
                _YF_SESSION = s
        return _YF_SESSION


def _yf_drop_session():
    global _YF_SESSION
    with _YF_LOCK:
        _YF_SESSION = None


YF_CALL_TIMEOUT = 30   # seconds per Yahoo call. Every call holds the single _YF_SEM,
                       # so one stalled request used to freeze the whole run (Oct 2026: 6 days)


class YahooTimeout(TimeoutError):
    pass


def _call_with_timeout(fn, timeout: float = YF_CALL_TIMEOUT):
    """Run fn() in a daemon thread and give up after `timeout` seconds.
    Daemon (not a ThreadPoolExecutor) so an abandoned, still-stuck request can't
    block interpreter exit either. Re-raises fn's own exceptions."""
    box: dict = {}

    def _run():
        try:
            box["val"] = fn()
        except BaseException as exc:          # noqa: BLE001 — re-raised below
            box["err"] = exc

    th = _threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise YahooTimeout(f"Yahoo call exceeded {timeout:.0f}s")
    if "err" in box:
        raise box["err"]
    return box.get("val")


def _yf_cache_load(ticker: str) -> dict | None:
    path = _os.path.join(_YF_CACHE_DIR, f"{ticker}.json")
    try:
        if _os.path.exists(path) and _time_mod.time() - _os.path.getmtime(path) < _YF_CACHE_TTL:
            with open(path, encoding="utf-8") as fh:
                return _json.load(fh)
    except Exception:
        pass
    return None


def _yf_cache_save(ticker: str, data: dict) -> None:
    try:
        _os.makedirs(_YF_CACHE_DIR, exist_ok=True)
        path = _os.path.join(_YF_CACHE_DIR, f"{ticker}.json")
        safe = {k: v for k, v in data.items()
                if isinstance(v, (str, int, float, bool, type(None)))}
        with open(path, "w", encoding="utf-8") as fh:
            _json.dump(safe, fh)
    except Exception:
        pass


def _is_auth_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(k in msg for k in ("401", "403", "Unauthorized", "Crumb", "crumb",
                                  "Too Many", "Rate limit", "rate limit"))


def _safe_info(ticker: str, retries: int = 3) -> dict:
    """Return yf.Ticker.info with 23-h disk cache + serialized live request."""
    cached = _yf_cache_load(ticker)
    if cached is not None:
        return cached
    with _YF_SEM:
        cached = _yf_cache_load(ticker)   # re-check after acquiring semaphore
        if cached is not None:
            return cached
        for attempt in range(retries):
            try:
                info = _call_with_timeout(
                    lambda: yf.Ticker(ticker, session=_yf_get_session()).info)
                _yf_cache_save(ticker, info)
                _time_mod.sleep(0.6)
                return info
            except YahooTimeout:
                print(f"      ⚠ {ticker}: info request timed out after {YF_CALL_TIMEOUT}s — skipped")
                _yf_drop_session()            # a hung session is likely unusable
                return {}
            except Exception as exc:
                if _is_auth_error(exc) and attempt < retries - 1:
                    _yf_drop_session()
                    _time_mod.sleep(8 * (attempt + 1))
                    continue
                return {}
    return {}


# ── attr cache (news, calendar, fast_info, etc.) ──────────────────────────────

class _AttrDict:
    """Thin dict wrapper with attribute access — makes cached fast_info behave like FastInfo."""
    def __init__(self, d: dict):
        self.__dict__.update({k: v for k, v in (d or {}).items() if not k.startswith("_")})
    def get(self, key, default=None):
        return self.__dict__.get(key, default)


def _attr_cache_load(ticker: str, attr: str):
    path = _os.path.join(_YF_ATTR_DIR, f"{ticker}__{attr}.pkl")
    try:
        if _os.path.exists(path) and _time_mod.time() - _os.path.getmtime(path) < _YF_ATTR_TTL:
            with open(path, "rb") as fh:
                val = _pickle.load(fh)
            if attr == "fast_info" and isinstance(val, dict):
                return _AttrDict(val)
            return val
    except Exception:
        pass
    return _YF_SENTINEL


def _attr_cache_save(ticker: str, attr: str, val) -> None:
    try:
        _os.makedirs(_YF_ATTR_DIR, exist_ok=True)
        path = _os.path.join(_YF_ATTR_DIR, f"{ticker}__{attr}.pkl")
        save_val = val
        if attr == "fast_info" and val is not None and not isinstance(val, dict):
            save_val = {k: getattr(val, k, None)
                        for k in ("last_price", "market_cap", "shares",
                                  "currency", "exchange", "quote_type",
                                  "three_month_average_volume", "long_name")}
        with open(path, "wb") as fh:
            _pickle.dump(save_val, fh)
    except Exception:
        pass


def _safe_attr(ticker: str, attr: str, retries: int = 3):
    """Fetch any yf.Ticker attribute with 4-h disk cache + serialized live request."""
    cached = _attr_cache_load(ticker, attr)
    if cached is not _YF_SENTINEL:
        return cached
    with _YF_SEM:
        cached = _attr_cache_load(ticker, attr)   # re-check after acquiring semaphore
        if cached is not _YF_SENTINEL:
            return cached
        for attempt in range(retries):
            try:
                val = _call_with_timeout(
                    lambda: getattr(yf.Ticker(ticker, session=_yf_get_session()), attr))
                _time_mod.sleep(0.5)
                _attr_cache_save(ticker, attr, val)
                return val
            except YahooTimeout:
                print(f"      ⚠ {ticker}.{attr}: timed out after {YF_CALL_TIMEOUT}s — skipped")
                _yf_drop_session()
                return None
            except Exception as exc:
                if _is_auth_error(exc) and attempt < retries - 1:
                    _yf_drop_session()
                    _time_mod.sleep(8 * (attempt + 1))
                    continue
                return None   # don't cache failures — retry next run
    return None


def fetch_fundamentals(tickers: list[str]) -> dict:
    """Parallel fetch of P/E, ROE, beta, and market-cap for factor scoring and classification.
    ETFs/index funds are skipped — they don't have per-stock fundamentals."""
    _skip  = INDEX_TICKERS | INTERNATIONAL_TICKERS
    result = {t: {"pe": None, "roe": None, "beta": None, "market_cap": None}
              for t in tickers if t in _skip}
    fetch_list = [t for t in tickers if t not in _skip]

    def _get(t):
        info = normalize_currency(_safe_info(t))   # ADR P/E otherwise mixes currencies
        pe   = info.get("forwardPE") or info.get("trailingPE")
        roe  = info.get("returnOnEquity")
        beta = info.get("beta")
        mcap = info.get("marketCap")
        return t, {
            "pe":         float(pe)   if pe   is not None else None,
            "roe":        float(roe)  if roe  is not None else None,
            "beta":       float(beta) if beta is not None else None,
            "market_cap": float(mcap) if mcap is not None else None,
        }
    with ThreadPoolExecutor(max_workers=8) as pool:
        result.update(dict(pool.map(_get, fetch_list)))
    return result


def classify_holding(ticker: str, plan_row: dict,
                     fundamentals: dict | None,
                     bl_views: list[dict] | None) -> tuple[str, str, str]:
    """
    Classify a portfolio holding into one of three risk/role buckets.

    Returns (bucket_name, hex_color, one-line description)

    Buckets
    -------
    Conviction    — explicitly modelled via BL investor views; event/catalyst driven
    Upside/Growth — high-momentum factor picks; growth oriented, higher volatility
    Bluechip Core — large-cap, dividend-paying anchors; lower beta, portfolio ballast

    Classification uses: bl_views membership → beta → market-cap → dividend → P/E + ROE.
    """
    # ── Conviction: investor has an explicit BL view on this ticker ────────────
    if any(v.get("ticker") == ticker for v in (bl_views or [])):
        return (
            "Conviction",
            "#ce93d8",
            "Thesis-driven — investor view modelled via Black-Litterman; event or catalyst based",
        )

    fund   = (fundamentals or {}).get(ticker, {}) or {}
    beta   = fund.get("beta")
    mcap   = fund.get("market_cap")
    pe     = fund.get("pe")
    roe    = fund.get("roe")

    # Parse dividend yield from the position row string e.g. "2.50%"
    div_str = str(plan_row.get("Div Yield", "—")).replace("%", "").strip()
    try:
        div_pct = float(div_str)
    except (ValueError, TypeError):
        div_pct = 0.0

    # ── Bluechip scoring ───────────────────────────────────────────────────────
    score = 0
    if div_pct >= 0.5:                            score += 2  # dividend payer — strongest signal
    if beta  is not None and beta  < 0.95:         score += 1  # lower market sensitivity
    if mcap  is not None and mcap  >= 50_000_000_000: score += 1  # large-cap ($50B+)
    if pe    is not None and 7 < pe < 28:          score += 1  # mature, profitable earnings
    if roe   is not None and roe   > 0.12:         score += 1  # strong capital efficiency

    if score >= 3:
        return (
            "Bluechip Core",
            "#4caf50",
            "Large-cap anchor — dividend payer, lower beta, portfolio stability and income",
        )

    # ── Default: momentum/growth pick ─────────────────────────────────────────
    return (
        "Upside / Growth",
        "#ff9800",
        "Factor-selected for momentum + quality + value composite score; higher growth potential",
    )


def fetch_valuation_data(tickers: list[str]) -> dict:
    """Fetch DCF + multiples fields for screened tickers in a single .info pass.
    ETFs and international funds are skipped — they have no fundamentals."""
    _fields = [
        "sector", "industry", "currentPrice", "regularMarketPrice",
        "beta", "freeCashflow", "netIncomeToCommon",
        "sharesOutstanding", "impliedSharesOutstanding",
        "totalDebt", "totalCash", "revenueGrowth", "earningsGrowth",
        "forwardEps", "trailingEps", "forwardPE", "trailingPE",
        "enterpriseToEbitda", "priceToBook", "marketCap",
        "shortPercentOfFloat", "shortRatio",
        "currency", "financialCurrency", "ebitda", "bookValue", "totalRevenue",
    ]
    _skip = INDEX_TICKERS | INTERNATIONAL_TICKERS
    result = {t: {} for t in tickers if t in _skip}
    fetch_list = [t for t in tickers if t not in _skip]

    def _get(t):
        info = _safe_info(t)
        row  = {f: info.get(f) for f in _fields}
        # Statement free cash flow, averaged over up to 3 fiscal years.
        # info["freeCashflow"] is Yahoo's levered-FCF estimate and can be a
        # fraction of reported FCF (MSFT: $16.5B vs $67B); one heavy-capex
        # year shouldn't decide the valuation either.
        try:
            cf = _safe_attr(t, "cashflow")
            if cf is not None and "Free Cash Flow" in cf.index:
                vals = cf.loc["Free Cash Flow"].dropna().iloc[:3]
                if len(vals):
                    row["fcfStatementAvg"] = float(vals.mean())
        except Exception:
            pass
        row["revenueGrowthFwd"] = forward_revenue_growth(t)
        # US filers: free cash flow from filed 10-K XBRL beats Yahoo's numbers
        try:
            import sec_edgar
            if sec_edgar.enabled():
                sec_fcf = sec_edgar.fcf_avg(t)
                if sec_fcf is not None:
                    row["fcfStatementAvg"] = sec_fcf
                    row["fcf_source"] = "SEC 10-K (3-yr avg)"
        except Exception:
            pass
        return t, normalize_currency(row)        # foreign filers → trading currency
    with ThreadPoolExecutor(max_workers=8) as pool:
        result.update(dict(pool.map(_get, fetch_list)))
    return result


# ─── FACTOR ANALYSIS & BLACK-LITTERMAN ────────────────────────────────────────

def multi_factor_screen(prices: pd.DataFrame, fundamentals: dict,
                        top_n: int = TOP_N_FACTORS,
                        catalyst_signals: dict | None = None,
                        force_custom: bool = True,
                        risk_penalties: dict | None = None,
                        revision_signals: dict | None = None) -> list[str]:
    """
    Composite score = 35% momentum + 25% quality (ROE) + 20% value (earnings yield)
                    + 10% estimate revisions + 10% catalyst (news/FDA/clinical events).
    Returns top_n tickers sorted by composite score descending.
    Tickers missing a factor get the neutral percentile rank (0.5).
    Catalyst factor uses live news + analyst upgrades; neutral (0.5) when no signal.
    """
    tickers = list(prices.columns)
    px = prices.ffill()

    # ── Momentum: risk-adjusted return (Sharpe-like) with freshness adjustment ─
    sub      = px.iloc[-252:].pct_change().dropna()
    mom_ret  = sub.mean() * 252
    mom_vol  = sub.std() * np.sqrt(252)
    momentum_raw = (mom_ret / mom_vol.replace(0, np.nan)).dropna()

    # Freshness signals — identify early-cycle vs late-cycle momentum
    raw_12m   = (px.iloc[-1] / px.iloc[-252].replace(0, np.nan) - 1).dropna()
    # 3-month return (≈63 trading days)
    _lo3 = max(0, len(px) - 63)
    raw_3m    = (px.iloc[-1] / px.iloc[_lo3].replace(0, np.nan) - 1).dropna()

    # 52-week range position: 0 = at 52wk low, 1 = at 52wk high
    high_52   = px.iloc[-252:].max()
    low_52    = px.iloc[-252:].min()
    range_pos = ((px.iloc[-1] - low_52) / (high_52 - low_52 + 1e-8)).clip(0, 1)

    # Late-cycle penalty: stock has run >MF_RUN_THRESHOLD AND is near 52wk high
    run_penalty   = (raw_12m - MF_RUN_THRESHOLD).clip(lower=0) * MF_RUN_PENALTY
    range_penalty = (range_pos - MF_RANGE_THRESHOLD).clip(lower=0) * MF_RANGE_PENALTY

    # Acceleration bonus: is recent 3m momentum above/below the expected quarterly pace?
    expected_3m = raw_12m / 4.0
    accel_raw   = (raw_3m - expected_3m) / (raw_12m.abs() + 0.05)
    accel_bonus = accel_raw.clip(-MF_ACCEL_CAP, MF_ACCEL_CAP)

    # Apply freshness adjustment — floor at 0.10x so stocks aren't zeroed out entirely
    adj_factor = ((1 - run_penalty - range_penalty) * (1 + accel_bonus)).clip(0.10, 2.0)
    momentum   = (momentum_raw * adj_factor.reindex(momentum_raw.index, fill_value=1.0)).dropna()

    # Quality: ROE — higher is better
    roe_s = pd.Series({
        t: v for t in tickers
        if (v := fundamentals.get(t, {}).get("roe")) is not None and -1 < v < 5
    })

    # Value: earnings yield = 1 / forward P/E — higher = cheaper
    val_s = pd.Series({
        t: 1.0 / v for t in tickers
        if (v := fundamentals.get(t, {}).get("pe")) is not None and v > 0
    })

    # Catalyst: news/analyst/FDA event score mapped to [0, 1]
    # score range is -1.0 … +1.0 from fetch_catalyst_signals → mapped to 0 … 1
    cat_raw = {}
    if catalyst_signals:
        for t in tickers:
            sig = catalyst_signals.get(t, {})
            if sig.get("score") is not None:
                cat_raw[t] = (float(sig["score"]) + 1.0) / 2.0  # -1…+1 → 0…1
    cat_s = pd.Series(cat_raw)

    # Estimate revisions (−1 … +1). Walk-forward passes none — Yahoo keeps no
    # history, so everyone gets the neutral 0.5 there (no lookahead).
    rev_s = pd.Series({t: float(s["score"]) for t, s in (revision_signals or {}).items()
                       if t in tickers and s.get("score") is not None}, dtype=float)

    all_t = set(tickers)

    def pct_rank(s: pd.Series) -> pd.Series:
        return s.rank(pct=True)

    mom_r = pct_rank(momentum.reindex(all_t))
    roe_r = pct_rank(roe_s.reindex(all_t))
    val_r = pct_rank(val_s.reindex(all_t))
    cat_r = pct_rank(cat_s.reindex(all_t))
    rev_r = pct_rank(rev_s.reindex(all_t))

    composite = (
        FACTOR_BLEND["momentum"]  * mom_r.fillna(0.5)
        + FACTOR_BLEND["quality"] * roe_r.fillna(0.5)
        + FACTOR_BLEND["value"]   * val_r.fillna(0.5)
        + FACTOR_BLEND["revisions"] * rev_r.fillna(0.5)
        + FACTOR_BLEND["catalyst"]* cat_r.fillna(0.5)
    ).dropna()

    # Structural / competitive risk penalty (see STRUCTURAL_RISKS)
    for t, r in (risk_penalties or {}).items():
        if t in composite.index:
            composite[t] -= 0.15 * r["severity"]

    top = composite.nlargest(top_n).index.tolist()
    if force_custom:
        # Hand-picked tickers AND everything an investor already holds are always
        # evaluated (the optimizer may still give them 0%) — otherwise a holding
        # outside the index universe (a recent IPO) is flagged "EVALUATE SELL" merely
        # because it was never screened. Not used in walk-forward (hindsight).
        for t in list(CUSTOM_TICKERS) + held_tickers():
            if t in tickers and t not in top:
                top.append(t)
    return top


def held_tickers() -> list[str]:
    """Every ticker any investor profile currently holds."""
    return sorted({t for p in INVESTOR_PROFILES.values()
                   for t, n in p.get("existing_shares", {}).items() if n})


def black_litterman_mu(tickers: list[str], daily_returns: pd.DataFrame,
                       market_caps: dict) -> dict:
    """
    No-views Black-Litterman: implied equilibrium returns blended with sample.
      Pi       = delta * Sigma * w_mkt  (market-cap-weighted equilibrium)
      mu_bl    = (1 - BL_ALPHA) * Pi + BL_ALPHA * mu_sample
    BL_ALPHA=0.4 preserves momentum signal while dampening estimation noise.
    """
    available = [t for t in tickers if t in daily_returns.columns]
    ret = daily_returns[available].dropna()

    total_mc = sum(market_caps.get(t, 1.0) for t in available)
    w_mkt    = np.array([market_caps.get(t, 1.0) / total_mc for t in available])

    cov       = ret.cov().values * 252
    Pi        = BL_DELTA * (cov @ w_mkt)          # annualised equilibrium returns
    mu_sample = ret.mean().values * 252

    mu_bl = (1.0 - BL_ALPHA) * Pi + BL_ALPHA * mu_sample
    return dict(zip(available, mu_bl))


def apply_bl_views(base_mu: dict, views: list[dict],
                   daily_returns: "pd.DataFrame") -> dict:
    """
    Apply explicit investor views onto a base BL mu using the full BL posterior formula.

      mu_post = [inv(tau*Sigma) + P'*inv(Omega)*P]^-1  @  [inv(tau*Sigma)*Pi + P'*inv(Omega)*Q]

    Each view dict: {ticker, expected_return (annualised), confidence (0–1), note}

    Absolute views — P is a K×N pick matrix with a single 1 per row.
    Omega[i,i] = (1 - confidence_i) * annualised_variance(stock_i) + eps
      → confidence=1 means certainty (very small omega → view dominates)
      → confidence=0 means no information (omega ≈ prior variance → view ignored)
    tau=0.05 follows standard BL literature (small relative to Sigma).
    """
    valid_views = [v for v in views
                   if v.get("ticker") in base_mu and v.get("expected_return") is not None]
    if not valid_views:
        return base_mu

    # Restrict to tickers present in both base_mu and daily_returns
    available = [t for t in base_mu if t in daily_returns.columns]
    if not available:
        return base_mu

    idx_map  = {t: i for i, t in enumerate(available)}
    Pi       = np.array([base_mu[t] for t in available])
    sub_ret  = daily_returns[available].dropna()
    Sigma    = sub_ret.cov().values * 252
    Sigma   += np.eye(len(available)) * 1e-8    # numerical stability

    tau = 0.05   # uncertainty scalar on Pi

    # Build K×N pick matrix P, return vector Q, diagonal uncertainty Omega
    valid_views = [v for v in valid_views if v["ticker"] in idx_map]
    K = len(valid_views)
    if K == 0:
        return base_mu

    P          = np.zeros((K, len(available)))
    Q          = np.zeros(K)
    omega_diag = np.zeros(K)

    for i, view in enumerate(valid_views):
        j              = idx_map[view["ticker"]]
        P[i, j]        = 1.0
        Q[i]           = float(view["expected_return"])
        conf           = float(max(0.01, min(0.99, view.get("confidence", 0.5))))
        var_j          = float(Sigma[j, j])
        omega_diag[i]  = (1.0 - conf) * var_j + 1e-8

    Omega  = np.diag(omega_diag)
    tS_inv = np.linalg.inv(tau * Sigma)
    O_inv  = np.linalg.inv(Omega)

    A = tS_inv + P.T @ O_inv @ P
    b = tS_inv @ Pi + P.T @ O_inv @ Q
    try:
        mu_post_arr = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return base_mu   # fallback if singular

    result = dict(base_mu)
    for i, t in enumerate(available):
        result[t] = float(mu_post_arr[i])
    return result


# ─── VALUATION ENGINE (DCF + multiples → price targets) ───────────────────────

SV_WACC_FLOOR      = 0.07   # very low betas (JNJ 0.2) would otherwise imply ~6% WACC → inflated TV
SV_MAX_GROWTH      = 0.20   # sustained >20%/yr FCF growth for 5 years is rare even for leaders
SV_CYCLICAL_GROWTH = 0.03   # peak-cycle cash flow mean-reverts; don't extrapolate a rebound
SV_NO_DCF_SECTORS  = {"Financial Services", "Financials", "Real Estate"}


def _sv_wacc(info: dict) -> float:
    raw  = float(info.get("beta") or 1.2)
    raw  = max(0.1, min(raw, 3.0))
    beta = 0.67 * raw + 0.33          # Blume adjustment: betas mean-revert toward 1.0
    return max(SV_WACC_FLOOR, RISK_FREE_RATE + beta * SV_EQUITY_RISK_PREMIUM)


def _sv_growth_rate(info: dict) -> float:
    """
    Phase-1 FCF growth. Free cash flow can't durably outgrow revenue, so take
    min(revenue growth, median of earnings-based signals). Trailing earnings growth
    and forward/trailing EPS are noisy (cyclical rebounds, GAAP vs adjusted EPS),
    so each is clipped before the median. Cyclical industries are capped low.
    """
    # Prefer analysts' NEXT-YEAR revenue growth: trailing reported growth is noisy and,
    # for foreign filers, distorted by currency translation (Roche: −1.2% trailing in
    # CHF vs +4.6% forward). Falls back to trailing when no estimate exists.
    rev_g  = info.get("revenueGrowthFwd")
    if rev_g is None:
        rev_g = info.get("revenueGrowth")
    earn_g = info.get("earningsGrowth")
    eps_f  = info.get("forwardEps")
    eps_t  = info.get("trailingEps")
    implied = ((float(eps_f) / float(eps_t)) - 1.0
               if eps_f and eps_t and float(eps_t) > 0 else None)
    # forwardEps is an analyst forecast → optimism haircut (bias grows with horizon)
    from analyst_signals import debias_growth
    implied = debias_growth(implied, "+1y")
    earn_sigs = [float(np.clip(x, -0.20, 0.40)) for x in (earn_g, implied) if x is not None]
    candidates = []
    if rev_g is not None:
        candidates.append(float(rev_g))
    if earn_sigs:
        candidates.append(float(np.median(earn_sigs)))
    g = min(candidates) if candidates else 0.08
    cap = SV_CYCLICAL_GROWTH if info.get("industry") in SV_CYCLICAL_INDUSTRIES else SV_MAX_GROWTH
    return float(np.clip(g, 0.0, cap))


def _sv_dcf(info: dict, growth_rate: float) -> dict | None:
    """Two-stage DCF: years 1-5 at g, years 6-10 fading linearly to terminal growth."""
    fcf = (info.get("fcfStatementAvg") or info.get("freeCashflow")
           or info.get("netIncomeToCommon"))
    if not fcf or float(fcf) <= 0:
        return None
    wacc = _sv_wacc(info)
    if wacc <= SV_TERMINAL_GROWTH:
        wacc = SV_TERMINAL_GROWTH + 0.03
    pv, cf, g1 = 0.0, float(fcf), growth_rate
    for yr in range(1, 6):
        cf *= (1 + g1); pv += cf / (1 + wacc) ** yr
    for yr in range(6, 11):
        g2 = g1 + (SV_TERMINAL_GROWTH - g1) * (yr - 5) / 5   # fade to terminal by year 10
        cf *= (1 + g2); pv += cf / (1 + wacc) ** yr
    tv = cf * (1 + SV_TERMINAL_GROWTH) / (wacc - SV_TERMINAL_GROWTH)
    pv += tv / (1 + wacc) ** 10
    ev = pv - float(info.get("totalDebt") or 0) + float(info.get("totalCash") or 0)
    if ev <= 0:
        return None
    shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
    if not shares or float(shares) <= 0:
        return None
    intrinsic = ev / float(shares)
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if not price:
        return None
    upside = (intrinsic - float(price)) / float(price)
    return {"intrinsic": round(intrinsic, 2), "upside_pct": round(upside * 100, 1),
            "signal": ("UNDERVALUED" if upside > 0.15 else
                       "OVERVALUED"  if upside < -0.15 else "FAIR VALUE")}


def _sv_multiples_composite(info: dict) -> float:
    """Quick composite: +1 ATTRACTIVE, 0 NEUTRAL, -1 EXPENSIVE for fwd_pe/ev_ebitda/pb."""
    sector = info.get("sector", "Default")
    bench  = SV_SECTOR_BENCHMARKS.get(sector, SV_SECTOR_BENCHMARKS["Default"])
    scores = []
    for val, bm in [
        (info.get("forwardPE"),          bench["fwd_pe"]),
        (info.get("enterpriseToEbitda"), bench["ev_ebitda"]),
        (info.get("priceToBook"),        bench["pb"]),
    ]:
        if val and float(val) > 0 and bm:
            r = float(val) / bm
            scores.append(1 if r < 0.75 else (-1 if r > 1.15 else 0))
    return float(np.mean(scores)) if scores else 0.0


def valuation_signal(ticker: str, info: dict) -> dict:
    """
    Returns verdict (UNDERVALUED / FAIR VALUE / OVERVALUED) + 3-scenario DCF
    price range (buy_low = conservative intrinsic, buy_high = optimistic intrinsic).
    ETFs and tickers with insufficient data return FAIR VALUE with no range.
    """
    if not info or ticker in INDEX_TICKERS or ticker in INTERNATIONAL_TICKERS:
        return {"verdict": "—", "dcf_upside": None,
                "buy_low": None, "buy_high": None, "composite": 0.0}
    g = _sv_growth_rate(info)
    if info.get("sector") in SV_NO_DCF_SECTORS:
        # Banks/insurers/REITs: deposits, float and property cash flows make an
        # FCF DCF meaningless — value on multiples only.
        dcf_cons = dcf_base = dcf_opti = None
    else:
        # Additive spreads — multiplicative ones (g×0.5, g×1.5) collapse to one value at g=0
        dcf_cons = _sv_dcf(info, max(0.0, g - 0.03))
        dcf_base = _sv_dcf(info, g)
        dcf_opti = _sv_dcf(info, min(SV_MAX_GROWTH * 1.25, g + 0.04))
    mult     = _sv_multiples_composite(info)
    signals: list[float] = []
    if dcf_base:
        if dcf_base["signal"] == "UNDERVALUED":  signals.extend([1.0, 1.0])
        elif dcf_base["signal"] == "OVERVALUED": signals.extend([-1.0, -1.0])
        else:                                    signals.append(0.0)
    signals.append(max(-1.0, min(1.0, mult)) * 2.0)
    composite = float(np.mean(signals)) if signals else 0.0
    if composite > 0.20:
        verdict = "UNDERVALUED"
    elif composite < -0.20:
        verdict = "OVERVALUED"
    else:
        verdict = "FAIR VALUE"
    return {
        "verdict":    verdict,
        "dcf_upside": dcf_base["upside_pct"] if dcf_base else None,
        "buy_low":    dcf_cons["intrinsic"] if dcf_cons else None,
        "buy_high":   dcf_opti["intrinsic"] if dcf_opti else None,
        "composite":  round(composite, 3),
        "multiples":  round(mult, 3),
    }


def rerank_valuation_verdicts(val_signals: dict, tercile: float = 1 / 3) -> dict:
    """
    Replace absolute DCF verdicts with a RELATIVE ranking across the universe.

    A conservative DCF in a high-multiple market labels most stocks OVERVALUED
    (68 of 97 in Oct 2026), so an absolute ±15% threshold carries no information.
    Instead score each stock on cheapness vs its peers in this run:
        value score = mean(percentile of DCF upside, percentile of multiples score)
    (multiples only where no DCF — banks, insurers, REITs). Cheapest third →
    UNDERVALUED, dearest third → OVERVALUED, rest FAIR VALUE. Ranges are kept.
    """
    rated = {t: v for t, v in val_signals.items() if v.get("verdict") != "—"}
    if len(rated) < 6:
        return val_signals
    dcf  = pd.Series({t: v["dcf_upside"] for t, v in rated.items()
                      if v.get("dcf_upside") is not None}).rank(pct=True)
    mult = pd.Series({t: v.get("multiples", 0.0) for t, v in rated.items()}).rank(pct=True)
    score = pd.concat([dcf, mult], axis=1).mean(axis=1)
    lo, hi = score.quantile(tercile), score.quantile(1 - tercile)
    out = dict(val_signals)
    for t, s in score.items():
        verdict = "UNDERVALUED" if s >= hi else "OVERVALUED" if s <= lo else "FAIR VALUE"
        out[t] = {**val_signals[t], "verdict": verdict, "value_pct": round(float(s), 3)}
    return out


def apply_valuation_tilt(mu_dict: dict, val_signals: dict) -> dict:
    """Tilt BL expected returns ±VALUATION_TILT_MAG based on DCF/multiples verdict.
    Uses |mu| so the tilt direction is correct even when mu is negative
    (mu × 1.10 on a negative mu would *penalise* an undervalued stock)."""
    tilted = {}
    for t, mu in mu_dict.items():
        v = val_signals.get(t, {}).get("verdict", "FAIR VALUE")
        if v == "UNDERVALUED":
            tilted[t] = mu + VALUATION_TILT_MAG * abs(mu)
        elif v == "OVERVALUED":
            tilted[t] = mu - VALUATION_TILT_MAG * abs(mu)
        else:
            tilted[t] = mu
    return tilted


# ─── OPTIMIZATION ─────────────────────────────────────────────────────────────

def _time_decay_weights(n_obs: int) -> np.ndarray:
    weights = np.ones(n_obs)
    for ds, de, mult in DECAY_TIERS:
        i0 = max(0, n_obs - de)
        i1 = max(0, n_obs - ds)
        weights[i0:i1] = mult
    return weights / weights.sum()


def add_sector_constraints(constraints, sector_map, tickers, cap=SECTOR_CAP):
    sectors: dict[str, list[int]] = {}
    for i, t in enumerate(tickers):
        s = sector_map.get(t, "Unknown")
        sectors.setdefault(s, []).append(i)
    for s, idx in sectors.items():
        if s == "Unknown":
            continue
        constraints.append({
            "type": "ineq",
            "fun":  lambda w, ix=idx: cap - sum(w[i] for i in ix),
        })
    return constraints


def equal_weight_capped(ranked: list[str], n: int, group_caps: list | None = None,
                        ticker_caps: dict | None = None,
                        locked: dict | None = None) -> dict:
    """1/N over the top-ranked names, respecting the same limits as the optimizer:
    names that would push an industry / binary-event group over its cap are skipped
    (next-ranked name instead), per-ticker caps (biotech 6%) are honoured, and
    locked positions (minimum holding period) keep their weight."""
    locked = dict(locked or {})
    caps = ticker_caps or {}
    groups = group_caps or []
    room = max(1.0 - sum(locked.values()), 0.0)
    slots = max(n - len(locked), 0)
    if slots == 0 or room <= 0:
        return locked
    w_eq = room / slots
    used = {i: sum(v for t, v in locked.items() if t in m) for i, (m, _) in enumerate(groups)}
    picks: dict = {}
    for t in ranked:
        if len(picks) >= slots:
            break
        if t in locked or t in picks:
            continue
        wt = min(w_eq, caps.get(t, w_eq))
        if any(t in m and used[i] + wt > c + 1e-9 for i, (m, c) in enumerate(groups)):
            continue
        picks[t] = wt
        for i, (m, _) in enumerate(groups):
            if t in m:
                used[i] += wt
    # Hand any capped-name shortfall to the uncapped picks
    short = room - sum(picks.values())
    free = [t for t in picks if caps.get(t, 1.0) > picks[t] + 1e-12]
    if short > 1e-9 and free:
        for t in free:
            picks[t] += short / len(free)
    return {**locked, **picks}


def binary_event_tickers(infos: dict, fda_tickers=()) -> set:
    """Names whose value hinges on a single binary event: clinical-stage biotechs,
    plus non-megacap names with a dated FDA decision (Roche's giredestrant isn't
    binary for a $250B company; a $5B company's PDUFA date is)."""
    out = set()
    for t, info in infos.items():
        info = info or {}
        small = (info.get("marketCap") or 0) < BINARY_MCAP_LIMIT   # unknown size → binary
        if small and (info.get("industry") == "Biotechnology" or t in fda_tickers):
            out.add(t)                    # VRTX/REGN-size biotechs have product revenue
    return out


def build_group_caps(infos: dict, fda_tickers=(),
                     industry_cap: float | None = None,
                     binary_cap: float | None = None) -> list:
    """[(tickers, cap), …] for optimize_weights: one group per industry with 2+ names,
    plus the aggregate binary-event group."""
    industry_cap = INDUSTRY_MAX_WEIGHT if industry_cap is None else industry_cap
    binary_cap   = BINARY_EVENT_MAX if binary_cap is None else binary_cap
    by_ind: dict[str, set] = {}
    for t, info in infos.items():
        ind = (info or {}).get("industry")
        if ind:
            by_ind.setdefault(ind, set()).add(t)
    groups = [(m, industry_cap) for m in by_ind.values() if len(m) > 1]
    binary = binary_event_tickers(infos, fda_tickers)
    if binary:
        groups.append((binary, binary_cap))
    return groups


def trim_to_max_holdings(weights: dict, max_n: int, max_weight: float,
                         caps: dict | None = None) -> dict:
    """
    Keep the top `max_n` positions and renormalise to 100% WITHOUT breaching
    any cap. Plain renormalisation scaled capped names above their limit
    (a 6% biotech cap became 6.7%); excess is instead redistributed pro-rata
    across positions that still have room (water-filling).
    """
    if not weights:
        return {}
    top  = dict(sorted(weights.items(), key=lambda x: -x[1])[:max_n])
    lim  = {t: min(max_weight, (caps or {}).get(t, max_weight)) for t in top}
    if sum(lim.values()) < 1.0 - 1e-9:          # caps can't reach 100% — just scale
        tot = sum(top.values())
        return {t: w / tot for t, w in top.items()}
    w = {t: min(v, lim[t]) for t, v in top.items()}
    for _ in range(50):
        gap = 1.0 - sum(w.values())
        if abs(gap) < 1e-9:
            break
        room = [t for t in w if lim[t] - w[t] > 1e-12]
        if not room:
            break
        base = sum(w[t] for t in room)
        for t in room:
            share = w[t] / base if base > 0 else 1 / len(room)
            w[t] = min(lim[t], w[t] + gap * share)
    return w


def _aligned_window(tickers: list[str], prices: pd.DataFrame,
                    min_coverage: float = 0.90) -> tuple[list[str], pd.DataFrame]:
    """
    Keep only tickers with price data for >= min_coverage of the window, then
    forward-fill and drop the (few) remaining leading-NaN rows.

    A bare `prices[tickers].ffill().dropna()` lets ONE short-history ticker
    (a recent IPO) truncate the whole matrix to its own history — or empty it
    entirely when that ticker has no data in the window.
    """
    cols = [t for t in tickers if t in prices.columns]
    if not cols:
        return [], prices.iloc[:, :0]
    coverage = prices[cols].notna().mean()
    keep     = [t for t in cols if coverage[t] >= min_coverage]
    return keep, prices[keep].ffill().dropna()


def optimize_weights(tickers: list[str], prices: pd.DataFrame,
                     objective: str, max_weight: float, min_weight: float,
                     sector_map: dict | None = None,
                     mu_override: dict | None = None,
                     int_floor: float = 0.0,
                     ticker_caps: dict | None = None,
                     group_caps: list | None = None,
                     prev_weights: dict | None = None,
                     turnover_penalty: float = 0.0,
                     floor_weights: dict | None = None) -> dict:
    """
    Mean-variance optimization.
    Objectives: max_sharpe | max_return | min_variance | max_sortino | equal_weight
    Extras: BL mu override, international ETF floor, sector cap, time-decay cov,
    per-ticker caps (e.g. binary-event biotechs) that override max_weight,
    group caps [(tickers, cap), …] (industry, aggregate binary-event exposure),
    and a turnover penalty vs prev_weights (annual-return units per 100% turnover).
    """
    if objective == "equal_weight":
        # 1/N benchmark (DeMiguel, Garlappi & Uppal 2009): the optimizer has to beat this
        avail = [t for t in tickers if t in prices.columns and prices[t].notna().any()]
        return {t: 1.0 / len(avail) for t in avail} if avail else {}
    available, sub = _aligned_window([t for t in tickers if t in prices.columns], prices)
    if len(available) < 2:
        return {}
    daily = sub.pct_change().dropna()

    obs_w     = _time_decay_weights(len(daily))
    mu_daily  = (daily.values * obs_w[:, None]).sum(axis=0)
    mu_sample = mu_daily * 252

    demeaned = daily.values - mu_daily
    cov      = (demeaned * obs_w[:, None]).T @ demeaned * 252
    cov     += np.eye(len(available)) * 1e-8

    if mu_override is not None:
        mu = np.array([mu_override.get(t, mu_sample[i]) for i, t in enumerate(available)])
    else:
        mu = mu_sample

    n  = len(available)
    w0 = np.ones(n) / n

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    if int_floor > 0:
        intl_idx = [i for i, t in enumerate(available) if t in INTERNATIONAL_TICKERS]
        if intl_idx:
            constraints.append({
                "type": "ineq",
                "fun":  lambda w, ix=intl_idx: sum(w[i] for i in ix) - int_floor,
            })

    if sector_map:
        constraints = add_sector_constraints(constraints, sector_map, available)

    # Floors: positions that can't be reduced yet (minimum holding period). Scaled down
    # if they'd exceed 100%; caps and group caps give way to a floor, never the reverse.
    floors = {t: v for t, v in (floor_weights or {}).items() if t in available and v > 0}
    if sum(floors.values()) > 0.999:
        k = 0.999 / sum(floors.values())
        floors = {t: v * k for t, v in floors.items()}

    for members, gcap in (group_caps or []):
        g_idx = [i for i, t in enumerate(available) if t in members]
        gcap = max(gcap, sum(floors.get(available[i], 0.0) for i in g_idx))
        if g_idx and gcap < 1.0:
            constraints.append({"type": "ineq",
                                "fun": lambda w, ix=g_idx, c=gcap: c - sum(w[i] for i in ix)})

    caps   = ticker_caps or {}
    bounds = []
    for t in available:
        hi = min(max_weight, caps.get(t, max_weight))
        lo = min(min_weight, hi)
        f  = floors.get(t, 0.0)
        bounds.append((max(lo, f), max(hi, f)))
    if floors:
        w0 = np.array([max(b[0], 1.0 / len(available)) for b in bounds])
        w0 = w0 / w0.sum()

    daily_arr = daily.values
    prev = np.array([(prev_weights or {}).get(t, 0.0) for t in available])
    lam  = turnover_penalty if prev_weights else 0.0

    def ret(w):
        """Expected return net of the turnover penalty (smooth |x| for SLSQP)."""
        r = float(w @ mu)
        if lam:
            r -= lam * float(np.sum(np.sqrt((w - prev) ** 2 + 1e-8)))
        return r

    if objective == "min_variance":
        def obj(w): return float(w @ cov @ w)

    elif objective == "max_sharpe":
        def obj(w):
            vol = float(np.sqrt(w @ cov @ w))
            return -(ret(w) - CASH_RATE) / vol if vol > 0 else 0.0

    elif objective == "max_sortino":
        def obj(w):
            pd_r  = daily_arr @ w
            neg   = pd_r[pd_r < 0]
            dv    = neg.std() * np.sqrt(252) if len(neg) > 0 else 1e-6
            return -(ret(w) - CASH_RATE) / dv

    else:  # max_return
        def obj(w): return -ret(w)

    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-9})

    if not np.all(np.isfinite(res.x)):
        print(f"      ⚠ optimizer returned non-finite weights ({res.message}) — no holdings")
        return {}
    raw   = {available[i]: float(res.x[i]) for i in range(n) if res.x[i] >= 0.01}
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {t: w / total for t, w in raw.items()}


# ─── PORTFOLIO ANALYTICS ──────────────────────────────────────────────────────

def compute_metrics(prices: pd.DataFrame, weights: dict, key: str) -> dict:
    available = [t for t in weights if t in prices.columns]
    w         = {t: weights[t] for t in available}
    total     = sum(w.values())
    w         = {t: v / total for t, v in w.items()}

    sub        = prices[available].ffill().dropna()
    daily_ret  = sub.pct_change().dropna()
    wvec       = np.array([w[t] for t in available])
    port_ret   = pd.Series(daily_ret.values @ wvec, index=daily_ret.index)
    cumulative = (1 + port_ret).cumprod() * INITIAL_INVESTMENT

    n_years      = len(port_ret) / 252
    total_return = cumulative.iloc[-1] / INITIAL_INVESTMENT - 1
    cagr         = (1 + total_return) ** (1 / n_years) - 1
    vol          = port_ret.std() * np.sqrt(252)
    sharpe       = (cagr - CASH_RATE) / vol if vol > 0 else 0.0

    neg_ret  = port_ret[port_ret < 0]
    dv_vol   = neg_ret.std() * np.sqrt(252) if len(neg_ret) > 0 else vol
    sortino  = (cagr - CASH_RATE) / dv_vol if dv_vol > 0 else 0.0

    roll_max = cumulative.expanding().max()
    max_dd   = ((cumulative - roll_max) / roll_max).min()

    var_thr  = np.percentile(port_ret, 5)
    var_95   = var_thr * INITIAL_INVESTMENT
    cvar_95  = port_ret[port_ret <= var_thr].mean() * INITIAL_INVESTMENT

    return {
        "key":          key,
        "weights":      w,
        "tickers":      available,
        "port_ret":     port_ret,
        "cumulative":   cumulative,
        "total_return": total_return,
        "cagr":         cagr,
        "volatility":   vol,
        "sharpe":       sharpe,
        "sortino":      sortino,
        "downside_vol": dv_vol,
        "max_dd":       max_dd,
        "var_95":       var_95,
        "cvar_95":      cvar_95,
        "final_value":  cumulative.iloc[-1],
        "gain":         cumulative.iloc[-1] - INITIAL_INVESTMENT,
        "start":        cumulative.index[0].strftime("%Y-%m-%d"),
        "end":          cumulative.index[-1].strftime("%Y-%m-%d"),
        "n_years":      n_years,
    }


def walk_forward_backtest(tickers: list[str], prices: pd.DataFrame,
                          objective: str, max_weight: float, min_weight: float,
                          sector_map: dict | None = None,
                          screen_universe: list[str] | None = None,
                          fundamentals: dict | None = None,
                          top_n: int | None = None) -> dict:
    """
    Out-of-sample test: screen + optimize on the first ~2/3 of history, evaluate
    on the rest, vs SPY over the same test period.

    When `screen_universe` is given, the factor screen is re-run on TRAINING
    prices only. Passing today's screened list instead leaks the test period:
    that list was picked by momentum over the last 252 days, which overlaps the
    test window, so the "out-of-sample" portfolio would be drawn from stocks
    already known to have done well. (Fundamentals are current values — a small
    residual leak; news catalysts are excluded for the same reason.)
    """
    n     = len(prices)
    split = int(n * 0.67)
    train = prices.iloc[:split]
    test  = prices.iloc[split:]

    if screen_universe is not None:
        cols    = [t for t in screen_universe if t in train.columns]
        tickers = multi_factor_screen(train[cols], fundamentals or {},
                                      top_n=top_n or len(tickers))

    holdings = optimize_weights(tickers, train, objective,
                                max_weight, min_weight, sector_map)

    avail = [t for t in holdings if t in test.columns]
    if not avail:
        # Report "no result" rather than a fabricated 0.0% return
        return {
            "train_period":   f"{train.index[0].date()} → {train.index[-1].date()}",
            "test_period":    f"{test.index[0].date()} → {test.index[-1].date()}",
            "oos_total": None, "oos_annualized": None,
            "oos_sharpe": None, "oos_max_dd": None,
            "spy_annualized": None, "alpha": None, "holdings": {},
        }
    wvec  = np.array([holdings[t] for t in avail])
    wvec /= wvec.sum()

    sub       = test[avail].ffill().dropna()
    daily_ret = sub.pct_change().dropna()
    port_rets = (daily_ret.values * wvec).sum(axis=1)
    cum       = (1 + port_rets).cumprod()

    total_ret = float(cum[-1]) - 1
    n_days    = len(cum)
    ann_ret   = (1 + total_ret) ** (252 / n_days) - 1
    oos_vol   = float(np.std(port_rets) * np.sqrt(252))
    oos_sharpe = (ann_ret - CASH_RATE) / oos_vol if oos_vol > 0 else None
    oos_max_dd = float(np.min(cum / np.maximum.accumulate(cum) - 1))

    spy_ann = None
    if "SPY" in test.columns:
        spy_p   = test["SPY"].ffill().dropna().pct_change().dropna()
        spy_c   = (1 + spy_p).cumprod()
        spy_tot = float(spy_c.iloc[-1]) - 1
        spy_ann = (1 + spy_tot) ** (252 / len(spy_c)) - 1

    return {
        "train_period":   f"{train.index[0].date()} → {train.index[-1].date()}",
        "test_period":    f"{test.index[0].date()} → {test.index[-1].date()}",
        "oos_total":      total_ret,
        "oos_annualized": ann_ret,
        "oos_sharpe":     oos_sharpe,
        "oos_max_dd":     oos_max_dd,
        "spy_annualized": spy_ann,
        "alpha":          (ann_ret - spy_ann) if spy_ann is not None else None,
        "holdings":       holdings,
    }


class DataQualityError(RuntimeError):
    """Raised when a run's outputs are too broken to act on."""


def data_quality_checks(prices: pd.DataFrame, results: dict, wf_results: dict,
                        div_yields: dict, fundamentals: dict, val_signals: dict,
                        layer_status: dict,
                        extra_warnings: list[str] | None = None) -> list[tuple[str, str]]:
    """
    Sanity-check the run's OUTPUTS rather than trusting every try/except upstream.
    Each check below corresponds to a real silent failure found in the Oct-2026 audit.
    Returns [(level, message)] with level CRITICAL or WARN.
    """
    issues: list[tuple[str, str]] = []
    crit = lambda m: issues.append(("CRITICAL", m))
    warn = lambda m: issues.append(("WARN", m))

    # Price history
    if len(prices) < 500:
        crit(f"price history only {len(prices)} days — need ≥500 for backtests")
    if "SPY" not in prices.columns:
        crit("SPY missing — regime, beta and benchmarks are invalid")

    # Short-history truncation (made metrics run on 20 months instead of 5 years)
    for k, r in results.items():
        if r.get("n_years", 0) < 2.0:
            crit(f"{k}: metrics cover only {r.get('n_years', 0):.1f} years — "
                 f"a short-history ticker is truncating the window")
        if not r.get("weights"):
            crit(f"{k}: optimizer produced no holdings")
        if r.get("cagr", 0) > 1.0:
            warn(f"{k}: in-sample CAGR {r['cagr']:.0%} — overfit, don't read as expected return")
        er = r.get("expected_return")
        if er is not None and not (0.0 <= er <= 0.25):
            warn(f"{k}: projection E[R] {er:.1%} outside 0–25%")

    # Walk-forward produced nothing (was a fake 0.0%)
    for k, wf in wf_results.items():
        if wf.get("oos_annualized") is None:
            warn(f"{k}: walk-forward produced no out-of-sample result")

    # Units (dividend yield came back in percent → 200%+)
    bad_dy = {t: y for t, y in (div_yields or {}).items() if y and y > 0.15}
    if bad_dy:
        warn(f"implausible dividend yields >15%: "
             + ", ".join(f"{t} {y:.0%}" for t, y in list(bad_dy.items())[:5]))

    # Fundamentals coverage
    if fundamentals:
        cov = np.mean([bool(v.get("pe")) for v in fundamentals.values()])
        if cov < 0.60:
            warn(f"only {cov:.0%} of tickers have P/E — quality/value factors degraded")

    # Valuation verdicts dominated by one class (absolute-DCF skew)
    verdicts = [v.get("verdict") for v in (val_signals or {}).values()
                if v.get("verdict") not in (None, "—")]
    if len(verdicts) >= 10:
        top_share = max(verdicts.count(x) for x in set(verdicts)) / len(verdicts)
        if top_share > 0.60:
            warn(f"{top_share:.0%} of valuation verdicts are one class — signal uninformative")

    # Data layers that silently returned nothing (FDA layer & pipeline never ran before)
    for layer, ok in (layer_status or {}).items():
        if not ok:
            warn(f"{layer} layer returned nothing — check its log line above")
    for msg in (extra_warnings or []):
        warn(msg)
    return issues


def report_data_quality(issues: list[tuple[str, str]]) -> None:
    if not issues:
        print("\n  ✓ Data quality: all checks passed")
        return
    print("\n" + "=" * 74 + "\nDATA QUALITY\n" + "=" * 74)
    for level, msg in issues:
        print(f"  {'✗' if level == 'CRITICAL' else '⚠'} {level:<8} {msg}")
    if any(level == "CRITICAL" for level, _ in issues):
        raise DataQualityError("critical data-quality issues — outputs not written "
                               "(fix the issues above, then re-run)")


SP500_HISTORY_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
                     "sp500_ticker_start_end.csv")


def sp500_membership(since: pd.Timestamp):
    """
    Point-in-time S&P 500 membership for the walk-forward (reduces survivorship bias:
    a backtest on TODAY's members only picks from companies that survived/grew).

    Current members  → join date from Wikipedia "Date added" (keeps the original date
                       across renames — FB→META, BK→BNY appear in the GitHub data as a
                       removal + a new addition on the same day).
    Former members   → github.com/fja05680/sp500 start/end dates; tickers that are just
                       an old symbol of a current member are dropped.
    Returns (members_at(date) -> set, ever_member: set, removed_since: list) or None.
    """
    import io
    import requests
    H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36"}
    try:
        cur = pd.read_html(io.StringIO(requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=H, timeout=20).text))[0]
        cur_add = dict(zip(cur["Symbol"].str.replace(".", "-", regex=False),
                           pd.to_datetime(cur["Date added"], errors="coerce")))
        hist = pd.read_csv(io.StringIO(requests.get(SP500_HISTORY_URL, timeout=30).text))
        hist["start_date"] = pd.to_datetime(hist["start_date"], errors="coerce")
        hist["end_date"]   = pd.to_datetime(hist["end_date"], errors="coerce")
        hist["ticker"]     = hist["ticker"].str.replace(".", "-", regex=False)
    except Exception as exc:
        print(f"      ⚠ S&P 500 membership history unavailable ({type(exc).__name__}) — "
              f"walk-forward uses today's members (survivorship bias)")
        return None

    # A removed ticker whose end date equals the GitHub start of a current member that
    # Wikipedia says joined EARLIER is that member's old symbol (rename) — not a removal.
    gh_start = hist[hist["end_date"].isna()].set_index("ticker")["start_date"].to_dict()
    renamed_on = {d for t, d in gh_start.items()
                  if t in cur_add and pd.notna(cur_add[t]) and pd.notna(d) and cur_add[t] < d}
    former = hist[hist["end_date"].notna() & ~hist["ticker"].isin(cur_add)
                  & ~hist["end_date"].isin(renamed_on)]
    former = former.groupby("ticker").agg(start=("start_date", "min"), end=("end_date", "max"))

    def members_at(d: pd.Timestamp) -> set:
        m = {t for t, a in cur_add.items() if pd.isna(a) or a <= d}
        m |= set(former[(former["start"] <= d) & (former["end"] > d)].index)
        return m

    removed_since = sorted(former[former["end"] >= since].index)
    return members_at, set(cur_add) | set(former.index), removed_since


WF_MIN_TRAIN   = 504     # first fold trains on ~2 years
WF_REBAL_DAYS  = 63      # re-screen + re-optimise every ~quarter
WF_COST_BPS    = 10      # trading cost per side, basis points of turnover
BACKTEST_YEARS = 10      # walk-forward history (covers 2018, 2020, 2022 sell-offs);
                         # the live optimizer keeps its 3-year window
WF_TURNOVER_PENALTY = 0.05   # annual-return cost per 100% turnover in the walk-forward
                             # optimizer — STARTING VALUE (≈ costs + tax drag), not fitted
# Investor rule: no short-term sales, unless a stock is clearly underperforming or
# something happens. Backtest proxy for "clearly underperforming": down ≥ 25% from
# entry (STARTING VALUE). News-driven exits can't be replayed historically.
WF_MIN_HOLD_DAYS       = 252
WF_EARLY_EXIT_DRAWDOWN = 0.25

# Taxable-account replay — personal rates come from TAX_OVERRIDES (profiles_local.py)
TAX_STATE_NAME   = TAX_OVERRIDES.get("state_name", "State")
TAX_STATE        = TAX_OVERRIDES.get("state_rate", 0.0)
TAX_FED_ORDINARY = TAX_OVERRIDES.get("fed_ordinary", 0.22)
TAX_FED_LT       = TAX_OVERRIDES.get("fed_lt", 0.15)
TAX_RATE_ST      = TAX_FED_ORDINARY + TAX_STATE   # short-term gains = ordinary income
TAX_RATE_LT      = TAX_FED_LT + TAX_STATE         # long-term gains

# Concentration groups — STARTING VALUES
INDUSTRY_MAX_WEIGHT = 0.25   # e.g. PBF+VLO+MPC+DINO (refiners, corr 0.81–0.88) ≤ 25% together
BINARY_EVENT_MAX    = 0.15   # all binary-event names together (clinical biotechs, small/mid
BINARY_MCAP_LIMIT   = 50e9   # caps with a dated FDA decision); megacaps aren't binary


def walk_forward_rolling(universe: list[str], prices: pd.DataFrame,
                         objective: str, max_weight: float, min_weight: float,
                         top_n: int, fundamentals: dict | None = None,
                         sector_map: dict | None = None,
                         min_train: int = WF_MIN_TRAIN,
                         rebal_days: int = WF_REBAL_DAYS,
                         cost_bps: float = WF_COST_BPS,
                         ticker_caps: dict | None = None,
                         membership=None, sp_ever: set | None = None,
                         fundamentals_at=None,
                         group_caps: list | None = None,
                         turnover_penalty: float = 0.0,
                         min_hold_days: int = 0,
                         early_exit_drawdown: float = 0.25,
                         progress: str | None = None) -> dict:
    """
    Rolling out-of-sample backtest — the process the analyzer would actually
    have followed: at each quarterly rebalance date, screen and optimise on data
    up to that date ONLY, hold the weights (drifting with prices) until the next
    rebalance, and pay costs on turnover. Hand-picked CUSTOM_TICKERS are NOT
    forced in, so convictions chosen with hindsight can't flatter the result.

    fundamentals_at(date) → {ticker: {pe, roe}} as known on that date (SEC filings,
    point_in_time.py). Without it, `fundamentals` (today's values) is used for every
    fold — a look-ahead leak; pass fundamentals={} for a momentum-only control.
    objective="equal_weight" holds the top MAX_HOLDINGS screened names at 1/N.

    Same keys as walk_forward_backtest plus n_folds, hit_rate, avg_turnover,
    daily / spy_daily / rsp_annualized and the weight schedule (for after_tax_replay).
    """
    px = prices.ffill()
    n  = len(px)
    if n < min_train + rebal_days or "SPY" not in px.columns:
        return walk_forward_backtest(universe, prices, objective, max_weight,
                                     min_weight, sector_map)

    port_daily: list[pd.Series] = []
    fold_rets, spy_fold_rets, turnovers = [], [], []
    prev_w: dict = {}
    last_holdings: dict = {}
    schedule: list[tuple] = []             # (first day of fold, last day of fold, weights)
    entry_idx: dict = {}                   # ticker → index of first purchase (current run)
    entry_px: dict = {}
    drift_w: dict = {}                     # weights at the end of the previous fold
    early_exits = 0
    folds_total = len(range(min_train, n, rebal_days))
    t_start = _time_mod.time()

    for fold_no, start in enumerate(range(min_train, n, rebal_days), 1):
        if progress and (fold_no % 8 == 0 or fold_no == folds_total):
            print(f"         {progress}: fold {fold_no}/{folds_total} "
                  f"({_time_mod.time() - t_start:.0f}s)", flush=True)
        # Minimum holding period: positions held < min_hold_days can't be reduced,
        # unless they're down ≥ early_exit_drawdown from entry ("clearly underperforming"
        # — and a loss carries no short-term gains tax anyway).
        locked: dict = {}
        if min_hold_days > 0:
            for t, dw in drift_w.items():
                p_now = px[t].iloc[start - 1] if t in px.columns else np.nan
                if pd.isna(p_now) or start - entry_idx.get(t, start) >= min_hold_days:
                    continue
                if p_now <= entry_px.get(t, p_now) * (1 - early_exit_drawdown):
                    early_exits += 1
                    continue
                locked[t] = dw
        end   = min(start + rebal_days, n)         # fold covers days [start, end)
        train = prices.iloc[:start]
        cols  = [t for t in universe if t in train.columns]
        if membership is not None:
            # Point-in-time: an S&P name is eligible only while it was in the index
            members = membership(prices.index[start])
            cols = [t for t in cols if t not in (sp_ever or set()) or t in members]
        fund_t = (fundamentals_at(prices.index[start - 1]) if fundamentals_at is not None
                  else (fundamentals or {}))
        cand  = multi_factor_screen(train[cols], fund_t, top_n=top_n, force_custom=False)
        if objective == "equal_weight":
            w = equal_weight_capped(cand, MAX_HOLDINGS, group_caps, ticker_caps, locked)
        else:
            cand = cand + [t for t in locked if t not in cand]
            w = optimize_weights(cand, train, objective, max_weight, min_weight, sector_map,
                                 ticker_caps=ticker_caps, group_caps=group_caps,
                                 prev_weights=prev_w or None,
                                 turnover_penalty=turnover_penalty, floor_weights=locked)
        if len(w) > MAX_HOLDINGS:          # same rule as the live portfolios; locked stay
            free = sorted((t for t in w if t not in locked), key=lambda t: -w[t])
            keep = free[:max(MAX_HOLDINGS - len(locked), 0)]
            kept_w = sum(w[t] for t in keep)
            room = 1.0 - sum(w[t] for t in locked if t in w)
            w = {**{t: w[t] for t in locked if t in w},
                 **{t: w[t] / kept_w * room for t in keep}} if kept_w > 0 else \
                {t: w[t] for t in locked if t in w}
        if not w:
            w = dict(prev_w)                       # keep previous book if optimiser fails
        if not w:
            continue
        held  = [t for t in w if t in px.columns]
        wvec  = np.array([w[t] for t in held], dtype=float)
        wvec /= wvec.sum()

        seg   = px[held].iloc[start - 1:end]       # prior close as base + fold days
        rel   = (seg / seg.iloc[0]).fillna(1.0)    # missing data → flat (cash-like)
        value = rel.values @ wvec                  # buy-and-hold drift within the fold
        daily = pd.Series(value[1:] / value[:-1] - 1, index=seg.index[1:])

        all_t = set(w) | set(prev_w)
        turn  = sum(abs(w.get(t, 0.0) - prev_w.get(t, 0.0)) for t in all_t)
        if not prev_w:
            turn = 1.0                             # initial purchase, one side
        cost = turn * cost_bps / 10_000
        daily.iloc[0] -= cost

        port_daily.append(daily)
        fold_rets.append(float((1 + daily).prod() - 1))
        spy_seg = px["SPY"].iloc[start - 1:end]
        spy_fold_rets.append(float(spy_seg.iloc[-1] / spy_seg.iloc[0] - 1))
        turnovers.append(turn)
        schedule.append((start, end, dict(zip(held, wvec))))
        for t in held:                              # holding-period clock per position
            if t not in drift_w:
                entry_idx[t], entry_px[t] = start, px[t].iloc[start - 1]
        for t in list(entry_idx):
            if t not in held:
                entry_idx.pop(t, None)
                entry_px.pop(t, None)
        end_val = wvec * rel.values[-1]
        drift_w = dict(zip(held, end_val / end_val.sum())) if end_val.sum() > 0 else {}
        prev_w, last_holdings = w, w

    if not port_daily:
        return walk_forward_backtest(universe, prices, objective, max_weight,
                                     min_weight, sector_map)

    port = pd.concat(port_daily)
    cum  = (1 + port).cumprod()
    yrs  = len(port) / 252
    tot  = float(cum.iloc[-1] - 1)
    ann  = (1 + tot) ** (1 / yrs) - 1
    vol  = float(port.std() * np.sqrt(252))

    spy   = px["SPY"].loc[port.index[0]:port.index[-1]]
    spy_base = px["SPY"].iloc[px.index.get_loc(port.index[0]) - 1]
    spy_ann  = (float(spy.iloc[-1]) / float(spy_base)) ** (1 / yrs) - 1
    rsp_ann  = None                        # equal-weight S&P 500: size-neutral benchmark
    if "RSP" in px.columns and px["RSP"].notna().all():
        rsp_base = px["RSP"].iloc[px.index.get_loc(port.index[0]) - 1]
        rsp_ann  = (float(px["RSP"].loc[port.index[-1]]) / float(rsp_base)) ** (1 / yrs) - 1

    return {
        "train_period":   f"rolling {min_train}d+ (expanding), rebalanced every {rebal_days}d",
        "test_period":    f"{port.index[0].date()} → {port.index[-1].date()}",
        "oos_total":      tot,
        "oos_annualized": ann,
        "oos_sharpe":     (ann - CASH_RATE) / vol if vol > 0 else None,
        "oos_max_dd":     float((cum / cum.cummax() - 1).min()),
        "spy_annualized": spy_ann,
        "alpha":          ann - spy_ann,
        "holdings":       last_holdings,
        "n_folds":        len(fold_rets),
        "hit_rate":       float(np.mean([p > s for p, s in zip(fold_rets, spy_fold_rets)])),
        "avg_turnover":   float(np.mean(turnovers[1:])) if len(turnovers) > 1 else None,
        "daily":          port,                  # OOS daily returns (robustness_grid.py)
        "spy_daily":      px["SPY"].pct_change().reindex(port.index),
        "rsp_annualized": rsp_ann,
        "alpha_rsp":      (ann - rsp_ann) if rsp_ann is not None else None,
        "schedule":       schedule,
        "early_exits":    early_exits,       # holding-period overrides on a ≥25% drop
    }


def after_tax_replay(schedule: list, prices: pd.DataFrame,
                     st_rate: float | None = None, lt_rate: float | None = None,
                     cost_bps: float = WF_COST_BPS) -> dict | None:
    """
    Replay a walk-forward weight schedule in a TAXABLE account: FIFO lots, gains
    realised on every trim/exit, short-term (< 252 trading days) vs long-term rates,
    losses netted within a rebalance and carried forward. Taxes are paid from the
    portfolio. Open positions at the end are NOT taxed (deferral is a real benefit).
    Returns pre-tax and after-tax annualised returns on identical trades.
    """
    st_rate = TAX_RATE_ST if st_rate is None else st_rate
    lt_rate = TAX_RATE_LT if lt_rate is None else lt_rate
    if not schedule:
        return None
    px = prices.ffill()

    def run(st, lt):
        cash, lots, carry = 1.0, {}, 0.0
        taxes = st_g = lt_g = 0.0
        for start, _end, w in schedule:
            i = start - 1
            p = px.iloc[i]
            nav = cash + sum(sum(u for _, u, _ in ls) * p[t] for t, ls in lots.items()
                             if pd.notna(p.get(t)))
            # Sells first (FIFO)
            st_net = lt_net = proceeds = traded = 0.0
            for t in list(lots):
                if pd.isna(p.get(t)):
                    continue
                held   = sum(u for _, u, _ in lots[t])
                target = w.get(t, 0.0) * nav / p[t]
                sell   = held - target
                if sell <= 1e-12:
                    continue
                proceeds += sell * p[t]
                traded   += sell * p[t]
                while sell > 1e-12 and lots[t]:
                    bi, u, c = lots[t][0]
                    take = min(u, sell)
                    gain = take * (p[t] - c)
                    if i - bi < 252:
                        st_net += gain
                    else:
                        lt_net += gain
                    sell -= take
                    if take >= u - 1e-12:
                        lots[t].pop(0)
                    else:
                        lots[t][0] = (bi, u - take, c)
                if not lots[t]:
                    del lots[t]
            # Net ST vs LT losses, then apply the carried-forward loss
            if st_net < 0 < lt_net:
                lt_net, st_net = max(lt_net + st_net, 0.0), min(lt_net + st_net, 0.0)
            elif lt_net < 0 < st_net:
                st_net, lt_net = max(st_net + lt_net, 0.0), min(st_net + lt_net, 0.0)
            carry += min(st_net, 0.0) + min(lt_net, 0.0)
            st_net, lt_net = max(st_net, 0.0), max(lt_net, 0.0)
            use = min(-carry, st_net); st_net -= use; carry += use
            use = min(-carry, lt_net); lt_net -= use; carry += use
            tax = st * st_net + lt * lt_net
            taxes += tax; st_g += st_net; lt_g += lt_net
            cash += proceeds - tax
            # Buys, scaled to the cash available after tax and costs
            buys = {t: max(wt * nav / p[t] - sum(u for _, u, _ in lots.get(t, [])), 0.0)
                    for t, wt in w.items() if pd.notna(p.get(t)) and p[t] > 0}
            spend = sum(u * p[t] for t, u in buys.items())
            traded += spend
            cash -= traded * cost_bps / 10_000
            scale = min(1.0, max(cash, 0.0) / spend) if spend > 0 else 0.0
            for t, u in buys.items():
                if u * scale > 1e-12:
                    lots.setdefault(t, []).append((i, u * scale, p[t]))
            cash -= spend * scale
        last = schedule[-1][1] - 1
        p = px.iloc[last]
        final = cash + sum(sum(u for _, u, _ in ls) * p[t] for t, ls in lots.items()
                           if pd.notna(p.get(t)))
        yrs = (last - (schedule[0][0] - 1)) / 252
        return {"ann": final ** (1 / yrs) - 1 if final > 0 and yrs > 0 else None,
                "taxes": taxes, "st_gains": st_g, "lt_gains": lt_g}

    pre, post = run(0.0, 0.0), run(st_rate, lt_rate)
    gains = post["st_gains"] + post["lt_gains"]
    return {
        "pre_tax_annualized":   pre["ann"],
        "after_tax_annualized": post["ann"],
        "tax_drag":             (pre["ann"] - post["ann"])
                                if pre["ann"] is not None and post["ann"] is not None else None,
        "short_term_share":     post["st_gains"] / gains if gains > 0 else None,
        "st_rate": st_rate, "lt_rate": lt_rate,
    }


def prepare_backtest(valid_tickers: list[str], years: int = BACKTEST_YEARS) -> dict:
    """Everything the walk-forward needs, built once (also used by robustness_grid.py):
    long price history + splits, survivorship (former S&P 500 members, point-in-time
    membership), point-in-time SEC fundamentals, biotech caps, industry and
    binary-event group caps. Market caps / industries are today's values (minor leak);
    the FDA calendar is NOT used historically (that would be hindsight)."""
    universe = list(valid_tickers)
    start = pd.Timestamp(END_DATE) - pd.DateOffset(years=years)
    sp = sp500_membership(start)
    member_fn, sp_ever, removed = (sp if sp is not None else (None, None, []))
    t0 = _time_mod.time()
    prices, split_close, splits = fetch_backtest_prices(universe + removed + ["SPY", "RSP"],
                                                        years)
    universe = [t for t in universe if t in prices.columns and prices[t].notna().any()]
    got = [t for t in removed if t in prices.columns and prices[t].notna().any()
           and t not in universe]
    universe += got
    print(f"      {len(prices)} days ({prices.index[0].date()} → {prices.index[-1].date()}) "
          f"for {len(universe)} tickers ({_time_mod.time() - t0:.0f}s)")
    if sp is not None:
        print(f"      Survivorship fix: {len(removed)} former S&P 500 members in window — "
              f"{len(got)} with price data added ({len(removed) - len(got)} acquired/"
              f"delisted, no data); current members gated by join date")
    else:
        print("      ⚠ No membership history — backtest uses today's members (survivorship bias)")

    infos = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for t, info in zip(universe, pool.map(lambda t: _safe_info(t) or {}, universe)):
            infos[t] = info
    sectors = {t: i.get("sector") or "Unknown" for t, i in infos.items()}
    caps = {t: BIOTECH_MAX_WEIGHT for t, i in infos.items() if i.get("industry") == "Biotechnology"}
    groups = build_group_caps(infos)

    pit = None
    try:
        from point_in_time import PointInTimeFundamentals
        pit = PointInTimeFundamentals.build(universe, split_close, splits)
    except Exception as exc:
        print(f"      ⚠ Point-in-time fundamentals failed ({type(exc).__name__}: {exc})")
    if pit is None:
        print("      ⚠ No point-in-time fundamentals (SEC EDGAR off) — backtest uses TODAY's "
              "P/E and ROE for every quarter: look-ahead bias, alpha overstated")
    return {"prices": prices, "universe": universe, "membership": member_fn,
            "sp_ever": sp_ever, "pit": pit, "sector_map": sectors, "caps": caps,
            "group_caps": groups, "infos": infos}


def _wf(bt: dict, objective: str, max_weight: float, min_weight: float, top_n: int,
        fundamentals: dict | None, use_pit: bool = True,
        group_caps: list | None = None, early_exit: float | None = None,
        progress: str | None = None) -> dict:
    """Walk-forward on a prepared backtest. group_caps / early_exit override the
    defaults (robustness_grid.py varies them)."""
    pit = bt["pit"] if use_pit else None
    return walk_forward_rolling(
        bt["universe"], bt["prices"], objective, max_weight, min_weight, top_n=top_n,
        fundamentals=fundamentals if pit is None else None, fundamentals_at=pit,
        sector_map=bt["sector_map"], ticker_caps=bt["caps"], membership=bt["membership"],
        sp_ever=bt["sp_ever"],
        group_caps=bt["group_caps"] if group_caps is None else group_caps,
        turnover_penalty=WF_TURNOVER_PENALTY if objective != "equal_weight" else 0.0,
        min_hold_days=WF_MIN_HOLD_DAYS,
        early_exit_drawdown=WF_EARLY_EXIT_DRAWDOWN if early_exit is None else early_exit,
        progress=progress)


def run_backtests(bt: dict, fundamentals_today: dict) -> dict:
    """Walk-forward for each scenario plus three checks, attached to each result:
      after_tax         — the same trades in a taxable account (FIFO lots, ST/LT rates)
      momentum_only     — quality/value switched off: how much do fundamentals add?
      equal_weight      — top-15 screened names at 1/N: does the optimizer add anything?
    """
    fund = fundamentals_today if bt["pit"] is None else None
    t0 = _time_mod.time()
    ew = _wf(bt, "equal_weight", 1.0, 0.0, TOP_N_FACTORS, fund)
    print(f"      Benchmarks: SPY {pct_or_na(ew.get('spy_annualized'))}/yr  ·  "
          f"RSP (equal-weight S&P) {pct_or_na(ew.get('rsp_annualized'))}/yr  ·  "
          f"screen top-15 at 1/N {pct_or_na(ew.get('oos_annualized'))}/yr  "
          f"({_time_mod.time() - t0:.0f}s)")
    out: dict = {}
    for key, pf in PORTFOLIOS.items():
        t0 = _time_mod.time()
        print(f"      → {pf['label']}")
        wf = _wf(bt, pf["objective"], pf["max_weight"], pf["min_weight"],
                 pf["screen_top_n"], fund, progress="strategy")
        mom = _wf(bt, pf["objective"], pf["max_weight"], pf["min_weight"],
                  pf["screen_top_n"], {}, use_pit=False, progress="momentum-only")
        wf["after_tax"] = after_tax_replay(wf.get("schedule") or [], bt["prices"])
        wf["momentum_only"] = {k: mom.get(k) for k in ("oos_annualized", "alpha")}
        wf["equal_weight"] = {k: ew.get(k) for k in ("oos_annualized", "alpha")}
        wf["pit_fundamentals"] = bt["pit"] is not None
        out[key] = wf
        if wf.get("n_folds"):
            print(f"         {wf['n_folds']} quarterly folds  ·  beat SPY in "
                  f"{wf['hit_rate']:.0%} of quarters  ·  avg turnover "
                  f"{(wf.get('avg_turnover') or 0):.0%}/qtr  ·  {WF_COST_BPS}bp costs  ·  "
                  f"1-yr min hold ({wf.get('early_exits', 0)} early exits on ≥"
                  f"{WF_EARLY_EXIT_DRAWDOWN:.0%} drops)")
        alpha_rsp = wf.get("alpha_rsp")
        print(f"         Test {wf['test_period']}: OOS {pct_or_na(wf['oos_annualized'])}/yr  "
              f"alpha vs SPY {pct_or_na(wf.get('alpha'))}  vs RSP {pct_or_na(alpha_rsp)}")
        at = wf["after_tax"] or {}
        if at.get("after_tax_annualized") is not None:
            print(f"         After tax (ST {at['st_rate']:.1%} / LT {at['lt_rate']:.1%}, "
                  f"{TAX_STATE_NAME} + federal {TAX_FED_ORDINARY:.0%}): "
                  f"{at['after_tax_annualized']:+.1%}/yr  "
                  f"(drag {at['tax_drag']:.1%}/yr, "
                  f"{(at.get('short_term_share') or 0):.0%} of realised gains short-term)")
        print(f"         Momentum-only control: {pct_or_na(mom.get('oos_annualized'))}/yr  "
              f"·  equal-weight benchmark: {pct_or_na(ew.get('oos_annualized'))}/yr  "
              f"({_time_mod.time() - t0:.0f}s)")
    return out


def stress_test(port_ret: pd.Series, prices: pd.DataFrame) -> list[dict]:
    """
    Estimate portfolio drawdown under historical crash scenarios using beta vs SPY.
    Beta = Cov(portfolio, SPY) / Var(SPY) computed over the full history.
    """
    if "SPY" not in prices.columns:
        return []
    spy_ret = prices["SPY"].pct_change().dropna()
    aligned = pd.concat([port_ret.rename("port"), spy_ret.rename("spy")],
                        axis=1).dropna()
    if len(aligned) < 50:
        return []
    cov_mat = aligned.cov()
    beta    = cov_mat.loc["port", "spy"] / cov_mat.loc["spy", "spy"]

    rows = []
    for name, spy_shock, duration_mo in STRESS_SCENARIOS:
        port_shock = max(beta * spy_shock, -0.95)
        trough     = INITIAL_INVESTMENT * (1 + port_shock)
        rows.append({
            "Scenario":                  name,
            "SPY Drawdown":              f"{spy_shock:.1%}",
            "Portfolio Beta":            round(float(beta), 2),
            "Est. Portfolio Drawdown":   f"{port_shock:.1%}",
            "Est. Trough Value ($)":     round(trough, 0),
            "Est. Loss ($)":             round(INITIAL_INVESTMENT - trough, 0),
            "Historical Duration (mo)":  duration_mo,
        })
    return rows


def detect_regime(prices: pd.DataFrame) -> dict:
    """Market regime classification via SPY 200-day moving average crossover."""
    if "SPY" not in prices.columns:
        return {"regime": "Unknown", "signal": "SPY not in universe",
                "spy_price": None, "ma200": None, "pct_vs_ma": "N/A"}
    spy = prices["SPY"].ffill().dropna()
    if len(spy) < 200:
        return {"regime": "Unknown", "signal": "Insufficient SPY history",
                "spy_price": None, "ma200": None, "pct_vs_ma": "N/A"}
    ma200    = float(spy.rolling(200).mean().iloc[-1])
    current  = float(spy.iloc[-1])
    pct_diff = (current - ma200) / ma200
    if current > ma200 * 1.02:
        regime = "Bull"
        signal = f"SPY {pct_diff:+.1%} above 200-day MA — uptrend intact"
    elif current < ma200 * 0.98:
        regime = "Bear"
        signal = f"SPY {pct_diff:+.1%} below 200-day MA — elevated downside risk"
    else:
        regime = "Neutral"
        signal = f"SPY within ±2% of 200-day MA — transitional zone"
    return {
        "regime":    regime,
        "signal":    signal,
        "spy_price": round(current, 2),
        "ma200":     round(ma200, 2),
        "pct_vs_ma": f"{pct_diff:+.1%}",
    }


def backtest_portfolio(holdings: dict, prices: pd.DataFrame, months: int) -> dict | None:
    """Hypothetical return if today's optimized weights had been deployed N months ago."""
    tickers = [t for t, w in holdings.items() if w > 0 and t in prices.columns]
    if not tickers:
        return None
    weights = np.array([holdings[t] for t in tickers])
    weights /= weights.sum()
    sub = prices[tickers].ffill()
    approx_days = int(months * 21.75)
    if len(sub) < approx_days + 10:
        return None
    sub = sub.iloc[-approx_days:].dropna(how="any")
    if len(sub) < 10:
        return None
    daily_rets = sub.pct_change().dropna()
    port_rets  = (daily_rets.values * weights).sum(axis=1)
    cum        = (1 + port_rets).cumprod()
    total_ret  = float(cum[-1]) - 1
    n_days     = len(cum)
    ann_ret    = (1 + total_ret) ** (252 / n_days) - 1
    end_val    = INITIAL_INVESTMENT * (1 + total_ret)
    return {
        "months":       months,
        "start_date":   daily_rets.index[0].strftime("%Y-%m-%d"),
        "end_date":     daily_rets.index[-1].strftime("%Y-%m-%d"),
        "end_value":    end_val,
        "gain":         end_val - INITIAL_INVESTMENT,
        "total_return": total_ret,
        "annualized":   ann_ret,
        "cumulative":   pd.Series(cum * INITIAL_INVESTMENT, index=daily_rets.index),
    }


def compute_rebalance_triggers(holdings: dict, prices: pd.DataFrame,
                                lookback_months: int = 12) -> list[dict]:
    """
    Simulate drift over lookback_months: which positions would need trimming or adding?
    Triggers when drift from target weight exceeds 4%.
    """
    approx_days = int(lookback_months * 21.75)
    tickers = [t for t in holdings if t in prices.columns and holdings[t] > 0]
    if len(prices) < approx_days or not tickers:
        return []
    p_start = prices[tickers].iloc[-approx_days]
    p_end   = prices[tickers].iloc[-1]
    current_vals: dict[str, float] = {}
    for t in tickers:
        init  = INITIAL_INVESTMENT * holdings[t]
        ratio = (float(p_end[t]) / float(p_start[t])
                 if pd.notna(p_start[t]) and float(p_start[t]) > 0 else 1.0)
        current_vals[t] = init * ratio
    total = sum(current_vals.values())
    triggers = []
    for t in tickers:
        target  = holdings[t]
        current = current_vals[t] / total if total > 0 else target
        drift   = current - target
        if abs(drift) >= 0.04:
            triggers.append({
                "Ticker":         t,
                "Target Weight":  f"{target:.1%}",
                "Drifted Weight": f"{current:.1%}",
                "Drift":          f"{drift:+.1%}",
                "Action":         "TRIM" if drift > 0 else "ADD",
            })
    return sorted(triggers, key=lambda x: abs(float(x["Drift"].rstrip("%"))), reverse=True)


def factor_exposure(holdings: dict, fundamentals: dict,
                    momentum_scores: dict) -> dict:
    """Weighted average factor exposures for the portfolio."""
    total_w = sum(holdings.values())
    w_pe  = sum(holdings[t] * (fundamentals.get(t, {}).get("pe")  or 22.0) for t in holdings) / total_w
    w_roe = sum(holdings[t] * (fundamentals.get(t, {}).get("roe") or 0.12) for t in holdings) / total_w
    w_mom = sum(holdings[t] * momentum_scores.get(t, 0.0)                  for t in holdings) / total_w
    tilt  = "Growth" if w_pe > 25 else ("Value" if w_pe < 15 else "Blended")
    return {
        "Weighted Avg Forward P/E":   round(w_pe, 1),
        "Weighted Avg ROE":           f"{w_roe:.1%}",
        "Weighted Avg 12m Momentum":  f"{w_mom:.1%}",
        "Style Tilt":                 tilt,
    }


def build_allocation_plan(holdings: dict, current_prices: dict,
                          div_yields: dict | None = None,
                          val_signals: dict | None = None):
    rows: list[dict] = []
    intl_cost = dom_cost = 0.0
    for ticker, weight in sorted(holdings.items(), key=lambda x: -x[1]):
        target  = INITIAL_INVESTMENT * weight
        price   = current_prices.get(ticker)
        shares  = int(target / price) if price else 0
        cost    = shares * price if price else 0.0
        dy      = (div_yields.get(ticker, 0.0) or 0.0) if div_yields else 0.0
        income  = cost * dy if dy else 0.0
        # Valuation verdict + DCF price range
        vsig     = (val_signals or {}).get(ticker, {})
        verdict  = vsig.get("verdict") or "—"
        buy_low  = vsig.get("buy_low")
        buy_high = vsig.get("buy_high")
        if buy_low and buy_high:
            dcf_range = f"${buy_low:,.0f}–${buy_high:,.0f}"
        elif buy_low:
            dcf_range = f"<${buy_low:,.0f}"
        elif buy_high:
            dcf_range = f"<${buy_high:,.0f}"
        else:
            dcf_range = "—"
        if ticker in INTERNATIONAL_TICKERS:
            typ = "INTL"
            intl_cost += cost
        elif ticker in INDEX_TICKERS:
            typ = "INDEX"
            dom_cost  += cost
        else:
            typ = "stock"
            dom_cost  += cost
        rows.append({
            "Type":               typ,
            "Ticker":             ticker,
            "Weight":             f"{weight:.0%}",
            "Target ($)":         round(target, 2),
            "Current Price ($)":  round(price, 2) if price else "N/A",
            "Shares to Buy":      shares,
            "Actual Cost ($)":    round(cost, 2),
            "Div Yield (TTM)":    f"{dy:.2%}" if dy else "—",
            "Est. Annual Income": round(income, 2),
            "Verdict":            verdict,
            "DCF Range ($)":      dcf_range,
        })
    total_cost   = sum(r["Actual Cost ($)"] for r in rows)
    total_income = sum(r["Est. Annual Income"] for r in rows)
    rows.append({
        "Type": "TOTAL", "Ticker": "", "Weight": "100%",
        "Target ($)": INITIAL_INVESTMENT,
        "Current Price ($)": "", "Shares to Buy": "",
        "Actual Cost ($)":    round(total_cost, 2),
        "Div Yield (TTM)":    "",
        "Est. Annual Income": round(total_income, 2),
        "Difference ($)":     round(INITIAL_INVESTMENT - total_cost, 2),
        "Verdict": "", "DCF Range ($)": "",
    })
    return rows, intl_cost, dom_cost


def forward_expected_return(port_ret: pd.Series, prices: pd.DataFrame) -> tuple[float, float]:
    """
    Forward-looking annual (arithmetic) expected return for projections:
        E[R] = risk-free + beta × equity risk premium   (CAPM)
    Beta is measured against SPY. The in-sample CAGR is deliberately NOT used —
    the weights were fitted to that history, so it reflects overfit, not expectation.
    Returns (expected_return, beta).
    """
    beta = 1.0
    if "SPY" in prices.columns:
        spy = prices["SPY"].ffill().pct_change().rename("spy")
        aligned = pd.concat([port_ret.rename("p"), spy], axis=1, sort=True).dropna()
        if len(aligned) > 60 and aligned["spy"].var() > 0:
            beta = float(aligned.cov().iloc[0, 1] / aligned["spy"].var())
    beta = float(np.clip(beta, 0.3, 2.5))
    return RISK_FREE_RATE + beta * SV_EQUITY_RISK_PREMIUM, beta


def monte_carlo(expected_return: float, vol: float) -> np.ndarray:
    """GBM projection. expected_return is an arithmetic annual mean (see
    forward_expected_return); log drift = E[R] − ½σ²."""
    expected_return = float(np.clip(np.nan_to_num(expected_return, nan=0.08), -0.05, 0.25))
    vol  = float(np.clip(np.nan_to_num(vol,  nan=0.20),  0.01, 1.50))
    np.random.seed(42)
    steps = PROJECTION_YEARS * 252
    dt    = 1 / 252
    mu    = expected_return - 0.5 * vol ** 2
    rand  = np.random.normal(0, 1, (MC_SIMULATIONS, steps))
    log_r = mu * dt + vol * np.sqrt(dt) * rand
    return INITIAL_INVESTMENT * np.exp(np.cumsum(log_r, axis=1))


# ─── VISUALIZATION ────────────────────────────────────────────────────────────

BG     = "#0f1117"
PANEL  = "#1a1d2e"
GRID_C = "#2a2d3e"
TEXT_C = "#e0e0e0"


def _style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.grid(color=GRID_C, linewidth=0.5, linestyle="--")
    ax.tick_params(colors=TEXT_C, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)


def plot_dashboard(results: dict, mc_results: dict,
                   backtest_results: dict | None = None,
                   stress_results: dict | None = None,
                   wf_results: dict | None = None,
                   regime: dict | None = None):
    colors = {k: PORTFOLIOS[k]["color"] for k in PORTFOLIOS}
    labels = {k: PORTFOLIOS[k]["label"]  for k in PORTFOLIOS}

    fig = plt.figure(figsize=(22, 40))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(6, 3, figure=fig, hspace=0.55, wspace=0.35)

    # ── Row 0: Historical growth (full width) ─────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    _style_ax(ax0)
    for key, res in results.items():
        cum = res["cumulative"]
        ax0.plot(cum.index, cum.values, color=colors[key], linewidth=2.2,
                 label=f"{labels[key]}  →  {fmt_dollars(res['final_value'])}  "
                       f"({res['total_return']:.0%} total, {res['cagr']:.1%} CAGR)")
    ax0.axhline(INITIAL_INVESTMENT, color="#888", linewidth=1, linestyle=":",
                alpha=0.7, label=f"Initial ${INITIAL_INVESTMENT:,}")
    regime_suffix = (f"   |   Regime: {regime['regime']} — {regime['signal']}"
                     if regime else "")
    ax0.set_title(f"Historical Portfolio Growth — ${INITIAL_INVESTMENT:,}{regime_suffix}",
                  color=TEXT_C, fontsize=12, fontweight="bold", pad=10)
    ax0.yaxis.set_major_formatter(FuncFormatter(fmt_dollars))
    ax0.set_xlabel("Date", color=TEXT_C)
    ax0.set_ylabel("Portfolio Value", color=TEXT_C)
    ax0.legend(loc="upper left", framealpha=0.3, facecolor=BG,
               labelcolor=TEXT_C, fontsize=9)

    # ── Row 1: Monte Carlo projections ────────────────────────────────────────
    x_years = np.linspace(0, PROJECTION_YEARS, PROJECTION_YEARS * 252)
    for col, (key, res) in enumerate(results.items()):
        ax = fig.add_subplot(gs[1, col])
        _style_ax(ax)
        paths = mc_results[key]
        for i in range(min(150, MC_SIMULATIONS)):
            ax.plot(x_years, paths[i], color=colors[key], alpha=0.03, linewidth=0.5)
        p5,  p25 = np.percentile(paths,  5, axis=0), np.percentile(paths, 25, axis=0)
        p50, p75 = np.percentile(paths, 50, axis=0), np.percentile(paths, 75, axis=0)
        p95      = np.percentile(paths, 95, axis=0)
        ax.fill_between(x_years, p5,  p95, color=colors[key], alpha=0.12)
        ax.fill_between(x_years, p25, p75, color=colors[key], alpha=0.22)
        ax.plot(x_years, p50, color=colors[key], linewidth=2.5,
                label=f"Median: {fmt_dollars(p50[-1])}")
        ax.plot(x_years, p95, color=colors[key], linewidth=0.9, linestyle="--",
                alpha=0.6, label=f"95th: {fmt_dollars(p95[-1])}")
        ax.plot(x_years, p5,  color=colors[key], linewidth=0.9, linestyle="--",
                alpha=0.6, label=f"5th:  {fmt_dollars(p5[-1])}")
        ax.axhline(INITIAL_INVESTMENT, color="#888", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_dollars))
        ax.set_title(f"Monte Carlo — {labels[key]}\n{PROJECTION_YEARS}-yr ({MC_SIMULATIONS:,} paths)",
                     color=TEXT_C, fontsize=9, fontweight="bold")
        ax.set_xlabel("Years from today", color=TEXT_C, fontsize=8)
        ax.set_ylabel("Value", color=TEXT_C, fontsize=8)
        ax.legend(framealpha=0.3, facecolor=BG, labelcolor=TEXT_C, fontsize=7)

    # ── Row 2: Buy plan bars ───────────────────────────────────────────────────
    for col, (key, pf) in enumerate(PORTFOLIOS.items()):
        ax = fig.add_subplot(gs[2, col])
        _style_ax(ax)
        tickers     = list(pf["holdings"].keys())
        dollar_amts = [INITIAL_INVESTMENT * w for w in pf["holdings"].values()]
        palette     = plt.cm.tab20(np.linspace(0, 1, max(len(tickers), 1)))
        ax.barh(tickers[::-1], dollar_amts[::-1], color=palette[::-1],
                edgecolor=PANEL, linewidth=0.8)
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_dollars))
        ax.set_title(f"{labels[key]}\nAllocation (${INITIAL_INVESTMENT:,})",
                     color=TEXT_C, fontsize=9, fontweight="bold")
        ax.set_xlabel("Dollar Amount", color=TEXT_C, fontsize=8)

    # ── Row 3: Risk/Return scatter | Enhanced metrics table | Stress test bars ─
    ax_rr = fig.add_subplot(gs[3, 0])
    _style_ax(ax_rr)
    for key, res in results.items():
        ax_rr.scatter(res["volatility"] * 100, res["cagr"] * 100,
                      color=colors[key], s=220, zorder=5,
                      edgecolors="white", linewidth=1)
        ax_rr.annotate(labels[key].split(" (")[0],
                       (res["volatility"] * 100, res["cagr"] * 100),
                       textcoords="offset points", xytext=(8, 4),
                       color=colors[key], fontsize=8, fontweight="bold")
    ax_rr.set_title("Risk vs. Return", color=TEXT_C, fontsize=10, fontweight="bold")
    ax_rr.set_xlabel("Annualised Volatility (%)", color=TEXT_C, fontsize=8)
    ax_rr.set_ylabel("CAGR (%)", color=TEXT_C, fontsize=8)

    ax_tbl = fig.add_subplot(gs[3, 1])
    ax_tbl.set_facecolor(PANEL)
    ax_tbl.axis("off")
    col_lbl   = ["CAGR", "Sharpe", "Sortino", "Max DD", "CVaR 95%", "Final Val"]
    tbl_rows, tbl_rlbl = [], []
    for key, res in results.items():
        tbl_rlbl.append(labels[key].split(" (")[0])
        tbl_rows.append([
            f"{res['cagr']:.1%}", f"{res['sharpe']:.2f}", f"{res['sortino']:.2f}",
            f"{res['max_dd']:.1%}", fmt_dollars(res["cvar_95"]),
            fmt_dollars(res["final_value"]),
        ])
    tbl = ax_tbl.table(cellText=tbl_rows, rowLabels=tbl_rlbl,
                       colLabels=col_lbl, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 2.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(PANEL if r > 0 else "#2a2d3e")
        cell.set_edgecolor("#3a3d4e")
        if r == 0:
            cell.set_text_props(color=TEXT_C, fontweight="bold")
        elif c == -1:
            cell.set_text_props(color=colors[list(results.keys())[r - 1]],
                                fontweight="bold")
        else:
            cell.set_text_props(color=TEXT_C)
    ax_tbl.set_title("Enhanced Metrics (Historical)",
                     color=TEXT_C, fontsize=10, fontweight="bold", pad=8)

    ax_st = fig.add_subplot(gs[3, 2])
    _style_ax(ax_st)
    if stress_results:
        sc_names  = [s["Scenario"] for s in next(iter(stress_results.values()))]
        n_sc, n_pf = len(sc_names), len(PORTFOLIOS)
        x_st    = np.arange(n_sc)
        offs_st = np.linspace(-(n_pf - 1) / 2, (n_pf - 1) / 2, n_pf) * 0.22
        for i, key in enumerate(PORTFOLIOS):
            troughs = [s["Est. Trough Value ($)"] for s in stress_results[key]]
            ax_st.bar(x_st + offs_st[i], troughs, 0.22,
                      color=colors[key], alpha=0.82,
                      label=labels[key].split(" (")[0],
                      edgecolor=PANEL, linewidth=0.6)
        ax_st.axhline(INITIAL_INVESTMENT, color="#aaa", linewidth=1.2,
                      linestyle=":", alpha=0.8)
        ax_st.set_xticks(x_st)
        ax_st.set_xticklabels([n.replace(" ", "\n") for n in sc_names],
                              color=TEXT_C, fontsize=6)
        ax_st.yaxis.set_major_formatter(FuncFormatter(fmt_dollars))
        ax_st.set_title("Stress Test — Estimated Trough Values\n(beta-adjusted)",
                        color=TEXT_C, fontsize=9, fontweight="bold")
        ax_st.legend(framealpha=0.3, facecolor=BG, labelcolor=TEXT_C, fontsize=7)

    # ── Row 4: Walk-forward results table ─────────────────────────────────────
    ax_wf = fig.add_subplot(gs[4, :])
    ax_wf.set_facecolor(PANEL)
    ax_wf.axis("off")
    if wf_results:
        wf_col = ["Scenario", "Train Period", "Test Period",
                  "OOS Return/yr", "SPY Return/yr", "Alpha vs SPY"]
        wf_rows = []
        for key, wf in wf_results.items():
            wf_rows.append([
                labels[key].split(" (")[0],
                wf["train_period"],
                wf["test_period"],
                pct_or_na(wf["oos_annualized"]),
                f"{wf['spy_annualized']:+.1%}" if wf["spy_annualized"] is not None else "N/A",
                f"{wf['alpha']:+.1%}" if wf["alpha"] is not None else "N/A",
            ])
        tbl_wf = ax_wf.table(cellText=wf_rows, colLabels=wf_col,
                             loc="center", cellLoc="center")
        tbl_wf.auto_set_font_size(False)
        tbl_wf.set_fontsize(9)
        tbl_wf.scale(1, 3.0)
        key_list = list(wf_results.keys())
        for (r, c), cell in tbl_wf.get_celld().items():
            cell.set_facecolor(PANEL if r > 0 else "#2a2d3e")
            cell.set_edgecolor("#3a3d4e")
            if r == 0:
                cell.set_text_props(color=TEXT_C, fontweight="bold")
            elif c == 0:
                cell.set_text_props(color=colors[key_list[r - 1]], fontweight="bold")
            elif c in (3, 5) and r > 0:
                try:
                    val = float(wf_rows[r - 1][c].rstrip("%").replace("+", ""))
                    tc  = "#4CAF50" if val >= 0 else "#f44336"
                    cell.set_text_props(color=tc)
                except ValueError:
                    cell.set_text_props(color=TEXT_C)
            else:
                cell.set_text_props(color=TEXT_C)
    ax_wf.set_title(
        "Walk-Forward Backtest — True Out-of-Sample Performance  (no look-ahead bias)",
        color=TEXT_C, fontsize=11, fontweight="bold", pad=10)

    # ── Row 5: Lookback backtest bars ─────────────────────────────────────────
    if backtest_results:
        ax_bt = fig.add_subplot(gs[5, :])
        _style_ax(ax_bt)
        bt_periods = [3, 6, 12]
        bt_labels  = ["3 Months Ago", "6 Months Ago", "12 Months Ago"]
        sk = list(PORTFOLIOS.keys())
        x  = np.arange(len(bt_periods))
        offs_bt = np.linspace(-(len(sk) - 1) / 2, (len(sk) - 1) / 2, len(sk)) * 0.22
        for i, key in enumerate(sk):
            vals = []
            for mo in bt_periods:
                bt = backtest_results.get(key, {}).get(mo)
                vals.append(bt["end_value"] if bt else INITIAL_INVESTMENT)
            bars = ax_bt.bar(x + offs_bt[i], vals, 0.22, color=colors[key], alpha=0.82,
                             label=labels[key], edgecolor=PANEL, linewidth=0.7)
            for bar, mo in zip(bars, bt_periods):
                bt = backtest_results.get(key, {}).get(mo)
                if bt:
                    ret = bt["total_return"]
                    tc  = "#4CAF50" if ret >= 0 else "#f44336"
                    ax_bt.text(bar.get_x() + bar.get_width() / 2,
                               bar.get_height() + 80,
                               f"{ret:+.1%}", ha="center", va="bottom",
                               fontsize=8, color=tc, fontweight="bold")
        ax_bt.axhline(INITIAL_INVESTMENT, color="#aaa", linewidth=1.2,
                      linestyle=":", alpha=0.8, label=f"Initial ${INITIAL_INVESTMENT:,}")
        ax_bt.set_xticks(x)
        ax_bt.set_xticklabels(bt_labels, color=TEXT_C, fontsize=10)
        ax_bt.yaxis.set_major_formatter(FuncFormatter(fmt_dollars))
        ax_bt.set_title(
            "Lookback Backtest — Hypothetical $15,000 if Deployed N Months Ago\n"
            "(current weights applied to past prices — look-ahead bias present)",
            color=TEXT_C, fontsize=11, fontweight="bold", pad=10)
        ax_bt.set_ylabel("Portfolio Value", color=TEXT_C)
        ax_bt.legend(loc="upper left", framealpha=0.3, facecolor=BG,
                     labelcolor=TEXT_C, fontsize=9)

    first_key = list(results.keys())[0]
    fig.suptitle(
        f"Advanced Portfolio Dashboard  —  ${INITIAL_INVESTMENT:,} Investment\n"
        f"Historical: {results[first_key]['start']} to {results[first_key]['end']}   |   "
        f"Projection: {PROJECTION_YEARS} years forward",
        color=TEXT_C, fontsize=14, fontweight="bold", y=0.998,
    )
    plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  ✓ Chart saved: {OUTPUT_CHART}")
    plt.close("all")  # non-blocking — chart already saved to disk


# ─── EXCEL EXPORT ─────────────────────────────────────────────────────────────

def export_excel(results: dict, mc_results: dict, current_prices: dict,
                 div_yields: dict | None = None,
                 backtest_results: dict | None = None,
                 wf_results: dict | None = None,
                 stress_results: dict | None = None,
                 regime: dict | None = None,
                 factor_exposures: dict | None = None,
                 val_signals: dict | None = None,
                 risk_contribs: dict | None = None,
                 thematic_exposures: dict | None = None,
                 insider_signals: dict | None = None,
                 short_signals: dict | None = None,
                 drip_data: dict | None = None,
                 rebalance_plan: list | None = None,
                 dca_schedule: list | None = None,
                 options_protection: list | None = None,
                 tax_loss_list: list | None = None):
    from openpyxl.styles import Font, PatternFill

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:

        # Buy plans
        for key, pf in PORTFOLIOS.items():
            plan, _, _ = build_allocation_plan(pf["holdings"], current_prices, div_yields, val_signals)
            df_plan = pd.DataFrame(plan)
            sheet   = f"BuyPlan_{key[:5]}"
            df_plan.to_excel(writer, sheet_name=sheet, index=False)
            ws   = writer.sheets[sheet]
            bold = ws.max_row
            fill = PatternFill("solid", fgColor="2A2D3E")
            for cell in ws[bold]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = fill

        # Summary
        summary = []
        for key, res in results.items():
            mc   = mc_results[key]
            p5e  = np.percentile(mc[:, -1],  5)
            p50e = np.median(mc[:, -1])
            p95e = np.percentile(mc[:, -1], 95)
            summary.append({
                "Scenario":               PORTFOLIOS[key]["label"],
                "Description":            PORTFOLIOS[key]["description"],
                "Initial Investment ($)": INITIAL_INVESTMENT,
                "Final Value ($)":        round(res["final_value"], 2),
                "Total Gain ($)":         round(res["gain"], 2),
                "Total Return":           f"{res['total_return']:.2%}",
                "CAGR":                   f"{res['cagr']:.2%}",
                "Volatility":             f"{res['volatility']:.2%}",
                "Sharpe Ratio":           round(res["sharpe"], 3),
                "Sortino Ratio":          round(res["sortino"], 3),
                "Max Drawdown":           f"{res['max_dd']:.2%}",
                "VaR 95% 1-day ($)":      round(res["var_95"], 2),
                "CVaR 95% 1-day ($)":     round(res["cvar_95"], 2),
                f"MC P5  {PROJECTION_YEARS}yr ($)": round(p5e,  2),
                f"MC P50 {PROJECTION_YEARS}yr ($)": round(p50e, 2),
                f"MC P95 {PROJECTION_YEARS}yr ($)": round(p95e, 2),
                "Analysis Period":        f"{res['start']} to {res['end']}",
            })
        pd.DataFrame(summary).to_excel(writer, sheet_name="Summary", index=False)

        # Allocations
        alloc = []
        for key, pf in PORTFOLIOS.items():
            for ticker, weight in pf["holdings"].items():
                alloc.append({
                    "Scenario":      PORTFOLIOS[key]["label"],
                    "Ticker":        ticker,
                    "Weight":        f"{weight:.0%}",
                    "Dollar Amount": round(INITIAL_INVESTMENT * weight, 2),
                })
        pd.DataFrame(alloc).to_excel(writer, sheet_name="Allocations", index=False)

        # Monthly history
        for key, res in results.items():
            try:
                monthly = res["cumulative"].resample("ME").last()
            except ValueError:
                monthly = res["cumulative"].resample("M").last()
            monthly.name = "Portfolio Value ($)"
            monthly.to_frame().to_excel(writer, sheet_name=f"History_{key[:5]}")

        # Monte Carlo yearly percentiles
        mc_rows: dict = {}
        for key, paths in mc_results.items():
            yi  = [min(int(i * 252), paths.shape[1] - 1) for i in range(PROJECTION_YEARS + 1)]
            lbl = PORTFOLIOS[key]["label"]
            mc_rows[f"{lbl} P5"]  = np.percentile(paths[:, yi],  5, axis=0)
            mc_rows[f"{lbl} P50"] = np.percentile(paths[:, yi], 50, axis=0)
            mc_rows[f"{lbl} P95"] = np.percentile(paths[:, yi], 95, axis=0)
        pd.DataFrame(mc_rows, index=range(PROJECTION_YEARS + 1)).to_excel(
            writer, sheet_name="MonteCarlo", index=True)

        # Walk-forward
        if wf_results:
            wf_rows = []
            for key, wf in wf_results.items():
                wf_rows.append({
                    "Scenario":         PORTFOLIOS[key]["label"],
                    "Train Period":     wf["train_period"],
                    "Test Period":      wf["test_period"],
                    "OOS Total Return": pct_or_na(wf["oos_total"], "+.2%"),
                    "OOS Annualized":   pct_or_na(wf["oos_annualized"], "+.2%"),
                    "SPY Annualized":   (f"{wf['spy_annualized']:+.2%}"
                                        if wf["spy_annualized"] is not None else "N/A"),
                    "Alpha vs SPY":     (f"{wf['alpha']:+.2%}"
                                        if wf["alpha"] is not None else "N/A"),
                })
            pd.DataFrame(wf_rows).to_excel(writer, sheet_name="WalkForward", index=False)

        # Stress test
        if stress_results:
            all_st = []
            for key, rows in stress_results.items():
                for row in rows:
                    all_st.append({"Portfolio": PORTFOLIOS[key]["label"], **row})
            pd.DataFrame(all_st).to_excel(writer, sheet_name="StressTest", index=False)

        # Lookback backtest
        if backtest_results:
            bt_rows = []
            for months, lbl in [(3, "3 Months Ago"), (6, "6 Months Ago"), (12, "12 Months Ago")]:
                row: dict = {"Period": lbl}
                for key in PORTFOLIOS:
                    bt  = backtest_results.get(key, {}).get(months)
                    plbl = PORTFOLIOS[key]["label"]
                    if bt:
                        row[f"{plbl} — End Value ($)"]  = round(bt["end_value"], 2)
                        row[f"{plbl} — Gain ($)"]       = round(bt["gain"], 2)
                        row[f"{plbl} — Total Return"]   = f"{bt['total_return']:+.2%}"
                        row[f"{plbl} — Ann. Return"]    = f"{bt['annualized']:+.2%}"
                    else:
                        for s in ("End Value ($)", "Gain ($)", "Total Return", "Ann. Return"):
                            row[f"{plbl} — {s}"] = "N/A"
                bt_rows.append(row)
            pd.DataFrame(bt_rows).to_excel(writer, sheet_name="Backtest", index=False)

        # Analytics: Factor Exposure + Regime
        analytics: list[dict] = []
        if regime:
            for k, v in regime.items():
                analytics.append({"Category": "Market Regime", "Key": k, "Value": str(v)})
        if factor_exposures:
            for key, exp in factor_exposures.items():
                plbl = PORTFOLIOS[key]["label"]
                for k, v in exp.items():
                    analytics.append({"Category": f"Factor Exposure — {plbl}",
                                      "Key": k, "Value": str(v)})
        if analytics:
            pd.DataFrame(analytics).to_excel(writer, sheet_name="Analytics", index=False)

        # Risk Contribution
        if risk_contribs:
            rc_rows = []
            for key, rows in risk_contribs.items():
                for row in rows:
                    rc_rows.append({"Scenario": PORTFOLIOS[key]["label"], **row})
            if rc_rows:
                pd.DataFrame(rc_rows).to_excel(writer, sheet_name="RiskContrib", index=False)

        # Thematic Exposure
        if thematic_exposures:
            th_rows = []
            for key, th_list in thematic_exposures.items():
                for theme, weight in th_list:
                    th_rows.append({
                        "Scenario": PORTFOLIOS[key]["label"],
                        "Theme": theme,
                        "Exposure": f"{weight:.1%}",
                    })
            if th_rows:
                pd.DataFrame(th_rows).to_excel(writer, sheet_name="Thematic", index=False)

        # Insider + Short signals
        sig_rows = []
        all_sigs_tickers = set()
        if insider_signals:
            all_sigs_tickers |= set(insider_signals.keys())
        if short_signals:
            all_sigs_tickers |= set(short_signals.keys())
        for t in sorted(all_sigs_tickers):
            ins  = (insider_signals or {}).get(t, {})
            shrt = (short_signals   or {}).get(t, {})
            sig_rows.append({
                "Ticker":           t,
                "Insider Signal":   ins.get("signal", "—"),
                "Net Buys (90d)":   ins.get("net_buys", "—"),
                "Short % Float":    f"{shrt['short_pct']:.1%}" if shrt.get("short_pct") else "—",
                "Short Ratio":      shrt.get("short_ratio", "—"),
                "Short Signal":     shrt.get("signal", "—"),
            })
        if sig_rows:
            pd.DataFrame(sig_rows).to_excel(writer, sheet_name="Signals", index=False)

        # DRIP
        if drip_data:
            drip_rows = []
            for key, dd in drip_data.items():
                if not dd or not dd.get("projection"):
                    continue
                for yr, (no_d, with_d) in dd["projection"].items():
                    drip_rows.append({
                        "Scenario":         PORTFOLIOS[key]["label"],
                        "Year":             yr,
                        "Without DRIP ($)": no_d,
                        "With DRIP ($)":    with_d,
                        "DRIP Bonus ($)":   round(with_d - no_d, 2),
                    })
            if drip_rows:
                pd.DataFrame(drip_rows).to_excel(writer, sheet_name="DRIP", index=False)

        # Rebalance Plan
        if rebalance_plan:
            pd.DataFrame(rebalance_plan).to_excel(
                writer, sheet_name="Rebalance", index=False)

        # DCA Schedule
        if dca_schedule:
            pd.DataFrame(dca_schedule).to_excel(
                writer, sheet_name="DCA", index=False)

        # Options protection
        if options_protection:
            pd.DataFrame(options_protection).to_excel(
                writer, sheet_name="OptionsRef", index=False)

        # Tax-loss harvesting
        if tax_loss_list:
            pd.DataFrame(tax_loss_list).to_excel(
                writer, sheet_name="TaxLoss", index=False)

    print(f"  ✓ Excel saved: {OUTPUT_EXCEL}")


# ─── ENHANCED ANALYSIS ────────────────────────────────────────────────────────

def deployment_timing_recommendation(capital: float,
                                     regime: str,
                                     val_signals: dict | None,
                                     spy_volatility_30d: float | None = None) -> dict:
    """
    Recommends LUMP SUM vs staged deployment (DCA) for a given capital amount.

    Inputs
    ------
    capital            : total dollar amount to deploy
    regime             : "BULL" | "BEAR" | "NEUTRAL" from detect_market_regime()
    val_signals        : {ticker: {"verdict": ...}} from the valuation pass
    spy_volatility_30d : annualised 30-day SPY vol (fetched here if None)

    Output
    ------
    dict with keys: strategy, tranches, interval_days, schedule, rationale, confidence
    """
    # ── Fetch SPY vol if not supplied ────────────────────────────────────────
    if spy_volatility_30d is None:
        try:
            spy = yf.download("SPY", period="60d", progress=False, auto_adjust=True)["Close"]
            spy_volatility_30d = float(spy.pct_change().dropna().std() * np.sqrt(252))
        except Exception:
            spy_volatility_30d = 0.18  # fallback: typical long-run SPY vol

    vol = spy_volatility_30d

    # ── Valuation heat: % of signals that are OVERVALUED ─────────────────────
    if val_signals:
        verdicts  = [v.get("verdict", "—") for v in val_signals.values()
                     if v.get("verdict") not in ("—", None)]
        n_over    = sum(1 for v in verdicts if v == "OVERVALUED")
        n_under   = sum(1 for v in verdicts if v == "UNDERVALUED")
        n_total   = len(verdicts) or 1
        over_pct  = n_over / n_total
        under_pct = n_under / n_total
    else:
        over_pct = under_pct = 0.0

    # ── Scoring: each unfavourable signal adds tranches ───────────────────────
    extra_tranches = 0
    rationale      = []

    # Regime — accept either string "Bull"/"Bear"/"Neutral" or the full dict from detect_regime()
    regime_str = regime["regime"] if isinstance(regime, dict) else str(regime)
    if regime_str.upper() == "BEAR":
        extra_tranches += 2
        rationale.append("Market is in a BEAR regime (SPY below 200-day MA) — phasing in reduces entry risk.")
    elif regime_str.upper() == "BULL":
        rationale.append("Market is in a BULL regime — lump-sum historically outperforms DCA ~66% of the time.")
    else:
        extra_tranches += 1
        rationale.append("Market regime is NEUTRAL — moderate staging provides a safety margin.")

    # Volatility
    if vol > 0.28:
        extra_tranches += 2
        rationale.append(f"Elevated 30-day SPY volatility ({vol:.0%} ann.) — DCA smooths entry price.")
    elif vol > 0.20:
        extra_tranches += 1
        rationale.append(f"Moderate SPY volatility ({vol:.0%} ann.) — slight caution warranted.")
    else:
        rationale.append(f"Low SPY volatility ({vol:.0%} ann.) — favours immediate deployment.")

    # Valuation heat
    if over_pct > 0.50:
        extra_tranches += 1
        rationale.append(f"{over_pct:.0%} of target stocks appear OVERVALUED — patience may yield better entry points.")
    elif under_pct > 0.40:
        rationale.append(f"{under_pct:.0%} of target stocks appear UNDERVALUED — favours buying sooner.")

    # Capital size
    if capital >= 50_000:
        extra_tranches += 2
        rationale.append(f"Large deployment (${capital:,.0f}) — staging limits single-day market-impact risk.")
    elif capital >= 20_000:
        extra_tranches += 1
        rationale.append(f"Mid-size deployment (${capital:,.0f}) — a modest split captures price variation.")
    else:
        rationale.append(f"Smaller deployment (${capital:,.0f}) — transaction costs favour fewer tranches.")

    # ── Final tranche count ───────────────────────────────────────────────────
    n_tranches = max(1, 1 + extra_tranches)

    if n_tranches == 1:
        strategy    = "LUMP SUM"
        interval    = 0
        confidence  = "HIGH"
        summary     = "Deploy all capital on Day 1."
    elif n_tranches == 2:
        strategy    = "2-STAGE"
        interval    = 14          # bi-weekly
        confidence  = "HIGH"
        summary     = "Split 50/50 across two entries ~2 weeks apart."
    elif n_tranches == 3:
        strategy    = "3-STAGE (Monthly)"
        interval    = 30
        confidence  = "MEDIUM"
        summary     = "Deploy in thirds over 3 months (~30 days apart)."
    elif n_tranches == 4:
        strategy    = "4-STAGE (Bi-weekly)"
        interval    = 14
        confidence  = "MEDIUM"
        summary     = "Deploy in four equal tranches every 2 weeks."
    else:
        n_tranches  = 6
        strategy    = "6-STAGE (Monthly)"
        interval    = 30
        confidence  = "LOW"
        summary     = "High-caution environment — spread over 6 months."

    # ── Build concrete schedule ───────────────────────────────────────────────
    tranche_amt = round(capital / n_tranches, 2)
    today       = datetime.now()
    schedule    = []
    for i in range(n_tranches):
        deploy_date = today + timedelta(days=interval * i)
        schedule.append({
            "Tranche":    i + 1,
            "Date":       deploy_date.strftime("%Y-%m-%d"),
            "Amount ($)": tranche_amt,
            "Note":       "Initial buy — focus on highest-conviction positions" if i == 0
                          else "Top-up underweight positions at current prices",
        })

    return {
        "strategy":    strategy,
        "summary":     summary,
        "tranches":    n_tranches,
        "tranche_amt": tranche_amt,
        "interval_days": interval,
        "schedule":    schedule,
        "rationale":   rationale,
        "confidence":  confidence,
        "spy_vol_30d": round(vol, 4),
        "market_regime": regime,
        "over_pct":    round(over_pct, 4),
        "under_pct":   round(under_pct, 4),
    }


REBAL_BAND    = 0.25   # trim only when a position is >25% above its target weight
MIN_TRADE_USD = 100    # ignore buys/trims smaller than this

# "Keep what I have" (profile["keep_existing"]): a holding the model doesn't pick is
# KEPT, not sold. A sale is suggested only on one of these negative signals, or when
# the holding is over the position cap by more than REBAL_BAND. STARTING VALUES.
EXIT_CATALYST_SCORE  = -0.7    # FDA rejection −1.0, trial failure −0.8, clinical hold −0.7
EXIT_REVISION_SCORE  = -0.5    # analysts broadly cutting estimates
EXIT_STRUCTURAL_SEV  = 0.7     # structural/competitive risk severity
EARNINGS_STAGGER_DAYS = 14     # new buys reporting within this many days → split the buy


def exit_signals(tickers, catalyst_signals=None, revision_signals=None,
                 structural=None) -> dict:
    """{ticker: reason} for holdings with a concrete reason to consider selling."""
    out = {}
    for t in tickers:
        reasons = []
        c = (catalyst_signals or {}).get(t) or {}
        if c.get("score") is not None and c["score"] <= EXIT_CATALYST_SCORE:
            reasons.append(f"{c.get('catalyst_type', 'negative event')}: "
                           f"{str(c.get('headline', ''))[:80]}")
        r = (revision_signals or {}).get(t) or {}
        if r.get("score") is not None and r["score"] <= EXIT_REVISION_SCORE:
            reasons.append(f"analysts cutting estimates (revision score {r['score']:+.2f})")
        s = (structural or {}).get(t) or {}
        if s.get("severity", 0) >= EXIT_STRUCTURAL_SEV:
            reasons.append(f"structural risk: {s.get('risk')} (severity {s['severity']})")
        if reasons:
            out[t] = "; ".join(reasons)
    return out


def load_tax_lots(profile_id: str) -> dict:
    """Cost basis / holding period per ticker from portfolio_tracker's state file
    (portfolio_state_<profile>.json). Empty dict when trades haven't been logged."""
    try:
        from pathlib import Path
        import portfolio_tracker as _pt
        path = Path(__file__).parent / f"portfolio_state_{profile_id}.json"
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as fh:
            return _pt._compute_positions(_json.load(fh))
    except Exception as exc:
        print(f"      ⚠ Could not read tax lots for '{profile_id}' ({exc})")
        return {}


def _tax_note(ticker: str, lot: dict | None, price: float) -> str:
    """One-line tax context for a sell/trim suggestion in a taxable account."""
    if not lot:
        return ("Cost basis unknown — log your purchase with "
                "`portfolio_tracker.py log buy` before selling")
    lots = lot.get("lots")
    if lots:
        lt = [l for l in lots if l["term"] == "LT"]
        st = [l for l in lots if l["term"] == "ST"]
        g  = lambda ls: sum(l["shares"] * price - l["cost"] for l in ls)
        parts = []
        if lt:
            parts.append(f"{sum(l['shares'] for l in lt):g} sh long-term (gain ${g(lt):,.0f})")
        if st:
            nxt = min(l["lt_date"] for l in st)
            parts.append(f"{sum(l['shares'] for l in st):g} sh SHORT-term (gain ${g(st):,.0f}, "
                         f"long-term from {nxt})")
        tip = ""
        if lt:
            best = max(lt, key=lambda l: l["price"])
            tip = (f" — if trimming, use specific-lot ID and sell the highest-cost long-term "
                   f"lot first ({best['date']} @ ${best['price']:,.2f}); FIFO would sell the "
                   f"oldest, lowest-cost lot and realise the biggest gain")
        if g(lots) < 0:
            tip = " — unrealised LOSS overall: selling harvests it (watch the 30-day wash-sale rule)"
        return "; ".join(parts) + tip
    gain = lot["shares"] * price - lot["cost_basis"]
    if gain < 0:
        return f"Unrealised LOSS ${-gain:,.0f} — selling harvests it (watch the 30-day wash-sale rule)"
    if lot.get("days_held", 0) < 365:
        lt_date = (datetime.strptime(lot["first_bought"], "%Y-%m-%d")
                   + timedelta(days=366)).strftime("%Y-%m-%d")
        return (f"SHORT-TERM gain ${gain:,.0f} (taxed as income) — becomes long-term "
                f"on {lt_date}; consider waiting or rebalancing with new cash instead")
    return f"Long-term gain ${gain:,.0f} (lower capital-gains rate)"


def build_personalized_plan(profile: dict, target_weights: dict,
                            current_prices: dict,
                            div_yields: dict | None = None,
                            val_signals: dict | None = None,
                            tax_lots: dict | None = None,
                            too_new: set | None = None,
                            sell_signals: dict | None = None,
                            position_cap: float | None = None,
                            binary_set: set | None = None,
                            earnings_days: dict | None = None) -> dict:
    """
    Translate optimal target weights into a personalised action plan for one investor.

    Handles three situations in a single function:
      - Fresh investor (no holdings): straightforward allocation of capital
      - Existing portfolio + new cash: reuse existing positions, fill gaps with new cash
      - Existing portfolio + pending capital: show now vs. in N days

    Tax-aware for taxable accounts: rebalancing is done with new cash first;
    TRIM only fires outside a ±REBAL_BAND drift band; trades under MIN_TRADE_USD
    are skipped; every TRIM / EVALUATE SELL carries a tax note (short-term gain,
    harvestable loss, or unknown basis) from the tracker's logged lots.
    """
    existing_shares   = profile.get("existing_shares", {})
    frac              = bool(profile.get("fractional_shares"))

    def _sh(amount: float, price: float):
        """Shares affordable with `amount`: whole shares, or 3-decimal fractions when
        the investor's broker supports them (profile["fractional_shares"])."""
        if price <= 0 or amount <= 0:
            return 0
        return math.floor(amount / price * 1000) / 1000 if frac else int(amount / price)

    immediate_cap     = float(profile.get("capital", 0))
    pending           = profile.get("pending_capital", {})
    pending_amt       = float(pending.get("amount", 0))
    pending_days      = int(pending.get("days", 0))
    monthly           = float(profile.get("monthly_contribution", 0))
    forced_min        = profile.get("forced_min_weight", {})   # e.g. {"GH": 0.08}

    # Cash reserved for anticipated events (e.g. an upcoming IPO) — withheld from deployment
    cash_reserves     = profile.get("cash_reserves", [])
    reserved_total    = sum(float(r.get("amount", 0)) for r in cash_reserves)
    immediate_cap     = max(immediate_cap - reserved_total, 0.0)

    # ── Step 1: value existing holdings at current prices ────────────────────
    existing_positions: dict = {}
    existing_value = 0.0
    for ticker, shares in existing_shares.items():
        px = current_prices.get(ticker) or 0.0
        if px > 0 and shares > 0:
            val = shares * px
            existing_value += val
            existing_positions[ticker] = {
                "shares": shares,
                "price":  round(px, 2),
                "value":  round(val, 2),
            }

    total_now     = existing_value + immediate_cap
    total_later   = total_now + pending_amt

    # ── Keep-what-I-have: holdings outside the target stay put unless flagged ─
    keep_mode = bool(profile.get("keep_existing"))
    sell_signals = sell_signals or {}
    cap = position_cap or 1.0
    kept: dict = {}            # ticker → value kept (not in target, no exit signal)
    if keep_mode:
        for t, pos in existing_positions.items():
            if t in target_weights or t in forced_min:
                continue
            if t in sell_signals:
                continue
            kept[t] = pos["value"]
    kept_val = sum(kept.values())
    # Target weights apply to what's left after the kept holdings
    base_now   = max(total_now - kept_val, 0.0)
    base_later = max(total_later - kept_val, 0.0)

    # ── Step 2: compute per-ticker targets and actions ───────────────────────
    # Include forced_min tickers even if they're not in the model's target
    all_tickers = sorted(set(target_weights) | set(existing_shares) | set(forced_min))
    positions   = []
    for ticker in all_tickers:
        model_w    = target_weights.get(ticker, 0.0)
        floor_w    = forced_min.get(ticker, 0.0)
        tgt_w      = max(model_w, floor_w)           # apply manual floor
        is_override = floor_w > model_w and floor_w > 0
        tgt_now    = base_now   * tgt_w
        tgt_later  = base_later * tgt_w
        cur        = existing_positions.get(ticker, {})
        cur_val    = cur.get("value", 0.0)
        cur_shares = cur.get("shares", 0)
        px         = current_prices.get(ticker) or 0.0

        delta_now   = tgt_now  - cur_val
        delta_later = tgt_later - cur_val
        shares_now  = _sh(delta_now, px)
        shares_later= _sh(delta_later, px)

        overweight = cur_val > tgt_now * (1 + REBAL_BAND) and -delta_now >= MIN_TRADE_USD
        exit_reason = ""
        if ticker not in target_weights and ticker in sell_signals and cur_val > 0:
            # Exit signal or the investor's own sell intent — even for too-new names
            action = "EVALUATE SELL"
            exit_reason = sell_signals[ticker]
        elif ticker not in target_weights and ticker in (too_new or set()):
            # <1 year of prices — the optimizer can't estimate its risk, so a 0%
            # target means "no model view", not "sell"
            action = "HOLD — too new to model"
        elif ticker in kept:
            # Kept holding: trim only the excess over cap × (1 + band)
            limit = total_now * cap
            if cur_val > limit * (1 + REBAL_BAND) and cur_val - limit >= MIN_TRADE_USD:
                tgt_now = tgt_later = limit
                delta_now = delta_later = limit - cur_val
                exit_reason = f"over the {cap:.0%} position cap — trim back to it"
                action = "TRIM"
            else:
                action = "KEEP"
        elif ticker not in target_weights and ticker not in forced_min:
            action = "EVALUATE SELL"
            exit_reason = sell_signals.get(ticker, "")
        elif cur_val == 0:
            action = "NEW BUY" if delta_now >= MIN_TRADE_USD else "SKIP (< $100)"
        elif delta_now >= MIN_TRADE_USD:
            action = "BUY MORE"
        elif overweight:
            action = "TRIM"
        else:
            action = "HOLD"            # inside the drift band — no trade, no tax event
        # No routine short-term sales (profile["no_short_term_sales"]): a TRIM uses
        # long-term shares only. Exit-signal sales (EVALUATE SELL) are still allowed —
        # that's the "unless something happens" case.
        lot = (tax_lots or {}).get(ticker) or {}
        if profile.get("no_short_term_sales") and action == "TRIM" and lot.get("lots") and px:
            lt_sh  = float(lot.get("lt_shares", 0) or 0)
            lt_val = lt_sh * px
            if lt_val < cur_val - tgt_now:
                if lt_val >= MIN_TRADE_USD:
                    tgt_now, delta_now = cur_val - lt_val, -lt_val
                    note = f"trim limited to the {lt_sh:g} long-term shares (no short-term sales)"
                else:
                    action = "HOLD — short-term"
                    nxt = min((l["lt_date"] for l in lot["lots"] if l["term"] == "ST"),
                              default="—")
                    note = f"overweight, but the shares are short-term until {nxt} — trim then"
                exit_reason = f"{exit_reason}; {note}" if exit_reason else note
        tax_note = (_tax_note(ticker, (tax_lots or {}).get(ticker), px)
                    if action in ("TRIM", "EVALUATE SELL") and cur_val > 0 and px else "")

        vsig      = (val_signals or {}).get(ticker, {})
        verdict   = vsig.get("verdict", "—")
        buy_low   = vsig.get("buy_low")
        buy_high  = vsig.get("buy_high")
        dcf_range = (f"${buy_low:,.0f}–${buy_high:,.0f}"
                     if buy_low and buy_high else "—")
        dy        = (div_yields or {}).get(ticker, 0.0)

        positions.append({
            "Ticker":             ticker,
            "In Target":          ticker in target_weights or ticker in forced_min,
            "Target Wt":          f"{tgt_w:.1%}",
            "Model Wt":           f"{model_w:.1%}",
            "Override":           is_override,
            "Override Floor":     f"{floor_w:.1%}" if is_override else "",
            "Target Now ($)":     round(tgt_now, 2),
            "Target w/Pending ($)": round(tgt_later, 2),
            "Current Shares":     cur_shares,
            "Current Value ($)":  round(cur_val, 2),
            "Price ($)":          round(px, 2) if px else "N/A",
            "Shares to Buy Now":  shares_now,
            "Cost Now ($)":       round(shares_now * px, 2) if px else 0,
            "Add w/Pending ($)":  round((shares_later - shares_now) * px, 2) if px and pending_amt else 0,
            "Action":             action,
            "Div Yield":          f"{dy:.2%}" if dy else "—",
            "Verdict":            verdict,
            "DCF Range ($)":      dcf_range,
            "Tax Note":           tax_note,
            "Exit Reason":        exit_reason,
        })

    # Sort: TRIM/SELL first (review), then BUY MORE, NEW BUY, HOLD
    _order = {"TRIM": 0, "EVALUATE SELL": 1, "BUY MORE": 2, "NEW BUY": 3, "HOLD": 4,
              "KEEP": 4, "HOLD — too new to model": 4, "HOLD — short-term": 4,
              "SKIP (< $100)": 5}
    positions.sort(key=lambda x: (_order.get(x["Action"], 5), -x["Target Now ($)"]))

    # ── Step 3: immediate capital deployment (fills BUY/NEW positions first) ─
    buy_rows   = [p for p in positions if p["Action"] in ("BUY MORE", "NEW BUY")]
    remaining  = immediate_cap
    deployment = []
    for p in sorted(buy_rows, key=lambda x: -x["Target Now ($)"]):
        if remaining <= 2:
            break
        alloc  = min(max(float(p["Target Now ($)"] - p["Current Value ($)"]), 0), remaining)
        px     = current_prices.get(p["Ticker"]) or 0.0
        shrs   = _sh(alloc, px)
        cost   = round(shrs * px, 2)
        remaining -= cost
        if shrs > 0:
            row = {
                "Ticker":    p["Ticker"],
                "Action":    p["Action"],
                "Shares":    shrs,
                "Price ($)": round(px, 2),
                "Cost ($)":  cost,
                "Verdict":   p["Verdict"],
            }
            d = (earnings_days or {}).get(p["Ticker"])
            if d is not None and 0 <= d <= EARNINGS_STAGGER_DAYS:
                row["Note"] = (f"reports earnings in {d}d — consider buying half now and "
                               f"half after the report")
            deployment.append(row)

    # ── Step 4: pending-capital deployment (2 weeks out) ─────────────────────
    pending_deployment = []
    if pending_amt > 0:
        rem2 = pending_amt + remaining  # leftover from now + new cash
        for p in sorted(buy_rows, key=lambda x: -x["Target w/Pending ($)"]):
            if rem2 <= 2:
                break
            already_bought = next(
                (d["Cost ($)"] for d in deployment if d["Ticker"] == p["Ticker"]), 0)
            still_needed = max(
                float(p["Target w/Pending ($)"]) - float(p["Current Value ($)"]) - already_bought, 0)
            alloc = min(still_needed, rem2)
            px    = current_prices.get(p["Ticker"]) or 0.0
            shrs  = _sh(alloc, px)
            cost  = round(shrs * px, 2)
            rem2 -= cost
            if shrs > 0:
                pending_deployment.append({
                    "Ticker":    p["Ticker"],
                    "Action":    "BUY (pending)",
                    "Shares":    shrs,
                    "Price ($)": round(px, 2),
                    "Cost ($)":  cost,
                    "Note":      f"Deploy when ${pending_amt:,.0f} arrives in ~{pending_days}d",
                })

    # ── Step 5: monthly contribution schedule ────────────────────────────────
    # Each month goes to the 1–2 most underweight targets (projected at today's
    # prices) instead of many small buys; re-check at the time of contribution.
    # Targets grow with each contribution: month i targets w × (base + i × monthly).
    deployed = {d["Ticker"]: d["Cost ($)"] for d in deployment}
    tgt_wts  = {t: max(target_weights.get(t, 0.0), forced_min.get(t, 0.0))
                for t in set(target_weights) | set(forced_min)}
    held_val = {p["Ticker"]: float(p["Current Value ($)"]) + deployed.get(p["Ticker"], 0.0)
                for p in positions}
    no_add   = {p["Ticker"] for p in positions if p["Action"] in ("TRIM", "EVALUATE SELL")}
    contrib_schedule = []
    if monthly > 0:
        today = datetime.now()
        for i in range(1, 7):
            date = today + timedelta(days=30 * i)
            base_i = base_now + monthly * i
            gaps = {t: w * base_i - held_val.get(t, 0.0) for t, w in tgt_wts.items()
                    if t not in no_add}
            left, picks = float(monthly), []
            for t in sorted((t for t, g in gaps.items() if g >= MIN_TRADE_USD),
                            key=lambda t: -gaps[t])[:2]:
                amt = min(gaps[t], left)
                if amt < MIN_TRADE_USD:
                    break
                picks.append(f"{t} ${amt:,.0f}")
                held_val[t] = held_val.get(t, 0.0) + amt
                left -= amt
            contrib_schedule.append({
                "Month":          i,
                "Date":           date.strftime("%Y-%m-%d"),
                "Amount ($)":     monthly,
                "Cumulative ($)": round(monthly * i, 2),
                "Suggestion":     (" + ".join(picks) + " (most underweight; re-check prices)"
                                   if picks else
                                   "Targets filled — add to the most underweight position"),
            })

    # ── Binary-event exposure after this plan (kept holdings included) ───────
    binary_exposure = None
    if binary_set:
        post = {}
        for p in positions:
            t = p["Ticker"]
            v = float(p["Current Value ($)"]) + deployed.get(t, 0.0)
            if p["Action"] in ("TRIM", "EVALUATE SELL"):
                v = min(v, float(p["Target Now ($)"]))
            post[t] = v
        names = {t: v for t, v in post.items() if t in binary_set and v > 0}
        val = sum(names.values())
        binary_exposure = {"value": round(val, 2),
                           "pct": val / total_now if total_now else 0.0,
                           "names": sorted(names), "limit": BINARY_EVENT_MAX}

    return {
        "name":                  profile["name"],
        "note":                  profile.get("note", ""),
        "existing_value":        round(existing_value, 2),
        "immediate_capital":     immediate_cap,
        "pending_capital":       pending_amt,
        "pending_days":          pending_days,
        "total_now":             round(total_now, 2),
        "total_later":           round(total_later, 2),
        "monthly_contribution":  monthly,
        "cash_reserves":         cash_reserves,
        "reserved_total":        round(reserved_total, 2),
        "bl_views":              profile.get("bl_views", []),
        "positions":             positions,
        "deployment_now":        deployment,
        "cash_leftover_now":     round(remaining, 2),
        "pending_deployment":    pending_deployment,
        "contribution_schedule": contrib_schedule,
        "binary_exposure":       binary_exposure,
        "fractional_shares":     frac,
    }


def compute_rebalance_plan(target_holdings: dict, current_holdings: dict,
                           current_prices: dict) -> list[dict]:
    """
    Compare CURRENT_HOLDINGS (dollar values) to target allocation.
    Returns rows with BUY / SELL / HOLD action and dollar deltas.
    """
    if not current_holdings or not target_holdings:
        return []
    total_target = INITIAL_INVESTMENT
    rows = []
    all_tickers = set(target_holdings.keys()) | set(current_holdings.keys())
    for t in sorted(all_tickers):
        target_val  = total_target * target_holdings.get(t, 0.0)
        current_val = float(current_holdings.get(t, 0.0))
        delta       = target_val - current_val
        price       = current_prices.get(t) or 0.0
        shares_delta = int(delta / price) if price > 0 else 0
        if abs(delta) < 10:
            action = "HOLD"
        elif delta > 0:
            action = "BUY"
        else:
            action = "SELL"
        rows.append({
            "Ticker":           t,
            "Current Value ($)": round(current_val, 2),
            "Target Value ($)":  round(target_val, 2),
            "Delta ($)":         round(delta, 2),
            "Shares Delta":      shares_delta,
            "Action":            action,
        })
    return sorted(rows, key=lambda x: abs(x["Delta ($)"]), reverse=True)


def correlation_analysis(holdings: dict, prices: pd.DataFrame) -> tuple:
    """
    Returns (corr_matrix_df, high_corr_pairs) for portfolio holdings.
    High-correlation pairs = absolute correlation > 0.80 (excluding self).
    """
    tickers = [t for t in holdings if t in prices.columns and holdings[t] > 0]
    if len(tickers) < 2:
        return pd.DataFrame(), []
    sub  = prices[tickers].ffill().dropna()
    rets = sub.pct_change().dropna()
    corr = rets.corr()
    pairs = []
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if j <= i:
                continue
            c = corr.loc[t1, t2] if t1 in corr.index and t2 in corr.columns else 0.0
            if abs(c) >= 0.80:
                pairs.append({
                    "Ticker A":    t1,
                    "Ticker B":    t2,
                    "Correlation": round(float(c), 3),
                    "Risk Note":   "HIGH" if abs(c) >= 0.90 else "ELEVATED",
                })
    pairs.sort(key=lambda x: -abs(x["Correlation"]))
    return corr, pairs


def risk_contribution(holdings: dict, prices: pd.DataFrame) -> list[dict]:
    """Marginal risk contribution per position as % of total portfolio variance."""
    tickers = [t for t in holdings if t in prices.columns and holdings[t] > 0]
    if not tickers:
        return []
    w     = np.array([holdings[t] for t in tickers])
    w    /= w.sum()
    sub   = prices[tickers].ffill().dropna()
    rets  = sub.pct_change().dropna().values
    cov   = np.cov(rets, rowvar=False) * 252
    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return []
    marginal = cov @ w
    contrib  = w * marginal / port_var
    rows = []
    for i, t in enumerate(tickers):
        rows.append({
            "Ticker":             t,
            "Weight":             f"{holdings[t]:.1%}",
            "Risk Contribution":  f"{contrib[i]:.1%}",
            "Risk/Weight Ratio":  round(contrib[i] / holdings[t], 2) if holdings[t] > 0 else 0,
        })
    return sorted(rows, key=lambda x: -float(x["Risk Contribution"].rstrip("%")))


def build_dca_schedule(holdings: dict, current_prices: dict,
                       tranches: int = DCA_TRANCHES,
                       interval_days: int = DCA_INTERVAL) -> list[dict]:
    """
    Split the investment across `tranches` purchases, `interval_days` apart.
    Each tranche gets an equal share of INITIAL_INVESTMENT.
    """
    if tranches <= 1:
        return []
    tranche_amt = INITIAL_INVESTMENT / tranches
    rows = []
    today = datetime.now()
    for i in range(tranches):
        date     = today + timedelta(days=i * interval_days)
        date_str = date.strftime("%Y-%m-%d")
        for ticker, weight in sorted(holdings.items(), key=lambda x: -x[1]):
            target_val = tranche_amt * weight
            price      = current_prices.get(ticker) or 0.0
            shares     = int(target_val / price) if price > 0 else 0
            cost       = shares * price
            rows.append({
                "Tranche":      i + 1,
                "Date":         date_str,
                "Ticker":       ticker,
                "Allocation ($)": round(target_val, 2),
                "Price ($)":    round(price, 2),
                "Shares":       shares,
                "Cost ($)":     round(cost, 2),
            })
    return rows


def fetch_earnings_calendar(tickers: list) -> dict:
    """
    Returns {ticker: {"date": "YYYY-MM-DD", "days_away": int}} for tickers
    with earnings within EARNINGS_WARN_DAYS. Others return {}.
    ETFs/index funds are skipped — they don't have earnings dates.
    """
    _skip  = INDEX_TICKERS | INTERNATIONAL_TICKERS
    tickers = [t for t in tickers if t not in _skip]
    today  = datetime.now().date()

    def _get(t):
        try:
            cal = _safe_attr(t, "calendar")
            if cal is None:
                return t, {}
            if isinstance(cal, dict):
                date_val = cal.get("Earnings Date")
                if date_val is None:
                    return t, {}
                if isinstance(date_val, (list, tuple)):
                    date_val = date_val[0]
            else:
                if hasattr(cal, "iloc"):
                    date_val = cal.iloc[0, 0] if not cal.empty else None
                else:
                    return t, {}
            if date_val is None:
                return t, {}
            ed = pd.Timestamp(date_val).date()
            days_away = (ed - today).days
            if 0 <= days_away <= EARNINGS_WARN_DAYS:
                return t, {"date": str(ed), "days_away": days_away}
            return t, {}
        except Exception:
            return t, {}

    with ThreadPoolExecutor(max_workers=8) as pool:
        raw = dict(pool.map(_get, tickers))
    return {t: v for t, v in raw.items() if v}


def fetch_insider_signals(tickers: list) -> dict:
    """
    Returns {ticker: {"net_buys": int, "signal": str}} using yfinance insider_transactions.
    net_buys = (# Buy transactions) - (# Sell transactions) in last 90 days.
    ETFs/index funds are skipped — they have no insider transactions.
    """
    _skip  = INDEX_TICKERS | INTERNATIONAL_TICKERS
    tickers = [t for t in tickers if t not in _skip]
    cutoff = datetime.now() - timedelta(days=90)

    def _get(t):
        try:
            ins = _safe_attr(t, "insider_transactions")
            if ins is None or ins.empty:
                return t, {}
            cols = {c.lower().replace(" ", "_"): c for c in ins.columns}
            date_col  = cols.get("start_date") or cols.get("date") or ins.columns[0]
            trans_col = cols.get("transaction") or cols.get("text") or ""
            ins["_date"] = pd.to_datetime(ins[date_col], errors="coerce")
            recent = ins[ins["_date"] >= cutoff]
            buys  = sells = 0
            if trans_col:
                buys  = recent[trans_col].str.contains("Buy|Purchase",  case=False, na=False).sum()
                sells = recent[trans_col].str.contains("Sell|Sale",     case=False, na=False).sum()
            net = int(buys) - int(sells)
            signal = "BULLISH" if net > 0 else ("BEARISH" if net < 0 else "NEUTRAL")
            return t, {"net_buys": net, "signal": signal}
        except Exception:
            return t, {}

    with ThreadPoolExecutor(max_workers=8) as pool:
        raw = dict(pool.map(_get, tickers))
    return {t: v for t, v in raw.items() if v}


_LEGAL_WORDS = {"inc", "corp", "corporation", "co", "company", "ltd", "llc", "plc", "se",
                "nv", "ag", "sa", "holding", "holdings", "group", "and", "the", "adr"}


def company_key(name: str) -> str:
    """Normalised two-word company key: 'Scholar Rock Holding Corporation' and
    'Scholar Rock, Inc.' → 'scholar rock'; 'Roche Holding AG' → 'roche'."""
    import re
    words = [w for w in re.sub(r"[^a-z0-9 ]", " ", (name or "").lower()).split()
             if w not in _LEGAL_WORDS]
    return " ".join(words[:2])


# Legal entities big companies file trials / NDAs under. Keyed by company_key(longName).
SPONSOR_ALIASES: dict[str, list[str]] = {
    "roche":           ["Hoffmann-La Roche", "Genentech", "Chugai Pharmaceutical"],
    "merck":           ["Merck Sharp & Dohme"],
    "johnson johnson": ["Janssen Research & Development", "Janssen Pharmaceutical",
                        "Janssen Biotech"],
    "novartis":        ["Novartis Pharmaceuticals"],
    "gsk":             ["GlaxoSmithKline"],
    "astrazeneca":     ["Alexion Pharmaceuticals"],
}

GNEWS_RSS = "https://news.google.com/rss/search"


def google_news(query: str) -> list[tuple[str, str, datetime]]:
    """
    Fallback news source: Google News RSS (free, no API key; aggregates Reuters,
    Bloomberg, trade press). Returns (headline, source, published-UTC-naive).
    Disk-cached 4h like other Yahoo attributes.
    """
    import re
    import requests
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    key = "GNEWS_" + re.sub(r"\W+", "_", query.lower())[:60]
    cached = _attr_cache_load(key, "gnews")
    if cached is not _YF_SENTINEL:
        return cached
    out: list = []
    try:
        r = requests.get(GNEWS_RSS, timeout=15,
                         params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                         headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            for it in ET.fromstring(r.content).iter("item"):
                title = it.findtext("title") or ""
                src   = it.findtext("source") or ""
                try:
                    dt = parsedate_to_datetime(it.findtext("pubDate") or "")
                    dt = dt.astimezone(tz=None).replace(tzinfo=None) if dt.tzinfo else dt
                except Exception:
                    continue
                if title:
                    out.append((title, src, dt))
        else:
            print(f"      ⚠ Google News HTTP {r.status_code} for '{query}'")
    except Exception as exc:
        print(f"      ⚠ Google News failed for '{query}' ({type(exc).__name__})")
        return out                                  # don't cache failures
    _attr_cache_save(key, "gnews", out)
    return out


def _is_subject(headline: str, company_first_word: str, max_pos: int = 15) -> bool:
    """
    Is the company the SUBJECT of the headline? A headline that names the company
    only late ("GSK's two FDA approvals in one week, Roche's $15B bet…",
    "Jefferies raises Zymeworks target on Roche deal") is someone else's catalyst.
    Headlines that don't name it at all pass — Yahoo tags those to the ticker.
    """
    if not company_first_word:
        return True
    pos = headline.lower().find(company_first_word)
    return pos == -1 or pos <= max_pos


def news_items(ticker: str) -> list[tuple[str, str, datetime]]:
    """
    Recent news as (headline, summary, published) — handles both yfinance formats:
      ≥1.0 : {"id", "content": {"title", "summary", "pubDate": "2026-08-30T01:07:50Z"}}
      <1.0 : {"title", "summary", "providerPublishTime": <unix ts>}
    Reading only the old top-level keys returned empty titles for every article,
    so the news half of the catalyst engine silently never fired. Items without a
    timestamp are dropped (they used to default to "now" and look fresh).
    """
    out = []
    for item in (_safe_attr(ticker, "news") or []):
        c = item.get("content") or {}
        title   = c.get("title") or item.get("title") or ""
        summary = c.get("summary") or c.get("description") or item.get("summary") or ""
        pub = None
        if c.get("pubDate"):
            try:
                ts_ = pd.Timestamp(c["pubDate"])
                if ts_.tzinfo is not None:
                    ts_ = ts_.tz_convert("UTC").tz_localize(None)
                pub = ts_.to_pydatetime()
            except Exception:
                pub = None
        elif item.get("providerPublishTime"):
            pub = datetime.fromtimestamp(item["providerPublishTime"])
        if title and pub:
            out.append((title, summary, pub))

    # Fallback: Yahoo's feed regularly returns nothing (RHHBY, Oct 2026 — missed
    # Roche's giredestrant priority review). Search Google News by company name.
    if not out:
        info = _safe_info(ticker) or {}
        key  = company_key(info.get("longName") or info.get("shortName") or "")
        if key:
            # Business/pharma qualifiers keep out same-name noise (athletes, judges…)
            q = (f'"{key}" (stock OR shares OR FDA OR trial OR drug OR approval '
                 f'OR earnings OR deal OR acquisition)')
            # Only headlines that name the company — otherwise a competitor's news
            # (Novartis "weaker-than-expected data") gets scored as Roche's catalyst
            first = key.split()[0]
            out = [(h, src, d) for h, src, d in google_news(q) if first in h.lower()]
    return out


def extend_universe(new: list[str], prices: pd.DataFrame, valid_tickers: list[str],
                    min_days: int = 20) -> list[str]:
    """Download history for tickers outside the index universe (pipeline picks,
    upcoming-FDA names, OTC ADRs) and add them in place. Returns those added."""
    new = [t for t in dict.fromkeys(new) if t not in valid_tickers]
    if not new:
        return []
    try:
        dl = yf.download(new, start=START_DATE, end=END_DATE,
                         auto_adjust=True, progress=False)["Close"]
    except Exception as exc:
        print(f"      ⚠ Price download failed for {', '.join(new)} ({type(exc).__name__})")
        return []
    if isinstance(dl, pd.Series):
        dl = dl.to_frame(new[0])
    added = []
    for t in new:
        if t in dl.columns and dl[t].notna().sum() >= min_days:
            prices[t] = dl[t]
            valid_tickers.append(t)
            added.append(t)
    return added


def upcoming_fda_signals(today: datetime | None = None) -> tuple[dict, list[str]]:
    """
    Turn UPCOMING_FDA_DECISIONS into catalyst signals.
    Returns ({ticker: catalyst_signal}, [warnings for past decisions lacking an outcome]).
    Score rises as the date approaches: +0.4 at 120 days → +0.7 inside 30 days.
    """
    today = today or datetime.now()
    sigs: dict = {}
    stale: list[str] = []
    for d in UPCOMING_FDA_DECISIONS:
        try:
            when = datetime.strptime(d["date"], "%Y-%m-%d")
        except (KeyError, ValueError):
            continue
        days = (when - today).days
        if days < 0:
            if not d.get("outcome"):
                stale.append(f"{d['ticker']} {d['drug']} FDA decision was due {d['date']} — "
                             f"record the outcome in UPCOMING_FDA_DECISIONS")
            continue
        if days > FDA_DECISION_WINDOW:
            continue
        score = 0.7 if days <= 30 else 0.4 + 0.3 * (FDA_DECISION_WINDOW - days) / (FDA_DECISION_WINDOW - 30)
        sigs[d["ticker"]] = {
            "signal":        "POSITIVE",
            "score":         round(score, 2),
            "catalyst_type": "Upcoming FDA Decision",
            "headline":      (f"{d['review']} — FDA decision on {d['drug']} due {d['date']} "
                              f"({days}d): {d['indication'][:80]}"),
            "days_ago":      0,
            "fda_decision":  d,
        }
    return sigs, stale


def assess_structural_risks(universe: list[str], lookback_days: int = 90) -> dict:
    """
    Map STRUCTURAL_RISKS onto tickers in the universe, escalating severity when
    recent news on an affected ticker mentions a watch term.
    Returns {ticker: {risk, severity, base_severity, note, evidence: [headlines]}}.
    """
    out: dict = {}
    cutoff = datetime.now() - timedelta(days=lookback_days)
    for risk in STRUCTURAL_RISKS:
        terms = [w.lower() for w in risk.get("watch_terms", [])]
        for t in risk["tickers"]:
            if t not in universe:
                continue
            hits = [h for h, s, d in news_items(t)
                    if d >= cutoff and any(w in (h + " " + s).lower() for w in terms)]
            sev = min(1.0, risk["severity"] + 0.1 * min(len(hits), 3))
            if sev > out.get(t, {}).get("severity", -1):
                out[t] = {"risk": risk["name"], "severity": round(sev, 2),
                          "base_severity": risk["severity"], "note": risk["note"],
                          "evidence": hits[:3]}
    return out


def fetch_catalyst_signals(tickers: list) -> dict:
    """
    Scans recent news headlines + analyst upgrades for event-driven catalysts.

    Detects:
      FDA / regulatory: approval, PDUFA date, NDA filing, clearance, CRL rejection
      Clinical trials:  Phase 2/3 results, primary endpoint, trial success/failure
      Analyst events:   upgrade to Buy/Strong Buy, price target raise, initiation
      Corporate:        M&A (acquisition, buyout, merger), buyback announcement
      Earnings:         beat + guidance raise, major miss + guidance cut

    Returns
    -------
    {ticker: {
        "signal":        "POSITIVE" | "NEGATIVE" | "NEUTRAL",
        "score":         float  -1.0 … +1.0,
        "catalyst_type": str   e.g. "FDA Approval", "Clinical Trial", "Analyst Upgrade"
        "headline":      str   most relevant recent headline
        "days_ago":      int
    }}
    Only tickers with a non-NEUTRAL signal are included.
    """
    _skip = INDEX_TICKERS | INTERNATIONAL_TICKERS
    fetch_list = [t for t in tickers if t not in _skip]

    # Keyword tables — (pattern, score, label)
    POS = [
        # ── Regulatory / clinical ─────────────────────────────────────────────
        (r"\bfda\b.{0,40}(approv|clear|grant|authoriz)",      +1.0, "FDA Approval"),
        (r"(approv|clear|grant).{0,30}\bfda\b",                +1.0, "FDA Approval"),
        (r"\bpdufa\b.{0,40}(accept|set|date)",                 +0.6, "PDUFA Date"),
        # "phase 3", "phase III", "phase 2/3" — Roman numerals were missed before
        (r"(phase (2|3|ii|iii)\b|pivotal).{0,80}(positive|met|success|significant)",
                                                                +0.8, "Clinical Trial"),
        (r"primary (endpoint|objective).{0,20}(met|achieved)",  +0.8, "Clinical Trial"),
        # Bare "breakthrough" matched chip news ("Micron's Next AI Memory Breakthrough",
        # "CXMT's Chip Breakthrough") — require the regulatory phrasing
        (r"(breakthrough (therapy|device)|fast.track (designation|status)|priority review"
         r"|\bfda\b.{0,40}(breakthrough|fast.track))",          +0.6, "FDA Designation"),
        # ── Analyst events ────────────────────────────────────────────────────
        (r"(upgraded?|initiat).{0,30}(buy|strong buy|overweight)", +0.5, "Analyst Upgrade"),
        (r"price target.{0,20}rais|target.{0,10}increas",       +0.4, "Price Target Raise"),
        # ── Corporate / financial ─────────────────────────────────────────────
        (r"(acqui|merger|buyout|takeover).{0,30}(agree|definitive|complet)", +0.6, "M&A"),
        (r"(beat|exceed).{0,20}(estimate|consensus|expect)",    +0.3, "Earnings Beat"),
        (r"rais.{0,20}(guidance|outlook|forecast)",             +0.4, "Guidance Raise"),
        (r"share.{0,10}(buyback|repurchase).{0,20}(authoriz|announc)", +0.3, "Buyback"),
        # ── Product launches & business development ───────────────────────────
        (r"(launch|unveil|introduc|reveal|announc).{0,50}(new|next.gen|next gen|gen \d)",
                                                                +0.3, "Product Launch"),
        (r"(launch|unveil|introduc|reveal).{0,60}(chip|model|phone|device|platform|service|feature|product)",
                                                                +0.3, "Product Launch"),
        (r"(new|next.gen).{0,30}(chip|model|phone|device|platform|ai|processor|gpu|cpu)",
                                                                +0.25, "Product Launch"),
        (r"(win|award|secur|sign).{0,40}(contract|deal|partnership|agreement).{0,30}(billion|million|\$[0-9])",
                                                                +0.4, "Major Contract"),
        (r"(partner|collaborat|joint venture).{0,30}(deal|agree|sign|announc).{0,30}(billion|million|\$[0-9]|exclusiv)",
                                                                +0.3, "Partnership"),
        (r"(record|all.time.high|best.ever).{0,30}(revenue|sales|profit|subscriber|unit|shipment)",
                                                                +0.3, "Record Performance"),
        (r"(market share|share gain).{0,20}(grow|expand|increas|gain|reach)",
                                                                +0.2, "Market Share Gain"),
        (r"expand.{0,30}(into|to).{0,30}(new market|new country|new region|international)",
                                                                +0.2, "Market Expansion"),
    ]
    NEG = [
        # ── Regulatory / clinical ─────────────────────────────────────────────
        (r"\bfda\b.{0,40}(reject|refus|complet.response|crl\b)", -1.0, "FDA Rejection"),
        (r"(reject|refus|crl\b).{0,30}\bfda\b",                  -1.0, "FDA Rejection"),
        (r"(phase (2|3|ii|iii)\b|pivotal).{0,80}(fail|miss|terminat|discontin)",
                                                                  -0.8, "Trial Failure"),
        (r"primary (endpoint|objective).{0,20}(not met|miss|fail)",    -0.8, "Trial Failure"),
        (r"(clinical.hold|full.hold)",                            -0.7, "FDA Clinical Hold"),
        # ── Analyst events ────────────────────────────────────────────────────
        (r"(downgrad|cut.{0,10}to).{0,30}(sell|underperform|underweight)", -0.5, "Analyst Downgrade"),
        (r"price target.{0,20}(cut|lower|reduc)",                 -0.3, "Price Target Cut"),
        # ── Financial ─────────────────────────────────────────────────────────
        (r"(miss|below).{0,20}(estimate|consensus|expect)",       -0.3, "Earnings Miss"),
        (r"lower.{0,20}(guidance|outlook|forecast)",              -0.4, "Guidance Cut"),
        (r"(sec|doj|ftc).{0,30}(investigat|charg|subpoen|fine)",  -0.5, "Regulatory Risk"),
        (r"(layoff|restructur|workforce.reduc).{0,30}(announc|plan)", -0.3, "Restructuring"),
        # ── Product / business issues ─────────────────────────────────────────
        (r"(delay|postpon|push.back|cancel).{0,40}(product|launch|release|shipment|rollout)",
                                                                  -0.3, "Launch Delay"),
        (r"(recall|defect|safety.concern|safety.issue).{0,40}(product|device|unit|vehicle)",
                                                                  -0.5, "Product Recall"),
        (r"(lose|lost|losing|lose.{0,10}to).{0,30}(market share|major contract|key customer)",
                                                                  -0.25, "Market Share Loss"),
        (r"(ban|prohibit|block).{0,30}(product|service|sale|import|export)",
                                                                  -0.4, "Product Ban"),
        (r"antitrust.{0,30}(investigat|suit|action|fine|block)",  -0.4, "Antitrust Risk"),
    ]

    import re
    cutoff = datetime.now() - timedelta(days=45)

    def _score_text(text: str):
        t = text.lower()
        best_pos = best_neg = 0.0
        best_pos_label = best_neg_label = ""
        for pattern, score, label in POS:
            if re.search(pattern, t):
                if score > best_pos:
                    best_pos, best_pos_label = score, label
        for pattern, score, label in NEG:
            if re.search(pattern, t):
                if abs(score) > abs(best_neg):
                    best_neg, best_neg_label = score, label
        net = best_pos + best_neg
        if net > 0:
            return net, best_pos_label
        elif net < 0:
            return net, best_neg_label
        return 0.0, ""

    def _get(t):
        try:
            best_score = 0.0
            best_label = ""
            best_headline = ""
            best_days = 999

            # ── News headlines ────────────────────────────────────────────
            co_first = (company_key((_safe_info(t) or {}).get("longName") or "")
                        .split() or [""])[0]
            for headline, summary, pub_dt in news_items(t):
                if pub_dt < cutoff:
                    continue
                if not _is_subject(headline, co_first):
                    continue      # another company's story that merely mentions this one
                sc, lbl = _score_text(headline + " " + summary)
                if abs(sc) > abs(best_score):
                    best_score = sc
                    best_label = lbl
                    best_headline = headline[:120]
                    best_days = (datetime.now() - pub_dt).days

            # ── Analyst upgrades/downgrades (last 30 days) ────────────────
            try:
                upg = _safe_attr(t, "upgrades_downgrades")
                if upg is not None and not upg.empty:
                    upg.index = pd.to_datetime(upg.index, errors="coerce")
                    recent = upg[upg.index >= pd.Timestamp(cutoff)]
                    for _, row in recent.iterrows():
                        grade = str(row.get("ToGrade", "")).lower()
                        action = str(row.get("Action", "")).lower()
                        if any(g in grade for g in ("buy", "strong buy", "outperform", "overweight")):
                            sc, lbl = +0.5, "Analyst Upgrade"
                        elif any(g in grade for g in ("sell", "underperform", "underweight")):
                            sc, lbl = -0.5, "Analyst Downgrade"
                        elif "up" in action:
                            sc, lbl = +0.3, "Analyst Upgrade"
                        elif "down" in action:
                            sc, lbl = -0.3, "Analyst Downgrade"
                        else:
                            continue
                        if abs(sc) > abs(best_score):
                            best_score = sc
                            best_label = lbl
                            best_headline = f"{row.get('Firm','?')} → {row.get('ToGrade','?')}"
                            best_days = (datetime.now() - row.name.to_pydatetime()).days
            except Exception:
                pass

            if best_label:
                signal = "POSITIVE" if best_score > 0 else "NEGATIVE"
                return t, {
                    "signal":        signal,
                    "score":         round(best_score, 3),
                    "catalyst_type": best_label,
                    "headline":      best_headline,
                    "days_ago":      best_days,
                }
        except Exception:
            pass
        return t, {}

    with ThreadPoolExecutor(max_workers=8) as pool:
        raw = dict(pool.map(_get, fetch_list))
    return {t: v for t, v in raw.items() if v}


def fetch_fda_pipeline(tickers: list, sector_map: dict | None = None) -> dict:
    """
    Two-source FDA/clinical intelligence layer for healthcare & biotech stocks.

    Source 1 — ClinicalTrials.gov REST API (free, no auth):
      Finds Phase 2 & 3 trials per company with primary completion dates in the
      next 12 months. These are upcoming binary readouts — major stock catalysts.

    Source 2 — FDA Novel Drug Approvals page (scraped HTML):
      Detects if a company received a recent drug/device approval in the last
      90 days. A fresh approval is one of the strongest positive catalysts.

    Returns
    -------
    {ticker: {
        "trials":              list[dict],   # upcoming Phase 2/3 readouts
        "recent_approval":     str | None,   # drug name if recently approved
        "fda_signal":          str,          # "RECENT_APPROVAL" | "UPCOMING_PHASE3"
                                             # | "UPCOMING_PHASE2" | "NONE"
        "fda_score":           float,        # +1.0 approval / +0.7 Ph3 / +0.4 Ph2
        "fda_summary":         str,          # human-readable one-liner
    }}
    Only tickers with a non-NONE signal are included.
    """
    import re as _re
    import requests
    from bs4 import BeautifulSoup

    HEADERS = {"User-Agent": "PortfolioAnalyzer/1.0 (educational use; contact: user@example.com)"}
    TIMEOUT = 12

    # Only run for healthcare / biotech sectors — skip others to save API calls
    BIOTECH_SECTORS = {
        "Healthcare", "Biotechnology", "Pharmaceutical", "Diagnostics",
        "Medical Devices", "Drug Manufacturers", "Specialty Chemicals",
    }
    if sector_map:
        target = [t for t in tickers
                  if any(bio.lower() in (sector_map.get(t) or "").lower()
                         for bio in BIOTECH_SECTORS)]
    else:
        target = list(tickers)

    if not target:
        return {}

    CT_BASE = "https://clinicaltrials.gov/api/v2/studies"

    _name_key = company_key

    # ── Source 2: FDA Novel Drug Approvals (last 90 days, scraped once) ──────
    # The data lives on per-year pages: No. | Drug Name | Active Ingredient |
    # Approval Date (M/D/YYYY) | Use. There is NO company column, so each
    # approved ingredient's lead sponsor is looked up on ClinicalTrials.gov.
    approved_by_sponsor: dict[str, str] = {}   # lower-case lead sponsor → "Brand (ingredient)"
    cutoff_date = datetime.now() - timedelta(days=90)
    for yr in sorted({cutoff_date.year, datetime.now().year}):
        try:
            fda_url = ("https://www.fda.gov/drugs/novel-drug-approvals-fda/"
                       f"novel-drug-approvals-{yr}")
            resp = requests.get(fda_url, headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            for row in soup.select("table tbody tr"):
                cells = [c.get_text(" ", strip=True) for c in row.find_all("td")]
                if len(cells) < 4:
                    continue
                brand, ingredient, date_str = cells[1], cells[2], cells[3]
                try:
                    appr_date = datetime.strptime(date_str.strip(), "%m/%d/%Y")
                except ValueError:
                    continue
                if appr_date < cutoff_date:
                    continue
                # FDA names ("insulin efsitora alfa-gobe") often differ from CT's
                # ("efsitora alfa") — retry with the most distinctive word.
                generic = {"insulin", "sodium", "hydrochloride", "alfa", "beta", "acid",
                           "potassium", "tartrate", "mesylate", "maleate"}
                words   = [w for w in _re.split(r"[\s\-]+", ingredient.lower())
                           if len(w) >= 6 and w not in generic]
                queries = [ingredient] + ([max(words, key=len)] if words else [])
                for qry in queries:
                    found = False
                    try:
                        r = requests.get(CT_BASE, headers=HEADERS, timeout=TIMEOUT, params={
                            "query.intr": qry, "pageSize": 5,
                            "fields": "protocolSection.sponsorCollaboratorsModule.leadSponsor"})
                        for st in (r.json().get("studies", []) if r.status_code == 200 else []):
                            sp = (st.get("protocolSection", {}).get("sponsorCollaboratorsModule", {})
                                    .get("leadSponsor", {}).get("name", ""))
                            if sp:
                                approved_by_sponsor[_name_key(sp)] = f"{brand} ({ingredient})"
                                found = True
                    except Exception:
                        pass
                    if found:
                        break
        except Exception:
            continue

    result  = {}
    today   = datetime.now()

    def _get_company_name(ticker: str) -> str:
        try:
            # fast_info has no longName — .info does (and is disk-cached)
            info = _safe_info(ticker)
            name = info.get("longName") or info.get("shortName") or ""
            # Search on the distinctive part of the name ('Scholar Rock', not
            # 'Scholar Rock Holding Corporation' — and no stray punctuation)
            return _name_key(name) or ticker
        except Exception:
            return ticker

    def _query_trials(company: str, accept_keys: set | None = None) -> list[dict]:
        accept_keys = accept_keys or {company}
        trials = []
        try:
            params = {
                # query.spons = sponsor/collaborator fields; query.term matched any
                # trial that merely mentioned the company name anywhere in its record
                "query.spons": company,
                "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING,ENROLLING_BY_INVITATION",
                # API v2 syntax is "phase:2 3" — "phase:PHASE2,PHASE3" returns HTTP 400
                "aggFilters": "phase:2 3",
                "fields": (
                    "protocolSection.identificationModule.briefTitle,"
                    "protocolSection.statusModule.overallStatus,"
                    "protocolSection.statusModule.primaryCompletionDateStruct,"
                    "protocolSection.designModule.phases,"
                    "protocolSection.sponsorCollaboratorsModule.leadSponsor"
                ),
                "pageSize": 25,
            }
            resp = requests.get(CT_BASE, params=params, headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code != 200:
                return []
            data = resp.json()
            for study in data.get("studies", []):
                ps  = study.get("protocolSection", {})
                lead = (ps.get("sponsorCollaboratorsModule", {})
                          .get("leadSponsor", {}).get("name", ""))
                if _name_key(lead) not in accept_keys:
                    continue          # company is only a collaborator — not its catalyst
                imod = ps.get("identificationModule", {})
                smod = ps.get("statusModule", {})
                dmod = ps.get("designModule", {})
                title  = imod.get("briefTitle", "")
                status = smod.get("overallStatus", "")
                phases = dmod.get("phases", [])
                pcdate = (smod.get("primaryCompletionDateStruct") or {}).get("date", "")
                days_away = None
                if pcdate:
                    try:
                        # CT dates are "YYYY-MM-DD" or "YYYY-MM" — Timestamp parses both
                        pc_dt = pd.Timestamp(pcdate).to_pydatetime()
                        days_away = (pc_dt - today).days
                    except Exception:
                        pass
                # Only include trials with upcoming readouts (within 18 months)
                if days_away is not None and -30 <= days_away <= 548:
                    trials.append({
                        "title":       title[:100],
                        "phase":       ", ".join(phases),
                        "status":      status,
                        "completion":  pcdate[:10] if pcdate else "?",
                        "days_away":   days_away,
                    })
        except Exception:
            pass
        return sorted(trials, key=lambda x: x["days_away"])

    MEGACAP = 50e9   # a single trial isn't material to a $50B+ pharma

    def _is_p3(tr):
        return "PHASE3" in tr["phase"].upper() or "PHASE 3" in tr["phase"].upper()

    for t in target:
        company  = _get_company_name(t)
        long_nm  = _safe_info(t).get("longName") or company
        key      = _name_key(long_nm)
        # Legal entities the company files trials/NDAs under (Roche → Genentech…)
        search_names = [key] + SPONSOR_ALIASES.get(key, [])
        keys         = {_name_key(n) for n in search_names}

        # Approval match on normalised name key (lead sponsor of the approved drug)
        approval_drug = next((approved_by_sponsor[k] for k in keys
                              if len(k) >= 4 and k in approved_by_sponsor), None)

        # Trials: all Phase 2/3 for small/mid caps. For megacaps, where any one trial
        # is rarely material and some Phase 3 is always running, only Phase 3
        # readouts inside 90 days count (and score lower). Their big dated catalysts
        # come through approvals above and the UPCOMING_FDA_DECISIONS calendar.
        mcap   = float(_safe_info(t).get("marketCap") or 0)
        trials = []
        for nm in search_names:
            for tr in _query_trials(nm, keys):
                if tr["title"] not in {x["title"] for x in trials}:
                    trials.append(tr)
        trials.sort(key=lambda x: x["days_away"])
        is_mega = mcap >= MEGACAP
        if is_mega:
            trials = [tr for tr in trials if _is_p3(tr) and 0 <= tr["days_away"] <= 90]

        if not trials and not approval_drug:
            continue

        # Score and signal
        if approval_drug:
            fda_score  = +0.5 if is_mega else +1.0   # one drug rarely moves a $50B+ company much
            fda_signal = "RECENT_APPROVAL"
            summary    = f"FDA approved {approval_drug} (last 90 days)"
        elif any(_is_p3(tr) for tr in trials):
            best = next(tr for tr in trials if _is_p3(tr))
            fda_score  = +0.3 if is_mega else +0.7
            fda_signal = "UPCOMING_PHASE3"
            summary    = (f"Phase 3 readout in ~{best['days_away']}d — {best['title'][:60]}")
        else:
            best = trials[0]
            fda_score  = +0.4
            fda_signal = "UPCOMING_PHASE2"
            summary    = (f"Phase 2 readout in ~{best['days_away']}d — {best['title'][:60]}")

        result[t] = {
            "trials":          trials,
            "recent_approval": approval_drug,
            "fda_signal":      fda_signal,
            "fda_score":       fda_score,
            "fda_summary":     summary,
        }

    return result


def extract_short_signals(tickers: list, val_data: dict) -> dict:
    """
    Returns {ticker: {"short_pct": float, "short_ratio": float, "signal": str}}
    using data already fetched in val_data (shortPercentOfFloat, shortRatio).
    High short = >15% float; elevated = >10%.
    """
    result = {}
    for t in tickers:
        info = val_data.get(t, {})
        sp   = info.get("shortPercentOfFloat")
        sr   = info.get("shortRatio")
        if sp is None:
            continue
        sp = float(sp)
        sr = float(sr) if sr else 0.0
        if sp >= 0.15:
            sig = "HIGH SHORT"
        elif sp >= 0.10:
            sig = "ELEVATED"
        else:
            sig = "NORMAL"
        result[t] = {"short_pct": round(sp, 4), "short_ratio": round(sr, 1), "signal": sig}
    return result


def thematic_exposure(holdings: dict) -> list[tuple]:
    """
    Returns sorted list of (theme_name, total_weight) for configured themes.
    A ticker can belong to multiple themes.
    """
    theme_weights: dict = {theme: 0.0 for theme in THEMES}
    for ticker, weight in holdings.items():
        for theme, members in THEMES.items():
            if ticker in members:
                theme_weights[theme] += weight
    return sorted(
        [(t, w) for t, w in theme_weights.items() if w > 0],
        key=lambda x: -x[1],
    )


def drip_projection(holdings: dict, div_yields: dict,
                    cagr: float, years: int = 10) -> dict:
    """
    Projects portfolio value with and without dividend reinvestment.
    Returns {years: (value_no_drip, value_drip)} dict.
    """
    total_yield = sum(
        holdings.get(t, 0.0) * (div_yields.get(t) or 0.0)
        for t in holdings
    )
    rows = {}
    val_no_drip = INITIAL_INVESTMENT
    val_drip    = INITIAL_INVESTMENT
    for yr in range(1, years + 1):
        val_no_drip *= (1 + cagr)
        val_drip    *= (1 + cagr + total_yield)
        rows[yr] = (round(val_no_drip, 2), round(val_drip, 2))
    return {"portfolio_yield": round(total_yield, 4), "projection": rows}


def options_protection_estimate(portfolio_value: float, beta: float) -> list[dict]:
    """
    Rough cost estimate for SPY put-based downside protection.
    Uses rule-of-thumb: ATM SPY put ≈ 2% of notional for 3-month, 3.5% for 6-month.
    Adjusts notional by beta to reflect actual hedging need.
    """
    hedged_notional = portfolio_value * max(0.5, min(beta, 2.5))
    rows = []
    for label, months, pct_cost in [
        ("3-month ATM SPY put", 3, 0.020),
        ("6-month ATM SPY put", 6, 0.035),
        ("12-month 10%-OTM SPY put", 12, 0.025),
    ]:
        cost = hedged_notional * pct_cost
        breakeven_drop = pct_cost / beta if beta > 0 else pct_cost
        rows.append({
            "Strategy":              label,
            "Duration (mo)":         months,
            "Hedged Notional ($)":   round(hedged_notional, 0),
            "Est. Premium ($)":      round(cost, 0),
            "Premium (% portfolio)": f"{cost/portfolio_value:.1%}",
            "Breakeven Drop":        f"{breakeven_drop:.1%}",
            "Note":                  "Rule-of-thumb estimate. Actual cost varies by IV.",
        })
    return rows


def tax_loss_candidates(current_holdings: dict, current_prices: dict,
                        target_holdings: dict) -> list[dict]:
    """
    Identifies positions in CURRENT_HOLDINGS trading below cost basis
    (approximated as current price vs. cost-basis proxy).
    Only meaningful when CURRENT_HOLDINGS is filled in by the user.
    """
    if not current_holdings:
        return []
    rows = []
    for ticker, current_val in current_holdings.items():
        price = current_prices.get(ticker) or 0.0
        if price <= 0 or current_val <= 0:
            continue
        # We don't have actual cost basis — flag if holding is NOT in target
        # (user should sell it anyway) and market price has fallen
        target_w = target_holdings.get(ticker, 0.0)
        if target_w == 0:
            rows.append({
                "Ticker":          ticker,
                "Current Value ($)": round(float(current_val), 2),
                "Target Weight":   "0%",
                "Recommended":     "SELL — not in optimal portfolio",
                "TLH Note":        "Potential tax-loss if below cost basis",
            })
    return rows


# ─── SCENARIO RECOMMENDATION ──────────────────────────────────────────────────

def recommend_scenario(results: dict, wf_results: dict, regime: dict) -> dict:
    """
    Score each scenario on 4 dimensions and return a recommendation with reasoning.
      Walk-forward OOS alpha   40%  — most objective; tests on unseen data
      Walk-forward OOS Sharpe  25%  — risk-adjusted return on unseen data
      Regime alignment         20%  — does the scenario suit the current market?
      Walk-forward OOS max DD  15%  — downside risk on unseen data
    In-sample Sharpe/drawdown are NOT used: weights were optimised on that same
    history, so they measure overfit, not skill. A scenario whose walk-forward
    produced no result scores 0 on the OOS dimensions (no evidence ≠ neutral).
    """
    regime_name = regime.get("regime", "Neutral")

    REGIME_FIT = {
        "Bull":    {"GlobalGrowth": 0.50, "AggressiveGrowth": 0.90, "HighConviction": 1.00},
        "Bear":    {"GlobalGrowth": 1.00, "AggressiveGrowth": 0.40, "HighConviction": 0.00},
        "Neutral": {"GlobalGrowth": 0.70, "AggressiveGrowth": 0.80, "HighConviction": 0.50},
    }
    fit = REGIME_FIT.get(regime_name, REGIME_FIT["Neutral"])

    scores = {}
    for key in PORTFOLIOS:
        wf       = wf_results[key]
        alpha    = wf.get("alpha")
        oos_sh   = wf.get("oos_sharpe")
        oos_dd   = wf.get("oos_max_dd")

        s_alpha  = 0.0 if alpha  is None else max(0.0, min(1.0, 0.5 + alpha / 0.20))
        s_sharpe = 0.0 if oos_sh is None else max(0.0, min(1.0, (oos_sh - 0.5) / 2.0))
        s_regime = fit.get(key, 0.5)
        s_dd     = 0.0 if oos_dd is None else max(0.0, min(1.0, 1.0 + oos_dd / 0.40))

        composite = (0.40 * s_alpha + 0.25 * s_sharpe +
                     0.20 * s_regime + 0.15 * s_dd)
        scores[key] = {
            "composite": round(composite, 4),
            "alpha": alpha if alpha is not None else 0.0,
            "alpha_missing": alpha is None,
            "sharpe": oos_sh if oos_sh is not None else 0.0,
            "max_dd": oos_dd if oos_dd is not None else 0.0,
            "regime_fit": s_regime,
            "s_alpha": round(s_alpha, 3), "s_sharpe": round(s_sharpe, 3),
            "s_regime": round(s_regime, 3), "s_dd": round(s_dd, 3),
        }

    ranked     = sorted(scores, key=lambda k: scores[k]["composite"], reverse=True)
    best_key   = ranked[0]
    second_key = ranked[1]
    res = results[best_key]
    sc  = scores[best_key]
    pf  = PORTFOLIOS[best_key]

    reasons = []
    alpha = sc["alpha"]
    if sc.get("alpha_missing"):
        reasons.append(
            "Walk-forward test produced no result for this scenario (the optimizer "
            "found no valid holdings in the training window) — it is recommended on "
            "regime fit alone, which is weak evidence.")
    elif alpha > 0.02:
        reasons.append(
            f"<strong>Best out-of-sample alpha: {alpha:+.1%}/yr above SPY</strong> in the "
            f"rolling walk-forward — re-screened and re-optimised each quarter on past data "
            f"only, with trading costs. This is the most reliable performance measure.")
    elif alpha >= -0.01:
        reasons.append(
            f"Competitive walk-forward performance: {alpha:+.1%}/yr vs SPY. The model "
            f"matched the benchmark with lower concentration than buying SPY directly.")
    else:
        reasons.append(
            f"Walk-forward alpha was {alpha:+.1%}/yr vs SPY, but other dimensions "
            f"(Sharpe, regime fit) support this scenario over the alternatives.")

    reasons.append(
        f"<strong>Out-of-sample Sharpe {sc['sharpe']:.2f}</strong> on the walk-forward test "
        f"period. Above 1.0 is strong. The test spans only ~3 years of mostly rising "
        f"markets, so treat it as directional rather than proof.")

    regime_notes = {
        "Bull":    "The <strong>current Bull market</strong> (SPY trading above its 200-day moving "
                   "average) favors momentum-driven, higher-concentration strategies that "
                   "fully capture upside.",
        "Bear":    "The <strong>current Bear market</strong> (SPY below its 200-day moving average) "
                   "favors internationally-diversified strategies that buffer US-specific drawdowns.",
        "Neutral": "The <strong>current Neutral regime</strong> (SPY near its 200-day moving average) "
                   "favors balanced strategies — enough concentration to outperform, "
                   "enough diversification to handle a turn lower.",
    }
    reasons.append(regime_notes.get(regime_name,
                   "Regime alignment supports this scenario."))

    reasons.append(
        f"<strong>Out-of-sample max drawdown {sc['max_dd']:.1%}</strong> during the walk-forward "
        f"test period. In-sample drawdown ({res['max_dd']:.1%}) is shown in the metrics table but "
        f"understates real risk, since the weights were fitted to that history. Stress tests "
        f"show deeper plausible losses.")

    caveats = []
    if best_key == "HighConviction":
        caveats.append(
            "High Conviction places up to 25% in a single position — the most concentrated "
            "of the three scenarios. If you prefer more diversification with similar upside, "
            "consider Aggressive Growth as an alternative.")

    alt_pf    = PORTFOLIOS[second_key]
    alt_sc    = scores[second_key]
    margin    = sc["composite"] - alt_sc["composite"]
    alt_desc  = ("More internationally diversified — adds a ≥15% international ETF floor."
                 if second_key == "GlobalGrowth" else
                 "Higher potential upside through maximum-return optimization.")
    if margin < 0.05:
        caveats.append(
            f"The margin over {alt_pf['label']} is narrow "
            f"({sc['composite']:.3f} vs {alt_sc['composite']:.3f}). "
            f"Both are valid — the runner-up suits investors who want: {alt_desc}")
    else:
        caveats.append(
            f"Runner-up: {alt_pf['label']} (score {alt_sc['composite']:.3f}). {alt_desc}")

    return {
        "recommended_key":   best_key,
        "recommended_label": pf["label"],
        "recommended_color": pf["color"],
        "ranked":            ranked,
        "scores":            scores,
        "reasons":           reasons,
        "caveats":           caveats,
        "regime":            regime_name,
    }


# ─── HTML REPORT ──────────────────────────────────────────────────────────────

# Shared across generate_html_report and generate_investor_html_report
VCOLOR = {"UNDERVALUED": "#4caf50", "OVERVALUED": "#f44336",
          "FAIR VALUE": "#9e9e9e", "—": "#555"}
VICON  = {"UNDERVALUED": "★", "OVERVALUED": "⚠", "FAIR VALUE": "◆", "—": "—"}


def generate_html_report(
    results: dict, mc_results: dict, current_prices: dict,
    div_yields: dict | None, backtest_results: dict | None,
    wf_results: dict | None, stress_results: dict | None,
    regime: dict | None, factor_exposures: dict | None,
    val_signals: dict | None, recommendation: dict,
    # Enhanced analysis outputs (all optional)
    risk_contribs: dict | None = None,
    thematic_exposures: dict | None = None,
    corr_data: dict | None = None,
    earnings_flags: dict | None = None,
    insider_signals: dict | None = None,
    short_signals: dict | None = None,
    drip_data: dict | None = None,
    rebalance_plan: list | None = None,
    dca_schedule: list | None = None,
    options_protection: list | None = None,
    tax_loss_list: list | None = None,
    personalized_plans: list | None = None,
) -> str:
    """Generate a self-contained HTML stakeholder report."""

    def esc(v) -> str:
        return str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    rec_key    = recommendation["recommended_key"]
    rec_label  = recommendation["recommended_label"]
    rec_color  = recommendation["recommended_color"]
    scores     = recommendation["scores"]
    regime_name = (regime or {}).get("regime", "Unknown")
    REGIME_C   = {"Bull": "#4caf50", "Bear": "#f44336", "Neutral": "#ff9800", "Unknown": "#888"}
    rc         = REGIME_C.get(regime_name, "#888")
    # VCOLOR and VICON are module-level constants (defined below generate_html_report)

    # ── CSS ────────────────────────────────────────────────────────────────────
    css = """
    :root {
      --bg: #0f1117; --panel: #1a1d2e; --border: #2a2d3e;
      --text: #e0e0e0; --muted: #9e9e9e; --accent: #00bcd4;
      --green: #4caf50; --red: #f44336; --orange: #ff9800; --gold: #ffd700;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg); color: var(--text);
           font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px;
           line-height: 1.6; padding: 0 0 40px 0; }
    a { color: var(--accent); }
    h1 { font-size: 1.8rem; font-weight: 700; }
    h2 { font-size: 1.2rem; font-weight: 600; color: var(--accent);
         margin: 28px 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
    h3 { font-size: 1rem; font-weight: 600; margin: 16px 0 8px; }
    .wrap  { max-width: 1100px; margin: 0 auto; padding: 0 24px; }
    .hero  { background: linear-gradient(135deg, #1a1d2e 0%, #0f1117 100%);
             border-bottom: 2px solid var(--border); padding: 32px 0 24px; margin-bottom: 8px; }
    .hero .subtitle { color: var(--muted); margin-top: 4px; font-size: 0.9rem; }
    .rec-box { border-radius: 8px; padding: 20px 24px; margin: 20px 0;
               border-left: 5px solid; }
    .rec-box h2 { border: none; padding: 0; margin: 0 0 10px; font-size: 1.1rem; }
    .rec-box ul { padding-left: 18px; margin-top: 8px; }
    .rec-box li { margin-bottom: 6px; }
    .rec-box .caveats { color: var(--muted); font-size: 0.88rem;
                        margin-top: 12px; border-top: 1px solid var(--border); padding-top: 10px; }
    .regime-badge { display: inline-block; border-radius: 20px; padding: 4px 14px;
                    font-size: 0.85rem; font-weight: 600; margin-bottom: 4px; }
    .score-bar-wrap { background: var(--border); border-radius: 4px; height: 8px;
                      width: 120px; display: inline-block; vertical-align: middle; margin-left: 8px; }
    .score-bar { height: 8px; border-radius: 4px; }
    table  { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
    th     { background: var(--panel); color: var(--accent); font-weight: 600;
             padding: 8px 12px; text-align: left; font-size: 0.82rem;
             border-bottom: 2px solid var(--border); }
    td     { padding: 7px 12px; border-bottom: 1px solid var(--border);
             font-size: 0.85rem; vertical-align: top; }
    tr:last-child td { border-bottom: none; }
    tr.total-row td  { font-weight: 600; background: var(--panel);
                       border-top: 2px solid var(--border); }
    tr:hover td      { background: rgba(255,255,255,0.03); }
    .rec-col   { background: rgba(0,188,212,0.08); }
    .num       { text-align: right; font-variant-numeric: tabular-nums; }
    .good      { color: var(--green); font-weight: 600; }
    .bad       { color: var(--red);   font-weight: 600; }
    .neutral   { color: var(--muted); }
    .tag       { border-radius: 4px; padding: 2px 7px; font-size: 0.78rem;
                 font-weight: 600; display: inline-block; }
    .cards     { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
                 gap: 14px; margin-bottom: 20px; }
    .card      { background: var(--panel); border: 1px solid var(--border);
                 border-radius: 8px; padding: 14px 16px; }
    .card h3   { color: var(--accent); font-size: 0.9rem; margin-bottom: 6px; }
    .card .def { color: var(--muted); font-size: 0.83rem; margin-bottom: 6px; }
    .card .good-val { color: var(--green); font-size: 0.82rem; }
    details summary { cursor: pointer; color: var(--accent); font-weight: 600;
                      padding: 8px 0; list-style: none; }
    details summary::before { content: '▶ '; font-size: 0.8rem; }
    details[open] summary::before { content: '▼ '; }
    .disclaimer { color: var(--muted); font-size: 0.8rem; margin-top: 32px;
                  border-top: 1px solid var(--border); padding-top: 16px; }
    """

    # ── Helpers ────────────────────────────────────────────────────────────────
    def pct(v, decimals=1):
        return f"{v:.{decimals}%}" if v is not None else "N/A"

    def dollar(v):
        if v is None: return "N/A"
        return f"${v:,.0f}"

    def metric_color(label: str, value) -> str:
        if value is None: return ""
        pos_good = label in ("CAGR", "Sharpe", "Sortino", "Alpha", "OOS Annualized")
        if pos_good:
            return "good" if value > 0 else "bad"
        if label == "Max Drawdown":
            return "good" if value > -0.25 else ("bad" if value < -0.45 else "neutral")
        return ""

    def score_bar(score: float, color: str) -> str:
        w = int(max(0, min(1, score)) * 100)
        return (f'<span class="score-bar-wrap">'
                f'<span class="score-bar" style="width:{w}%;background:{color}"></span>'
                f'</span>')

    # ── Header ─────────────────────────────────────────────────────────────────
    spy_info = ""
    if regime:
        spy_info = (f'SPY ${regime.get("spy_price","N/A")} · '
                    f'200-day MA ${regime.get("ma200","N/A")} · '
                    f'{regime.get("pct_vs_ma","N/A")}')

    parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Portfolio Analysis Report — {datetime.now().strftime('%Y-%m-%d')}</title>
<style>{css}</style>
</head>
<body>
<div class="hero">
<div class="wrap">
  <h1>Portfolio Analysis Report</h1>
  <div class="subtitle">
    ${INITIAL_INVESTMENT:,} Investment &nbsp;·&nbsp;
    3-Year Historical Analysis &nbsp;·&nbsp;
    {PROJECTION_YEARS}-Year Projection &nbsp;·&nbsp;
    Generated {datetime.now().strftime('%B %d, %Y')}
  </div>
</div>
</div>
<div class="wrap">"""]

    # ── Recommendation box ─────────────────────────────────────────────────────
    parts.append(f"""
<h2>Scenario Recommendation</h2>
<div class="rec-box" style="background:rgba{_hex_to_rgba(rec_color, 0.08)};
     border-color:{rec_color}">
  <h2 style="color:{rec_color}">
    RECOMMENDED: {esc(rec_label)}
  </h2>
  <p style="color:var(--muted);font-size:0.85rem;margin-bottom:8px">
    Scored across 4 dimensions:
    walk-forward alpha (40%) · Sharpe ratio (25%) · regime alignment (20%) · max drawdown (15%)
  </p>
  <ul>""")
    for r in recommendation["reasons"]:
        parts.append(f"    <li>{r}</li>")
    parts.append("  </ul>")
    if recommendation["caveats"]:
        parts.append('  <div class="caveats"><strong>Considerations:</strong><ul>')
        for c in recommendation["caveats"]:
            parts.append(f"    <li>{c}</li>")
        parts.append("  </ul></div>")
    parts.append("</div>")

    # ── Market Regime ──────────────────────────────────────────────────────────
    if regime:
        icon = {"Bull": "▲", "Bear": "▼", "Neutral": "►"}.get(regime_name, "?")
        parts.append(f"""
<h2>Market Regime</h2>
<span class="regime-badge" style="background:{rc};color:#000">{icon} {regime_name}</span>
&nbsp; {esc(regime.get('signal',''))} &nbsp;·&nbsp; {esc(spy_info)}
<p style="color:var(--muted);font-size:0.83rem;margin-top:6px">
  The 200-day moving average regime signal is one of the most widely-watched institutional
  indicators. A Bull regime means the majority of market participants are profitable on their
  recent positions, reducing selling pressure. This signal does <em>not</em> change the portfolio
  weights, but should inform your timing decision on deploying capital.
</p>""")

    # ── Scenario Comparison ────────────────────────────────────────────────────
    parts.append("<h2>Scenario Comparison</h2>")
    parts.append("<table>")
    parts.append("""<thead><tr>
  <th>Metric</th>""")
    for key, pf in PORTFOLIOS.items():
        rec_mark = " ★" if key == rec_key else ""
        parts.append(
            f'  <th class="{"rec-col" if key == rec_key else ""}">'
            f'{esc(pf["label"])}{rec_mark}</th>')
    parts.append("</tr></thead><tbody>")

    def comparison_row(label, values_fn, format_fn=esc, color_fn=None):
        row = f'<tr><td><strong>{label}</strong></td>'
        for key in PORTFOLIOS:
            v = values_fn(key)
            cls = f'num {"rec-col" if key == rec_key else ""}'
            color_cls = (color_fn(label, v) if color_fn else "")
            if color_cls:
                cls += f" {color_cls}"
            row += f'<td class="{cls}">{format_fn(v)}</td>'
        row += "</tr>"
        parts.append(row)

    comparison_row("CAGR (3yr Historical)",
        lambda k: results[k]["cagr"], pct, metric_color)
    comparison_row("Sharpe Ratio",
        lambda k: results[k]["sharpe"],
        lambda v: f"{v:.2f}" if v is not None else "N/A", metric_color)
    comparison_row("Sortino Ratio",
        lambda k: results[k]["sortino"],
        lambda v: f"{v:.2f}" if v is not None else "N/A", metric_color)
    comparison_row("Max Drawdown",
        lambda k: results[k]["max_dd"], pct, metric_color)
    comparison_row("CVaR 95% (daily, $)",
        lambda k: results[k]["cvar_95"], dollar)
    comparison_row("Volatility (annual)",
        lambda k: results[k]["volatility"], pct)

    if mc_results:
        comparison_row(f"MC Median ({PROJECTION_YEARS}yr, $)",
            lambda k: float(np.median(mc_results[k][:, -1])), dollar)
        comparison_row(f"MC P5 ({PROJECTION_YEARS}yr, $)",
            lambda k: float(np.percentile(mc_results[k][:, -1], 5)), dollar)
        comparison_row(f"MC P95 ({PROJECTION_YEARS}yr, $)",
            lambda k: float(np.percentile(mc_results[k][:, -1], 95)), dollar)

    if wf_results:
        comparison_row("Walk-Forward Alpha vs SPY",
            lambda k: wf_results[k].get("alpha"), pct, metric_color)
        comparison_row("Walk-Forward OOS Annualized",
            lambda k: wf_results[k].get("oos_annualized"), pct, metric_color)

    # Scenario recommendation scores
    parts.append('<tr><td><strong>Recommendation Score</strong><br>'
                 '<span style="color:var(--muted);font-size:0.78rem">'
                 'Composite of 4 weighted dimensions</span></td>')
    for key in PORTFOLIOS:
        sc = scores[key]
        bar = score_bar(sc["composite"], PORTFOLIOS[key]["color"])
        cls = f'num {"rec-col" if key == rec_key else ""}'
        parts.append(
            f'<td class="{cls}">'
            f'<strong>{sc["composite"]:.3f}</strong> / 1.000 {bar}</td>')
    parts.append("</tr>")
    parts.append("</tbody></table>")

    # Score breakdown
    parts.append("""<details style="margin-bottom:20px">
<summary>How the recommendation score is calculated</summary>
<div style="padding:12px 0">
<table>
<thead><tr><th>Dimension</th><th>Weight</th>""")
    for key in PORTFOLIOS:
        rec_mark = " ★" if key == rec_key else ""
        parts.append(
            f'<th class="{"rec-col" if key == rec_key else ""}">'
            f'{esc(PORTFOLIOS[key]["label"])}{rec_mark}</th>')
    parts.append("</tr></thead><tbody>")
    dim_labels = [
        ("OOS Alpha vs SPY", "s_alpha", "40%",
         "Walk-forward out-of-sample alpha. +10%/yr alpha → score 1.0; −10%/yr → score 0.0"),
        ("OOS Sharpe Ratio", "s_sharpe","25%",
         "Walk-forward Sharpe. 0.5 → score 0.0; 2.5 → score 1.0"),
        ("Regime Fit",       "s_regime","20%",
         f"How well the scenario suits the current {regime_name} regime"),
        ("OOS Max Drawdown", "s_dd",    "15%",
         "Walk-forward max DD. 0% → score 1.0; −40% → score 0.0"),
    ]
    for label, key_s, weight, tooltip in dim_labels:
        parts.append(f'<tr><td title="{tooltip}"><strong>{label}</strong><br>'
                     f'<span style="color:var(--muted);font-size:0.78rem">'
                     f'Weight: {weight} — {tooltip}</span></td>'
                     f'<td class="num">{weight}</td>')
        for key in PORTFOLIOS:
            val = scores[key][key_s]
            bar = score_bar(val, PORTFOLIOS[key]["color"])
            parts.append(
                f'<td class="num {"rec-col" if key == rec_key else ""}">'
                f'{val:.3f} {bar}</td>')
        parts.append("</tr>")
    parts.append("</tbody></table></div></details>")

    # ── Metric Glossary ────────────────────────────────────────────────────────
    METRICS = [
        ("CAGR",
         "Compound Annual Growth Rate",
         "The constant annual return that would produce the same final value as the actual uneven returns.",
         "Example: $15,000 → $24,000 over 3 years = 17.0% CAGR.",
         ">15% is strong for an equity portfolio"),
        ("Sharpe Ratio",
         "Return per unit of total risk",
         "Measures how much return you earn for each unit of volatility you tolerate. "
         "A higher Sharpe means you're being paid well for the risk you take.",
         "Formula: (CAGR − Risk-Free Rate) / Annual Volatility. "
         "With 5% risk-free rate, a 20% CAGR and 15% volatility gives Sharpe = (20%−5%)/15% = 1.0.",
         ">1.0 strong · >2.0 exceptional"),
        ("Sortino Ratio",
         "Return per unit of downside risk only",
         "Like Sharpe, but only penalizes bad days (negative returns). "
         "Upside volatility is not counted against the portfolio. "
         "A Sortino significantly higher than Sharpe means most volatility is upside — favorable.",
         "Formula: (CAGR − Risk-Free Rate) / Downside Volatility",
         "Sortino > Sharpe is a good sign"),
        ("Max Drawdown",
         "Worst peak-to-trough decline",
         "The largest percentage loss from any portfolio peak to the following trough. "
         "Tests your psychological resilience: could you hold without panic-selling through this?",
         "Example: Portfolio peaks at $20,000, falls to $13,000 → Max Drawdown = −35%.",
         "−20% to −40% typical for aggressive equity portfolios"),
        ("CVaR 95%",
         "Conditional Value at Risk (Expected Shortfall)",
         "The average loss on the worst 5% of trading days — a more complete picture of tail "
         "risk than standard VaR. Preferred by institutional risk managers because it captures "
         "the severity of extreme events, not just their probability.",
         "If VaR is −$180 but CVaR is −$450, the portfolio has dangerous tail behavior on bad days.",
         "Lower (less negative) is better"),
        ("Walk-Forward Alpha",
         "Excess return vs SPY in the out-of-sample test period",
         "Each quarter the screen and optimizer are re-run using only data available at that "
         "date, the weights are held for the next quarter, and trading costs are deducted. "
         "Alpha is how much this rolling process beat (or lagged) SPY on data it never saw. "
         "Positive alpha over many quarters is evidence of skill; over a few, it may be luck.",
         "Train: May 2022→2024 → Optimize. Test: May 2024→2025 → +5% alpha = model adds value.",
         ">0% means the model beat buy-and-hold SPY"),
        ("Monte Carlo P5/P50/P95",
         "Range of possible 15-year outcomes",
         "Simulates 1,000 independent 15-year futures using the portfolio's historical return "
         "and volatility. P5 = only 5% of paths end below this (pessimistic). "
         "P50 = median outcome. P95 = only 5% end above this (optimistic). "
         "The wide range honestly captures how uncertain long-term equity returns are.",
         f"Based on {MC_SIMULATIONS:,} simulations × {PROJECTION_YEARS} years × 252 daily steps.",
         "Wide range is honest, not a flaw"),
        ("Valuation Verdict",
         "DCF-based buy/hold/caution signal per position",
         "Each stock in the portfolio is assessed using a two-stage Discounted Cash Flow model "
         "(conservative / base / optimistic growth scenarios) plus valuation multiples vs. "
         "sector medians. The verdict reflects whether today's price offers a margin of safety.",
         "Verdicts are relative to this run's universe (DCF upside + multiples, ranked): "
         "★ UNDERVALUED = cheapest third. ◆ FAIR VALUE = middle third. "
         "⚠ OVERVALUED = most expensive third.",
         "ETFs show — (DCF not applicable)"),
    ]

    parts.append("<h2>Understanding Your Metrics</h2>")
    parts.append('<div class="cards">')
    for name, tagline, explanation, example, benchmark in METRICS:
        parts.append(
            f'<div class="card">'
            f'<h3>{esc(name)}</h3>'
            f'<div class="def">{esc(tagline)}</div>'
            f'<p style="font-size:0.82rem;margin-bottom:6px">{explanation}</p>'
            f'<p style="font-size:0.8rem;color:var(--muted)">{esc(example)}</p>'
            f'<div class="good-val">{esc(benchmark)}</div>'
            f'</div>')
    parts.append("</div>")

    # ── Walk-Forward ───────────────────────────────────────────────────────────
    if wf_results:
        parts.append("<h2>Walk-Forward Backtest (Out-of-Sample)</h2>")
        parts.append("""<p style="color:var(--muted);font-size:0.83rem;margin-bottom:10px">
  Train on first 2 years → optimize weights → apply to last 1 year (data the optimizer never saw).
  This eliminates look-ahead bias and gives the most honest performance estimate.
</p><table><thead><tr>
  <th>Scenario</th><th>Train Period</th><th>Test Period</th>
  <th class="num">OOS Annualized</th><th class="num">SPY (same period)</th>
  <th class="num">Alpha vs SPY</th>
</tr></thead><tbody>""")
        for key, pf in PORTFOLIOS.items():
            wf = wf_results[key]
            alpha = wf.get("alpha")
            ac = "good" if (alpha or 0) > 0 else "bad"
            parts.append(
                f'<tr class="{"rec-col" if key == rec_key else ""}">'
                f'<td><strong>{esc(pf["label"])}</strong>'
                f'{"  ★" if key == rec_key else ""}</td>'
                f'<td>{esc(wf.get("train_period","N/A"))}</td>'
                f'<td>{esc(wf.get("test_period","N/A"))}</td>'
                f'<td class="num">{pct(wf.get("oos_annualized"))}</td>'
                f'<td class="num neutral">{pct(wf.get("spy_annualized"))}</td>'
                f'<td class="num {ac}">{pct(alpha)}</td>'
                f'</tr>')
        parts.append("</tbody></table>")

    # ── Lookback Backtest ──────────────────────────────────────────────────────
    if backtest_results:
        parts.append("""<h2>Lookback Backtest — Hypothetical Returns</h2>
<p style="color:var(--muted);font-size:0.83rem;margin-bottom:10px">
  What would $15,000 be worth today if today's optimal weights had been deployed 3, 6,
  or 12 months ago? <strong>Note: look-ahead bias is present</strong> — the optimizer used
  data including those periods. Figures are illustrative, not predictive.
</p>
<table><thead><tr><th>Period</th>""")
        for key, pf in PORTFOLIOS.items():
            parts.append(
                f'<th class="{"rec-col num" if key == rec_key else "num"}">'
                f'{esc(pf["label"])}'
                f'{"  ★" if key == rec_key else ""}</th>')
        parts.append("</tr></thead><tbody>")
        for mo, lbl in [(3, "3 months ago"), (6, "6 months ago"), (12, "12 months ago")]:
            parts.append(f"<tr><td><strong>{lbl}</strong></td>")
            for key in PORTFOLIOS:
                bt  = (backtest_results.get(key) or {}).get(mo)
                cls = f'num {"rec-col" if key == rec_key else ""}'
                if bt:
                    sign = "+" if bt["gain"] >= 0 else ""
                    gc   = "good" if bt["gain"] >= 0 else "bad"
                    parts.append(
                        f'<td class="{cls}">'
                        f'<span class="{gc}">{dollar(bt["end_value"])} '
                        f'({sign}{bt["total_return"]:.1%})</span><br>'
                        f'<span style="color:var(--muted);font-size:0.78rem">'
                        f'{sign}{bt["annualized"]:.1%}/yr ann.</span></td>')
                else:
                    parts.append(f'<td class="{cls}">N/A</td>')
            parts.append("</tr>")
        parts.append("</tbody></table>")

    # ── Stress Test ────────────────────────────────────────────────────────────
    if stress_results:
        parts.append("""<h2>Stress Test — Beta-Adjusted Historical Drawdowns</h2>
<p style="color:var(--muted);font-size:0.83rem;margin-bottom:10px">
  Estimates portfolio loss in four historical market crashes using portfolio beta vs SPY.
  Beta > 1.0 means the portfolio amplifies SPY moves — higher potential upside <em>and</em>
  higher crash exposure. Trough value = estimated lowest portfolio value during the crash.
</p>
<table><thead><tr><th>Scenario</th><th class="num">SPY Shock</th>""")
        for key, pf in PORTFOLIOS.items():
            parts.append(
                f'<th class="{"rec-col num" if key == rec_key else "num"}">'
                f'{esc(pf["label"].split("(")[0].strip())}'
                f'{"  ★" if key == rec_key else ""}</th>')
        parts.append("</tr></thead><tbody>")
        sample = next(iter(stress_results.values()))
        for i, s_row in enumerate(sample):
            name = s_row["Scenario"]
            spy_shock = s_row.get("SPY Historical Shock", "")
            parts.append(f"<tr><td><strong>{esc(name)}</strong></td>"
                         f'<td class="num bad">{esc(spy_shock)}</td>')
            for key in PORTFOLIOS:
                st  = stress_results[key][i]
                tv  = st.get("Est. Trough Value ($)", "N/A")
                dd  = st.get("Est. Portfolio Drawdown", "N/A")
                cls = f'num {"rec-col" if key == rec_key else ""}'
                parts.append(
                    f'<td class="{cls}"><span class="bad">{dollar(tv)}</span>'
                    f'<br><span style="color:var(--muted);font-size:0.78rem">'
                    f'drawdown: {esc(dd)}</span></td>')
            parts.append("</tr>")
        parts.append("</tbody></table>")

    # ── Buy Plans ──────────────────────────────────────────────────────────────
    parts.append("<h2>Buy Plans</h2>")
    parts.append(
        f'<p style="color:var(--muted);font-size:0.83rem;margin-bottom:16px">'
        f'Recommended scenario expanded by default. '
        f'Valuation verdicts: '
        f'<span style="color:var(--green)">★ UNDERVALUED</span> (cheapest third vs universe) · '
        f'<span style="color:var(--muted)">◆ FAIR VALUE</span> · '
        f'<span style="color:var(--red)">⚠ OVERVALUED</span> (dearest third) · '
        f'— = ETF / no DCF data available'
        f'</p>')

    for key, pf in PORTFOLIOS.items():
        plan, intl_cost, dom_cost = build_allocation_plan(
            pf["holdings"], current_prices, div_yields, val_signals)
        totals = plan[-1]
        fe = (factor_exposures or {}).get(key, {})
        is_rec = key == rec_key
        open_attr = "open" if is_rec else ""
        label_suffix = " — RECOMMENDED ★" if is_rec else ""

        parts.append(
            f'<details {open_attr} style="margin-bottom:16px">'
            f'<summary style="color:{"var(--accent)" if is_rec else "var(--muted)"}">'
            f'{esc(pf["label"])}{label_suffix}'
            f'  &nbsp;·&nbsp; Intl: {dollar(intl_cost)} · Domestic: {dollar(dom_cost)}'
            f'  &nbsp;·&nbsp; Style: {esc(fe.get("Style Tilt","—"))}'
            f'  · P/E: {esc(str(fe.get("Weighted Avg Forward P/E","—")))}'
            f'  · ROE: {esc(fe.get("Weighted Avg ROE","—"))}'
            f'</summary>')
        parts.append("""<table style="margin-top:10px"><thead><tr>
  <th>Type</th><th>Ticker</th><th class="num">Weight</th>
  <th class="num">Target $</th><th class="num">Price</th>
  <th class="num">Shares</th><th class="num">Cost $</th>
  <th class="num">Div Yield</th><th class="num">Ann. Income</th>
  <th>Verdict</th><th>DCF Range</th>
</tr></thead><tbody>""")

        for row in plan[:-1]:
            verd  = row.get("Verdict", "—")
            dcfr  = row.get("DCF Range ($)", "—")
            vc    = VCOLOR.get(verd, "#555")
            vico  = VICON.get(verd, "—")
            px_str = (f"${row['Current Price ($)']:,.2f}"
                      if isinstance(row["Current Price ($)"], float) else "N/A")
            inc_str = (f"${row['Est. Annual Income']:,.2f}"
                       if row.get("Est. Annual Income") else "—")
            typ_tag = (f'<span class="tag" style="background:#1a3a4a;color:var(--accent)">'
                       f'{esc(row["Type"])}</span>'
                       if row["Type"] in ("INTL", "INDEX") else
                       f'<span class="tag" style="background:#1e2a1e;color:var(--green)">'
                       f'stock</span>')
            parts.append(
                f'<tr>'
                f'<td>{typ_tag}</td>'
                f'<td><strong>{esc(row["Ticker"])}</strong></td>'
                f'<td class="num">{esc(row["Weight"])}</td>'
                f'<td class="num">{dollar(row["Target ($)"])}</td>'
                f'<td class="num">{esc(px_str)}</td>'
                f'<td class="num">{esc(str(row["Shares to Buy"]))}</td>'
                f'<td class="num">{dollar(row["Actual Cost ($)"])}</td>'
                f'<td class="num neutral">{esc(row["Div Yield (TTM)"])}</td>'
                f'<td class="num neutral">{esc(inc_str)}</td>'
                f'<td><span style="color:{vc};font-weight:600">'
                f'{vico} {esc(verd)}</span></td>'
                f'<td style="color:{vc}">{esc(dcfr)}</td>'
                f'</tr>')
        parts.append(
            f'<tr class="total-row">'
            f'<td colspan="2"><strong>TOTAL</strong></td>'
            f'<td class="num">100%</td>'
            f'<td class="num">{dollar(totals["Target ($)"])}</td>'
            f'<td></td><td></td>'
            f'<td class="num">{dollar(totals["Actual Cost ($)"])}</td>'
            f'<td></td>'
            f'<td class="num">{dollar(totals["Est. Annual Income"])}</td>'
            f'<td colspan="2" style="color:var(--muted);font-size:0.8rem">'
            f'Leftover: {dollar(totals.get("Difference ($)",0))}</td>'
            f'</tr>')
        parts.append("</tbody></table></details>")

    # ── Factor Exposure ────────────────────────────────────────────────────────
    if factor_exposures:
        parts.append("""<h2>Factor Exposure by Scenario</h2>
<p style="color:var(--muted);font-size:0.83rem;margin-bottom:10px">
  Weighted average characteristics across all holdings — tells you what kind of companies
  make up each portfolio: growth vs. value, high-quality vs. average, recent outperformers vs.
  laggards.
</p><table><thead><tr>
  <th>Scenario</th><th class="num">Avg Fwd P/E</th>
  <th class="num">Avg ROE</th><th class="num">12m Momentum</th><th>Style</th>
</tr></thead><tbody>""")
        for key, fe in factor_exposures.items():
            pf = PORTFOLIOS[key]
            parts.append(
                f'<tr class="{"rec-col" if key == rec_key else ""}">'
                f'<td><strong>{esc(pf["label"])}'
                f'{"  ★" if key == rec_key else ""}</strong></td>'
                f'<td class="num">{esc(str(fe.get("Weighted Avg Forward P/E","—")))}</td>'
                f'<td class="num">{esc(fe.get("Weighted Avg ROE","—"))}</td>'
                f'<td class="num">{esc(fe.get("Weighted Avg 12m Momentum","—"))}</td>'
                f'<td><span class="tag" style="background:var(--panel)">'
                f'{esc(fe.get("Style Tilt","—"))}</span></td>'
                f'</tr>')
        parts.append("</tbody></table>")

    # ── Risk Contribution ─────────────────────────────────────────────────────
    if risk_contribs:
        parts.append("""<h2>Risk Contribution by Position</h2>
<p style="color:var(--muted);font-size:0.83rem;margin-bottom:10px">
  What percentage of total portfolio variance does each position contribute?
  A position with a high risk/weight ratio is adding more risk than its size warrants.
</p>""")
        for key, pf in PORTFOLIOS.items():
            rc_rows = risk_contribs.get(key, [])
            if not rc_rows:
                continue
            is_rec = key == rec_key
            parts.append(
                f'<details {"open" if is_rec else ""} style="margin-bottom:12px">'
                f'<summary style="color:{"var(--accent)" if is_rec else "var(--muted)"}">'
                f'{esc(pf["label"])}{"  ★" if is_rec else ""}</summary>')
            parts.append("""<table style="margin-top:8px"><thead><tr>
  <th>Ticker</th><th class="num">Weight</th>
  <th class="num">Risk Contribution</th><th class="num">Risk/Weight</th>
</tr></thead><tbody>""")
            for row in rc_rows:
                rw = float(row["Risk/Weight Ratio"])
                rw_c = "bad" if rw > 1.5 else ("good" if rw < 0.8 else "neutral")
                parts.append(
                    f'<tr><td><strong>{esc(row["Ticker"])}</strong></td>'
                    f'<td class="num">{esc(row["Weight"])}</td>'
                    f'<td class="num">{esc(row["Risk Contribution"])}</td>'
                    f'<td class="num {rw_c}">{rw:.2f}x</td></tr>')
            parts.append("</tbody></table></details>")

    # ── Thematic Concentration ─────────────────────────────────────────────────
    if thematic_exposures:
        parts.append("""<h2>Thematic Concentration</h2>
<p style="color:var(--muted);font-size:0.83rem;margin-bottom:10px">
  Exposure to cross-sector investment themes. High concentration in a single theme
  (e.g. AI &amp; Semiconductors >40%) adds thematic risk beyond GICS sector caps.
</p><table><thead><tr>
  <th>Scenario</th>""")
        theme_names = list(THEMES.keys())
        for tn in theme_names:
            parts.append(f'<th class="num">{esc(tn)}</th>')
        parts.append("</tr></thead><tbody>")
        for key, pf in PORTFOLIOS.items():
            th_list = thematic_exposures.get(key, [])
            th_dict = dict(th_list)
            parts.append(
                f'<tr class="{"rec-col" if key == rec_key else ""}">'
                f'<td><strong>{esc(pf["label"])}{"  ★" if key == rec_key else ""}</strong></td>')
            for tn in theme_names:
                w = th_dict.get(tn, 0.0)
                c = "bad" if w >= 0.40 else ("neutral" if w >= 0.20 else "")
                parts.append(f'<td class="num {c}">{w:.0%}</td>')
            parts.append("</tr>")
        parts.append("</tbody></table>")

    # ── Correlation Heatmap (high pairs) ──────────────────────────────────────
    if corr_data:
        for key, pf in PORTFOLIOS.items():
            pairs = corr_data.get(key, {}).get("pairs", [])
            if not pairs:
                continue
            is_rec = key == rec_key
            if is_rec:
                parts.append(
                    f'<h2>High-Correlation Pairs — {esc(pf["label"])}</h2>'
                    f'<p style="color:var(--muted);font-size:0.83rem;margin-bottom:10px">'
                    f'Pairs with absolute correlation &gt;0.80. Holding both adds limited '
                    f'diversification benefit — consider whether both are needed.</p>'
                    f'<table><thead><tr>'
                    f'<th>Ticker A</th><th>Ticker B</th>'
                    f'<th class="num">Correlation</th><th>Risk Note</th>'
                    f'</tr></thead><tbody>')
                for p in pairs[:10]:
                    c = "bad" if p["Risk Note"] == "HIGH" else "neutral"
                    parts.append(
                        f'<tr><td>{esc(p["Ticker A"])}</td><td>{esc(p["Ticker B"])}</td>'
                        f'<td class="num {c}">{p["Correlation"]:.3f}</td>'
                        f'<td><span class="tag" style="background:var(--panel)">'
                        f'{esc(p["Risk Note"])}</span></td></tr>')
                parts.append("</tbody></table>")
                break

    # ── Earnings Calendar ──────────────────────────────────────────────────────
    if earnings_flags:
        all_flags = []
        for key, pf in PORTFOLIOS.items():
            ef = earnings_flags.get(key, {})
            for ticker, info in ef.items():
                if ticker not in [r["Ticker"] for r in all_flags]:
                    all_flags.append({
                        "Ticker": ticker,
                        "Earnings Date": info.get("date", "—"),
                        "Days Away": info.get("days_away", "—"),
                    })
        if all_flags:
            all_flags.sort(key=lambda x: x["Days Away"] if isinstance(x["Days Away"], int) else 999)
            parts.append(f"""<h2>Earnings Calendar Warning</h2>
<p style="color:var(--muted);font-size:0.83rem;margin-bottom:10px">
  Portfolio positions with earnings within the next {EARNINGS_WARN_DAYS} days.
  Consider sizing positions before large earnings surprises.
</p><table><thead><tr>
  <th>Ticker</th><th>Earnings Date</th><th class="num">Days Away</th>
</tr></thead><tbody>""")
            for row in all_flags:
                da = row["Days Away"]
                c = "bad" if isinstance(da, int) and da <= 7 else "neutral"
                parts.append(
                    f'<tr><td><strong>{esc(row["Ticker"])}</strong></td>'
                    f'<td>{esc(row["Earnings Date"])}</td>'
                    f'<td class="num {c}">{esc(str(da))}</td></tr>')
            parts.append("</tbody></table>")

    # ── Insider + Short Signals ────────────────────────────────────────────────
    if insider_signals or short_signals:
        parts.append("""<h2>Insider Transactions &amp; Short Interest</h2>
<p style="color:var(--muted);font-size:0.83rem;margin-bottom:10px">
  Insider net-buy signal: BULLISH = more buys than sells in last 90 days.
  Short interest: HIGH = &gt;15% of float, ELEVATED = 10–15%.
  High short + improving fundamentals can signal a short-squeeze catalyst.
</p><table><thead><tr>
  <th>Ticker</th><th>Insider Signal</th><th class="num">Net Buys (90d)</th>
  <th class="num">Short % Float</th><th class="num">Short Ratio (days)</th><th>Short Signal</th>
</tr></thead><tbody>""")
        all_tickers_set: set = set()
        if insider_signals:
            all_tickers_set |= set(insider_signals.keys())
        if short_signals:
            all_tickers_set |= set(short_signals.keys())
        for t in sorted(all_tickers_set):
            ins  = (insider_signals  or {}).get(t, {})
            shrt = (short_signals    or {}).get(t, {})
            ins_sig  = ins.get("signal",  "—")
            net_buys = ins.get("net_buys", "—")
            sp       = shrt.get("short_pct",   "—")
            sr       = shrt.get("short_ratio", "—")
            ss       = shrt.get("signal",       "—")
            ic = "good" if ins_sig == "BULLISH" else ("bad" if ins_sig == "BEARISH" else "neutral")
            sc2 = "bad" if ss == "HIGH SHORT" else ("neutral" if ss == "ELEVATED" else "")
            sp_str = f"{sp:.1%}" if isinstance(sp, float) else str(sp)
            sr_str = str(sr)
            parts.append(
                f'<tr><td><strong>{esc(t)}</strong></td>'
                f'<td class="{ic}">{esc(ins_sig)}</td>'
                f'<td class="num">{esc(str(net_buys))}</td>'
                f'<td class="num">{esc(sp_str)}</td>'
                f'<td class="num">{esc(sr_str)}</td>'
                f'<td class="{sc2}">{esc(ss)}</td></tr>')
        parts.append("</tbody></table>")

    # ── DRIP Projection ───────────────────────────────────────────────────────
    if drip_data:
        for key, pf in PORTFOLIOS.items():
            dd = drip_data.get(key)
            if not dd or not dd.get("projection"):
                continue
            is_rec = key == rec_key
            if is_rec:
                yr_data = dd["projection"]
                port_yield = dd.get("portfolio_yield", 0.0)
                parts.append(
                    f'<h2>Dividend Reinvestment (DRIP) Projection</h2>'
                    f'<p style="color:var(--muted);font-size:0.83rem;margin-bottom:10px">'
                    f'Based on recommended scenario ({esc(pf["label"])}) · '
                    f'Portfolio dividend yield: <strong>{port_yield:.2%}</strong> · '
                    f'DRIP compounds dividends back into the portfolio each year.'
                    f'</p><table><thead><tr>'
                    f'<th>Year</th><th class="num">Without DRIP ($)</th>'
                    f'<th class="num">With DRIP ($)</th><th class="num">DRIP Bonus ($)</th>'
                    f'</tr></thead><tbody>')
                for yr, (no_d, with_d) in yr_data.items():
                    bonus = with_d - no_d
                    parts.append(
                        f'<tr><td>Year {yr}</td>'
                        f'<td class="num">{dollar(no_d)}</td>'
                        f'<td class="num good">{dollar(with_d)}</td>'
                        f'<td class="num neutral">+{dollar(bonus)}</td></tr>')
                parts.append("</tbody></table>")
                break

    # ── Rebalance Plan (Current Holdings) ─────────────────────────────────────
    if rebalance_plan:
        parts.append("""<h2>Rebalance-to-Target Plan</h2>
<p style="color:var(--muted);font-size:0.83rem;margin-bottom:10px">
  Based on your current holdings (CURRENT_HOLDINGS config). Shows the trades
  needed to move from your existing positions to the recommended target allocation.
</p><table><thead><tr>
  <th>Ticker</th><th class="num">Current ($)</th><th class="num">Target ($)</th>
  <th class="num">Delta ($)</th><th class="num">Shares</th><th>Action</th>
</tr></thead><tbody>""")
        for row in rebalance_plan:
            ac = "good" if row["Action"] == "BUY" else ("bad" if row["Action"] == "SELL" else "neutral")
            parts.append(
                f'<tr><td><strong>{esc(row["Ticker"])}</strong></td>'
                f'<td class="num">{dollar(row["Current Value ($)"])}</td>'
                f'<td class="num">{dollar(row["Target Value ($)"])}</td>'
                f'<td class="num {ac}">{dollar(row["Delta ($)"])}</td>'
                f'<td class="num">{esc(str(row["Shares Delta"]))}</td>'
                f'<td class="{ac}"><strong>{esc(row["Action"])}</strong></td></tr>')
        parts.append("</tbody></table>")

    # ── DCA Schedule ──────────────────────────────────────────────────────────
    if dca_schedule:
        tranche_nums = sorted(set(r["Tranche"] for r in dca_schedule))
        tranche_amt  = INITIAL_INVESTMENT / len(tranche_nums)
        parts.append(
            f'<h2>Dollar-Cost Averaging Schedule ({DCA_TRANCHES} Tranches)</h2>'
            f'<p style="color:var(--muted);font-size:0.83rem;margin-bottom:10px">'
            f'Splits ${INITIAL_INVESTMENT:,} across {DCA_TRANCHES} purchases, '
            f'{DCA_INTERVAL} days apart. Each tranche ≈ ${tranche_amt:,.0f}.'
            f'</p>')
        for tranche in tranche_nums:
            tranche_rows = [r for r in dca_schedule if r["Tranche"] == tranche]
            date_str = tranche_rows[0]["Date"] if tranche_rows else "—"
            parts.append(
                f'<details style="margin-bottom:8px">'
                f'<summary>Tranche {tranche} — {esc(date_str)}'
                f'  (${tranche_amt:,.0f})</summary>'
                f'<table style="margin-top:6px"><thead><tr>'
                f'<th>Ticker</th><th class="num">Allocation ($)</th>'
                f'<th class="num">Price ($)</th><th class="num">Shares</th>'
                f'<th class="num">Cost ($)</th></tr></thead><tbody>')
            for row in tranche_rows:
                parts.append(
                    f'<tr><td>{esc(row["Ticker"])}</td>'
                    f'<td class="num">{dollar(row["Allocation ($)"])}</td>'
                    f'<td class="num">{dollar(row["Price ($)"])}</td>'
                    f'<td class="num">{esc(str(row["Shares"]))}</td>'
                    f'<td class="num">{dollar(row["Cost ($)"])}</td></tr>')
            parts.append("</tbody></table></details>")

    # ── Options Protection ─────────────────────────────────────────────────────
    if options_protection:
        parts.append("""<h2>Options-Based Downside Protection Estimates</h2>
<p style="color:var(--muted);font-size:0.83rem;margin-bottom:10px">
  Rule-of-thumb cost to hedge portfolio downside using SPY put options.
  Beta-adjusted notional reflects the portfolio's actual market sensitivity.
  <strong>These are estimates only</strong> — actual option premiums vary significantly
  with implied volatility and market conditions.
</p><table><thead><tr>
  <th>Strategy</th><th class="num">Hedged Notional ($)</th>
  <th class="num">Est. Premium ($)</th><th class="num">% of Portfolio</th>
  <th class="num">Breakeven Drop</th><th>Note</th>
</tr></thead><tbody>""")
        for row in options_protection:
            parts.append(
                f'<tr><td><strong>{esc(row["Strategy"])}</strong></td>'
                f'<td class="num">{dollar(row["Hedged Notional ($)"])}</td>'
                f'<td class="num bad">{dollar(row["Est. Premium ($)"])}</td>'
                f'<td class="num">{esc(row["Premium (% portfolio)"])}</td>'
                f'<td class="num">{esc(row["Breakeven Drop"])}</td>'
                f'<td style="color:var(--muted);font-size:0.8rem">'
                f'{esc(row["Note"])}</td></tr>')
        parts.append("</tbody></table>")

    # ── Personalised Investor Plans ────────────────────────────────────────────
    if personalized_plans:
        ACOLOR = {"EVALUATE SELL": "#f44336", "TRIM": "#ff9800",
                  "BUY MORE": "#4caf50", "NEW BUY": "#00bcd4", "HOLD": "#888", "KEEP": "#9e9e9e"}
        parts.append("""<h2>Personalised Investor Plans</h2>
<p style="color:var(--muted);font-size:0.83rem;margin-bottom:16px">
  Each plan is tailored to the investor's current holdings and available capital.
  Targets are based on the recommended scenario.
  <strong>EVALUATE SELL</strong> = position not in optimal portfolio (consider tax impact before selling).
  <strong>KEEP</strong> = existing holding the model doesn't pick — kept, no new money.
  <strong>TRIM</strong> = overweight vs target.
  <strong>BUY MORE / NEW BUY</strong> = deploy capital here.
</p>""")
        for plan in personalized_plans:
            parts.append(
                f'<details open style="margin-bottom:20px;border:1px solid var(--grid);'
                f'border-radius:8px;padding:14px">'
                f'<summary style="font-size:1.1rem;font-weight:700;color:var(--accent);'
                f'cursor:pointer">{esc(plan["name"])}</summary>')

            # Stats bar
            parts.append(
                f'<div style="display:flex;gap:24px;flex-wrap:wrap;margin:12px 0 16px">'
                f'<div><span style="color:var(--muted)">Existing Portfolio</span><br>'
                f'<strong>{dollar(plan["existing_value"])}</strong></div>'
                f'<div><span style="color:var(--muted)">Capital to Deploy Now</span><br>'
                f'<strong style="color:var(--green)">{dollar(plan["immediate_capital"])}</strong></div>')
            if plan["pending_capital"] > 0:
                parts.append(
                    f'<div><span style="color:var(--muted)">Pending Capital '
                    f'(~{plan["pending_days"]}d)</span><br>'
                    f'<strong style="color:var(--accent)">'
                    f'{dollar(plan["pending_capital"])}</strong></div>')
            parts.append(
                f'<div><span style="color:var(--muted)">Total Investable</span><br>'
                f'<strong>{dollar(plan["total_later"] if plan["pending_capital"] else plan["total_now"])}'
                f'</strong></div>')
            if plan["monthly_contribution"] > 0:
                parts.append(
                    f'<div><span style="color:var(--muted)">Monthly Contribution</span><br>'
                    f'<strong style="color:var(--green)">'
                    f'{dollar(plan["monthly_contribution"])}/mo</strong></div>')
            if plan.get("note"):
                parts.append(
                    f'<div style="color:var(--muted);font-size:0.82rem;'
                    f'align-self:center;font-style:italic">{esc(plan["note"])}</div>')
            parts.append("</div>")

            # Full position table
            parts.append("""<table><thead><tr>
  <th>Ticker</th><th>Action</th>
  <th class="num">Target Wt</th><th class="num">Current ($)</th>
  <th class="num">Target Now ($)</th><th class="num">Shares to Buy</th>
  <th class="num">Cost ($)</th><th class="num">Price ($)</th>
  <th>Verdict</th><th>DCF Range</th>
</tr></thead><tbody>""")
            for row in plan["positions"]:
                ac    = ACOLOR.get(row["Action"], "#888")
                verd  = row.get("Verdict", "—")
                vc    = VCOLOR.get(verd, "#555")
                vico  = VICON.get(verd, "")
                px_s  = (f"${row['Price ($)']:,.2f}"
                         if isinstance(row["Price ($)"], float) else "N/A")
                not_in_tgt = (
                    ' <span style="color:#f44336;font-size:0.75rem">[not in target]</span>'
                    if not row["In Target"] else ""
                )
                parts.append(
                    f'<tr>'
                    f'<td><strong>{esc(row["Ticker"])}</strong>{not_in_tgt}</td>'
                    f'<td><span style="color:{ac};font-weight:600">'
                    f'{esc(row["Action"])}</span></td>'
                    f'<td class="num">{esc(row["Target Wt"])}</td>'
                    f'<td class="num">{dollar(row["Current Value ($)"])}</td>'
                    f'<td class="num">{dollar(row["Target Now ($)"])}</td>'
                    f'<td class="num">{esc(str(row["Shares to Buy Now"]))}</td>'
                    f'<td class="num">{dollar(row["Cost Now ($)"])}</td>'
                    f'<td class="num neutral">{esc(px_s)}</td>'
                    f'<td><span style="color:{vc};font-weight:600">'
                    f'{vico} {esc(verd)}</span></td>'
                    f'<td style="color:{vc}">{esc(row["DCF Range ($)"])}</td>'
                    f'</tr>')
            parts.append("</tbody></table>")

            # Immediate deployment summary
            if plan["deployment_now"]:
                parts.append(
                    f'<h3 style="margin-top:16px;color:var(--green)">Deploy Now: '
                    f'{dollar(plan["immediate_capital"])} → '
                    f'{dollar(plan["immediate_capital"] - plan["cash_leftover_now"])} deployed, '
                    f'{dollar(plan["cash_leftover_now"])} leftover</h3>'
                    f'<table><thead><tr><th>Ticker</th><th>Action</th>'
                    f'<th class="num">Shares</th><th class="num">Price</th>'
                    f'<th class="num">Cost ($)</th><th>Verdict</th>'
                    f'</tr></thead><tbody>')
                for row in plan["deployment_now"]:
                    verd = row.get("Verdict", "—")
                    vc   = VCOLOR.get(verd, "#555")
                    vico = VICON.get(verd, "")
                    parts.append(
                        f'<tr><td><strong>{esc(row["Ticker"])}</strong></td>'
                        f'<td style="color:var(--green)">{esc(row["Action"])}</td>'
                        f'<td class="num">{esc(str(row["Shares"]))}</td>'
                        f'<td class="num">{dollar(row["Price ($)"])}</td>'
                        f'<td class="num good">{dollar(row["Cost ($)"])}</td>'
                        f'<td><span style="color:{vc}">{vico} {esc(verd)}</span></td></tr>')
                parts.append("</tbody></table>")

            # Pending capital deployment
            if plan["pending_deployment"]:
                parts.append(
                    f'<h3 style="margin-top:14px;color:var(--accent)">'
                    f'Deploy in ~{plan["pending_days"]} days: '
                    f'{dollar(plan["pending_capital"])}</h3>'
                    f'<table><thead><tr><th>Ticker</th><th>Action</th>'
                    f'<th class="num">Shares</th><th class="num">Price ($)</th>'
                    f'<th class="num">Cost ($)</th><th>Note</th>'
                    f'</tr></thead><tbody>')
                for row in plan["pending_deployment"]:
                    parts.append(
                        f'<tr><td><strong>{esc(row["Ticker"])}</strong></td>'
                        f'<td style="color:var(--accent)">{esc(row["Action"])}</td>'
                        f'<td class="num">{esc(str(row["Shares"]))}</td>'
                        f'<td class="num">{dollar(row["Price ($)"])}</td>'
                        f'<td class="num">{dollar(row["Cost ($)"])}</td>'
                        f'<td style="color:var(--muted);font-size:0.8rem">'
                        f'{esc(row["Note"])}</td></tr>')
                parts.append("</tbody></table>")

            # Monthly contribution schedule
            if plan["contribution_schedule"]:
                parts.append(
                    f'<h3 style="margin-top:14px">Monthly Contributions: '
                    f'{dollar(plan["monthly_contribution"])}/month</h3>'
                    f'<table><thead><tr>'
                    f'<th>Month</th><th>Date</th>'
                    f'<th class="num">Amount ($)</th>'
                    f'<th class="num">Cumulative ($)</th>'
                    f'<th>Suggestion</th>'
                    f'</tr></thead><tbody>')
                for row in plan["contribution_schedule"]:
                    parts.append(
                        f'<tr><td>Month {row["Month"]}</td>'
                        f'<td>{esc(row["Date"])}</td>'
                        f'<td class="num good">{dollar(row["Amount ($)"])}</td>'
                        f'<td class="num">{dollar(row["Cumulative ($)"])}</td>'
                        f'<td style="color:var(--muted);font-size:0.82rem">'
                        f'{esc(row["Suggestion"])}</td></tr>')
                parts.append("</tbody></table>")

            parts.append("</details>")

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    parts.append(f"""
<div class="disclaimer">
  <strong>Disclaimer:</strong> This report is generated by an algorithmic portfolio analyzer
  for educational and research purposes only. It does not constitute financial advice or a
  personalized investment recommendation. Past performance does not guarantee future results.
  All figures are based on historical data ending {END_DATE}. Monte Carlo projections are
  illustrative — actual outcomes will differ. Walk-forward test uses sample returns (no
  Black-Litterman views applied). Lookback backtest contains look-ahead bias.
  Consult a licensed financial advisor before making investment decisions.
  &nbsp;·&nbsp; Analysis generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
  &nbsp;·&nbsp; Output files: {OUTPUT_CHART} &nbsp;·&nbsp; {OUTPUT_EXCEL}
</div>
</div>
</body>
</html>""")

    return "\n".join(parts)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' to '(R, G, B, alpha)' for CSS rgba()."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"({r}, {g}, {b}, {alpha})"


# ─── INVESTOR HTML REPORT ─────────────────────────────────────────────────────

def generate_investor_html_report(
    plan: dict,
    rec_key: str,
    results: dict,
    mc_results: dict,
    stress_results: dict | None,
    wf_results: dict | None,
    catalyst_signals: dict | None = None,
    # Full analysis data (consolidated report)
    recommendation: dict | None = None,
    backtest_results: dict | None = None,
    factor_exposures: dict | None = None,
    val_signals: dict | None = None,
    risk_contribs: dict | None = None,
    thematic_exposures: dict | None = None,
    corr_data: dict | None = None,
    earnings_flags: dict | None = None,
    insider_signals: dict | None = None,
    short_signals: dict | None = None,
    drip_data: dict | None = None,
    rebalance_plan: list | None = None,
    dca_schedule: list | None = None,
    options_protection: list | None = None,
    tax_loss_list: list | None = None,
    regime: dict | None = None,
    current_prices: dict | None = None,
    knowledge_graph=None,
    fundamentals: dict | None = None,
    prices: "pd.DataFrame | None" = None,
    structural_risks: dict | None = None,
) -> str:
    """
    Self-contained dark-theme HTML report for a single investor profile.
    Shows their personalised action plan, key portfolio metrics, stress
    scenarios, and a 6-month outlook — no cross-investor data.
    """

    def esc(v) -> str:
        return str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def dollar(v) -> str:
        try:
            f = float(v)
            return f"${f:,.2f}"
        except Exception:
            return str(v)

    def _make_rationale(ticker, row, vsig, csig, fund):
        bullets = []
        try:
            wt_val = float(str(row.get("Target Wt", "0%")).replace("%", "").strip())
            action = row.get("Action", "HOLD")
            if wt_val >= 8:
                conv = f"High-conviction position at {wt_val:.1f}% of portfolio"
            elif wt_val >= 4:
                conv = f"Core position at {wt_val:.1f}% of portfolio"
            elif wt_val >= 1.5:
                conv = f"Supporting position at {wt_val:.1f}% of portfolio"
            else:
                conv = f"Diversifier at {wt_val:.1f}% of portfolio"
            if action == "BUY MORE":
                conv += " — model wants to increase exposure"
            elif action == "NEW BUY":
                conv += " — new entry flagged by multi-factor screen"
            elif action == "TRIM":
                conv += " — exceeds optimal weight, reduce to rebalance"
            elif action == "EVALUATE SELL":
                conv += " — not in optimal portfolio; consider exit (check tax impact first)"
            elif action == "HOLD":
                conv += " — within tolerance band, no trade required"
            elif action == "KEEP":
                conv = ("Existing holding — not among the model's picks, but kept "
                        "(no exit signal); new money goes elsewhere, so its weight shrinks over time")
            bullets.append(conv)
            if row.get("Exit Reason"):
                bullets.append(f"⚠ Exit signal: {row['Exit Reason']}")
        except Exception:
            pass
        # Valuation
        if vsig:
            verdict   = vsig.get("verdict", "—")
            dcf_up    = vsig.get("dcf_upside")
            buy_low   = vsig.get("buy_low")
            buy_high  = vsig.get("buy_high")
            # Verdicts are RELATIVE (cheapest / dearest third of this run's universe)
            dcf_txt = (f"; DCF base case {dcf_up:+.0f}% vs price" if dcf_up is not None else "")
            rng_txt = (f", range ${buy_low:,.0f}–${buy_high:,.0f}" if buy_low and buy_high else "")
            if verdict == "UNDERVALUED":
                vb = f"Valuation: among the cheapest third of the universe{dcf_txt}{rng_txt}"
            elif verdict == "OVERVALUED":
                vb = f"Valuation: among the most expensive third of the universe{dcf_txt}{rng_txt}"
            elif verdict == "FAIR VALUE":
                vb = f"Valuation: mid-range versus the universe{dcf_txt}{rng_txt}"
            else:
                vb = "Valuation: No DCF available (ETF / international / insufficient data)"
            bullets.append(vb)
        # Catalyst
        if csig:
            signal   = csig.get("signal", "")
            headline = str(csig.get("headline", ""))[:120]
            days_ago = csig.get("days_ago", "?")
            cat_type = csig.get("catalyst_type", "")
            kg_src   = csig.get("kg_source")
            if signal == "POSITIVE":
                cb = f'Catalyst ▲ {cat_type}: "{headline}" ({days_ago}d ago)'
                if kg_src:
                    cb += f" [indirect signal via {kg_src} supply-chain relationship]"
            elif signal == "NEGATIVE":
                cb = f'Catalyst ▼ {cat_type}: "{headline}" ({days_ago}d ago) — monitor closely'
            else:
                cb = "Catalyst: No significant recent news signal detected"
            bullets.append(cb)
        # Fundamentals
        if fund:
            pe  = fund.get("pe")
            roe = fund.get("roe")
            fp  = []
            if pe is not None:
                if pe < 15:
                    fp.append(f"P/E {pe:.1f}× (cheap vs ~20× market avg)")
                elif pe < 30:
                    fp.append(f"P/E {pe:.1f}× (in line with market)")
                else:
                    fp.append(f"P/E {pe:.1f}× (premium multiple — growth priced in)")
            if roe is not None:
                rp = roe * 100
                if rp >= 20:
                    fp.append(f"ROE {rp:.1f}% (strong)")
                elif rp >= 10:
                    fp.append(f"ROE {rp:.1f}% (adequate)")
                elif rp >= 0:
                    fp.append(f"ROE {rp:.1f}% (low — watch)")
                else:
                    fp.append(f"ROE {rp:.1f}% (negative)")
            if fp:
                bullets.append("Fundamentals: " + " · ".join(fp))
        # Tax context for any sell/trim suggestion (taxable account)
        if row.get("Tax Note"):
            bullets.append("Tax: " + row["Tax Note"])
        return bullets

    res    = results[rec_key]
    mc     = mc_results[rec_key]
    pf     = PORTFOLIOS[rec_key]
    name   = plan["name"]
    now    = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Colour constants (same palette as main report)
    ACOLOR = {
        "EVALUATE SELL": "#f44336", "TRIM":    "#ff9800",
        "BUY MORE":      "#4caf50", "NEW BUY": "#00bcd4",
        "HOLD":          "#888888", "KEEP":    "#9e9e9e",
    }

    CSS = """
:root{--bg:#0f1117;--panel:#1a1d2e;--grid:#2a2d3e;--text:#e0e0e0;
      --muted:#8888aa;--accent:#00bcd4;--green:#4caf50;--red:#f44336;
      --orange:#ff9800;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',sans-serif;
     font-size:14px;padding:32px;}
h1{font-size:1.8rem;margin-bottom:4px;color:var(--accent);}
h2{font-size:1.2rem;margin:28px 0 10px;color:var(--text);
   border-bottom:1px solid var(--grid);padding-bottom:6px;}
h3{font-size:1rem;margin:18px 0 8px;color:var(--muted);}
p{color:var(--muted);font-size:0.85rem;margin-bottom:10px;line-height:1.5;}
table{width:100%;border-collapse:collapse;margin-bottom:16px;font-size:0.83rem;}
th{background:var(--panel);color:var(--muted);padding:8px 10px;
   text-align:left;border-bottom:1px solid var(--grid);}
td{padding:7px 10px;border-bottom:1px solid var(--grid);}
tr:hover td{background:rgba(255,255,255,0.02);}
.num{text-align:right;}
.good{color:var(--green);}
.bad{color:var(--red);}
.neutral{color:var(--muted);}
.hero{background:var(--panel);border-radius:10px;padding:24px 28px;
      margin-bottom:28px;border-left:4px solid var(--accent);}
.stats{display:flex;gap:28px;flex-wrap:wrap;margin-top:14px;}
.stat label{display:block;font-size:0.75rem;color:var(--muted);margin-bottom:2px;}
.stat strong{font-size:1.3rem;}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));
             gap:14px;margin-bottom:24px;}
.metric{background:var(--panel);border-radius:8px;padding:14px 16px;}
.metric .label{font-size:0.72rem;color:var(--muted);margin-bottom:4px;}
.metric .value{font-size:1.25rem;font-weight:700;}
.tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.75rem;}
.disclaimer{margin-top:36px;padding:14px 18px;background:var(--panel);
            border-radius:8px;color:var(--muted);font-size:0.78rem;line-height:1.6;}
details summary{cursor:pointer;padding:6px 0;color:var(--muted);font-size:0.85rem;}
details[open] summary{color:var(--text);}
"""

    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        f'<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width">',
        f'<title>Investment Plan — {esc(name)}</title>',
        f'<style>{CSS}</style></head>',
        '<body>',
    ]

    # ── Hero ──────────────────────────────────────────────────────────────────
    pending_line = ""
    if plan["pending_capital"] > 0:
        pending_line = (
            f'  <div class="stat"><label>Pending Capital (~{plan["pending_days"]}d)</label>'
            f'<strong style="color:var(--accent)">{dollar(plan["pending_capital"])}</strong></div>')

    monthly_line = ""
    if plan["monthly_contribution"] > 0:
        monthly_line = (
            f'  <div class="stat"><label>Monthly Contribution</label>'
            f'<strong style="color:var(--green)">{dollar(plan["monthly_contribution"])}/mo</strong></div>')

    reserved_line = ""
    if plan.get("reserved_total", 0) > 0:
        reserved_line = (
            f'  <div class="stat"><label>Cash Reserved (events)</label>'
            f'<strong style="color:#ce93d8">{dollar(plan["reserved_total"])}</strong></div>')

    bl_views_tag = ""
    if plan.get("bl_views"):
        bl_views_tag = (
            f'<p style="color:#90caf9;font-size:0.82rem;margin-top:6px">'
            f'&#9656; Portfolio optimized with {len(plan["bl_views"])} personal BL conviction view(s) '
            f'— weights differ from the shared scenario baseline.</p>')

    parts.append(f"""
<div class="hero">
  <h1>Investment Plan — {esc(name)}</h1>
  <p style="color:var(--muted);margin-top:4px">
    Based on <strong style="color:var(--accent)">{esc(pf["label"])}</strong>
    (recommended scenario) &nbsp;·&nbsp; Generated {esc(now)}
  </p>
  {f'<p style="color:var(--muted);font-style:italic;margin-top:6px">{esc(plan["note"])}</p>' if plan.get("note") else ""}
  {bl_views_tag}
  <div class="stats">
    <div class="stat"><label>Existing Portfolio</label>
      <strong>{dollar(plan["existing_value"])}</strong></div>
    <div class="stat"><label>Capital to Deploy Now</label>
      <strong style="color:var(--green)">{dollar(plan["immediate_capital"])}</strong></div>
    {pending_line}
    <div class="stat"><label>Total Investable</label>
      <strong>{dollar(plan["total_later"] if plan["pending_capital"] else plan["total_now"])}</strong></div>
    {monthly_line}
    {reserved_line}
  </div>
</div>""")

    # ── Portfolio performance snapshot ────────────────────────────────────────
    p5  = np.percentile(mc[:, -1],  5)
    p50 = np.median(mc[:, -1])
    p95 = np.percentile(mc[:, -1], 95)
    wf  = (wf_results or {}).get(rec_key, {})
    alpha_s = (f'{wf["alpha"]:+.1%}' if wf.get("alpha") is not None else "N/A")
    alpha_c = "good" if (wf.get("alpha") or 0) > 0 else "bad"

    parts.append('<h2>Recommended Scenario — Key Metrics</h2>')
    parts.append('<div class="metric-grid">')
    for label, value, cls in [
        ("3yr CAGR",           f"{res['cagr']:.1%}",            "good" if res['cagr'] > 0.10 else ""),
        ("Sharpe Ratio",       f"{res['sharpe']:.2f}",           "good" if res['sharpe'] > 1.0 else ""),
        ("Sortino Ratio",      f"{res['sortino']:.2f}",          "good" if res['sortino'] > 1.0 else ""),
        ("Max Drawdown",       f"{res['max_dd']:.1%}",           "bad"),
        ("Volatility",         f"{res['volatility']:.1%}",       "neutral"),
        (f"MC P50 {PROJECTION_YEARS}yr", f"${p50:,.0f}",         "good"),
        (f"MC P5  {PROJECTION_YEARS}yr", f"${p5:,.0f}",          "neutral"),
        (f"MC P95 {PROJECTION_YEARS}yr", f"${p95:,.0f}",         "good"),
        ("OOS Alpha vs SPY",   alpha_s,                           alpha_c),
    ]:
        parts.append(
            f'<div class="metric"><div class="label">{esc(label)}</div>'
            f'<div class="value {cls}">{esc(value)}</div></div>')
    parts.append("</div>")

    # ── Conviction Views (BL) ────────────────────────────────────────────────
    if plan.get("bl_views"):
        parts.append('<h2>Conviction Views</h2>')
        parts.append(
            '<p>These views were incorporated into your personal Black-Litterman optimization. '
            'The expected return for each view stock was shifted toward your stated belief, '
            'proportional to your confidence level. Higher confidence compresses the view\'s '
            'uncertainty so the optimizer takes it more literally.</p>')
        parts.append("""<table><thead><tr>
  <th>Ticker</th>
  <th class="num">Expected Return</th>
  <th class="num">Confidence</th>
  <th>Thesis</th>
</tr></thead><tbody>""")
        for v in plan["bl_views"]:
            conf     = float(v.get("confidence", 0.5))
            conf_pct = f"{conf:.0%}"
            if conf >= 0.75:
                conf_color = "#4caf50"
            elif conf >= 0.55:
                conf_color = "#ff9800"
            else:
                conf_color = "#9e9e9e"
            ret_pct = f"{float(v.get('expected_return', 0)):.0%}"
            parts.append(
                f'<tr>'
                f'<td><strong>{esc(v["ticker"])}</strong></td>'
                f'<td class="num good">{esc(ret_pct)} / yr</td>'
                f'<td class="num"><span style="color:{conf_color};font-weight:600">'
                f'{esc(conf_pct)}</span></td>'
                f'<td style="color:var(--muted);font-size:0.85rem">{esc(v.get("note",""))}</td>'
                f'</tr>')
        parts.append("</tbody></table>")

    # ── Cash Reserves ─────────────────────────────────────────────────────────
    if plan.get("cash_reserves"):
        parts.append('<h2>Cash Reserves (Earmarked)</h2>')
        parts.append(
            '<p>The following amounts are held in cash (e.g. HYSA) and excluded from '
            'portfolio deployment. They will be released when the anticipated event occurs.</p>')
        parts.append("""<table><thead><tr>
  <th>Reason</th><th class="num">Amount</th><th>Note</th>
</tr></thead><tbody>""")
        for r in plan["cash_reserves"]:
            parts.append(
                f'<tr>'
                f'<td><strong>{esc(r.get("reason",""))}</strong></td>'
                f'<td class="num" style="color:#ce93d8">{dollar(r.get("amount",0))}</td>'
                f'<td style="color:var(--muted);font-size:0.85rem">{esc(r.get("note",""))}</td>'
                f'</tr>')
        parts.append("</tbody></table>")

    # ── Portfolio Architecture ────────────────────────────────────────────────
    bl_views_list = plan.get("bl_views", [])
    target_split  = {}
    # Look up target split from INVESTOR_PROFILES by matching plan name
    for _pid, _prof in INVESTOR_PROFILES.items():
        if _prof.get("name") == plan.get("name"):
            target_split = _prof.get("portfolio_split", {})
            break

    # Classify every in-target position
    arch: dict[str, list] = {"Conviction": [], "Upside / Growth": [], "Bluechip Core": []}
    total_portfolio = plan.get("total_now") or plan.get("total_later") or 1.0
    for row in plan["positions"]:
        if not row.get("In Target"):
            continue
        tk = row["Ticker"]
        try:
            w = float(str(row.get("Target Wt", "0%")).replace("%", "").strip()) / 100
        except Exception:
            w = 0.0
        bname, bcolor, bdesc = classify_holding(
            tk, row, fundamentals, bl_views_list)
        arch[bname].append({"ticker": tk, "weight": w, "color": bcolor})

    BUCKET_META = {
        "Conviction":    {"color": "#ce93d8", "icon": "★",
                          "role": "Thesis-driven plays — calibrated by event studies and BL views. "
                                  "Highest potential upside; requires active monitoring."},
        "Upside / Growth": {"color": "#ff9800", "icon": "▲",
                             "role": "Factor-selected for momentum, quality, and value composite score. "
                                     "Core growth engine of the portfolio."},
        "Bluechip Core": {"color": "#4caf50", "icon": "◆",
                          "role": "Large-cap dividend payers with lower beta. "
                                  "Provide income, reduce drawdown, and anchor the portfolio in downturns."},
    }

    parts.append('<h2>Portfolio Architecture</h2>')
    parts.append(
        '<p>Your portfolio is divided into three risk/role buckets. '
        'Bars show actual allocation vs your target split.</p>')
    parts.append('<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:20px">')

    for bname, meta in BUCKET_META.items():
        items    = arch[bname]
        actual_w = sum(i["weight"] for i in items)
        target_w = target_split.get(bname, 0.0)
        tickers_str = "  ".join(
            f'<span style="background:{meta["color"]}22;color:{meta["color"]};'
            f'padding:2px 7px;border-radius:4px;font-size:0.78rem;font-weight:600">'
            f'{esc(i["ticker"])}</span>'
            for i in sorted(items, key=lambda x: -x["weight"])
        ) or '<span style="color:#555;font-size:0.8rem">— none —</span>'
        bar_actual = min(actual_w * 100, 100)
        bar_target = min(target_w * 100, 100)
        drift_color = "#4caf50" if abs(actual_w - target_w) < 0.08 else "#ff9800"
        parts.append(f"""
<div style="background:var(--panel);border-radius:10px;padding:16px 18px;
            border-top:3px solid {meta['color']}">
  <div style="font-size:1rem;font-weight:700;color:{meta['color']};margin-bottom:4px">
    {meta['icon']} {esc(bname)}
  </div>
  <div style="font-size:0.78rem;color:var(--muted);margin-bottom:10px;line-height:1.45">
    {esc(meta['role'])}
  </div>
  <div style="margin-bottom:8px">{tickers_str}</div>
  <div style="font-size:0.82rem;color:var(--muted);margin-bottom:4px">
    Actual <strong style="color:{meta['color']}">{actual_w:.0%}</strong>
    &nbsp;vs&nbsp; target <strong style="color:#aaa">{target_w:.0%}</strong>
    &nbsp;<span style="color:{drift_color};font-size:0.75rem">
      {'✓ on-target' if abs(actual_w - target_w) < 0.08 else f'({actual_w - target_w:+.0%} drift)'}
    </span>
  </div>
  <div style="background:#222;border-radius:4px;height:6px;margin-bottom:2px;position:relative">
    <div style="background:{meta['color']};border-radius:4px;height:6px;width:{bar_actual:.1f}%"></div>
  </div>
  <div style="background:#333;border-radius:4px;height:3px;width:{bar_target:.1f}%;opacity:0.5"></div>
</div>""")
    parts.append('</div>')

    # ── Full position action table ─────────────────────────────────────────────
    parts.append('<h2>Your Position Plan</h2>')
    parts.append(
        '<p>What to do with each position. '
        '<span style="color:#4caf50">BUY MORE / NEW BUY</span> = add shares. '
        '<span style="color:#ff9800">TRIM</span> = reduce overweight position. '
        '<span style="color:#f44336">EVALUATE SELL</span> = not in optimal portfolio '
        '(check tax impact first). '
        '<span style="color:#9e9e9e">KEEP</span> = existing holding kept; new money goes '
        'to the model\'s picks.</p>')
    parts.append("""<table><thead><tr>
  <th>Ticker</th><th>Action</th>
  <th class="num">Target Wt</th><th class="num">Current ($)</th>
  <th class="num">Target Now ($)</th>
  <th class="num">Shares to Buy</th><th class="num">Cost ($)</th>
  <th class="num">Price ($)</th><th>Verdict</th><th>DCF Range</th><th>Catalyst</th>
</tr></thead><tbody>""")
    for row in plan["positions"]:
        ac   = ACOLOR.get(row["Action"], "#888")
        verd = row.get("Verdict", "—")
        vc   = VCOLOR.get(verd, "#555")
        vico = VICON.get(verd, "")
        px_s = (f"${row['Price ($)']:,.2f}"
                if isinstance(row["Price ($)"], float) else "N/A")
        not_in_tgt = (
            (' <span style="color:#9e9e9e;font-size:0.72rem">[kept]</span>'
             if row["Action"] == "KEEP" else
             ' <span style="color:#f44336;font-size:0.72rem">[not in target]</span>')
            if not row["In Target"] else ""
        )
        override_badge = ""
        if row.get("Override"):
            override_badge = (
                f' <span style="background:#7b1fa2;color:#fff;font-size:0.68rem;'
                f'padding:1px 6px;border-radius:3px;vertical-align:middle">'
                f'OVERRIDE {esc(row["Override Floor"])} '
                f'(model: {esc(row["Model Wt"])})</span>'
            )
        tgt_wt_cell = f'{esc(row["Target Wt"])}{override_badge}'
        cat        = (catalyst_signals or {}).get(row["Ticker"], {})
        cat_hl     = esc(str(cat.get("headline", ""))[:100])
        cat_days   = cat.get("days_ago", "?")
        cat_type   = esc(cat.get("catalyst_type", ""))
        cat_tip    = f'{cat_hl} ({cat_days}d ago)'
        if cat.get("signal") == "POSITIVE":
            cat_cell = (
                f'<span style="color:#4caf50;font-size:0.75rem" title="{cat_tip}">'
                f'▲ {cat_type}</span>'
            )
        elif cat.get("signal") == "NEGATIVE":
            cat_cell = (
                f'<span style="color:#f44336;font-size:0.75rem" title="{cat_tip}">'
                f'▼ {cat_type}</span>'
            )
        else:
            cat_cell = '<span style="color:#555">—</span>'
        parts.append(
            f'<tr>'
            f'<td><strong>{esc(row["Ticker"])}</strong>{not_in_tgt}</td>'
            f'<td><span style="color:{ac};font-weight:600">{esc(row["Action"])}</span></td>'
            f'<td class="num">{tgt_wt_cell}</td>'
            f'<td class="num">{dollar(row["Current Value ($)"])}</td>'
            f'<td class="num">{dollar(row["Target Now ($)"])}</td>'
            f'<td class="num">{esc(str(row["Shares to Buy Now"]))}</td>'
            f'<td class="num">{dollar(row["Cost Now ($)"])}</td>'
            f'<td class="num neutral">{esc(px_s)}</td>'
            f'<td><span style="color:{vc};font-weight:600">{vico} {esc(verd)}</span></td>'
            f'<td style="color:{vc}">{esc(row["DCF Range ($)"])}</td>'
            f'<td>{cat_cell}</td>'
            f'</tr>')
    parts.append("</tbody></table>")

    # ── Position Rationale Cards ──────────────────────────────────────────────
    parts.append('<h2>Why Each Position?</h2>')
    parts.append(
        '<p>Each card explains why the model allocated that specific percentage — '
        'covering the investment thesis, valuation, catalyst, and risk profile.</p>')

    def _wt_sort(r):
        try:
            return float(str(r.get("Target Wt", "0%")).replace("%", "").strip())
        except Exception:
            return 0.0

    rec_pf_max_weight = PORTFOLIOS.get(rec_key, {}).get("max_weight", 0.25)

    for row in sorted(plan["positions"], key=_wt_sort, reverse=True):
        tk   = row["Ticker"]
        ac   = ACOLOR.get(row["Action"], "#888")
        vsig = (val_signals or {}).get(tk, {})
        csig = (catalyst_signals or {}).get(tk, {})
        fund = (fundamentals or {}).get(tk, {}) if fundamentals else {}
        buls = _make_rationale(tk, row, vsig, csig, fund)

        # Classify for bucket badge
        bname, bcolor, bdesc = classify_holding(tk, row, fundamentals, bl_views_list)
        bucket_badge = (
            f'<span style="background:{bcolor}22;color:{bcolor};'
            f'padding:2px 8px;border-radius:4px;font-size:0.72rem;font-weight:700;'
            f'margin-left:8px;vertical-align:middle">{esc(bname)}</span>'
        )

        # "Why this weight?" headline
        try:
            tgt_w = float(str(row.get("Target Wt", "0%")).replace("%", "").strip()) / 100
        except Exception:
            tgt_w = 0.0
        tgt_dollar = round(tgt_w * total_portfolio, 0)

        # Determine weight driver
        at_cap = tgt_w >= rec_pf_max_weight - 0.01
        is_conviction = any(v.get("ticker") == tk for v in bl_views_list)
        if is_conviction and at_cap:
            weight_driver = (f"At the {rec_pf_max_weight:.0%} scenario cap — "
                             f"conviction view drives maximum allowed allocation")
        elif is_conviction:
            weight_driver = (f"BL conviction view ({next((v['expected_return'] for v in bl_views_list if v['ticker']==tk), 0):.0%} "
                             f"expected return, {next((v['confidence'] for v in bl_views_list if v['ticker']==tk), 0):.0%} confidence) "
                             f"pulls allocation above model baseline")
        elif at_cap:
            weight_driver = (f"At the {rec_pf_max_weight:.0%} scenario cap — "
                             f"optimizer would go higher without the constraint")
        elif tgt_w < 0.05:
            weight_driver = "Diversifier role — low correlation with other holdings reduces portfolio CVaR"
        else:
            weight_driver = (f"Optimizer's natural allocation — best risk/return contribution "
                             f"at this weight given covariance with other positions")

        # Risk note
        verdict = vsig.get("verdict", "—")
        beta_v  = fund.get("beta")
        srisk   = (structural_risks or {}).get(tk)
        if srisk:
            ev = (f" Recent: “{srisk['evidence'][0][:80]}”" if srisk.get("evidence") else "")
            risk_note = (f"STRUCTURAL — {srisk['risk']} (severity {srisk['severity']}): "
                         f"{srisk['note']}.{ev}")
        elif beta_v and beta_v > 1.3:
            risk_note = f"High beta ({beta_v:.1f}×) — amplifies market moves in both directions"
        elif beta_v and beta_v < 0.7:
            risk_note = f"Low beta ({beta_v:.1f}×) — defensive; lags in rallies, holds better in selloffs"
        elif verdict == "OVERVALUED":
            risk_note = "Stretched valuation — DCF suggests limited margin of safety at current price"
        elif bname == "Conviction":
            risk_note = "Binary catalyst risk — thesis must materialise; monitor FDA/approval timeline"
        else:
            risk_note = "Standard equity risk — diversified across sector and factor exposures"

        try:
            sh = int(float(str(row.get("Shares to Buy Now", 0))))
        except Exception:
            sh = 0
        act = row["Action"]
        if act in ("BUY MORE", "NEW BUY") and sh > 0:
            act_label = f"Buy {sh} share{'s' if sh != 1 else ''}"
        elif act == "TRIM":
            act_label = "Trim position"
        elif act == "EVALUATE SELL":
            act_label = "Evaluate selling"
        elif act == "KEEP":
            act_label = "Keep — no new money"
        else:
            act_label = "Hold — no trade"

        parts.append(
            f'<div style="margin-bottom:12px;background:var(--panel);border-radius:10px;'
            f'padding:14px 18px;border-left:4px solid {bcolor}">'
            # Header row
            f'<div style="display:flex;align-items:baseline;justify-content:space-between;'
            f'flex-wrap:wrap;gap:6px;margin-bottom:8px">'
            f'  <div>'
            f'    <strong style="font-size:1.05rem;color:var(--text)">{esc(tk)}</strong>'
            f'    {bucket_badge}'
            f'    <span style="color:var(--muted);font-size:0.82rem;margin-left:10px">'
            f'      {esc(act_label)}</span>'
            f'  </div>'
            f'  <div style="text-align:right">'
            f'    <span style="font-size:1.1rem;font-weight:700;color:{bcolor}">'
            f'      {esc(row["Target Wt"])}</span>'
            f'    <span style="color:var(--muted);font-size:0.8rem;margin-left:6px">'
            f'      ≈ ${tgt_dollar:,.0f}</span>'
            f'  </div>'
            f'</div>'
            # Why this weight
            f'<div style="background:#ffffff0a;border-radius:6px;padding:8px 12px;'
            f'margin-bottom:8px;font-size:0.83rem">'
            f'  <span style="color:{bcolor};font-weight:600">Why {esc(row["Target Wt"])}?</span>'
            f'  &ensp;<span style="color:var(--text)">{esc(weight_driver)}</span>'
            f'</div>'
        )
        # Rationale bullets
        parts.append('<ul style="margin:0 0 6px 16px;line-height:1.8;font-size:0.84rem">')
        for b in buls:
            parts.append(f'<li style="color:var(--text)">{esc(b)}</li>')
        if not buls:
            parts.append('<li style="color:var(--muted)">No detailed rationale available.</li>')
        # Risk note footer
        parts.append(
            f'</ul><div style="font-size:0.78rem;color:#ff9800;margin-top:4px">'
            f'⚠ Risk: {esc(risk_note)}</div>'
            f'</div>'
        )

    # ── "What If You Had Bought Earlier?" ────────────────────────────────────
    if prices is not None and not prices.empty:
        buy_rows = [r for r in plan["positions"] if r["Action"] in ("BUY MORE", "NEW BUY")]
        if buy_rows:
            # Find nearest date for each lookback window
            def _lookup_px(ticker, days_back):
                if ticker not in prices.columns:
                    return None
                col = prices[ticker].dropna()
                if col.empty:
                    return None
                try:
                    idx = col.index
                    target = idx[-1] - pd.Timedelta(days=days_back)
                    pos = idx.searchsorted(target)
                    pos = max(0, min(pos, len(idx) - 1))
                    return float(col.iloc[pos])
                except Exception:
                    return None

            WINDOWS = [("3 months", 91), ("6 months", 182), ("12 months", 365)]
            # Build rows
            sim_rows = []
            for r in buy_rows:
                tk = r["Ticker"]
                try:
                    sh = int(float(str(r.get("Shares to Buy Now", 0))))
                except Exception:
                    sh = 0
                try:
                    cur_px = float(r["Price ($)"])
                except Exception:
                    cur_px = None
                if sh <= 0 or cur_px is None:
                    continue
                gains = []
                for lbl, days in WINDOWS:
                    hist_px = _lookup_px(tk, days)
                    if hist_px and hist_px > 0:
                        gain    = (cur_px - hist_px) * sh
                        pct_ret = (cur_px - hist_px) / hist_px
                        gains.append((lbl, hist_px, gain, pct_ret))
                    else:
                        gains.append((lbl, None, None, None))
                sim_rows.append((tk, r["Action"], sh, cur_px, gains))

            if sim_rows:
                parts.append('<h2>What If You Had Bought Earlier?</h2>')
                parts.append(
                    '<p>Paper gain you would hold <em>today</em> had you executed these '
                    'exact buy recommendations 3, 6, or 12 months ago. '
                    'Green = profit, red = loss at the time. '
                    '<strong>Past performance does not guarantee future results.</strong></p>')
                parts.append("""<table><thead><tr>
  <th>Ticker</th><th>Action</th><th class="num">Shares</th>
  <th class="num">Now</th>
  <th class="num">3m ago price</th><th class="num">3m gain</th>
  <th class="num">6m ago price</th><th class="num">6m gain</th>
  <th class="num">12m ago price</th><th class="num">12m gain</th>
</tr></thead><tbody>""")
                totals = {lbl: 0.0 for lbl, _ in WINDOWS}
                for tk, action, sh, cur_px, gains in sim_rows:
                    ac2 = ACOLOR.get(action, "#888")
                    cells = [
                        f'<td><strong>{esc(tk)}</strong></td>',
                        f'<td><span style="color:{ac2}">{esc(action)}</span></td>',
                        f'<td class="num">{sh}</td>',
                        f'<td class="num neutral">${cur_px:,.2f}</td>',
                    ]
                    for lbl, hist_px, gain, pct_ret in gains:
                        if hist_px is None:
                            cells.append('<td class="num neutral">—</td>')
                            cells.append('<td class="num neutral">—</td>')
                        else:
                            totals[lbl] += gain
                            sign  = "+" if gain >= 0 else ""
                            gcls  = "good" if gain >= 0 else "bad"
                            cells.append(f'<td class="num neutral">${hist_px:,.2f}</td>')
                            cells.append(
                                f'<td class="num {gcls}">{sign}${gain:,.0f} '
                                f'({sign}{pct_ret:.1%})</td>')
                    parts.append(f'<tr>{"".join(cells)}</tr>')
                # Totals row
                total_cells = [
                    '<td colspan="3"><strong>Combined gain</strong></td>',
                    '<td></td>',
                ]
                for lbl, _ in WINDOWS:
                    t = totals[lbl]
                    sign = "+" if t >= 0 else ""
                    gcls = "good" if t >= 0 else "bad"
                    total_cells.append('<td></td>')
                    total_cells.append(
                        f'<td class="num {gcls}"><strong>{sign}${t:,.0f}</strong></td>')
                parts.append(
                    f'<tr style="border-top:2px solid var(--grid)">{"".join(total_cells)}</tr>')
                parts.append("</tbody></table>")

    # ── FDA & Clinical Pipeline ───────────────────────────────────────────────
    fda_rows = [
        (t, sig) for t, sig in (catalyst_signals or {}).items()
        if sig.get("fda_trials") or sig.get("fda_approval")
    ]
    if fda_rows:
        parts.append('<h2>FDA &amp; Clinical Pipeline</h2>')
        parts.append(
            '<p>Active Phase 2/3 trials and recent FDA approvals detected via '
            '<strong>ClinicalTrials.gov</strong> and the <strong>FDA Novel Drug '
            'Approvals</strong> database. These are binary catalyst events that '
            'can move a stock dramatically — factor them into position sizing.</p>')
        for t, sig in sorted(fda_rows, key=lambda x: -x[1]["score"]):
            sig_color = "#4caf50" if sig["signal"] == "POSITIVE" else "#f44336"
            parts.append(
                f'<details style="margin-bottom:10px;background:var(--panel);'
                f'border-radius:8px;padding:10px 16px">'
                f'<summary style="cursor:pointer;font-weight:600">'
                f'<span style="color:{sig_color}">'
                f'{"▲" if sig["signal"] == "POSITIVE" else "▼"} '
                f'{esc(t)}</span> &nbsp; '
                f'<span style="color:var(--muted);font-size:0.85rem">'
                f'{esc(sig.get("catalyst_type",""))}</span></summary>')
            if sig.get("fda_approval"):
                parts.append(
                    f'<p style="color:#4caf50;margin-top:8px">'
                    f'★ Recent FDA Approval: <strong>{esc(sig["fda_approval"])}</strong>'
                    f' (last 90 days)</p>')
            if sig.get("fda_trials"):
                parts.append("""<table style="margin-top:8px"><thead><tr>
  <th>Trial</th><th>Phase</th><th>Status</th>
  <th class="num">Completion</th><th class="num">Days Away</th>
</tr></thead><tbody>""")
                for tr in sig["fda_trials"][:5]:
                    d = tr["days_away"]
                    urgency = ("bad" if d <= 90 else
                               "neutral" if d <= 270 else "good")
                    parts.append(
                        f'<tr>'
                        f'<td style="font-size:0.8rem">{esc(tr["title"])}</td>'
                        f'<td style="font-size:0.8rem">{esc(tr["phase"])}</td>'
                        f'<td style="font-size:0.8rem;color:var(--muted)">'
                        f'{esc(tr["status"])}</td>'
                        f'<td class="num">{esc(tr["completion"])}</td>'
                        f'<td class="num {urgency}">{d}d</td></tr>')
                parts.append("</tbody></table>")
            parts.append("</details>")

    # ── Knowledge Graph relationships ────────────────────────────────────────
    if knowledge_graph is not None:
        try:
            portfolio_tickers = list(
                PORTFOLIOS.get(rec_key, {}).get("holdings", {}).keys()
            )
            kg_html = knowledge_graph.to_html_section(
                portfolio_tickers, catalyst_signals
            )
            if kg_html:
                parts.append(kg_html)
        except Exception as _kg_exc:
            pass  # KG section is non-critical — skip silently

    # ── Deployment timing recommendation ─────────────────────────────────────
    dt = plan.get("deployment_timing")
    if dt:
        conf_color = {"HIGH": "#4caf50", "MEDIUM": "#ff9800", "LOW": "#f44336"}.get(
            dt["confidence"], "#888")
        dt_regime_str = (dt["market_regime"]["regime"]
                         if isinstance(dt["market_regime"], dict)
                         else str(dt["market_regime"]))
        bear_note = (" — exercise extra caution with sizing"
                     if dt_regime_str.upper() == "BEAR" else "")
        parts.append(f'<h2>Deployment Strategy</h2>')
        parts.append(
            f'<p>Based on current market conditions (regime: <strong>'
            f'{esc(dt_regime_str)}</strong>, 30d SPY vol: '
            f'<strong>{dt["spy_vol_30d"]:.0%}</strong>, valuation heat: '
            f'{dt["over_pct"]:.0%} overvalued / {dt["under_pct"]:.0%} undervalued), '
            f'the model recommends:</p>')
        interval_note = (f', every {dt["interval_days"]} days'
                         if dt["interval_days"] > 0 else "")
        parts.append(
            f'<div style="background:var(--panel);border-radius:8px;padding:18px 22px;'
            f'margin-bottom:16px;border-left:4px solid {conf_color}">'
            f'<div style="font-size:1.4rem;font-weight:700;color:{conf_color}">'
            f'{esc(dt["strategy"])}</div>'
            f'<div style="color:var(--muted);margin-top:4px">{esc(dt["summary"])}'
            f'{esc(bear_note)}</div>'
            f'<div style="margin-top:8px;font-size:0.8rem;color:var(--muted)">'
            f'Confidence: <span style="color:{conf_color};font-weight:600">'
            f'{esc(dt["confidence"])}</span> &nbsp;·&nbsp; '
            f'{dt["tranches"]} tranche(s) of '
            f'{dollar(dt["tranche_amt"])} each{esc(interval_note)}'
            f'</div></div>')
        parts.append('<ul style="margin:0 0 14px 20px;color:var(--muted);font-size:0.85rem">')
        for reason in dt["rationale"]:
            parts.append(f'<li style="margin-bottom:4px">{esc(reason)}</li>')
        parts.append('</ul>')
        # Tranche schedule table
        parts.append("""<table><thead><tr>
  <th>Tranche</th><th>Target Date</th>
  <th class="num">Amount ($)</th><th>Suggested Focus</th>
</tr></thead><tbody>""")
        for row in dt["schedule"]:
            parts.append(
                f'<tr>'
                f'<td><strong>#{row["Tranche"]}</strong></td>'
                f'<td>{esc(row["Date"])}</td>'
                f'<td class="num good">{dollar(row["Amount ($)"])}</td>'
                f'<td style="color:var(--muted);font-size:0.82rem">{esc(row["Note"])}</td>'
                f'</tr>')
        parts.append("</tbody></table>")

    # ── Deploy now ────────────────────────────────────────────────────────────
    if plan["deployment_now"]:
        deployed = plan["immediate_capital"] - plan["cash_leftover_now"]
        parts.append(
            f'<h2 style="color:var(--green)">Step 1 — Deploy Now: '
            f'{dollar(plan["immediate_capital"])}</h2>'
            f'<p>{dollar(deployed)} deployed across {len(plan["deployment_now"])} positions. '
            f'{dollar(plan["cash_leftover_now"])} leftover (hold in cash / money-market).</p>')
        parts.append("""<table><thead><tr>
  <th>Ticker</th><th>Action</th>
  <th class="num">Shares</th><th class="num">Price ($)</th>
  <th class="num">Cost ($)</th><th>Verdict</th><th>Note</th>
</tr></thead><tbody>""")
        for row in plan["deployment_now"]:
            verd = row.get("Verdict", "—")
            vc   = VCOLOR.get(verd, "#555")
            vico = VICON.get(verd, "")
            parts.append(
                f'<tr>'
                f'<td><strong>{esc(row["Ticker"])}</strong></td>'
                f'<td style="color:var(--green)">{esc(row["Action"])}</td>'
                f'<td class="num">{esc(str(row["Shares"]))}</td>'
                f'<td class="num">{dollar(row["Price ($)"])}</td>'
                f'<td class="num good">{dollar(row["Cost ($)"])}</td>'
                f'<td><span style="color:{vc}">{vico} {esc(verd)}</span></td>'
                f'<td style="color:#ff9800;font-size:0.8rem">{esc(row.get("Note", ""))}</td>'
                f'</tr>')
        parts.append("</tbody></table>")
    be = plan.get("binary_exposure")
    if be:
        over = be["pct"] > be["limit"]
        parts.append(
            f'<p style="color:{"#f44336" if over else "var(--muted)"}">'
            f'Binary-event exposure after this plan: {be["pct"]:.1%} '
            f'({esc(", ".join(be["names"]))}) — {"OVER" if over else "within"} the '
            f'{be["limit"]:.0%} limit for names that hinge on a single FDA/clinical event.</p>')
    fv = plan.get("fractional_view")
    if fv and fv["deployment_now"]:
        whole_left = plan["cash_leftover_now"]
        parts.append(
            f'<h3>Alternative — with fractional shares</h3>'
            f'<p style="color:var(--muted)">Same targets, if your broker allowed fractional '
            f'shares: leftover {dollar(fv["cash_leftover_now"])} instead of '
            f'{dollar(whole_left)} — {dollar(whole_left - fv["cash_leftover_now"])} more '
            f'invested. Your current broker is whole shares only, so Step 1 above is the '
            f'one to follow.</p>')
        parts.append('<table><thead><tr><th>Ticker</th><th class="num">Shares</th>'
                     '<th class="num">Cost ($)</th></tr></thead><tbody>')
        for row in fv["deployment_now"]:
            parts.append(f'<tr><td>{esc(row["Ticker"])}</td>'
                         f'<td class="num">{row["Shares"]:g}</td>'
                         f'<td class="num">{dollar(row["Cost ($)"])}</td></tr>')
        parts.append("</tbody></table>")
    elif plan["existing_value"] == 0 and plan["immediate_capital"] == 0:
        parts.append(
            '<h2>Step 1 — Capital Pending</h2>'
            '<p>No capital available to deploy right now.</p>')

    # ── Pending capital ───────────────────────────────────────────────────────
    if plan["pending_deployment"]:
        parts.append(
            f'<h2 style="color:var(--accent)">Step 2 — Deploy in ~{plan["pending_days"]} Days: '
            f'{dollar(plan["pending_capital"])}</h2>'
            f'<p>When the {dollar(plan["pending_capital"])} arrives, make these purchases '
            f'to continue filling toward target weights.</p>')
        parts.append("""<table><thead><tr>
  <th>Ticker</th><th class="num">Shares</th>
  <th class="num">Price ($)</th><th class="num">Cost ($)</th><th>Note</th>
</tr></thead><tbody>""")
        for row in plan["pending_deployment"]:
            parts.append(
                f'<tr>'
                f'<td><strong>{esc(row["Ticker"])}</strong></td>'
                f'<td class="num">{esc(str(row["Shares"]))}</td>'
                f'<td class="num">{dollar(row["Price ($)"])}</td>'
                f'<td class="num good">{dollar(row["Cost ($)"])}</td>'
                f'<td style="color:var(--muted);font-size:0.8rem">'
                f'{esc(row["Note"])}</td></tr>')
        parts.append("</tbody></table>")

    # ── Monthly contributions ─────────────────────────────────────────────────
    if plan["contribution_schedule"]:
        parts.append(
            f'<h2>Ongoing — Monthly Contributions: '
            f'{dollar(plan["monthly_contribution"])}/month</h2>'
            f'<p>Each month goes to the one or two most underweight targets (projected at '
            f'today\'s prices — re-check when you contribute). Next 6 months:</p>')
        parts.append("""<table><thead><tr>
  <th>Month</th><th>Date</th>
  <th class="num">Amount ($)</th><th class="num">Cumulative ($)</th>
  <th>Suggestion</th>
</tr></thead><tbody>""")
        for row in plan["contribution_schedule"]:
            parts.append(
                f'<tr>'
                f'<td>Month {row["Month"]}</td>'
                f'<td>{esc(row["Date"])}</td>'
                f'<td class="num good">{dollar(row["Amount ($)"])}</td>'
                f'<td class="num">{dollar(row["Cumulative ($)"])}</td>'
                f'<td style="color:var(--muted);font-size:0.82rem">'
                f'{esc(row["Suggestion"])}</td></tr>')
        parts.append("</tbody></table>")
    else:
        parts.append(
            '<h2>Ongoing Contributions</h2>'
            '<p style="color:var(--muted)">Set <code>monthly_contribution</code> in '
            '<code>INVESTOR_PROFILES</code> to generate a contribution schedule.</p>')

    # ── Stress test ───────────────────────────────────────────────────────────
    if stress_results and rec_key in stress_results:
        parts.append("""<h2>Stress Test — Estimated Drawdowns</h2>
<p>Beta-adjusted estimated portfolio loss in historical market crashes.
These figures apply to the recommended scenario's portfolio composition.</p>
<table><thead><tr>
  <th>Scenario</th><th class="num">SPY Drawdown</th>
  <th class="num">Portfolio Beta</th>
  <th class="num">Est. Portfolio DD</th>
  <th class="num">Est. Trough Value ($)</th>
</tr></thead><tbody>""")
        for row in stress_results[rec_key]:
            parts.append(
                f'<tr>'
                f'<td><strong>{esc(row["Scenario"])}</strong></td>'
                f'<td class="num bad">{esc(row["SPY Drawdown"])}</td>'
                f'<td class="num">{esc(str(row["Portfolio Beta"]))}</td>'
                f'<td class="num bad">{esc(row["Est. Portfolio Drawdown"])}</td>'
                f'<td class="num">{dollar(row["Est. Trough Value ($)"])}</td>'
                f'</tr>')
        parts.append("</tbody></table>")

    # ── Monte Carlo outlook ───────────────────────────────────────────────────
    total_inv = plan["total_later"] if plan["pending_capital"] else plan["total_now"]
    if total_inv <= 0:
        total_inv = INITIAL_INVESTMENT
    scale = total_inv / INITIAL_INVESTMENT
    parts.append(
        f'<h2>{PROJECTION_YEARS}-Year Monte Carlo Outlook</h2>'
        f'<p>Based on {MC_SIMULATIONS:,} simulations using the recommended scenario\'s '
        f'historical CAGR ({res["cagr"]:.1%}) and volatility ({res["volatility"]:.1%}). '
        f'Scaled to your investable amount of {dollar(total_inv)}.</p>')
    parts.append("""<table><thead><tr>
  <th>Outcome</th><th class="num">Final Value ($)</th>
  <th class="num">Total Return</th><th>Interpretation</th>
</tr></thead><tbody>""")
    for label, pct_val, interp in [
        ("Pessimistic (P5)",  np.percentile(mc[:, -1],  5), "Only 5% of paths end below this"),
        ("Median (P50)",      np.median(mc[:, -1]),          "Most likely central outcome"),
        ("Optimistic (P95)",  np.percentile(mc[:, -1], 95), "Only 5% of paths exceed this"),
    ]:
        scaled = pct_val * scale
        ret    = scaled / total_inv - 1
        c      = "good" if ret > 0 else "bad"
        parts.append(
            f'<tr><td><strong>{esc(label)}</strong></td>'
            f'<td class="num {c}">{dollar(scaled)}</td>'
            f'<td class="num {c}">{ret:+.0%}</td>'
            f'<td style="color:var(--muted);font-size:0.82rem">{esc(interp)}</td></tr>')
    parts.append("</tbody></table>")

    # ── Full Portfolio Analysis (collapsible) ─────────────────────────────────
    parts.append("""
<details style="margin-top:32px">
<summary style="font-size:1.1rem;font-weight:700;color:var(--accent);cursor:pointer;
  padding:12px 0;border-top:1px solid var(--grid)">
  ▶ Full Portfolio Analysis — All Scenarios &amp; Signals
</summary>
<div style="margin-top:16px">""")

    # Scenario scoring
    if recommendation:
        scores = recommendation.get("scores", {})
        rec_k  = recommendation.get("recommended_key", rec_key)
        parts.append('<h2>Scenario Recommendation Scoring</h2>')
        parts.append('<p>Composite score = OOS Alpha 40% + Sharpe 25% + Regime Fit 20% + MaxDD 15%</p>')
        parts.append("""<table><thead><tr>
  <th>Scenario</th><th class="num">Composite</th><th class="num">OOS Alpha</th>
  <th class="num">Sharpe</th><th class="num">Max DD</th>
  <th class="num">CAGR</th><th class="num">Volatility</th><th>Recommended</th>
</tr></thead><tbody>""")
        for key, sc in sorted(scores.items(), key=lambda x: -x[1]["composite"]):
            pf  = PORTFOLIOS[key]
            res = results.get(key, {})
            star = "★ YES" if key == rec_k else "—"
            sc_color = "good" if key == rec_k else ""
            parts.append(
                f'<tr>'
                f'<td><strong>{esc(pf["label"])}</strong></td>'
                f'<td class="num {sc_color}">{sc["composite"]:.3f}</td>'
                f'<td class="num">{sc["alpha"]:+.1%}</td>'
                f'<td class="num">{sc["sharpe"]:.2f}</td>'
                f'<td class="num bad">{sc["max_dd"]:+.1%}</td>'
                f'<td class="num">{res.get("cagr",0):.1%}</td>'
                f'<td class="num">{res.get("volatility",0):.1%}</td>'
                f'<td style="color:var(--accent);font-weight:600">{esc(star)}</td></tr>')
        parts.append("</tbody></table>")

    # All-scenario Monte Carlo comparison
    parts.append('<h2>Monte Carlo — All Scenarios</h2>')
    parts.append(f'<p>{MC_SIMULATIONS:,} simulations over {PROJECTION_YEARS} years starting from ${INITIAL_INVESTMENT:,}.</p>')
    parts.append("""<table><thead><tr>
  <th>Scenario</th><th class="num">P5 (Pessimistic)</th>
  <th class="num">P50 (Median)</th><th class="num">P95 (Optimistic)</th>
  <th class="num">CAGR</th><th class="num">Sharpe</th>
</tr></thead><tbody>""")
    for key in PORTFOLIOS:
        mc  = mc_results.get(key)
        res = results.get(key, {})
        if mc is None:
            continue
        p5_v  = np.percentile(mc[:, -1],  5)
        p50_v = np.median(mc[:, -1])
        p95_v = np.percentile(mc[:, -1], 95)
        star  = " ★" if key == rec_key else ""
        parts.append(
            f'<tr>'
            f'<td><strong>{esc(PORTFOLIOS[key]["label"])}{esc(star)}</strong></td>'
            f'<td class="num neutral">${p5_v:,.0f}</td>'
            f'<td class="num good">${p50_v:,.0f}</td>'
            f'<td class="num good">${p95_v:,.0f}</td>'
            f'<td class="num">{res.get("cagr",0):.1%}</td>'
            f'<td class="num">{res.get("sharpe",0):.2f}</td></tr>')
    parts.append("</tbody></table>")

    # Walk-forward backtest
    if wf_results:
        parts.append('<h2>Walk-Forward Backtest (Out-of-Sample)</h2>')
        _pit = any(w.get("pit_fundamentals") for w in wf_results.values())
        parts.append(
            f'<p>Rolling: re-screened and re-optimised every quarter on past data only, '
            f'{BACKTEST_YEARS}-year history, {WF_COST_BPS}bp costs, point-in-time S&amp;P '
            f'membership. Fundamentals: '
            + ('as filed with the SEC at each date (no look-ahead).' if _pit else
               '<span class="bad">today\'s values for every quarter — look-ahead bias, '
               'alpha overstated.</span>')
            + ' Checks: <em>After tax</em> replays the same trades in a taxable account '
            f'(ST {TAX_RATE_ST:.1%} / LT {TAX_RATE_LT:.1%}: {esc(TAX_STATE_NAME)} '
            f'{TAX_STATE:.2%} + federal, '
            f'{TAX_FED_ORDINARY:.0%} bracket assumed); '
            '<em>Momentum-only</em> switches off quality/value; <em>1/N</em> holds the top 15 '
            'screened names equally — the optimizer should beat it.</p>')
        parts.append("""<table><thead><tr>
  <th>Scenario</th><th>Train Period</th><th>Test Period</th>
  <th class="num">OOS Return</th><th class="num">SPY Same Period</th><th class="num">Alpha</th>
  <th class="num">vs RSP</th><th class="num">After tax</th>
  <th class="num">Momentum-only</th><th class="num">1/N</th>
</tr></thead><tbody>""")
        for key, wf in wf_results.items():
            alpha     = wf.get("alpha")
            spy_ann   = wf.get("spy_annualized")
            ac        = "good" if (alpha or 0) > 0 else "bad"
            oos_c     = "good" if (wf["oos_annualized"] or 0) > 0 else "bad"
            spy_str   = "N/A" if spy_ann is None else f"{spy_ann:+.1%}/yr"
            alpha_str = "N/A" if alpha  is None else f"{alpha:+.1%}"
            star_str  = "  ★" if key == rec_key else ""
            parts.append(
                f'<tr>'
                f'<td><strong>{esc(PORTFOLIOS[key]["label"])}{esc(star_str)}</strong></td>'
                f'<td style="color:var(--muted);font-size:0.8rem">{esc(wf["train_period"])}</td>'
                f'<td style="color:var(--muted);font-size:0.8rem">{esc(wf["test_period"])}</td>'
                f'<td class="num {oos_c}">{pct_or_na(wf["oos_annualized"])}/yr</td>'
                f'<td class="num neutral">{esc(spy_str)}</td>'
                f'<td class="num {ac}">{esc(alpha_str)}</td>'
                f'<td class="num">{pct_or_na(wf.get("alpha_rsp"))}</td>'
                f'<td class="num">{pct_or_na((wf.get("after_tax") or {}).get("after_tax_annualized"))}</td>'
                f'<td class="num">{pct_or_na((wf.get("momentum_only") or {}).get("oos_annualized"))}</td>'
                f'<td class="num">{pct_or_na((wf.get("equal_weight") or {}).get("oos_annualized"))}</td>'
                f'</tr>')
        parts.append("</tbody></table>")

    # Factor exposure
    if factor_exposures:
        parts.append('<h2>Factor Exposure</h2>')
        parts.append("""<table><thead><tr>
  <th>Scenario</th><th class="num">Fwd P/E</th><th class="num">ROE</th>
  <th class="num">12m Momentum</th><th>Style</th>
</tr></thead><tbody>""")
        for key, fe in factor_exposures.items():
            parts.append(
                f'<tr>'
                f'<td><strong>{esc(PORTFOLIOS[key]["label"])}</strong></td>'
                f'<td class="num">{esc(fe.get("Weighted Avg Forward P/E","—"))}</td>'
                f'<td class="num">{esc(fe.get("Weighted Avg ROE","—"))}</td>'
                f'<td class="num">{esc(fe.get("Weighted Avg 12m Momentum","—"))}</td>'
                f'<td><span class="tag" style="background:var(--panel)">'
                f'{esc(fe.get("Style Tilt","—"))}</span></td></tr>')
        parts.append("</tbody></table>")

    # Thematic exposure
    if thematic_exposures:
        parts.append('<h2>Thematic Exposure</h2>')
        parts.append("""<table><thead><tr>
  <th>Scenario</th><th>Theme</th><th class="num">Exposure</th>
</tr></thead><tbody>""")
        for key, th_list in thematic_exposures.items():
            for theme, weight in (th_list or [])[:6]:
                parts.append(
                    f'<tr><td>{esc(PORTFOLIOS[key]["label"])}</td>'
                    f'<td>{esc(theme)}</td>'
                    f'<td class="num">{weight:.0%}</td></tr>')
        parts.append("</tbody></table>")

    # Correlation (high-corr pairs)
    if corr_data:
        all_pairs = []
        for key, cd in corr_data.items():
            for p in (cd.get("pairs") or []):
                p["Scenario"] = PORTFOLIOS[key]["label"]
                all_pairs.append(p)
        if all_pairs:
            parts.append('<h2>High-Correlation Pairs</h2>')
            parts.append('<p>Pairs with |ρ| ≥ 0.80 — these positions move together and reduce diversification.</p>')
            parts.append("""<table><thead><tr>
  <th>Scenario</th><th>Ticker A</th><th>Ticker B</th>
  <th class="num">Correlation</th><th>Risk Note</th>
</tr></thead><tbody>""")
            for p in all_pairs[:15]:
                c = abs(p["Correlation"])
                cc = "bad" if c >= 0.90 else "neutral"
                parts.append(
                    f'<tr><td style="color:var(--muted);font-size:0.8rem">'
                    f'{esc(p.get("Scenario",""))}</td>'
                    f'<td><strong>{esc(p["Ticker A"])}</strong></td>'
                    f'<td><strong>{esc(p["Ticker B"])}</strong></td>'
                    f'<td class="num {cc}">{p["Correlation"]:.3f}</td>'
                    f'<td style="color:var(--muted);font-size:0.8rem">'
                    f'{esc(p.get("Risk Note",""))}</td></tr>')
            parts.append("</tbody></table>")

    # Earnings calendar
    if earnings_flags:
        all_earn = []
        for key, ef in earnings_flags.items():
            for t, row in ef.items():
                row["Scenario"] = PORTFOLIOS[key]["label"]
                all_earn.append(row)
        if all_earn:
            parts.append('<h2>Upcoming Earnings Calendar</h2>')
            parts.append(f'<p>Positions reporting earnings within {EARNINGS_WARN_DAYS} days — expect elevated volatility.</p>')
            parts.append("""<table><thead><tr>
  <th>Ticker</th><th>Earnings Date</th><th class="num">Days Away</th>
</tr></thead><tbody>""")
            seen = set()
            for row in sorted(all_earn, key=lambda x: x.get("days_away") or 999):
                t = row.get("Ticker","")
                if t in seen:
                    continue
                seen.add(t)
                da = row.get("days_away")
                c  = "bad" if da is not None and da <= 14 else "neutral"
                parts.append(
                    f'<tr><td><strong>{esc(t)}</strong></td>'
                    f'<td>{esc(row.get("Earnings Date","?"))}</td>'
                    f'<td class="num {c}">{da if da is not None else "?"}</td></tr>')
            parts.append("</tbody></table>")

    # Insider + short signals
    if insider_signals or short_signals:
        all_t = sorted(set(list(insider_signals or {}) + list(short_signals or {})))
        if all_t:
            parts.append('<h2>Insider &amp; Short Interest Signals</h2>')
            parts.append("""<table><thead><tr>
  <th>Ticker</th><th>Insider Signal</th><th class="num">Net Buys (90d)</th>
  <th class="num">Short % Float</th><th>Short Signal</th>
</tr></thead><tbody>""")
            for t in all_t:
                ins  = (insider_signals or {}).get(t, {})
                shrt = (short_signals   or {}).get(t, {})
                ic   = "good" if ins.get("signal") == "BULLISH" else ("bad" if ins.get("signal") == "BEARISH" else "")
                sc2  = "bad"  if shrt.get("signal") == "HIGH SHORT" else ("neutral" if shrt.get("signal") == "ELEVATED" else "")
                sp_str = f"{shrt['pct_float']:.1f}%" if shrt.get("pct_float") else "—"
                sr_str = f"{shrt['days_to_cover']:.1f}d" if shrt.get("days_to_cover") else "—"
                parts.append(
                    f'<tr><td><strong>{esc(t)}</strong></td>'
                    f'<td class="{ic}">{esc(ins.get("signal","—"))}</td>'
                    f'<td class="num">{esc(str(ins.get("net_buys","—")))}</td>'
                    f'<td class="num">{esc(sp_str)} / {esc(sr_str)}</td>'
                    f'<td class="{sc2}">{esc(shrt.get("signal","—"))}</td></tr>')
            parts.append("</tbody></table>")

    # DRIP projection
    if drip_data and rec_key in drip_data:
        dd = drip_data[rec_key]
        if dd and dd.get("projection"):
            port_yield = dd.get("portfolio_yield", 0.0)
            parts.append('<h2>DRIP Projection (Recommended Scenario)</h2>')
            parts.append(
                f'<p>Dividend reinvestment over 10 years at portfolio CAGR. '
                f'Portfolio dividend yield: <strong>{port_yield:.2%}</strong>. '
                f'DRIP compounds dividends back into the portfolio each year.</p>')
            parts.append("""<table><thead><tr>
  <th>Year</th><th class="num">Without DRIP ($)</th>
  <th class="num">With DRIP ($)</th><th class="num">DRIP Bonus ($)</th>
</tr></thead><tbody>""")
            for yr, (no_d, with_d) in sorted(dd["projection"].items()):
                bonus = with_d - no_d
                parts.append(
                    f'<tr><td>Year {yr}</td>'
                    f'<td class="num">${no_d:,.0f}</td>'
                    f'<td class="num good">${with_d:,.0f}</td>'
                    f'<td class="num neutral">+${bonus:,.0f}</td></tr>')
            parts.append("</tbody></table>")

    # DCA Schedule
    if dca_schedule:
        parts.append('<h2>DCA Schedule</h2>')
        parts.append(f'<p>{DCA_TRANCHES} equal tranches of ${INITIAL_INVESTMENT/DCA_TRANCHES:,.0f} deployed every {DCA_INTERVAL} days.</p>')
        parts.append("""<table><thead><tr>
  <th>Tranche</th><th>Deploy Date</th><th>Ticker</th>
  <th class="num">Shares</th><th class="num">Price ($)</th><th class="num">Cost ($)</th>
</tr></thead><tbody>""")
        for row in dca_schedule[:20]:
            parts.append(
                f'<tr><td>#{row.get("Tranche","")}</td>'
                f'<td>{esc(row.get("Deploy Date",""))}</td>'
                f'<td><strong>{esc(row.get("Ticker",""))}</strong></td>'
                f'<td class="num">{esc(str(row.get("Shares","")))}</td>'
                f'<td class="num">${float(row.get("Price ($)",0)):,.2f}</td>'
                f'<td class="num good">${float(row.get("Cost ($)",0)):,.0f}</td></tr>')
        parts.append("</tbody></table>")

    # Options protection
    if options_protection:
        parts.append('<h2>Options Hedge Reference</h2>')
        parts.append('<p>Estimated cost of 3-month SPY put protection (beta-adjusted). For reference only.</p>')
        parts.append("""<table><thead><tr>
  <th>Strike</th><th class="num">Notional ($)</th>
  <th class="num">Est. Premium ($)</th><th class="num">Premium %</th><th>Note</th>
</tr></thead><tbody>""")
        for row in options_protection:
            parts.append(
                f'<tr><td>{esc(row.get("Strike",""))}</td>'
                f'<td class="num">${float(row.get("Notional ($)",0)):,.0f}</td>'
                f'<td class="num">${float(row.get("Est. Premium ($)",0)):,.0f}</td>'
                f'<td class="num">{esc(str(row.get("Premium %","")))}</td>'
                f'<td style="color:var(--muted);font-size:0.8rem">'
                f'{esc(row.get("Note",""))}</td></tr>')
        parts.append("</tbody></table>")

    # Tax-loss harvesting
    if tax_loss_list:
        parts.append('<h2>Tax-Loss Harvesting Candidates</h2>')
        parts.append('<p>Positions in your current portfolio that are not in the optimal target — review for potential tax-loss realisation.</p>')
        parts.append("""<table><thead><tr>
  <th>Ticker</th><th class="num">Current Value ($)</th>
  <th class="num">Target Value ($)</th><th class="num">Delta ($)</th><th>Action</th>
</tr></thead><tbody>""")
        for row in tax_loss_list:
            ac = "bad" if row.get("Action") == "SELL" else "neutral"
            parts.append(
                f'<tr><td><strong>{esc(row.get("Ticker",""))}</strong></td>'
                f'<td class="num">${float(row.get("Current ($)",0)):,.0f}</td>'
                f'<td class="num">${float(row.get("Target ($)",0)):,.0f}</td>'
                f'<td class="num bad">${float(row.get("Delta ($)",0)):+,.0f}</td>'
                f'<td class="{ac};font-weight:600">{esc(row.get("Action",""))}</td></tr>')
        parts.append("</tbody></table>")

    # Rebalance plan
    if rebalance_plan:
        parts.append('<h2>Rebalance Plan (Current vs Target)</h2>')
        parts.append("""<table><thead><tr>
  <th>Ticker</th><th class="num">Current ($)</th><th class="num">Target ($)</th>
  <th class="num">Delta ($)</th><th class="num">Shares Delta</th><th>Action</th>
</tr></thead><tbody>""")
        for row in rebalance_plan:
            ac = "good" if row.get("Action") == "BUY" else ("bad" if row.get("Action") == "SELL" else "neutral")
            parts.append(
                f'<tr><td><strong>{esc(row.get("Ticker",""))}</strong></td>'
                f'<td class="num">${float(row.get("Current ($)",0)):,.0f}</td>'
                f'<td class="num">${float(row.get("Target ($)",0)):,.0f}</td>'
                f'<td class="num {ac}">${float(row.get("Delta ($)",0)):+,.0f}</td>'
                f'<td class="num">{esc(str(row.get("Shares Delta","")))}</td>'
                f'<td class="{ac};font-weight:600">{esc(row.get("Action",""))}</td></tr>')
        parts.append("</tbody></table>")

    # Market regime
    if regime:
        icon = {"Bull": "▲", "Bear": "▼", "Neutral": "►"}.get(regime.get("regime",""), "?")
        parts.append(
            f'<p style="margin-top:16px;color:var(--muted);font-size:0.85rem">'
            f'Market Regime: <strong style="color:var(--text)">{icon} {esc(regime.get("regime","?"))}</strong>'
            f' — {esc(regime.get("signal",""))} &nbsp;·&nbsp; '
            f'SPY ${regime.get("spy_price","?")} vs 200d MA ${regime.get("ma200","?")} '
            f'({esc(regime.get("pct_vs_ma",""))})</p>')

    parts.append("</div></details>")  # close full-analysis block

    # ── Disclaimer ────────────────────────────────────────────────────────────
    parts.append(f"""
<div class="disclaimer">
  <strong>Disclaimer:</strong> This plan is generated by an algorithmic portfolio
  analyzer for educational purposes only. It does not constitute personalised
  financial advice. Past performance does not guarantee future results.
  All prices are live quotes at time of generation and will change.
  Consult a licensed financial advisor before making investment decisions.
  &nbsp;·&nbsp; Generated: {now}
  &nbsp;·&nbsp; Scenario: {esc(pf["label"])}
</div>
</body></html>""")

    return "\n".join(parts)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def _load_portfolio_state() -> dict:
    """
    Load portfolio_state.json (written by portfolio_tracker.py) if it exists.
    Returns {"current_holdings": {ticker: dollar_value}, "target_scenario": str}.
    """
    import json
    from pathlib import Path
    state_file = Path(__file__).parent / "portfolio_state.json"
    if not state_file.exists():
        return {}
    try:
        with open(state_file, encoding="utf-8") as f:
            state = json.load(f)
        trades    = state.get("trades", [])
        positions: dict = {}
        for trade in trades:
            t      = trade["ticker"]
            shares = float(trade["shares"])
            price  = float(trade["price"])
            action = trade["action"].upper()
            if t not in positions:
                positions[t] = {"shares": 0.0, "total_cost": 0.0}
            if action == "BUY":
                positions[t]["shares"]     += shares
                positions[t]["total_cost"] += shares * price
            elif action == "SELL":
                if positions[t]["shares"] > 0:
                    ratio = min(shares / positions[t]["shares"], 1.0)
                    positions[t]["total_cost"] -= positions[t]["total_cost"] * ratio
                positions[t]["shares"] = max(0.0, positions[t]["shares"] - shares)
        # Convert to dollar values using average cost (no live price needed at load time)
        current_holdings = {
            t: round(pos["total_cost"], 2)
            for t, pos in positions.items()
            if pos["shares"] > 0.001
        }
        return {
            "current_holdings":  current_holdings,
            "target_scenario":   state.get("target_scenario", ""),
            "target_weights":    state.get("target_weights", {}),
        }
    except Exception as e:
        print(f"  ⚠ Could not load portfolio_state.json: {e}")
        return {}


def _export_target_weights(scenario_key: str, holdings: dict) -> None:
    """Write the recommended scenario's weights to target_weights_export.json
    so portfolio_tracker.py can import them via set-target."""
    import json
    from pathlib import Path
    out = Path(__file__).parent / "target_weights_export.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(holdings, f, indent=2)
    print(f"  ✓ Target weights exported → {out.name}")
    print(f"    Run: python portfolio_tracker.py set-target {scenario_key} "
          f"target_weights_export.json")


def _price_prescreen(prices: pd.DataFrame, tickers: list, top_n: int = 200) -> list:
    """
    Pre-screen tickers by 12-month price momentum (no API calls).
    Reduces the universe fed to API-heavy steps 4 and 5 from 500+ to top_n,
    cutting live yfinance requests by ~60% without affecting momentum scoring
    (multi_factor_screen still runs on the full valid_tickers price matrix).
    """
    if len(tickers) <= top_n:
        return list(tickers)
    try:
        sub   = prices[tickers].ffill().dropna(how="all")
        n     = len(sub)
        ret12 = sub.pct_change(min(252, n - 1)).iloc[-1]
        ret1  = sub.pct_change(min(21,  n - 1)).iloc[-1]
        score = ret12.fillna(-99.0) - ret1.fillna(0.0)
        return score.nlargest(top_n).index.tolist()
    except Exception:
        return list(tickers[:top_n])


def main():
    print("=" * 74)
    print("ADVANCED PORTFOLIO ANALYZER  —  Institutional-Grade Edition")
    print(f"Investment: ${INITIAL_INVESTMENT:,}  |  3 Aggressive Scenarios  |  "
          f"3yr history  |  {PROJECTION_YEARS}yr projection")
    print("=" * 74)

    import os as _os_main
    _os_main.makedirs(OUTPUT_DIR, exist_ok=True)

    # Auto-load existing portfolio state from portfolio_tracker.py
    portfolio_state = _load_portfolio_state()
    active_holdings = portfolio_state.get("current_holdings", CURRENT_HOLDINGS) or CURRENT_HOLDINGS
    if active_holdings:
        print(f"\n  📂 Portfolio state loaded: {len(active_holdings)} existing position(s)  "
              f"(target: {portfolio_state.get('target_scenario','—')})")

    print(f"      {refresh_market_rates()}")

    # [1/9] Universe
    print(f"\n[1/9] Building universe (S&P 500 + NASDAQ-100 + "
          f"{len(INTERNATIONAL_TICKERS)} intl ETFs + custom tickers)...")
    raw_universe = fetch_universe()
    print(f"      {len(raw_universe)} total tickers")

    # [2/9] Prices
    print(f"\n[2/9] Downloading 3yr price history ({len(raw_universe)} tickers)...")
    prices = fetch_prices(raw_universe)
    print(f"      {len(prices)} trading days loaded")
    def _min_days(t: str) -> int:
        return EMERGING_MIN_HISTORY_DAYS if t in EMERGING_TICKERS else MIN_HISTORY_DAYS

    valid_tickers = [t for t in raw_universe
                     if t in prices.columns
                     and prices[t].notna().sum() >= _min_days(t)]
    n_emerging_pass = sum(1 for t in valid_tickers if t in EMERGING_TICKERS)
    dropped = len(raw_universe) - len(valid_tickers)
    print(f"      {len(valid_tickers)} passed history screen  |  {dropped} dropped  |  "
          f"{n_emerging_pass} emerging tickers included "
          f"(≥{EMERGING_MIN_HISTORY_DAYS}d threshold)")

    # Holdings and FDA-calendar names outside the index lists (OTC ADRs, new
    # listings) join BEFORE fundamentals so they get the same checks as everyone
    _extra = [t for t in held_tickers() + [d["ticker"] for d in UPCOMING_FDA_DECISIONS]
              if t not in valid_tickers]
    if _extra:
        added = extend_universe(_extra, prices, valid_tickers)
        print(f"      Added holdings / FDA-calendar names: {', '.join(added) or 'none (no data)'}")

    # [3/9] Regime detection
    print(f"\n[3/9] Detecting market regime (SPY 200-day MA)...")
    regime = detect_regime(prices)
    icon   = {"Bull": "▲", "Bear": "▼", "Neutral": "►"}.get(regime["regime"], "?")
    print(f"      {icon} {regime['signal']}")

    # [4/9] Fundamentals for the WHOLE universe. The old momentum gate (top 200 by
    # 12-month return) meant ~64% of stocks never had value/quality measured, so a
    # cheap, high-quality stock that hadn't moved yet could never be found.
    # .info is disk-cached for 23h: only the first run of the day pays for this.
    print(f"\n[4/9] Fetching fundamentals ({len(valid_tickers)} tickers, "
          f"factor blend: " + " / ".join(f"{k} {v:.0%}" for k, v in FACTOR_BLEND.items())
          + ")...")
    try:
        fundamentals = fetch_fundamentals(valid_tickers)
        n_pe  = sum(1 for t in valid_tickers if fundamentals.get(t, {}).get("pe"))
        n_roe = sum(1 for t in valid_tickers if fundamentals.get(t, {}).get("roe"))
        print(f"      {n_pe} with P/E  |  {n_roe} with ROE")
    except Exception as e:
        print(f"      Fundamentals failed ({e}), using momentum-only screening")
        fundamentals = {t: {} for t in valid_tickers}

    # API tier for the expensive per-ticker checks (news, FDA, SEC, valuation):
    # top names by the FULL factor composite (momentum + quality + value), not by
    # momentum alone — plus every holding and custom ticker.
    api_tickers = multi_factor_screen(prices[valid_tickers], fundamentals,
                                      top_n=API_TIER_N, force_custom=True)
    print(f"      API tier: top {API_TIER_N} of {len(valid_tickers)} by factor composite "
          f"(+ holdings/custom → {len(api_tickers)}) get news, FDA, SEC and valuation checks")

    # [5/9] Catalyst signals + multi-factor screen (per scenario)
    print(f"\n[5/9] Fetching catalyst signals (news, analyst, FDA, clinical)...")
    layer_status: dict[str, bool] = {}       # read by data_quality_checks()
    try:
        catalyst_signals = fetch_catalyst_signals(api_tickers)
        layer_status["News catalyst"] = bool(catalyst_signals)
        n_pos = sum(1 for v in catalyst_signals.values() if v.get("signal") == "POSITIVE")
        n_neg = sum(1 for v in catalyst_signals.values() if v.get("signal") == "NEGATIVE")
        print(f"      {n_pos} POSITIVE catalyst(s)  |  {n_neg} NEGATIVE catalyst(s)")
        for t, sig in sorted(catalyst_signals.items(),
                              key=lambda x: -abs(x[1].get("score", 0)))[:6]:
            arrow = "▲" if sig["signal"] == "POSITIVE" else "▼"
            print(f"      {arrow} {t:<6}  [{sig['catalyst_type']}]  "
                  f"{sig['days_ago']}d ago — {sig['headline'][:70]}")
    except Exception as e:
        print(f"      ⚠ Catalyst fetch failed ({type(e).__name__}: {e}), skipping")
        catalyst_signals = {}
        layer_status["News catalyst"] = False

    # FDA pipeline layer (ClinicalTrials.gov + FDA approvals page)
    print(f"      FDA pipeline — ClinicalTrials.gov + FDA approvals (healthcare/biotech)...")
    try:
        # Sector map for ALL api tickers — slicing [:30] silently excluded the rest
        fda_data = fetch_fda_pipeline(api_tickers, sector_map=fetch_sector_map(api_tickers))
        if fda_data:
            print(f"      FDA signals: {len(fda_data)} ticker(s)")
            for t, fd in sorted(fda_data.items(), key=lambda x: -x[1]["fda_score"])[:5]:
                print(f"      ★ {t:<6}  [{fd['fda_signal']}]  {fd['fda_summary'][:75]}")
            # Merge FDA scores into catalyst_signals (FDA takes precedence for score)
            for t, fd in fda_data.items():
                existing = catalyst_signals.get(t, {})
                if fd["fda_score"] > existing.get("score", 0):
                    catalyst_signals[t] = {
                        "signal":        "POSITIVE",
                        "score":         fd["fda_score"],
                        "catalyst_type": fd["fda_signal"].replace("_", " ").title(),
                        "headline":      fd["fda_summary"],
                        "days_ago":      0,
                        "fda_trials":    fd.get("trials", []),
                        "fda_approval":  fd.get("recent_approval"),
                    }
        else:
            print(f"      No upcoming FDA catalysts detected in universe")
    except Exception as e:
        print(f"      ⚠ FDA pipeline failed ({type(e).__name__}: {e}), skipping")
        fda_data = {}
    layer_status["FDA / ClinicalTrials"] = bool(fda_data)

    # Upcoming FDA decisions (hand-maintained PDUFA calendar)
    fda_decisions, fda_stale = upcoming_fda_signals()
    for t, sig in fda_decisions.items():
        if abs(catalyst_signals.get(t, {}).get("score", 0)) < sig["score"]:
            catalyst_signals[t] = sig
        print(f"      📅 {t:<6} {sig['headline'][:100]}")
    _new = [t for t in fda_decisions if t not in valid_tickers]
    if _new:
        added = extend_universe(_new, prices, valid_tickers)
        if added:
            print(f"      Added to universe for upcoming FDA decision: {', '.join(added)}")
    # SEC EDGAR 8-K red flags (bankruptcy, delisting notice, non-reliance, auditor
    # change, impairment…) — official filings, often ahead of the news
    try:
        import sec_edgar
        if sec_edgar.enabled():
            n_flags = 0
            for t in api_tickers:
                sig = sec_edgar.red_flag_signal(t)
                if sig and abs(sig["score"]) > abs(catalyst_signals.get(t, {}).get("score", 0)):
                    catalyst_signals[t] = sig
                    n_flags += 1
                    print(f"      ▼ {t:<6} {sig['headline']}")
            print(f"      SEC EDGAR: {n_flags} red-flag 8-K(s) in {len(api_tickers)} tickers")
            layer_status["SEC EDGAR"] = sec_edgar.cik("AAPL") is not None   # reachability probe
        else:
            print("      SEC EDGAR off — set SEC_USER_AGENT (name + email) to enable 8-K checks")
    except Exception as e:
        print(f"      ⚠ SEC EDGAR check failed ({type(e).__name__}: {e})")
        layer_status["SEC EDGAR"] = False

    # Candidates for the calendar — news mentioning a PDUFA date / FDA designation
    _listed = {d["ticker"] for d in UPCOMING_FDA_DECISIONS}
    _cands = [(t, s["headline"]) for t, s in catalyst_signals.items()
              if s.get("catalyst_type") in ("PDUFA Date", "FDA Designation") and t not in _listed]
    if _cands:
        print("      Possible FDA dates in the news — verify, then add to UPCOMING_FDA_DECISIONS:")
        for t, h in _cands[:8]:
            print(f"        ? {t:<6} {h[:95]}")

    # Biotech Pipeline Intelligence — score curated clinical-stage universe
    if RUN_BIOTECH_PIPELINE:
        print(f"      Biotech pipeline scoring (ClinicalTrials.gov + mechanism validation)...")
        try:
            from biotech_pipeline import score_pipeline, as_catalyst_signals
            pipeline_scores = score_pipeline(verbose=False)
            pipeline_signals = as_catalyst_signals(pipeline_scores)
            # Merge pipeline signals — don't overwrite stronger news-based signals
            merged_count = 0
            for t, sig in pipeline_signals.items():
                existing_score = abs(catalyst_signals.get(t, {}).get("score", 0))
                if existing_score < abs(sig.get("score", 0)):
                    catalyst_signals[t] = sig
                    merged_count += 1
            # Auto-add high-scoring pipeline candidates to valid_tickers
            pipeline_add = list(dict.fromkeys(
                s["ticker"] for s in pipeline_scores
                if s["pipeline_score"] >= PIPELINE_AUTO_ADD_SCORE
                and not s.get("fallen_angel")          # fallen angels need human review
                and s["ticker"] not in valid_tickers
            ))
            if pipeline_add:
                added = extend_universe(pipeline_add, prices, valid_tickers)
                print(f"      Pipeline: auto-added {len(added)} candidate(s): {', '.join(added)}")
            top_pipeline = sorted(pipeline_scores, key=lambda x: -x["pipeline_score"])[:5]
            print(f"      Pipeline top picks: "
                  + "  ".join(f"{s['ticker']} {s['pipeline_score']:.2f}"
                               for s in top_pipeline))
            layer_status["Biotech pipeline"] = bool(pipeline_scores)
        except Exception as _bp_exc:
            import traceback
            print(f"      ⚠ Biotech pipeline failed ({type(_bp_exc).__name__}: {_bp_exc}) — skipping")
            traceback.print_exc()
            layer_status["Biotech pipeline"] = False

    # [5.5/9] Knowledge Graph — propagate direct signals to related tickers
    print(f"\n[5.5/9] Building knowledge graph + propagating signals...")
    kg = None
    if _KG_AVAILABLE:
        try:
            kg = _KGClass()
            kg.build(valid_tickers)
            catalyst_signals = kg.propagate_signals(catalyst_signals)
            n_kg = sum(1 for v in catalyst_signals.values() if v.get("kg_source"))
            smry = kg.summary()
            print(f"      KG: {smry['nodes']} nodes  {smry['edges']} relationships  "
                  f"{n_kg} second-order signal(s) derived")
        except Exception as _kg_exc:
            print(f"      KG failed ({_kg_exc}), continuing without propagation")
            kg = None
    else:
        print(f"      KG skipped (knowledge_graph.py unavailable)")

    structural = assess_structural_risks(valid_tickers)
    for t, r in structural.items():
        esc = (f" (escalated from {r['base_severity']} — {len(r['evidence'])} recent headline(s))"
               if r["severity"] > r["base_severity"] else "")
        print(f"      ⚠ Structural risk {t}: {r['risk']} — severity {r['severity']}{esc}")

    # Analyst estimate revisions (API tier) + point-in-time snapshot for later calibration
    print(f"      Analyst estimate revisions (API tier)...")
    try:
        import analyst_signals as _as
        revision_signals = _as.fetch_revision_signals(
            [t for t in api_tickers if t not in INDEX_TICKERS | INTERNATIONAL_TICKERS],
            _safe_attr, _safe_info)   # ETFs have no estimates (Yahoo 404s)
        _last = prices.ffill().iloc[-1]
        snap = _as.save_snapshot(revision_signals,
                                 {t: float(_last[t]) for t in revision_signals
                                  if t in _last.index and pd.notna(_last[t])})
        n_up = sum(1 for s in revision_signals.values() if s["score"] > 0.25)
        n_dn = sum(1 for s in revision_signals.values() if s["score"] < -0.25)
        print(f"      {len(revision_signals)} covered  |  {n_up} rising  |  {n_dn} falling"
              + (f"  →  snapshot {snap.name}" if snap else ""))
        layer_status["Analyst revisions"] = bool(revision_signals)
    except Exception as e:
        print(f"      ⚠ Analyst revisions failed ({type(e).__name__}: {e}) — neutral for all")
        revision_signals = {}
        layer_status["Analyst revisions"] = False

    print(f"\n      Multi-factor screening (per scenario: "
          + " / ".join(f"{k} {v:.0%}" for k, v in FACTOR_BLEND.items()) + ")...")
    screened_by_scenario: dict[str, list[str]] = {}
    for key, pf in PORTFOLIOS.items():
        screened = multi_factor_screen(prices[valid_tickers], fundamentals,
                                       top_n=pf["screen_top_n"],
                                       catalyst_signals=catalyst_signals,
                                       risk_penalties=structural,
                                       revision_signals=revision_signals)
        screened_by_scenario[key] = screened
        print(f"      {pf['label']}: {len(screened)} tickers selected "
              f"(top {pf['screen_top_n']})")

    all_screened = sorted(set(t for s in screened_by_scenario.values() for t in s))

    # [6/9] Sector map + market caps + Black-Litterman mu + Valuation tilt
    print(f"\n[6/9] Sector map, market caps, Black-Litterman + valuation ({len(all_screened)} tickers)...")
    sector_map  = fetch_sector_map(all_screened)
    market_caps = fetch_market_caps(all_screened)
    # _aligned_window: a bare .dropna() let one short-history ticker shrink the
    # BL covariance window for every stock (same bug fixed in optimize_weights)
    _bl_tickers, _bl_px = _aligned_window(all_screened, prices)
    sub_daily   = _bl_px.pct_change().dropna()

    # Binary-event concentration cap for biotechs (info is disk-cached)
    biotech_caps = {t: BIOTECH_MAX_WEIGHT for t in set(all_screened) | set(api_tickers)
                    if (_safe_info(t) or {}).get("industry") == "Biotechnology"}
    if biotech_caps:
        print(f"      Biotech cap {BIOTECH_MAX_WEIGHT:.0%}/name on {len(biotech_caps)} ticker(s)")
    # Industry cap + aggregate binary-event cap (live: the FDA calendar counts)
    _infos = {t: _safe_info(t) or {} for t in set(all_screened) | set(api_tickers)}
    group_caps = build_group_caps(_infos, fda_tickers={d["ticker"] for d in UPCOMING_FDA_DECISIONS})
    binary_set = binary_event_tickers(_infos, {d["ticker"] for d in UPCOMING_FDA_DECISIONS})
    print(f"      Group caps: industry ≤{INDUSTRY_MAX_WEIGHT:.0%} "
          f"({sum(1 for _, c in group_caps if c == INDUSTRY_MAX_WEIGHT)} industries)  ·  "
          f"binary-event names ≤{BINARY_EVENT_MAX:.0%} together ({len(binary_set)} names)")
    bl_mu       = black_litterman_mu(all_screened, sub_daily, market_caps)
    print(f"      BL blend: {1 - BL_ALPHA:.0%} equilibrium + {BL_ALPHA:.0%} sample returns")

    if VALUATION_TILT:
        print(f"      Fetching DCF/multiples data for valuation tilt...")
        val_data    = fetch_valuation_data(all_screened)
        val_signals = {t: valuation_signal(t, val_data.get(t, {})) for t in all_screened}
        val_signals = rerank_valuation_verdicts(val_signals)   # relative, not absolute
        n_under = sum(1 for v in val_signals.values() if v["verdict"] == "UNDERVALUED")
        n_over  = sum(1 for v in val_signals.values() if v["verdict"] == "OVERVALUED")
        n_fair  = sum(1 for v in val_signals.values() if v["verdict"] == "FAIR VALUE")
        print(f"      Verdicts: {n_under} UNDERVALUED  {n_fair} FAIR  {n_over} OVERVALUED")
        bl_mu = apply_valuation_tilt(bl_mu, val_signals)
        print(f"      Applied ±{VALUATION_TILT_MAG:.0%} mu tilt to {n_under + n_over} tickers")
    else:
        val_signals = {}

    # Structural risk haircut on BL expected returns
    for t, r in structural.items():
        if t in bl_mu:
            bl_mu[t] -= STRUCTURAL_MU_HAIRCUT * r["severity"]

    if kg is not None:
        try:
            _ind_map = ({t: val_data.get(t, {}).get("industry") for t in all_screened}
                        if VALUATION_TILT else None)
            kg.enhance_with_sectors(sector_map, _ind_map)
        except Exception as _kg_exc:
            print(f"      KG sector enhancement failed ({_kg_exc})")

    # [7/9] Current prices + dividend yields
    print(f"\n[7/9] Fetching current prices and dividend yields...")
    current_prices = fetch_current_prices(all_screened)
    div_yields     = fetch_dividend_yields(all_screened)
    n_div = sum(1 for y in div_yields.values() if y > 0)
    print(f"      {n_div}/{len(all_screened)} tickers pay a dividend")

    # [8/9] Optimize, metrics, MC, stress, walk-forward, backtests, factor exposure
    print(f"\n[8/9] Optimizing (time-decay cov + BL + sector cap ≤{SECTOR_CAP:.0%})...")
    for key, pf in PORTFOLIOS.items():
        mu_ovr = bl_mu if pf["use_bl"] else None
        print(f"      → {pf['label']}  "
              f"[{pf['objective']}  max:{pf['max_weight']:.0%}  "
              f"intl_floor:{pf['int_floor']:.0%}"
              f"{'  BL' if pf['use_bl'] else ''}]")
        pf["holdings"] = optimize_weights(
            screened_by_scenario[key], prices, pf["objective"],
            pf["max_weight"], pf["min_weight"],
            sector_map=sector_map,
            mu_override=mu_ovr,
            int_floor=pf["int_floor"],
            ticker_caps=biotech_caps,
            group_caps=group_caps,
        )
        if len(pf["holdings"]) > MAX_HOLDINGS:
            pf["holdings"] = trim_to_max_holdings(pf["holdings"], MAX_HOLDINGS,
                                                  pf["max_weight"], biotech_caps)
        sorted_w = sorted(pf["holdings"].items(), key=lambda x: -x[1])
        print("         " + "  ".join(f"{t} {w:.1%}" for t, w in sorted_w[:8])
              + ("  ..." if len(sorted_w) > 8 else ""))

    print("\n      Computing historical metrics + CVaR + Sortino...")
    results: dict = {}
    for key, pf in PORTFOLIOS.items():
        results[key] = compute_metrics(prices, pf["holdings"], key)
        r = results[key]
        print(f"      {pf['label']}: CAGR {r['cagr']:.1%}  "
              f"Sharpe {r['sharpe']:.2f}  Sortino {r['sortino']:.2f}  "
              f"MaxDD {r['max_dd']:.1%}  CVaR {fmt_dollars(r['cvar_95'])}")

    print("\n      Monte Carlo projections...")
    mc_results: dict = {}
    for key, res in results.items():
        exp_ret, beta = forward_expected_return(res["port_ret"], prices)
        res["expected_return"], res["beta"] = exp_ret, beta
        mc_results[key] = monte_carlo(exp_ret, res["volatility"])
        mc = mc_results[key]
        print(f"      {PORTFOLIOS[key]['label']}: E[R] {exp_ret:.1%} (β {beta:.2f})  "
              f"P5 {fmt_dollars(np.percentile(mc[:,-1], 5))}  "
              f"P50 {fmt_dollars(np.median(mc[:,-1]))}  "
              f"P95 {fmt_dollars(np.percentile(mc[:,-1], 95))}")

    print("\n      Stress tests (beta-adjusted, 4 historical scenarios)...")
    stress_results: dict = {}
    for key, res in results.items():
        stress_results[key] = stress_test(res["port_ret"], prices)

    print(f"\n      Rolling walk-forward ({BACKTEST_YEARS}y history, quarterly re-screen + "
          f"re-optimise on past data only, costs included, vs SPY / RSP)...")
    bt = prepare_backtest(valid_tickers)
    wf_results = run_backtests(bt, fundamentals)

    print("\n      Lookback backtests (3m / 6m / 12m)...")
    BACKTEST_PERIODS = [3, 6, 12]
    backtest_results: dict = {}
    for key, pf in PORTFOLIOS.items():
        backtest_results[key] = {
            mo: backtest_portfolio(pf["holdings"], prices, mo)
            for mo in BACKTEST_PERIODS
        }

    print("\n      Factor exposure report...")
    sub_252    = prices[all_screened].ffill().iloc[-252:].pct_change().dropna()
    mom_scores = (sub_252.mean() * 252 / (sub_252.std() * np.sqrt(252)).replace(0, np.nan)).to_dict()
    factor_exposures = {
        key: factor_exposure(pf["holdings"], fundamentals, mom_scores)
        for key, pf in PORTFOLIOS.items()
    }

    print("\n      Rebalancing drift analysis (12m lookback)...")
    rebalance: dict = {}
    for key, pf in PORTFOLIOS.items():
        triggers = compute_rebalance_triggers(pf["holdings"], prices, lookback_months=12)
        rebalance[key] = triggers
        n_tri = len(triggers)
        if n_tri:
            print(f"      {pf['label']}: {n_tri} position(s) would need rebalancing  "
                  + "  ".join(f"{t['Ticker']} {t['Drift']} ({t['Action']})"
                               for t in triggers[:3]))
        else:
            print(f"      {pf['label']}: no rebalancing needed (all within ±4%)")

    # ── ENHANCED ANALYSIS ─────────────────────────────────────────────────────
    print("\n      Enhanced analysis — risk, themes, signals, DRIP, DCA...")

    # Recommended scenario key for analyses that need a single portfolio
    wf_tmp = {k: wf_results[k] for k in PORTFOLIOS}
    rec_tmp = recommend_scenario(results, wf_tmp, regime)
    rec_key_tmp = rec_tmp["recommended_key"]
    rec_holdings = PORTFOLIOS[rec_key_tmp]["holdings"]

    # Per-scenario: risk contribution, thematic, DRIP
    risk_contribs:      dict = {}
    thematic_exposures: dict = {}
    drip_data:          dict = {}
    for key, pf in PORTFOLIOS.items():
        risk_contribs[key]      = risk_contribution(pf["holdings"], prices)
        thematic_exposures[key] = thematic_exposure(pf["holdings"])
        drip_data[key]          = drip_projection(
            pf["holdings"], div_yields, results[key]["cagr"], years=10)

    # Correlation analysis (all scenarios)
    corr_data: dict = {}
    for key, pf in PORTFOLIOS.items():
        corr_mat, pairs = correlation_analysis(pf["holdings"], prices)
        corr_data[key] = {"matrix": corr_mat, "pairs": pairs}
        if pairs:
            print(f"      {pf['label']}: {len(pairs)} high-corr pair(s)  "
                  + "  ".join(f"{p['Ticker A']}/{p['Ticker B']} {p['Correlation']:.2f}"
                               for p in pairs[:3]))

    # Earnings calendar (recommended portfolio tickers)
    print("      Fetching earnings calendar...")
    all_holdings_tickers = list({
        t for pf in PORTFOLIOS.values() for t in pf["holdings"]
    })
    earnings_raw = fetch_earnings_calendar(all_holdings_tickers)
    earnings_flags: dict = {}
    for key, pf in PORTFOLIOS.items():
        earnings_flags[key] = {
            t: earnings_raw[t] for t in pf["holdings"] if t in earnings_raw
        }
    n_warn = sum(len(v) for v in earnings_flags.values())
    if n_warn:
        print(f"      ⚠ {len(earnings_raw)} ticker(s) report earnings within "
              f"{EARNINGS_WARN_DAYS} days")

    # Insider signals
    print("      Fetching insider transaction signals...")
    insider_raw = fetch_insider_signals(all_holdings_tickers)
    insider_signals_data: dict = {
        t: v for t, v in insider_raw.items() if v.get("signal") != "NEUTRAL"
    }
    n_bull = sum(1 for v in insider_signals_data.values() if v.get("signal") == "BULLISH")
    n_bear = sum(1 for v in insider_signals_data.values() if v.get("signal") == "BEARISH")
    print(f"      Insider: {n_bull} BULLISH  {n_bear} BEARISH  "
          f"(90-day net buy/sell counts)")

    # Short interest (from val_data already fetched)
    if VALUATION_TILT and val_signals:
        short_signals_data = extract_short_signals(all_holdings_tickers, val_data)
        n_hi = sum(1 for v in short_signals_data.values() if v["signal"] == "HIGH SHORT")
        n_el = sum(1 for v in short_signals_data.values() if v["signal"] == "ELEVATED")
        if n_hi + n_el:
            print(f"      Short interest: {n_hi} HIGH SHORT  {n_el} ELEVATED")
    else:
        short_signals_data = {}

    # Rebalance-to-target plan (uses live portfolio state if tracker is active)
    rebalance_plan_data = compute_rebalance_plan(
        rec_holdings, active_holdings, current_prices)
    if rebalance_plan_data:
        actions = [r for r in rebalance_plan_data if r["Action"] != "HOLD"]
        print(f"      Rebalance plan: {len(actions)} trade(s) to move to target")

    # DCA schedule (recommended portfolio)
    dca_sched = build_dca_schedule(rec_holdings, current_prices,
                                   DCA_TRANCHES, DCA_INTERVAL)
    if dca_sched:
        print(f"      DCA: {DCA_TRANCHES} tranches × "
              f"${INITIAL_INVESTMENT / DCA_TRANCHES:,.0f}, "
              f"{DCA_INTERVAL} days apart")

    # Options protection estimate (recommended portfolio)
    rec_beta = 1.0
    if rec_holdings and results.get(rec_key_tmp):
        port_ret = results[rec_key_tmp]["port_ret"]
        if "SPY" in prices.columns:
            spy_ret = prices["SPY"].pct_change().dropna()
            aligned = pd.concat([port_ret, spy_ret.rename("spy")], axis=1).dropna()
            if len(aligned) > 50:
                cm = aligned.cov()
                rec_beta = float(cm.iloc[0, 1] / cm.iloc[1, 1])
    opt_protect = options_protection_estimate(INITIAL_INVESTMENT, rec_beta)
    print(f"      Options hedge: portfolio beta {rec_beta:.2f}  "
          f"3mo put est. ${opt_protect[0]['Est. Premium ($)']:,.0f}")

    # Tax-loss harvesting candidates
    tax_loss_data = tax_loss_candidates(active_holdings, current_prices, rec_holdings)
    if tax_loss_data:
        print(f"      Tax-loss candidates: {len(tax_loss_data)} position(s) not in target")

    # Personalised investor plans (all profiles in INVESTOR_PROFILES config)
    print("\n      Building personalised investor plans...")
    personalized_plans = []
    all_profile_tickers = {
        t for profile in INVESTOR_PROFILES.values()
        for t in list(profile.get("existing_shares", {}))
                 + list(profile.get("forced_min_weight", {}))
    }
    if all_profile_tickers - set(current_prices):
        extra = all_profile_tickers - set(current_prices)
        extra_prices = fetch_current_prices(list(extra))
        current_prices = {**current_prices, **extra_prices}

    # Holdings with <1 year of prices can't be optimised — shown as "too new to model"
    too_new = {t for t in held_tickers()
               if t not in prices.columns or prices[t].notna().sum() < 252}
    if too_new:
        print(f"      Too new to model (<1yr history, held): {', '.join(sorted(too_new))}")

    for pid, profile in INVESTOR_PROFILES.items():
        _held = list(profile.get("existing_shares", {}))
        _sells = exit_signals(_held, catalyst_signals, revision_signals, structural)
        for t, why in profile.get("sell_intent", {}).items():      # investor's own call
            _sells[t] = f"your stated intent — {why}" + (f"; {_sells[t]}" if t in _sells else "")
        _plan_kw = dict(tax_lots=load_tax_lots(pid), too_new=too_new, sell_signals=_sells,
                        position_cap=PORTFOLIOS[rec_key_tmp]["max_weight"],
                        binary_set=binary_set | binary_event_tickers(
                            {t: _safe_info(t) or {} for t in _held},
                            {d["ticker"] for d in UPCOMING_FDA_DECISIONS}),
                        earnings_days={t: e["days_away"] for t, e in (earnings_raw or {}).items()
                                       if e and e.get("days_away") is not None})
        for t, why in _sells.items():
            print(f"      [{profile['name']}] exit signal {t}: {why}")
        # Per-investor Black-Litterman: if they have explicit views, re-optimize
        # on the recommended scenario universe using their view-adjusted mu.
        investor_views = profile.get("bl_views", [])
        if investor_views:
            personal_mu = apply_bl_views(bl_mu, investor_views, sub_daily)
            rec_pf = PORTFOLIOS[rec_key_tmp]
            print(f"      [{profile['name']}] Applying {len(investor_views)} BL view(s) → "
                  f"re-optimizing on {rec_pf['label']} universe...")
            # A BL view is a deliberate conviction — it may exceed the 6% biotech cap,
            # but not the investor's Conviction bucket (run10 put RVMD at the 20%
            # scenario cap against a 15% Conviction target)
            view_tickers  = {v["ticker"] for v in investor_views}
            conv_cap      = profile.get("portfolio_split", {}).get("Conviction", rec_pf["max_weight"])
            personal_caps = {t: c for t, c in biotech_caps.items() if t not in view_tickers}
            personal_caps.update({t: min(conv_cap, rec_pf["max_weight"]) for t in view_tickers})
            personal_weights = optimize_weights(
                screened_by_scenario[rec_key_tmp], prices,
                rec_pf["objective"], rec_pf["max_weight"], rec_pf["min_weight"],
                sector_map=sector_map,
                mu_override=personal_mu,
                int_floor=rec_pf["int_floor"],
                ticker_caps=personal_caps,
                group_caps=group_caps,
            )
            if len(personal_weights) > MAX_HOLDINGS:
                personal_weights = trim_to_max_holdings(personal_weights, MAX_HOLDINGS,
                                                        rec_pf["max_weight"], personal_caps)
            plan_weights = personal_weights
        else:
            plan_weights = rec_holdings
        plan = build_personalized_plan(
            profile, plan_weights, current_prices, div_yields, val_signals, **_plan_kw)
        if profile.get("show_fractional_view") and not profile.get("fractional_shares"):
            # Same targets, fractional shares — shows what whole-share rounding costs
            alt = build_personalized_plan({**profile, "fractional_shares": True}, plan_weights,
                                          current_prices, div_yields, val_signals, **_plan_kw)
            plan["fractional_view"] = {"deployment_now": alt["deployment_now"],
                                       "cash_leftover_now": alt["cash_leftover_now"]}

        # Compute deployment timing recommendation for any fresh capital
        deploy_capital = float(profile.get("capital", 0)) + float(
            profile.get("pending_capital", {}).get("amount", 0))
        if deploy_capital > 0:
            plan["deployment_timing"] = deployment_timing_recommendation(
                deploy_capital, regime, val_signals)
        else:
            plan["deployment_timing"] = None
        fv = plan.get("fractional_view")
        if fv:
            print(f"      [{profile['name']}] whole shares leave "
                  f"${plan['cash_leftover_now']:,.0f} uninvested; fractional view "
                  f"${fv['cash_leftover_now']:,.0f}")
        be = plan.get("binary_exposure")
        if be:
            flag = "⚠ OVER" if be["pct"] > be["limit"] else "within"
            print(f"      [{profile['name']}] binary-event exposure {be['pct']:.1%} "
                  f"({', '.join(be['names'])}) — {flag} the {be['limit']:.0%} limit")
        personalized_plans.append(plan)
        ex    = plan["existing_value"]
        now   = plan["immediate_capital"]
        pend  = plan["pending_capital"]
        rsrv  = plan.get("reserved_total", 0)
        views_tag = f"  [BL views: {len(investor_views)}]" if investor_views else ""
        rsrv_tag  = f"  [reserved: ${rsrv:,.0f}]" if rsrv else ""
        dtiming   = plan.get("deployment_timing")
        timing_str = f"  [{dtiming['strategy']}]" if dtiming else ""
        print(f"      {plan['name']}: existing ${ex:,.0f}  "
              f"+ now ${now:,.0f}  + pending ${pend:,.0f}"
              f"{views_tag}{rsrv_tag}{timing_str}")

    # Quality gate — stop before writing any output a person might act on
    report_data_quality(data_quality_checks(
        prices, results, wf_results, div_yields, fundamentals, val_signals, layer_status,
        extra_warnings=fda_stale))

    # [9/9] Outputs
    print(f"\n[9/9] Generating outputs...")
    plot_dashboard(results, mc_results, backtest_results, stress_results, wf_results, regime)
    try:
        export_excel(results, mc_results, current_prices, div_yields,
                     backtest_results, wf_results, stress_results, regime, factor_exposures,
                     val_signals,
                     risk_contribs=risk_contribs,
                     thematic_exposures=thematic_exposures,
                     insider_signals=insider_signals_data,
                     short_signals=short_signals_data,
                     drip_data=drip_data,
                     rebalance_plan=rebalance_plan_data,
                     dca_schedule=dca_sched,
                     options_protection=opt_protect,
                     tax_loss_list=tax_loss_data)
    except Exception as _exc:
        import traceback
        print(f"\n  ⚠ Excel export failed (continuing): {_exc}")
        traceback.print_exc()

    # ── RESULTS SUMMARY ───────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("RESULTS SUMMARY")
    print("=" * 74)
    for key, res in results.items():
        mc  = mc_results[key]
        fe  = factor_exposures[key]
        wf  = wf_results[key]
        pf  = PORTFOLIOS[key]
        print(f"\n{pf['label']}")
        print(f"  Historical:   CAGR {res['cagr']:.1%}  "
              f"Sharpe {res['sharpe']:.2f}  Sortino {res['sortino']:.2f}  "
              f"MaxDD {res['max_dd']:.1%}  CVaR {fmt_dollars(res['cvar_95'])}")
        print(f"  {PROJECTION_YEARS}yr MC:       "
              f"P5 {fmt_dollars(np.percentile(mc[:,-1], 5))}  "
              f"P50 {fmt_dollars(np.median(mc[:,-1]))}  "
              f"P95 {fmt_dollars(np.percentile(mc[:,-1], 95))}")
        spy_s   = f"  vs SPY {wf['spy_annualized']:+.1%}/yr" if wf["spy_annualized"] is not None else ""
        alpha_s = f"  alpha {wf['alpha']:+.1%}" if wf["alpha"] is not None else ""
        print(f"  OOS Test:     {pct_or_na(wf['oos_annualized'])}/yr{spy_s}{alpha_s}  "
              f"(test: {wf['test_period']})")
        print(f"  Factor Tilt:  P/E {fe['Weighted Avg Forward P/E']}  "
              f"ROE {fe['Weighted Avg ROE']}  "
              f"Momentum {fe['Weighted Avg 12m Momentum']}  "
              f"Style: {fe['Style Tilt']}")

    # ── STRESS TEST SUMMARY ───────────────────────────────────────────────────
    if stress_results:
        print("\n" + "=" * 74)
        print("STRESS TEST — Beta-Adjusted Drawdown Estimates")
        print("=" * 74)
        sample_rows = next(iter(stress_results.values()))
        col_w = 22
        hdr   = f"  {'Scenario':<30}" + "".join(
            f"  {PORTFOLIOS[k]['label'].split(' (')[0]:<{col_w}}" for k in PORTFOLIOS)
        print(hdr)
        print("  " + "─" * (30 + (col_w + 2) * len(PORTFOLIOS)))
        for i, _ in enumerate(sample_rows):
            name = stress_results[list(stress_results.keys())[0]][i]["Scenario"]
            line = f"  {name:<30}"
            for key in PORTFOLIOS:
                st   = stress_results[key][i]
                cell = f"${st['Est. Trough Value ($)']:,.0f} ({st['Est. Portfolio Drawdown']})"
                line += f"  {cell:<{col_w}}"
            print(line)

    # ── LOOKBACK BACKTEST SUMMARY ─────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("LOOKBACK BACKTEST — Hypothetical $15,000 Using Today's Weights")
    print("=" * 74)
    col_w = 28
    print(f"  {'Period':<14}" + "".join(
        f"  {PORTFOLIOS[k]['label'].split(' (')[0]:<{col_w}}" for k in PORTFOLIOS))
    print("  " + "─" * (14 + (col_w + 2) * len(PORTFOLIOS)))
    for mo in BACKTEST_PERIODS:
        lbl    = {3: "3 months ago", 6: "6 months ago", 12: "12 months ago"}[mo]
        line_v = f"  {lbl:<14}"
        line_a = f"  {'':14}"
        for key in PORTFOLIOS:
            bt = backtest_results[key].get(mo)
            if bt:
                sign = "+" if bt["gain"] >= 0 else ""
                vcell = f"${bt['end_value']:,.0f} ({sign}{bt['total_return']:.1%})"
                acell = f"{sign}{bt['annualized']:.1%}/yr ann."
            else:
                vcell, acell = "N/A", ""
            line_v += f"  {vcell:<{col_w}}"
            line_a += f"  {acell:<{col_w}}"
        print(line_v)
        print(line_a)
    print("  " + "─" * (14 + (col_w + 2) * len(PORTFOLIOS)))
    print("  Note: look-ahead bias present — current weights applied to past prices.")

    # ── BUY RECOMMENDATIONS ───────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print(f"BUY RECOMMENDATIONS  —  ${INITIAL_INVESTMENT:,} to deploy today")
    print("=" * 96)
    for key, pf in PORTFOLIOS.items():
        plan, intl_cost, dom_cost = build_allocation_plan(
            pf["holdings"], current_prices, div_yields, val_signals)
        totals = plan[-1]
        print(f"\n{'─'*96}")
        print(f"  {pf['label']}")
        print(f"  International ETFs: ${intl_cost:,.0f}  |  Domestic: ${dom_cost:,.0f}")
        fe = factor_exposures[key]
        print(f"  Style: {fe['Style Tilt']}  |  "
              f"P/E {fe['Weighted Avg Forward P/E']}  |  "
              f"ROE {fe['Weighted Avg ROE']}  |  "
              f"Momentum {fe['Weighted Avg 12m Momentum']}")
        print(f"{'─'*96}")
        print(f"  {'Type':<6} {'Ticker':<7} {'Wt':>5}  {'Target$':>8}  "
              f"{'Price':>8}  {'Shares':>6}  {'Cost$':>9}  {'DivYld':>7}  {'AnnInc':>9}  "
              f"{'Verdict':<12}  {'DCF Buy Range':>14}")
        print(f"  {'─'*5} {'─'*6} {'─'*5}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*9}  "
              f"{'─'*7}  {'─'*9}  {'─'*12}  {'─'*14}")
        for row in plan[:-1]:
            px  = (f"${row['Current Price ($)']:>7.2f}"
                   if isinstance(row["Current Price ($)"], float) else "    N/A ")
            inc = (f"${row['Est. Annual Income']:>7.2f}"
                   if row["Est. Annual Income"] else "        —")
            tag = (f"[{row['Type']:<4}]" if row["Type"] in ("INTL", "INDEX")
                   else f" {row['Type']:<5}")
            verd = row.get("Verdict", "—")
            dcfr = row.get("DCF Range ($)", "—")
            vicon = ("★" if verd == "UNDERVALUED" else
                     "⚠" if verd == "OVERVALUED"  else
                     "◆" if verd == "FAIR VALUE"   else " ")
            vdisp = f"{vicon} {verd}"[:12] if verd not in ("—", "") else "—"
            print(f"  {tag} {row['Ticker']:<7} {row['Weight']:>5}  "
                  f"${row['Target ($)']:>7,.0f}  {px}  "
                  f"{row['Shares to Buy']:>6}  ${row['Actual Cost ($)']:>8,.2f}  "
                  f"{row['Div Yield (TTM)']:>7}  {inc}  {vdisp:<12}  {dcfr:>14}")
        print(f"  {'─'*5} {'─'*6} {'─'*5}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*9}  "
              f"{'─'*7}  {'─'*9}  {'─'*12}  {'─'*14}")
        print(f"  {'TOTAL':<13} {'100%':>5}  ${totals['Target ($)']:>7,.0f}  "
              f"{'':>8}  {'':>6}  ${totals['Actual Cost ($)']:>8,.2f}  "
              f"{'':>7}  ${totals['Est. Annual Income']:>7.2f}")
        print(f"  Leftover: ${totals['Difference ($)']:,.2f}  |  "
              f"Est. annual dividends: ${totals['Est. Annual Income']:,.2f}")
        if VALUATION_TILT and val_signals:
            print(f"  ★UNDERVALUED (cheapest third vs universe)  "
                  f"◆FAIR VALUE  ⚠OVERVALUED (dearest third)  — = ETF/no data")

        if rebalance.get(key):
            print(f"  Rebalance triggers (if deployed 12m ago): "
                  + "  ".join(f"{t['Ticker']} {t['Drift']} ({t['Action']})"
                               for t in rebalance[key]))

    # ── SCENARIO RECOMMENDATION ───────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("SCENARIO RECOMMENDATION")
    print("=" * 96)
    recommendation = recommend_scenario(results, wf_results, regime)
    rec_key   = recommendation["recommended_key"]
    rec_label = recommendation["recommended_label"]
    _export_target_weights(rec_key, PORTFOLIOS[rec_key]["holdings"])

    # Decision log — record what was recommended today so it can be scored later
    try:
        from decision_log import record_run
        spy_px = (float(prices["SPY"].dropna().iloc[-1]) if "SPY" in prices.columns else None)
        entry_px = {t: (current_prices.get(t) or
                        (float(prices[t].dropna().iloc[-1]) if t in prices.columns else None))
                    for pf in PORTFOLIOS.values() for t in pf["holdings"]}
        record_run(ts, rec_key, {k: pf["holdings"] for k, pf in PORTFOLIOS.items()},
                   entry_px, spy_px,
                   bl_views=[v for p in INVESTOR_PROFILES.values()
                             for v in p.get("bl_views", [])],
                   regime=regime.get("regime"))
        print("  ✓ Recommendation logged → outputs/decision_log.jsonl "
              "(score later: python decision_log.py)")
    except Exception as _dl_exc:
        print(f"  ⚠ Decision log write failed ({type(_dl_exc).__name__}: {_dl_exc})")
    print(f"\n  RECOMMENDED: {rec_label}")
    print(f"  Regime: {recommendation['regime']}  |  Scored on alpha(40%) + "
          f"Sharpe(25%) + regime(20%) + drawdown(15%)\n")
    for r in recommendation["reasons"]:
        clean = r.replace("<strong>", "").replace("</strong>", "").replace("<em>", "").replace("</em>", "")
        print(f"  ✓ {clean}")
    if recommendation["caveats"]:
        print()
        for c in recommendation["caveats"]:
            clean = c.replace("<strong>", "").replace("</strong>", "")
            print(f"  ℹ  {clean}")
    print()
    sc_table_rows = [(k, recommendation["scores"][k]) for k in PORTFOLIOS]
    print(f"  {'Scenario':<30}  {'Score':>6}  {'Alpha':>8}  {'OOS Shp':>7}  "
          f"{'RegimeFit':>10}  {'OOS DD':>7}")
    print("  " + "─" * 70)
    for key, sc in sc_table_rows:
        marker = " ★" if key == rec_key else "  "
        print(f"  {PORTFOLIOS[key]['label']:<30}{marker}"
              f"  {sc['composite']:.3f}  "
              f"{sc['alpha']:>+7.1%}  "
              f"{sc['sharpe']:>7.2f}  "
              f"{sc['regime_fit']:>10.2f}  "
              f"{sc['max_dd']:>+6.1%}")

    # ── HTML REPORT ───────────────────────────────────────────────────────────
    print(f"\n      Generating HTML stakeholder report...")
    try:
        html = generate_html_report(
            results, mc_results, current_prices, div_yields,
            backtest_results, wf_results, stress_results, regime,
            factor_exposures, val_signals, recommendation,
            risk_contribs=risk_contribs,
            thematic_exposures=thematic_exposures,
            corr_data=corr_data,
            earnings_flags=earnings_flags,
            insider_signals=insider_signals_data,
            short_signals=short_signals_data,
            drip_data=drip_data,
            rebalance_plan=rebalance_plan_data,
            dca_schedule=dca_sched,
            options_protection=opt_protect,
            tax_loss_list=tax_loss_data,
            personalized_plans=personalized_plans,
        )
        with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  ✓ HTML report saved: {OUTPUT_HTML}")
    except Exception as _exc:
        import traceback
        print(f"\n  ⚠ Main HTML report failed: {_exc}")
        traceback.print_exc()

    # ── Per-investor HTML plans ───────────────────────────────────────────────
    investor_html_files = []
    for pid, plan in zip(INVESTOR_PROFILES.keys(), personalized_plans):
        safe_name     = pid.lower().replace(" ", "_")
        inv_html_path = os.path.join(OUTPUT_DIR, f"plan_{safe_name}_{ts}.html")
        try:
            inv_html = generate_investor_html_report(
                plan, rec_key, results, mc_results, stress_results, wf_results,
                catalyst_signals=catalyst_signals,
                recommendation=recommendation,
                backtest_results=backtest_results,
                factor_exposures=factor_exposures,
                val_signals=val_signals,
                risk_contribs=risk_contribs,
                thematic_exposures=thematic_exposures,
                corr_data=corr_data,
                earnings_flags=earnings_flags,
                insider_signals=insider_signals_data,
                short_signals=short_signals_data,
                drip_data=drip_data,
                rebalance_plan=rebalance_plan_data,
                dca_schedule=dca_sched,
                options_protection=opt_protect,
                tax_loss_list=tax_loss_data,
                regime=regime,
                current_prices=current_prices,
                knowledge_graph=kg,
                fundamentals=fundamentals,
                prices=prices,
                structural_risks=structural,
            )
            with open(inv_html_path, "w", encoding="utf-8") as f:
                f.write(inv_html)
            investor_html_files.append(inv_html_path)
            print(f"  ✓ Investor plan saved: {inv_html_path}  ({plan['name']})")
        except Exception as _exc:
            import traceback
            print(f"\n  ⚠ Investor HTML failed for {pid}: {_exc}")
            traceback.print_exc()

    print(f"\nFiles: {OUTPUT_CHART}  |  {OUTPUT_EXCEL}  |  {OUTPUT_HTML}")
    print("Investor plans: " + "  |  ".join(investor_html_files))
    print(f"\nMarket Regime: {regime['regime']} — {regime['signal']}")
    print("\n⚠  Disclaimer: Past performance does not guarantee future results.")
    print("   This tool is for educational purposes only, not financial advice.")
    print("   Walk-forward test uses no-views BL; lookback test has look-ahead bias.")


if __name__ == "__main__":
    main()
