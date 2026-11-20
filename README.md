# Equity Research Agents

A Python equity-research platform for an individual investor: multi-factor screening,
Black-Litterman portfolio optimization, SEC-based DCF valuation, FDA / SEC / news catalyst
scoring, Monte Carlo forecasting and tax-aware planning — validated with a point-in-time,
survivorship-adjusted walk-forward backtest and deflated-Sharpe robustness testing. A weekly
scan produces 3–5 research leads for the investor to review.

> **The headline result is a negative one, on purpose.** An early backtest showed the strategy
> beating the S&P 500 by ~28%/yr. After removing look-ahead bias (point-in-time SEC
> fundamentals), survivorship bias (former index members), extending the test to 10 years and
> modelling taxes and a 1-year holding rule, the optimized portfolios **did not** reliably beat
> SPY. The project was redesigned around that finding: the model became a research and
> risk assistant that generates and tracks ideas, rather than an automatic stock picker.

---

## What it does

| Area | Module | Highlights |
|---|---|---|
| Screening | `portfolio_analyzer_advanced.py` | Momentum (risk-adjusted, late-cycle penalty), quality (ROE), value (earnings yield), analyst revisions, catalysts; values-based exclusions; position / sector / industry / binary-event caps |
| Optimization | `portfolio_analyzer_advanced.py` | Mean-variance (SLSQP) with time-decay covariance, Black-Litterman with investor views, max-holdings trim, turnover penalty, 1/N benchmark |
| Valuation | `portfolio_analyzer_advanced.py`, `stock_analyst.py` | Two-stage DCF from SEC 10-K free cash flow, Blume-adjusted CAPM discount rate with a floor, horizon-debiased analyst growth, scenario ranges, relative valuation terciles, ADR currency normalization |
| Analyst signals | `analyst_signals.py` | Estimate-revision factor (90-day EPS trend, revision breadth, beat rate); optimism haircut on forecasts by horizon; daily snapshots for future point-in-time tests |
| Catalysts & risk | `portfolio_analyzer_advanced.py`, `sec_edgar.py`, `biotech_pipeline.py` | News scoring (Yahoo + Google News), FDA approvals, ClinicalTrials.gov, PDUFA calendar, SEC 8-K red flags, structural/competitive-risk layer, biotech pipeline scoring with phase base rates |
| Event studies | `fda_event_study.py` | Abnormal returns around FDA approvals to calibrate investor views |
| Forecasting | `portfolio_analyzer_advanced.py` | CAPM expected returns from live FRED rates, Monte Carlo projections, historical stress tests (1987, 2008, 2020, 2022) |
| Validation | `point_in_time.py`, `robustness_grid.py` | Rolling walk-forward, as-filed fundamentals, point-in-time S&P 500 membership, costs, after-tax replay, robustness grid with deflated Sharpe |
| Planning | `portfolio_analyzer_advanced.py`, `portfolio_tracker.py` | Tax-lot (FIFO) tracking, keep-existing-holdings mode, no-short-term-sales rule, exit signals, monthly contribution routing, fractional/whole-share views |
| Weekly scan | `scan.py` | 3–5 research leads + an "AI/tech pullbacks" table (price vs estimate direction) + trading-activity flags; every pick logged for forward scoring |
| Accountability | `decision_log.py` | Every recommendation scored against SPY at 3 / 6 / 12 months |

## Methodology notes

**Avoiding look-ahead bias.** Backtests use fundamentals *as originally filed* with the SEC on
each date (`point_in_time.py`): the earliest 10-K filing wins over restatements, EPS is put on
the split-adjusted share basis, and filings older than ~18 months are ignored. Prices for P/E
use split-only (not dividend-adjusted) closes.

**Avoiding survivorship bias.** S&P 500 membership is reconstructed point-in-time: former
members (acquired, delisted) are added back where price data exists, and current members are
only eligible after their join date.

**Realistic constraints.** 10 bp trading costs, a one-year minimum holding period (with an early
exit on a ≥25% drop), and a taxable-account replay with FIFO lots and short- vs long-term rates.

**Robustness instead of tuning.** `robustness_grid.py` re-runs the walk-forward across sampled
combinations of factor weights, position caps, screen size, optimizer objective, industry /
binary-event caps and exit rules. The verdict rule (≥70% beat SPY = robust, <50% = fragile) is
fixed *before* the run, and the best variation is judged with the Deflated Sharpe Ratio
(Bailey & López de Prado, 2014) rather than adopted.

**Research basis** (selected): Jensen, Kelly & Pedersen (2023) on factor replication;
McLean & Pontiff (2016) on post-publication decay; van Binsbergen, Han & Lopez-Lira (2023) on
analyst forecast bias by horizon; DeMiguel, Garlappi & Uppal (2009) on 1/N; Cohen, Malloy &
Pomorski (2012) on opportunistic insider trades; Asquith, Pathak & Ritter (2005) and
Hong et al. (2015) on short interest.

## Results (10-year walk-forward, Oct 2018 – Oct 2026)

| | Return / yr |
|---|---|
| SPY (S&P 500) | +15.6% to +15.8% |
| RSP (equal-weight S&P 500) | +12.4% to +12.6% |
| Optimized scenarios (single run) | +10.1% to +11.1% (≈ −5%/yr vs SPY) |

Robustness grid, 60 variations: **22% beat SPY**, median −1.7%/yr vs SPY, verdict **FRAGILE**;
the best variation's Deflated Sharpe was 0.52 (consistent with luck). The screen did beat the
equal-weight index (RSP) by ~1.5%/yr — it picks better-than-average stocks, but not well enough
to beat a market led by a handful of mega-caps.

## Running it

```bash
pip install -r requirements.txt
cp profiles_example.py profiles_local.py      # your own profiles, tax rates, exclusions (git-ignored)
export SEC_USER_AGENT="Your Name your.email@example.com"   # required by SEC EDGAR

python scan.py                       # weekly research leads → outputs/scan_<date>.md
python portfolio_analyzer_advanced.py  # full analysis, backtests, HTML/Excel reports
python robustness_grid.py --n 60     # robustness test
python stock_analyst.py MSFT         # single-stock deep dive
python decision_log.py               # score past recommendations
python -m pytest tests -q            # 126 offline tests (no network)
```

Data sources: Yahoo Finance (`yfinance`), SEC EDGAR (XBRL company facts, 8-K filings), FRED,
ClinicalTrials.gov, FDA, Google News RSS, Wikipedia / Nasdaq index lists, historical S&P 500
membership (fja05680/sp500), Kenneth French data library.

## Limitations

Free data only: annual (not quarterly) point-in-time fundamentals; no point-in-time history of
analyst estimates or news, so those signals can only be forward-tested; US filers only for SEC
data; Yahoo rate limits. Backtests cover one decade dominated by mega-cap growth.

## How it was built

Built by Luis Roman in collaboration with **Claude Code** (Anthropic's AI coding agent). I set the
goals and research questions, made the methodology and investment decisions, and challenged
results (for example rejecting outdated research and late-cycle picks); Claude implemented most
of the code, tests and analysis under that direction.

*For research and education only — not investment advice.*
