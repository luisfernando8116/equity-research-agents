"""
Point-in-time fundamentals for the walk-forward backtest, from SEC 10-K filings.

The live screen uses TODAY's P/E and ROE. Feeding those into a backtest of 2018
means the model "knew" in 2018 which companies would end up cheap and profitable
— look-ahead bias. Here each value is what had been FILED with the SEC on the
date in question:

  * fiscal-year EPS (diluted), net income, stockholders' equity from 10-K XBRL;
  * the ORIGINAL filing is used (earliest `filed` date), not later restatements;
  * a filing older than STALE_DAYS is ignored (company stopped filing / was acquired);
  * EPS is put on today's share basis using split history, because Yahoo's
    adjusted prices are split-adjusted and filed EPS is not (NVDA: 10:1 in 2024).

Limits, stated rather than hidden:
  * annual figures only (up to ~15 months stale at times) and trailing GAAP P/E,
    whereas the live screen prefers forward P/E — there is no free point-in-time
    history of analyst forecasts;
  * US filers only. Foreign filers (20-F) and former members SEC can't map get
    no value → neutral rank in the screen, the same as missing data live.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import sec_edgar

STALE_DAYS = 550          # ~18 months: one missed annual filing
CACHE_TTL  = 7 * 86400    # compact extracts; past filings never change
CACHE_DIR  = sec_edgar.CACHE_DIR


def _first_filed(units: list, instant: bool = False) -> dict[str, tuple[str, float]]:
    """{period_end: (earliest filed date, value)} from 10-K full-year (or instant) facts."""
    out: dict[str, tuple[str, float]] = {}
    for u in units:
        if u.get("form") not in ("10-K", "10-K405") or "end" not in u or "filed" not in u:
            continue
        if instant:
            if "start" in u:
                continue
        else:
            if u.get("fp") != "FY" or "start" not in u:
                continue
            try:
                span = (datetime.strptime(u["end"], "%Y-%m-%d")
                        - datetime.strptime(u["start"], "%Y-%m-%d")).days
            except ValueError:
                continue
            if not 340 <= span <= 380:
                continue
        e = u["end"]
        if e not in out or u["filed"] < out[e][0]:
            out[e] = (u["filed"], float(u["val"]))
    return out


def extract_records(facts: dict) -> list[dict]:
    """[{filed, end, eps, ni, equity}] sorted by filing date, one per fiscal year."""
    g = facts.get("facts", {}).get("us-gaap", {})

    def units(concept, unit):
        return g.get(concept, {}).get("units", {}).get(unit, [])

    eps = (_first_filed(units("EarningsPerShareDiluted", "USD/shares"))
           or _first_filed(units("EarningsPerShareBasic", "USD/shares")))
    ni  = _first_filed(units("NetIncomeLoss", "USD"))
    eq  = (_first_filed(units("StockholdersEquity", "USD"), instant=True)
           or _first_filed(units(
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "USD"),
               instant=True))
    recs = []
    for end in sorted(set(eps) | set(ni)):
        filed = min(x[end][0] for x in (eps, ni) if end in x)
        recs.append({"filed": filed, "end": end,
                     "eps": eps[end][1] if end in eps else None,
                     "ni": ni[end][1] if end in ni else None,
                     "equity": eq[end][1] if end in eq else None})
    return sorted(recs, key=lambda r: r["filed"])


def load_records(ticker: str) -> list[dict]:
    """Compact per-ticker extract, cached on disk (the raw facts JSON is several MB)."""
    c = sec_edgar.cik(ticker)
    if c is None:
        return []
    path = CACHE_DIR / f"pit_{c}.json"
    if path.exists() and time.time() - path.stat().st_mtime < CACHE_TTL:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    facts = sec_edgar._get_json(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{c:010d}.json",
        f"facts_{c}", sec_edgar.CACHE_TTL["facts"], store=False)
    if not facts:
        return []
    recs = extract_records(facts)
    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(recs), encoding="utf-8")
    return recs


class PointInTimeFundamentals:
    """Callable: pit(date) → {ticker: {"pe", "roe"}} as known on that date."""

    def __init__(self, records: dict[str, list[dict]], prices: pd.DataFrame,
                 splits: pd.DataFrame | None = None):
        self.records = {t: r for t, r in records.items() if r}
        self.px = prices.ffill()
        self.splits = splits if splits is not None else pd.DataFrame()
        self._memo: dict = {}

    @classmethod
    def build(cls, tickers, prices, splits=None, workers: int = 4, verbose: bool = True):
        if not sec_edgar.enabled():
            return None
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            recs = dict(zip(tickers, pool.map(load_records, tickers)))
        obj = cls(recs, prices, splits)
        if verbose:
            print(f"      Point-in-time fundamentals: {len(obj.records)}/{len(tickers)} "
                  f"tickers with SEC 10-K history ({time.time() - t0:.0f}s)")
        return obj

    def split_factor_after(self, ticker: str, date: str) -> float:
        """Product of split ratios after `date` (EPS filed then ÷ this = today's basis)."""
        if ticker not in self.splits.columns:
            return 1.0
        s = self.splits[ticker]
        s = s[(s.index > pd.Timestamp(date)) & (s > 0)]
        return float(s.prod()) if len(s) else 1.0

    def at(self, ticker: str, date: pd.Timestamp) -> dict | None:
        d = pd.Timestamp(date)
        cutoff = (d - timedelta(days=STALE_DAYS)).strftime("%Y-%m-%d")
        ds = d.strftime("%Y-%m-%d")
        known = [r for r in self.records.get(ticker, []) if cutoff <= r["filed"] <= ds]
        if not known:
            return None
        r = known[-1]
        out = {"pe": None, "roe": None}
        if r.get("eps") and r["eps"] > 0 and ticker in self.px.columns:
            p = self.px[ticker].asof(d) if len(self.px) else None
            if p is not None and pd.notna(p) and p > 0:
                out["pe"] = float(p / (r["eps"] / self.split_factor_after(ticker, r["filed"])))
        if r.get("ni") is not None and r.get("equity") and r["equity"] > 0:
            out["roe"] = float(r["ni"] / r["equity"])
        return out

    def __call__(self, date) -> dict:
        key = pd.Timestamp(date).normalize()
        if key not in self._memo:
            self._memo[key] = {t: v for t in self.records
                               if (v := self.at(t, key)) is not None}
        return self._memo[key]
