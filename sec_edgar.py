"""
SEC EDGAR — official, free, authoritative data for US-listed filers
====================================================================
Two uses in the analyzer:
  1. 8-K material-event filings — filed within 4 business days of the event, often
     before the news. Red-flag items become negative catalysts; all recent 8-Ks
     are listed in stock_analyst.py.
  2. Fundamentals from 10-K XBRL ("company facts") — free cash flow straight from
     filed statements (operating cash flow − capex), replacing Yahoo's estimates
     for US filers. Foreign filers (20-F / IFRS) return None → Yahoo path is kept.

SEC fair-access policy REQUIRES a User-Agent with a contact email, and caps traffic
at 10 requests/second. Requests without one get HTTP 403. Set it once:
    PowerShell:  setx SEC_USER_AGENT "Your Name your.email@example.com"
or put it in SEC_USER_AGENT below. With no contact set, every function returns
None / [] and the analyzer logs that EDGAR is off.
"""

import os
import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "")   # "Name email@domain.com"
CACHE_DIR      = Path(__file__).parent / ".sec_cache"
CACHE_TTL      = {"tickers": 7 * 86400, "submissions": 6 * 3600, "facts": 24 * 3600}
_MIN_INTERVAL  = 0.15                                     # ≤ ~7 req/s, under the 10/s cap
_last_call     = [0.0]
_RATE_LOCK     = threading.Lock()                         # threads share one rate limit

# 8-K item codes (Form 8-K General Instructions)
ITEM_MEANING = {
    "1.01": "Material agreement entered",       "1.02": "Material agreement terminated",
    "1.03": "Bankruptcy / receivership",        "1.05": "Cybersecurity incident",
    "2.01": "Acquisition / disposal completed", "2.02": "Results of operations",
    "2.03": "New material debt obligation",     "2.05": "Restructuring / exit costs",
    "2.06": "Material impairment",              "3.01": "Delisting / listing-rule notice",
    "3.02": "Unregistered equity sale",         "4.01": "Auditor change",
    "4.02": "Prior financials no longer reliable",
    "5.02": "Director / officer change",        "5.07": "Shareholder vote results",
    "7.01": "Reg FD disclosure",                "8.01": "Other material event",
    "9.01": "Financial statements / exhibits",
}
# Items that are bad news on their own, with catalyst scores (−1 … 0)
RED_FLAGS = {"1.03": -1.0, "3.01": -0.8, "4.02": -0.8, "1.05": -0.4,
             "4.01": -0.3, "2.06": -0.3, "1.02": -0.3, "2.05": -0.2}


def enabled() -> bool:
    return "@" in SEC_USER_AGENT


def _get_json(url: str, cache_key: str, ttl: int, store: bool = True):
    """GET with disk cache, rate limiting and the required User-Agent.
    store=False skips writing the response (callers that cache a compact extract)."""
    if not enabled():
        return None
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{cache_key}.json"
    if path.exists() and time.time() - path.stat().st_mtime < ttl:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    with _RATE_LOCK:
        wait = _MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
    try:
        r = requests.get(url, headers={"User-Agent": SEC_USER_AGENT,
                                       "Accept-Encoding": "gzip, deflate"}, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None
    if store:
        path.write_text(json.dumps(data), encoding="utf-8")
    return data


def cik(ticker: str) -> int | None:
    data = _get_json("https://www.sec.gov/files/company_tickers.json",
                     "company_tickers", CACHE_TTL["tickers"])
    if not data:
        return None
    t = ticker.upper().replace("-", ".")
    for v in data.values():
        if v.get("ticker", "").upper() in (t, ticker.upper()):
            return int(v["cik_str"])
    return None


def recent_8k(ticker: str, days: int = 30) -> list[dict]:
    """8-K filings in the last `days`: [{date, items:[codes], meaning:[...], url}]."""
    c = cik(ticker)
    if c is None:
        return []
    sub = _get_json(f"https://data.sec.gov/submissions/CIK{c:010d}.json",
                    f"sub_{c}", CACHE_TTL["submissions"])
    if not sub:
        return []
    rec = sub.get("filings", {}).get("recent", {})
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    out = []
    for form, date, items, acc, doc in zip(rec.get("form", []), rec.get("filingDate", []),
                                           rec.get("items", []), rec.get("accessionNumber", []),
                                           rec.get("primaryDocument", [])):
        if form not in ("8-K", "8-K/A") or date < cutoff:
            continue
        codes = [i.strip() for i in (items or "").split(",") if i.strip()]
        out.append({
            "date": date, "form": form, "items": codes,
            "meaning": [ITEM_MEANING.get(i, i) for i in codes if i != "9.01"],
            "url": (f"https://www.sec.gov/Archives/edgar/data/{c}/"
                    f"{acc.replace('-', '')}/{doc}"),
        })
    return out


def red_flag_signal(ticker: str, days: int = 60) -> dict | None:
    """Worst red-flag 8-K in the window as a catalyst-signal dict, else None."""
    worst = None
    for f in recent_8k(ticker, days):
        for code in f["items"]:
            if code in RED_FLAGS and (worst is None or RED_FLAGS[code] < worst[0]):
                worst = (RED_FLAGS[code], code, f)
    if worst is None:
        return None
    score, code, f = worst
    return {
        "signal":        "NEGATIVE",
        "score":         score,
        "catalyst_type": f"SEC 8-K {code}",
        "headline":      f"8-K {f['date']}: {ITEM_MEANING.get(code, code)}",
        "days_ago":      (datetime.now() - datetime.strptime(f["date"], "%Y-%m-%d")).days,
        "url":           f["url"],
    }


def _annual_values(facts: dict, concept: str) -> dict[int, float]:
    """{fiscal_year: value} from 10-K full-year USD facts for a us-gaap concept."""
    units = (facts.get("facts", {}).get("us-gaap", {}).get(concept, {})
             .get("units", {}).get("USD", []))
    out: dict[int, tuple[str, float]] = {}
    for u in units:
        if u.get("form") != "10-K" or u.get("fp") != "FY" or "start" not in u:
            continue
        # full-year periods only (~12 months)
        try:
            span = (datetime.strptime(u["end"], "%Y-%m-%d")
                    - datetime.strptime(u["start"], "%Y-%m-%d")).days
        except ValueError:
            continue
        if not 340 <= span <= 380:
            continue
        fy = int(u["end"][:4])
        # keep the latest-filed value for each fiscal year (restatements win)
        if fy not in out or u.get("filed", "") > out[fy][0]:
            out[fy] = (u.get("filed", ""), float(u["val"]))
    return {fy: v for fy, (_, v) in out.items()}


def fcf_avg(ticker: str, years: int = 3) -> float | None:
    """Average free cash flow (operating cash flow − capex) over the last `years`
    fiscal years, from filed 10-K XBRL. None for non-US filers or missing data."""
    c = cik(ticker)
    if c is None:
        return None
    facts = _get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{c:010d}.json",
                      f"facts_{c}", CACHE_TTL["facts"])
    if not facts:
        return None
    ocf = (_annual_values(facts, "NetCashProvidedByUsedInOperatingActivities")
           or _annual_values(facts, "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"))
    # Companies switch capex tags over time (NVDA: PP&E tag until 2012, then
    # "ProductiveAssets") — merge per year, preferring the PP&E tag.
    capex: dict[int, float] = {}
    for concept in ("PaymentsToAcquireProductiveAssets",
                    "PaymentsToAcquirePropertyPlantAndEquipment"):
        capex.update(_annual_values(facts, concept))
    if not ocf:
        return None
    # Use the LATEST fiscal years only — never fall back to stale ones (matching the
    # years two tags shared once produced NVDA FCF from 2010–2012: $0.6B vs ~$60B).
    recent = sorted(ocf, reverse=True)[:years]
    if recent[0] < datetime.now().year - 2 or any(y not in capex for y in recent):
        return None
    return sum(ocf[y] - capex[y] for y in recent) / len(recent)
