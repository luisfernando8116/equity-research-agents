"""
Weekly scan — 3–5 stocks worth researching, using the analyzer's full signal stack
without the slow parts (no 10-year backtest, no optimizer).

    python scan.py                 # writes outputs/scan_<date>.md and logs the picks
    python scan.py --n 5 --tier 120

These are RESEARCH LEADS, not buy signals: the robustness test showed the screen
picks better-than-average stocks but did not reliably beat SPY as a portfolio.
Every pick is logged with its price (outputs/candidates_log.jsonl) so the list
can be scored against SPY at 3 / 6 / 12 months.

Selection: factor composite (momentum with late-cycle penalty, quality, value,
analyst revisions, catalysts) → drop holdings, ETFs, values exclusions, SEC red
flags, severe structural risks, falling estimates and the dearest valuation third
→ at most one per industry and one binary-event biotech.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import portfolio_analyzer_advanced as pa

BUDGET = 500            # monthly contribution — "what $500 buys" in whole shares
PRICEY = 500            # flag share prices above this (whole shares only)
FALLING_REVISIONS = -0.25

# Theme section: large AI/tech names that have pulled back. The main ranking
# penalises falling prices (momentum), so these need their own look.
THEME_SECTORS = {"Technology", "Communication Services"}
THEME_MIN_DRAWDOWN = 0.15     # ≥15% below the 52-week high
THEME_MIN_MCAP = 10e9         # established companies only
THEME_MAX = 40                # names that get the full checks


# Trading-activity signals — only those with return-predictive evidence at a
# months-long horizon (insider purchases: Cohen, Malloy & Pomorski 2012; short
# interest: Asquith-Pathak-Ritter 2005, Hong et al. 2015). Selling pressure is a
# timing note (stage purchases), not a stock-selection signal.
INSIDER_LOOKBACK_DAYS = 90
INSIDER_MIN_BUY_USD = 50_000      # ignore token purchases
SHORT_PCT_WARN = 0.10             # ≥10% of float sold short
SHORT_DAYS_WARN = 5               # ≥5 days of average volume to cover
SELLING_VOL_RATIO = 1.5           # 20-day volume vs the prior ~3 months
SELLING_DOWN_UP_RATIO = 1.3       # down-day volume vs up-day volume


def activity_signals(ticker: str) -> dict:
    """Insider buying flag, short-interest warning and selling-pressure note."""
    out = {"insider_buy": None, "short_warn": None, "selling_pressure": None}
    try:
        it = pa._safe_attr(ticker, "insider_transactions")
        if it is not None and len(it):
            d = it[pd.to_datetime(it["Start Date"], errors="coerce")
                   >= pd.Timestamp.now() - pd.Timedelta(days=INSIDER_LOOKBACK_DAYS)]
            buys = d[d["Text"].fillna("").str.contains("Purchase", case=False)]
            val = float(pd.to_numeric(buys.get("Value"), errors="coerce").fillna(0).sum())
            if val >= INSIDER_MIN_BUY_USD:
                out["insider_buy"] = (f"insiders bought ${val:,.0f} on the open market in "
                                      f"{INSIDER_LOOKBACK_DAYS}d ({len(buys)} trade(s))")
    except Exception:
        pass
    info = pa._safe_info(ticker) or {}
    sp, sr = info.get("shortPercentOfFloat"), info.get("shortRatio")
    if (sp or 0) >= SHORT_PCT_WARN or (sr or 0) >= SHORT_DAYS_WARN:
        out["short_warn"] = (f"high short interest: {sp:.0%} of float" if sp else "high short interest") \
            + (f", {sr:.1f} days to cover" if sr else "")
    try:
        import yfinance as yf
        h = pa._call_with_timeout(lambda: yf.Ticker(ticker).history(period="6mo", auto_adjust=True))
        if h is not None and len(h) > 100:
            v, ch = h["Volume"], h["Close"].pct_change()
            base = v.iloc[-90:-20].mean()
            last = h.iloc[-20:]
            dn, upv = last[ch.iloc[-20:] < 0]["Volume"].mean(), last[ch.iloc[-20:] > 0]["Volume"].mean()
            ratio = v.iloc[-20:].mean() / base if base else 0
            if ratio >= SELLING_VOL_RATIO and upv and dn / upv >= SELLING_DOWN_UP_RATIO:
                out["selling_pressure"] = (f"heavy selling: volume {ratio:.1f}× normal, down days "
                                           f"{dn / upv:.1f}× up days — consider staging purchases")
    except Exception:
        pass
    return out


def _activity_text(a: dict) -> str:
    parts = []
    if a.get("insider_buy"):
        parts.append(f"🟢 {a['insider_buy']}")
    if a.get("short_warn"):
        parts.append(f"🔴 {a['short_warn']}")
    if a.get("selling_pressure"):
        parts.append(f"⏳ {a['selling_pressure']}")
    return " · ".join(parts) or "no notable insider, short or selling signals"


def _keep_awake():
    """Ask Windows not to sleep while the scan runs (released on exit)."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass


def _pct(a, b):
    return float(a / b - 1) if b else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="max picks")
    ap.add_argument("--tier", type=int, default=120,
                    help="top names (by base composite) that get the news/FDA/SEC/revision checks")
    args = ap.parse_args()
    _keep_awake()
    today = date.today().isoformat()
    print(pa.refresh_market_rates())

    # 1. Universe + prices
    universe = pa.fetch_universe()
    prices = pa.fetch_prices(universe)
    valid = [t for t in universe if t in prices.columns
             and prices[t].notna().sum() >= pa.MIN_HISTORY_DAYS]
    print(f"{len(valid)} tickers with enough history")

    # 2. Fundamentals (also warms the .info cache used below)
    fundamentals = pa.fetch_fundamentals(valid)
    infos = {t: pa._safe_info(t) or {} for t in valid}
    held = set(pa.held_tickers())
    skip = {t: r for t in valid if (r := pa.excluded_by_preference(t, infos.get(t)))}
    etfs = pa.INDEX_TICKERS | pa.INTERNATIONAL_TICKERS
    pool = [t for t in valid if t not in skip and t not in etfs]
    print(f"Excluded by your values screen: {len(skip)}  "
          f"({', '.join(sorted(skip)[:12])}{'…' if len(skip) > 12 else ''})")

    # 3. Expensive checks only for the top tier + the AI/tech pullback names
    tier = pa.multi_factor_screen(prices[pool], fundamentals, top_n=args.tier, force_custom=False)
    pxf = prices.ffill()
    last, high52 = pxf.iloc[-1], pxf.iloc[-252:].max()
    theme = [t for t in valid if t not in etfs and not skip.get(t)
             and infos.get(t, {}).get("sector") in THEME_SECTORS
             and (infos.get(t, {}).get("marketCap") or 0) >= THEME_MIN_MCAP
             and last[t] <= high52[t] * (1 - THEME_MIN_DRAWDOWN)]
    theme = sorted(theme, key=lambda t: -(infos[t].get("marketCap") or 0))[:THEME_MAX]
    theme += [t for t in held if t in valid and t not in theme
              and infos.get(t, {}).get("sector") in THEME_SECTORS]
    checked = list(dict.fromkeys(tier + theme))
    print(f"Checking news, FDA, SEC filings and analyst revisions for {len(checked)} names "
          f"(top {len(tier)} + {len(checked) - len(tier)} AI/tech pullbacks)...")
    catalysts = pa.fetch_catalyst_signals(checked)
    fda_dec, _ = pa.upcoming_fda_signals()
    for t, s in fda_dec.items():
        if t in checked and abs(catalysts.get(t, {}).get("score", 0)) < s["score"]:
            catalysts[t] = s
    red = {}
    try:
        import sec_edgar
        if sec_edgar.enabled():
            for t in checked:
                s = sec_edgar.red_flag_signal(t)
                if s:
                    red[t] = s["headline"]
    except Exception as e:
        print("SEC check failed:", e)
    structural = pa.assess_structural_risks(checked)
    import analyst_signals as an
    revisions = an.fetch_revision_signals(checked, pa._safe_attr, pa._safe_info)

    ranked = pa.multi_factor_screen(prices[tier], fundamentals, top_n=len(tier),
                                    catalyst_signals=catalysts, risk_penalties=structural,
                                    revision_signals=revisions, force_custom=False)

    # 4. Valuation (relative terciles across everything checked)
    val_data = pa.fetch_valuation_data(checked)
    val = pa.rerank_valuation_verdicts({t: pa.valuation_signal(t, val_data.get(t, {}))
                                        for t in checked})
    binary = pa.binary_event_tickers({t: infos.get(t, {}) for t in ranked},
                                     {d["ticker"] for d in pa.UPCOMING_FDA_DECISIONS})

    # 5. Filters + diversity
    picks, reasons_out, industries, n_binary = [], {}, set(), 0
    for t in ranked:
        why_not = None
        if t in held:
            why_not = "already held"
        elif t in red:
            why_not = f"SEC red flag: {red[t]}"
        elif structural.get(t, {}).get("severity", 0) >= pa.EXIT_STRUCTURAL_SEV:
            why_not = f"structural risk: {structural[t]['risk']}"
        elif revisions.get(t, {}).get("score", 0) <= FALLING_REVISIONS:
            why_not = "analysts cutting estimates"
        elif val.get(t, {}).get("verdict") == "OVERVALUED":
            why_not = "dearest valuation third"
        elif (ind := infos.get(t, {}).get("industry")) in industries:
            why_not = f"already have a {ind} pick"
        elif t in binary and n_binary >= 1:
            why_not = "one binary-event biotech per week"
        if why_not:
            reasons_out[t] = why_not
            continue
        picks.append(t)
        industries.add(infos.get(t, {}).get("industry"))
        n_binary += t in binary
        if len(picks) >= args.n:
            break

    # 6. Report
    px = prices.ffill()
    rows = []
    for rank, t in enumerate(picks, 1):
        info, s = infos.get(t, {}), px[t]
        price = float(s.iloc[-1])
        hi, lo = float(s.iloc[-252:].max()), float(s.iloc[-252:].min())
        rev = revisions.get(t, {})
        ctx = rev.get("context") or {}
        cat = catalysts.get(t, {})
        v = val.get(t, {})
        rows.append({
            "rank": rank, "ticker": t, "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"), "industry": info.get("industry"),
            "price": round(price, 2), "market_cap_b": round((info.get("marketCap") or 0) / 1e9, 1),
            "ret_12m": _pct(price, float(s.iloc[-252])), "ret_3m": _pct(price, float(s.iloc[-63])),
            "range_pos": (price - lo) / (hi - lo) if hi > lo else None,
            "pe": fundamentals.get(t, {}).get("pe"), "roe": fundamentals.get(t, {}).get("roe"),
            "revision_score": rev.get("score"), "revision_metrics": rev.get("metrics"),
            "analyst_target": ctx.get("target_mean"), "analyst_upside": ctx.get("upside"),
            "analyst_flags": ctx.get("flags"),
            "catalyst": (f"{cat.get('catalyst_type')}: {cat.get('headline')}" if cat else None),
            "valuation": v.get("verdict"), "dcf_range": (v.get("buy_low"), v.get("buy_high")),
            "binary_event": t in binary, "pricey": price > PRICEY,
            "shares_for_budget": int(BUDGET // price),
            "structural": structural.get(t, {}).get("risk"),
            "activity": activity_signals(t),
        })

    out_dir = Path(pa.OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)
    md = [f"# Stocks to review — {today}",
          f"_Research leads, not buy signals. Values screen applied ({len(skip)} companies "
          f"excluded); holdings excluded._", ""]
    for r in rows:
        f = lambda x, spec="+.0%": format(x, spec) if x is not None else "n/a"
        md += [f"## {r['rank']}. {r['ticker']} — {r['name']}",
               f"- {r['sector']} / {r['industry']} · ${r['price']:,.2f} · mkt cap ${r['market_cap_b']}B"
               + ("  ⚠ binary-event biotech" if r["binary_event"] else ""),
               f"- Momentum: 12m {f(r['ret_12m'])} · 3m {f(r['ret_3m'])} · "
               f"52-wk range position {f(r['range_pos'], '.0%')}",
               f"- P/E {f(r['pe'], '.1f')} · ROE {f(r['roe'], '.0%')} · valuation: {r['valuation']}"
               + (f" (DCF range ${r['dcf_range'][0]:,.0f}–${r['dcf_range'][1]:,.0f})"
                  if r["dcf_range"][0] and r["dcf_range"][1] else ""),
               f"- Analyst revisions score {f(r['revision_score'], '+.2f')} · mean target "
               + (f"${r['analyst_target']:,.0f} ({f(r['analyst_upside'])})" if r["analyst_target"] else "n/a")
               + (f" · ⚑ {'; '.join(r['analyst_flags'])}" if r["analyst_flags"] else ""),
               f"- Catalyst: {r['catalyst'] or 'none detected'}",
               f"- Trading activity: {_activity_text(r['activity'])}",
               f"- $500 buys {r['shares_for_budget']} whole share(s)"
               + ("  ⚠ share price above $500" if r["pricey"] else ""),
               ""]
    # AI/tech pullbacks: is the drop an opportunity or a warning?
    theme_rows = []
    for t in theme:
        info, s = infos.get(t, {}), pxf[t]
        price = float(s.iloc[-1])
        rev = revisions.get(t, {})
        m = rev.get("metrics") or {}
        sc, tr = rev.get("score"), m.get("trend_90d")
        if (sc is not None and sc <= FALLING_REVISIONS) or (tr is not None and tr < -0.02):
            verdict = "Estimates falling — the market may be right"
        elif (sc is not None and sc >= 0.25) or (tr is not None and tr > 0.02):
            verdict = "Price down, estimates up — worth a look"
        else:
            verdict = "Mixed / no clear signal"
        if t in red:
            verdict = f"SEC red flag — {red[t]}"
        theme_rows.append({
            "ticker": t, "name": info.get("longName") or info.get("shortName"),
            "held": t in held, "price": round(price, 2),
            "from_high": price / float(high52[t]) - 1,
            "ret_3m": _pct(price, float(s.iloc[-63])),
            "revision_score": sc, "eps_trend_90d": tr,
            "valuation": val.get(t, {}).get("verdict"), "pe": fundamentals.get(t, {}).get("pe"),
            "analyst_upside": (rev.get("context") or {}).get("upside"),
            "structural": structural.get(t, {}).get("risk"), "verdict": verdict,
            "activity": activity_signals(t),
        })
    order = {"Price down, estimates up — worth a look": 0, "Mixed / no clear signal": 1}
    theme_rows.sort(key=lambda r: (order.get(r["verdict"], 2), -(r["revision_score"] or -9)))
    md += ["## AI / tech pullbacks — opportunity or warning?",
           f"_Large Technology & Communication Services names ≥{THEME_MIN_DRAWDOWN:.0%} below their "
           f"52-week high, plus your holdings in those sectors. A falling price with rising analyst "
           f"estimates can be an opportunity; falling estimates usually mean the drop has a reason._", "",
           "| Ticker | From 52w high | 3m | Est. change 90d | Revisions | Valuation | P/E | Analyst upside | Read | Activity |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda x, spec: format(x, spec) if x is not None else "n/a"
    for r in theme_rows:
        md.append(f"| {r['ticker']}{' (held)' if r['held'] else ''} | {fmt(r['from_high'], '+.0%')} | "
                  f"{fmt(r['ret_3m'], '+.0%')} | {fmt(r['eps_trend_90d'], '+.1%')} | "
                  f"{fmt(r['revision_score'], '+.2f')} | {r['valuation'] or 'n/a'} | "
                  f"{fmt(r['pe'], '.1f')} | {fmt(r['analyst_upside'], '+.0%')} | {r['verdict']}"
                  + (f" · ⚠ {r['structural']}" if r["structural"] else "")
                  + f" | {_activity_text(r['activity'])} |")
    md.append("")
    md += ["### Next-ranked names that were filtered out", ""]
    md += [f"- {t}: {why}" for t, why in list(reasons_out.items())[:12]]
    path = out_dir / f"scan_{today}.md"
    path.write_text("\n".join(md), encoding="utf-8")
    (out_dir / f"scan_{today}.json").write_text(
        json.dumps({"picks": rows, "tech_pullbacks": theme_rows}, indent=1, default=str),
        encoding="utf-8")
    spy = float(px["SPY"].iloc[-1]) if "SPY" in px.columns else None
    with open(out_dir / "candidates_log.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"date": today, "spy": spy,
                             "picks": {r["ticker"]: r["price"] for r in rows}}) + "\n")
    print("\n".join(md))
    print(f"\nSaved {path.name}; picks logged for scoring vs SPY.")


if __name__ == "__main__":
    main()
