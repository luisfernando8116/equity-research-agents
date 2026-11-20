"""
Decision Log — the analyzer's track record
==========================================
Every run of portfolio_analyzer_advanced.py appends what it recommended (weights
per scenario, entry prices, SPY price, BL views) to outputs/decision_log.jsonl.
This module scores those past calls against what actually happened.

Why: backtests describe the past the model was fitted to. Only recommendations
recorded at the time — then scored later — show whether the signals add value.

Usage
-----
  python decision_log.py              # scoreboard: every logged run at 3/6/12 months
  python decision_log.py --detail     # per-run breakdown

Each entry is scored as a buy-and-hold of the recommended weights from the run
date (no rebalancing), versus buying SPY on the same date.
"""

import sys
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOG_PATH = Path(__file__).parent / "outputs" / "decision_log.jsonl"
HORIZONS = {"3m": 63, "6m": 126, "12m": 252}   # trading days


# ─── WRITE ────────────────────────────────────────────────────────────────────

def record_run(run_id: str, recommended_key: str, scenarios: dict,
               entry_prices: dict, spy_price: float | None,
               bl_views: list[dict] | None = None,
               regime: str | None = None) -> None:
    """
    Append one run. `scenarios` = {key: {ticker: weight}}. Prices are the
    run-date closes used to size the trades.
    """
    tickers = {t for w in scenarios.values() for t in w}
    entry = {
        "run_id":      run_id,
        "date":        datetime.now().strftime("%Y-%m-%d"),
        "recommended": recommended_key,
        "regime":      regime,
        "spy_price":   spy_price,
        "scenarios":   {k: {t: round(float(w), 6) for t, w in ws.items()}
                        for k, ws in scenarios.items()},
        "entry_prices": {t: float(entry_prices[t]) for t in tickers
                         if entry_prices.get(t)},
        "bl_views":    [{"ticker": v["ticker"], "expected_return": v["expected_return"],
                         "confidence": v["confidence"]} for v in (bl_views or [])],
    }
    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ─── SCORE ────────────────────────────────────────────────────────────────────

def _fetch_closes(tickers: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(sorted(set(tickers)), start=start, auto_adjust=True,
                     progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df.ffill()


def score_entry(entry: dict, closes: pd.DataFrame) -> dict:
    """Return {scenario: {horizon: (port_ret, spy_ret)}} for elapsed horizons."""
    run_date = pd.Timestamp(entry["date"])
    after    = closes.loc[closes.index > run_date]   # first close after the run
    out = {}
    for key, weights in entry["scenarios"].items():
        res = {}
        for h, days in HORIZONS.items():
            if len(after) < days:
                continue                             # horizon not reached yet
            row = after.iloc[days - 1]
            port, wsum = 0.0, 0.0
            for t, w in weights.items():
                p0 = entry["entry_prices"].get(t)
                p1 = row.get(t) if t in row.index else None
                if p0 and p1 is not None and not np.isnan(p1):
                    port += w * (p1 / p0 - 1)
                    wsum += w
            spy0, spy1 = entry.get("spy_price"), row.get("SPY")
            if wsum > 0 and spy0 and spy1 is not None:
                res[h] = (port / wsum, spy1 / spy0 - 1)
        out[key] = res
    return out


def scoreboard(detail: bool = False) -> None:
    log = load_log()
    if not log:
        print(f"No runs logged yet ({LOG_PATH}). Run portfolio_analyzer_advanced.py first.")
        return
    tickers = {t for e in log for w in e["scenarios"].values() for t in w} | {"SPY"}
    closes  = _fetch_closes(sorted(tickers), min(e["date"] for e in log))

    rows = []
    for e in log:
        scored = score_entry(e, closes)
        for key, res in scored.items():
            for h, (p, s) in res.items():
                rows.append({"run": e["run_id"], "date": e["date"], "scenario": key,
                             "recommended": key == e["recommended"], "horizon": h,
                             "port": p, "spy": s, "alpha": p - s})
    print(f"\nDECISION LOG — {len(log)} run(s) logged, "
          f"{LOG_PATH.name}\n" + "=" * 72)
    if not rows:
        first = min(e["date"] for e in log)
        print(f"No horizon reached yet — first scores appear 3 months after {first}.")
        return
    df = pd.DataFrame(rows)
    for label, sub in [("Recommended scenario only", df[df.recommended]),
                       ("All scenarios", df)]:
        print(f"\n{label}")
        print(f"  {'Horizon':<8} {'Runs':>5} {'Avg port':>9} {'Avg SPY':>8} "
              f"{'Avg alpha':>10} {'Beat SPY':>9}")
        for h in HORIZONS:
            g = sub[sub.horizon == h]
            if g.empty:
                continue
            print(f"  {h:<8} {len(g):>5} {g.port.mean():>+9.1%} {g.spy.mean():>+8.1%} "
                  f"{g.alpha.mean():>+10.1%} {(g.alpha > 0).mean():>9.0%}")
    print("\n  Note: runs only weeks apart overlap heavily — treat them as one observation,"
          "\n  not many. Meaningful evidence needs a year+ of logged runs.")
    if detail:
        print("\nPer run:")
        print(df.sort_values(["date", "scenario", "horizon"]).to_string(
            index=False, formatters={c: "{:+.1%}".format for c in ("port", "spy", "alpha")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score past analyzer recommendations")
    ap.add_argument("--detail", action="store_true", help="per-run breakdown")
    scoreboard(detail=ap.parse_args().detail)
