"""
Analyst estimate signals: revisions factor, horizon debiasing, snapshot logging.

Research basis (what we use and what we don't):
  * Changes in analyst estimates carry information that prices absorb slowly;
    "earnings momentum / revisions" is among the factor themes that replicate
    out-of-sample in Jensen, Kelly & Pedersen (JF 2023, "Is There a Replication
    Crisis in Finance?").  Published anomalies decay after publication
    (McLean & Pontiff, JF 2016), so the factor gets a modest 10% weight.
  * The LEVEL of analyst forecasts is optimistic, and the bias grows with
    horizon (van Binsbergen, Han & Lopez-Lira, RFS 2023).  We therefore use
    estimates for direction of change (revisions) and haircut their level
    before they feed the DCF.
  * Price targets and buy/sell ratings are shown as context only — they are
    not a factor.

All numeric scalings below are heuristics chosen here (marked STARTING VALUE),
not values taken from the papers.  The snapshot log exists so they can be
calibrated against realised outcomes once several quarters have accumulated.
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import pandas as pd

# Optimism haircut on POSITIVE analyst growth forecasts, by horizon.
# STARTING VALUE — direction from van Binsbergen-Han-Lopez-Lira (bias grows with
# horizon); magnitudes are ours. Recalibrate from outputs/analyst_snapshots/.
OPTIMISM_HAIRCUT = {"0q": 0.05, "+1q": 0.05, "0y": 0.10, "+1y": 0.20}

# Revision-score blend and scaling — STARTING VALUES
REV_TREND_FULL_SCALE = 0.10     # a 10% 90-day change in the +1y EPS estimate = max signal
REV_WEIGHTS = {"trend": 0.5, "breadth": 0.3, "surprise": 0.2}   # beats are partly guided → lowest

# Context flags for price targets / ratings
CROWDED_RATING_MAX = 1.8        # recommendationMean ≤ this (1 = strong buy) …
CROWDED_MIN_ANALYSTS = 10       # … with at least this many analysts
STRETCHED_TARGET_UPSIDE = 0.50  # mean target ≥ 50% above price

SNAPSHOT_DIR = Path(__file__).resolve().parent / "outputs" / "analyst_snapshots"


def _num(x) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _col(df: pd.DataFrame, name: str):
    """Case-insensitive column lookup (yfinance mixes 'downLast7Days'/'upLast7days')."""
    for c in df.columns:
        if str(c).lower() == name.lower():
            return c
    return None


def debias_growth(g: float | None, horizon: str = "+1y") -> float | None:
    """Shrink an optimistic analyst growth forecast toward zero. Negative
    forecasts are left alone — the documented bias is upward."""
    if g is None:
        return None
    g = float(g)
    return g * (1.0 - OPTIMISM_HAIRCUT.get(horizon, 0.20)) if g > 0 else g


def revision_metrics(eps_trend=None, eps_revisions=None, earnings_history=None) -> dict:
    """Raw revision inputs from the three yfinance frames (any may be None)."""
    out = {"trend_90d": None, "breadth": None, "surprise_hit": None, "n_quarters": 0}

    if isinstance(eps_trend, pd.DataFrame) and not eps_trend.empty:
        cur, old = _col(eps_trend, "current"), _col(eps_trend, "90daysAgo")
        for period in ("+1y", "0y"):
            if cur and old and period in eps_trend.index:
                c, o = _num(eps_trend.loc[period, cur]), _num(eps_trend.loc[period, old])
                if c is not None and o not in (None, 0):
                    out["trend_90d"] = (c - o) / abs(o)
                    out["trend_period"] = period
                    break

    if isinstance(eps_revisions, pd.DataFrame) and not eps_revisions.empty:
        up, dn = _col(eps_revisions, "upLast30days"), _col(eps_revisions, "downLast30days")
        if up and dn:
            ups = downs = 0.0
            for period in ("0y", "+1y"):
                if period in eps_revisions.index:
                    ups   += _num(eps_revisions.loc[period, up]) or 0.0
                    downs += _num(eps_revisions.loc[period, dn]) or 0.0
            if ups + downs > 0:
                out["breadth"] = (ups - downs) / (ups + downs)

    if isinstance(earnings_history, pd.DataFrame) and not earnings_history.empty:
        act, est = _col(earnings_history, "epsActual"), _col(earnings_history, "epsEstimate")
        if act and est:
            rows = earnings_history[[act, est]].dropna().tail(4)
            if len(rows):
                out["surprise_hit"] = float((rows[act] > rows[est]).mean())
                out["n_quarters"] = int(len(rows))
    return out


def revision_score(m: dict) -> float | None:
    """Combine the metrics into −1 … +1. None when no component is available."""
    parts = {}
    if m.get("trend_90d") is not None:
        parts["trend"] = max(-1.0, min(1.0, m["trend_90d"] / REV_TREND_FULL_SCALE))
    if m.get("breadth") is not None:
        parts["breadth"] = float(m["breadth"])
    if m.get("surprise_hit") is not None and m.get("n_quarters", 0) >= 2:
        parts["surprise"] = (m["surprise_hit"] - 0.5) * 2.0
    if not parts:
        return None
    w = sum(REV_WEIGHTS[k] for k in parts)
    return sum(REV_WEIGHTS[k] * v for k, v in parts.items()) / w


def consensus_context(info: dict) -> dict:
    """Price target / rating context with contrarian flags. Not a factor input."""
    price = _num(info.get("currentPrice") or info.get("regularMarketPrice"))
    tgt   = _num(info.get("targetMeanPrice"))
    rec   = _num(info.get("recommendationMean"))
    n     = int(_num(info.get("numberOfAnalystOpinions")) or 0)
    upside = (tgt / price - 1.0) if tgt and price else None
    flags = []
    if rec is not None and rec <= CROWDED_RATING_MAX and n >= CROWDED_MIN_ANALYSTS:
        flags.append("crowded: near-unanimous buy — little left to upgrade")
    if upside is not None and upside >= STRETCHED_TARGET_UPSIDE:
        flags.append(f"stretched target: +{upside:.0%} implied — targets are optimistic on average")
    if upside is not None and upside < 0:
        flags.append("price above mean target — analysts lag or the stock ran ahead")
    return {
        "target_mean": tgt, "target_low": _num(info.get("targetLowPrice")),
        "target_high": _num(info.get("targetHighPrice")), "upside": upside,
        "rating_mean": rec, "rating_key": info.get("recommendationKey"),
        "n_analysts": n, "flags": flags,
    }


def fetch_revision_signals(tickers, get_attr, get_info=None) -> dict:
    """{ticker: {score, metrics, context}} using the caller's cached fetchers
    (get_attr(ticker, attr) and get_info(ticker)). Tickers with no coverage are omitted."""
    out = {}
    for t in tickers:
        try:
            m = revision_metrics(get_attr(t, "eps_trend"), get_attr(t, "eps_revisions"),
                                 get_attr(t, "earnings_history"))
        except Exception:
            continue
        s = revision_score(m)
        if s is None:
            continue
        row = {"score": s, "metrics": m}
        if get_info is not None:
            try:
                row["context"] = consensus_context(get_info(t) or {})
            except Exception:
                pass
        out[t] = row
    return out


def save_snapshot(signals: dict, prices: dict | None = None, asof: date | None = None,
                  directory: Path | None = None) -> Path | None:
    """Write today's analyst data to <dir>/<YYYY-MM-DD>.json so revision weights and
    the optimism haircut can later be tested point-in-time (Yahoo keeps no history)."""
    if not signals:
        return None
    d = Path(directory or SNAPSHOT_DIR)
    d.mkdir(parents=True, exist_ok=True)
    asof = asof or date.today()
    payload = {"asof": asof.isoformat(), "haircut": OPTIMISM_HAIRCUT,
               "weights": REV_WEIGHTS, "tickers": {}}
    for t, s in signals.items():
        payload["tickers"][t] = {**s, "price": (prices or {}).get(t)}
    path = d / f"{asof.isoformat()}.json"
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return path
