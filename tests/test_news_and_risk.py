"""Offline tests for the news normaliser and structural-risk layer."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import portfolio_analyzer_advanced as pa  # noqa: E402

RECENT = (datetime.utcnow() - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
NEWS = {
    # yfinance >= 1.0 format
    "MU":   [{"id": "1", "content": {"title": "CXMT ramps DRAM output after IPO",
                                     "summary": "ChangXin adds capacity", "pubDate": RECENT}}],
    # legacy format
    "OLD":  [{"title": "FDA approves new drug", "summary": "",
              "providerPublishTime": int((datetime.now() - timedelta(days=2)).timestamp())}],
    # no timestamp → must be dropped, not treated as fresh
    "NOTS": [{"id": "2", "content": {"title": "Undated story"}}],
}


def _fake_attr(t, attr):
    return NEWS.get(t, []) if attr == "news" else None


def test_news_items_handles_both_formats(monkeypatch):
    monkeypatch.setattr(pa, "_safe_attr", _fake_attr)
    monkeypatch.setattr(pa, "google_news", lambda q: [])     # keep the test offline
    mu = pa.news_items("MU")
    assert mu and mu[0][0].startswith("CXMT") and isinstance(mu[0][2], datetime)
    assert pa.news_items("OLD")[0][0] == "FDA approves new drug"
    assert pa.news_items("NOTS") == []


def test_google_fallback_used_when_yahoo_empty_and_filters_other_companies(monkeypatch):
    now = datetime.now()
    monkeypatch.setattr(pa, "_safe_attr", lambda t, a: [] if a == "news" else None)
    monkeypatch.setattr(pa, "_safe_info", lambda t: {"longName": "Roche Holding AG"})
    monkeypatch.setattr(pa, "google_news", lambda q: [
        ("Roche phase 3 trial met primary endpoint", "Reuters", now),
        ("Novartis presents weaker-than-expected data", "Reuters", now)])
    items = pa.news_items("RHHBY")
    assert [h for h, _, _ in items] == ["Roche phase 3 trial met primary endpoint"]


def _catalyst_for(monkeypatch, ticker, name, headline):
    now = datetime.now()
    monkeypatch.setattr(pa, "news_items", lambda t: [(headline, "Wire", now)])
    monkeypatch.setattr(pa, "_safe_attr", lambda t, a: None)
    monkeypatch.setattr(pa, "_safe_info", lambda t: {"longName": name})
    monkeypatch.setattr(pa, "google_news", lambda q: [])
    return pa.fetch_catalyst_signals([ticker]).get(ticker, {})


def test_chip_breakthrough_is_not_an_fda_designation(monkeypatch):
    sig = _catalyst_for(monkeypatch, "MU", "Micron Technology, Inc.",
                        "Micron's Next AI Memory Breakthrough Has a Cooling Problem")
    assert sig.get("catalyst_type") != "FDA Designation"


def test_real_breakthrough_therapy_designation_still_scores(monkeypatch):
    sig = _catalyst_for(monkeypatch, "BNTX", "BioNTech SE",
                        "BioNTech receives Breakthrough Therapy designation for BNT327")
    assert sig.get("catalyst_type") == "FDA Designation"


def test_company_key():
    assert pa.company_key("Roche Holding AG") == "roche"
    assert pa.company_key("Scholar Rock Holding Corporation") == "scholar rock"


def test_structural_risk_escalates_on_watch_term_news(monkeypatch):
    monkeypatch.setattr(pa, "_safe_attr", _fake_attr)
    monkeypatch.setattr(pa, "google_news", lambda q: [])     # keep the test offline
    monkeypatch.setattr(pa, "STRUCTURAL_RISKS", [{
        "name": "memory oversupply", "tickers": {"MU", "SNDK"}, "severity": 0.5,
        "watch_terms": ["CXMT"], "note": "x"}])
    out = pa.assess_structural_risks(["MU", "SNDK", "AAPL"])
    assert out["MU"]["severity"] == 0.6 and out["MU"]["evidence"]
    assert out["SNDK"]["severity"] == 0.5          # no news → base severity
    assert "AAPL" not in out


def test_fda_calendar_scores_by_proximity_and_flags_missing_outcomes(monkeypatch):
    today = datetime(2026, 10, 24)
    monkeypatch.setattr(pa, "UPCOMING_FDA_DECISIONS", [
        {"ticker": "SOON", "drug": "a", "indication": "x", "date": "2026-11-15",
         "review": "Priority Review", "source": "s", "outcome": None},
        {"ticker": "LATER", "drug": "b", "indication": "x", "date": "2027-01-20",
         "review": "Standard", "source": "s", "outcome": None},
        {"ticker": "FAR", "drug": "c", "indication": "x", "date": "2027-09-01",
         "review": "Standard", "source": "s", "outcome": None},
        {"ticker": "PAST", "drug": "d", "indication": "x", "date": "2026-10-01",
         "review": "Standard", "source": "s", "outcome": None},
        {"ticker": "DONE", "drug": "e", "indication": "x", "date": "2026-09-01",
         "review": "Standard", "source": "s", "outcome": "approved"},
    ])
    sigs, stale = pa.upcoming_fda_signals(today)
    assert sigs["SOON"]["score"] == 0.7                   # inside 30 days
    assert 0.4 <= sigs["LATER"]["score"] < 0.7            # 88 days out
    assert "FAR" not in sigs                              # beyond the 120-day window
    assert len(stale) == 1 and "PAST" in stale[0]         # no outcome recorded


def test_screen_penalty_lowers_rank():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2023-01-02", periods=300)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, (300, 8)), axis=0)),
                      index=idx, columns=[f"T{i}" for i in range(8)])
    base = pa.multi_factor_screen(px, {}, top_n=3, force_custom=False)
    victim = base[0]
    pen = pa.multi_factor_screen(px, {}, top_n=3, force_custom=False,
                                 risk_penalties={victim: {"severity": 1.0}})
    assert pen.index(victim) > 0 if victim in pen else True
