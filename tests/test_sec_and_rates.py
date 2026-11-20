"""Offline tests for sec_edgar.py and the FRED rate refresh."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sec_edgar as se  # noqa: E402
import portfolio_analyzer_advanced as pa  # noqa: E402

TODAY = datetime.now().strftime("%Y-%m-%d")
OLD   = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")

FAKE = {
    "company_tickers": {"0": {"cik_str": 123, "ticker": "ABC"}},
    "sub_123": {"filings": {"recent": {
        "form":            ["8-K", "8-K", "10-Q", "8-K"],
        "filingDate":      [TODAY, TODAY, TODAY, OLD],
        "items":           ["2.02,9.01", "4.02", "", "1.03"],
        "accessionNumber": ["a-1", "a-2", "a-3", "a-4"],
        "primaryDocument": ["x.htm", "y.htm", "z.htm", "w.htm"],
    }}},
    "facts_123": {"facts": {"us-gaap": {
        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
            {"form": "10-K", "fp": "FY", "start": f"{y}-01-01", "end": f"{y}-12-31",
             "val": v, "filed": f"{y + 1}-02-15"} for y, v in [(2023, 100), (2024, 120), (2025, 140)]
        ] + [{"form": "10-Q", "fp": "Q1", "start": "2025-01-01", "end": "2025-03-31",
              "val": 999, "filed": "2025-05-01"}]}},
        "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [
            {"form": "10-K", "fp": "FY", "start": f"{y}-01-01", "end": f"{y}-12-31",
             "val": 20, "filed": f"{y + 1}-02-15"} for y in (2023, 2024, 2025)]}},
    }}},
}


@pytest.fixture
def fake_sec(monkeypatch):
    monkeypatch.setattr(se, "SEC_USER_AGENT", "Test test@example.com")
    monkeypatch.setattr(se, "_get_json", lambda url, key, ttl: FAKE.get(key))


def test_disabled_without_contact(monkeypatch):
    monkeypatch.setattr(se, "SEC_USER_AGENT", "")
    assert not se.enabled() and se.recent_8k("ABC") == [] and se.fcf_avg("ABC") is None


def test_recent_8k_window_and_meanings(fake_sec):
    f = se.recent_8k("ABC", days=30)
    assert len(f) == 2                                   # 10-Q and the 200-day-old 8-K excluded
    assert "Results of operations" in f[0]["meaning"]


def test_red_flag_picks_worst_recent_item(fake_sec):
    sig = se.red_flag_signal("ABC", days=60)
    assert sig["catalyst_type"] == "SEC 8-K 4.02" and sig["score"] == -0.8   # 1.03 is too old


def test_fcf_from_10k_full_years_only(fake_sec):
    assert se.fcf_avg("ABC") == pytest.approx(((100 - 20) + (120 - 20) + (140 - 20)) / 3)


def test_fcf_ignores_stale_years_when_capex_tag_changed(monkeypatch):
    y0 = datetime.now().year
    fy = lambda y, v: {"form": "10-K", "fp": "FY", "start": f"{y}-01-01",
                       "end": f"{y}-12-31", "val": v, "filed": f"{y + 1}-02-15"}
    facts = {"facts": {"us-gaap": {
        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
            fy(2010, 5), fy(2011, 6), fy(2012, 7),
            fy(y0 - 2, 600), fy(y0 - 1, 800), fy(y0, 1000)]}},
        "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [
            fy(2010, 1), fy(2011, 1), fy(2012, 1)]}},            # old tag, stopped
        "PaymentsToAcquireProductiveAssets": {"units": {"USD": [
            fy(y0 - 2, 100), fy(y0 - 1, 100), fy(y0, 100)]}},     # new tag
    }}}
    monkeypatch.setattr(se, "SEC_USER_AGENT", "T t@example.com")
    monkeypatch.setattr(se, "_get_json", lambda url, key, ttl:
                        {"0": {"cik_str": 1, "ticker": "NV"}} if key == "company_tickers" else facts)
    assert se.fcf_avg("NV") == pytest.approx((500 + 700 + 900) / 3)   # not the 2010-12 years


def test_fred_failure_keeps_defaults(monkeypatch):
    import requests
    monkeypatch.setattr(pa, "RISK_FREE_RATE", 0.05)
    monkeypatch.setattr(pa, "CASH_RATE", 0.045)
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    msg = pa.refresh_market_rates()
    assert "unreachable" in msg and pa.RISK_FREE_RATE == 0.05 and pa.CASH_RATE == 0.045
