"""Offline tests from the RHHBY (Roche ADR) test case, Oct 2026."""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import portfolio_analyzer_advanced as pa  # noqa: E402

ROCHE = {   # real Oct-2026 Yahoo fields: price in USD, fundamentals in CHF
    "currency": "USD", "financialCurrency": "CHF", "currentPrice": 61.17,
    "trailingEps": 2.41, "forwardEps": 3.33733, "trailingPE": 25.38, "forwardPE": 18.33,
    "bookValue": 6.2223, "priceToBook": 9.83, "freeCashflow": 10.318e9,
    "totalDebt": 33.308e9, "totalCash": 9.734e9, "ebitda": 23.964e9,
    "netIncomeToCommon": 12.341e9,
    "sharesOutstanding": 5.5206e9, "sector": "Healthcare",
    "industry": "Drug Manufacturers - General", "beta": 0.327,
    "revenueGrowth": -0.012, "earningsGrowth": -0.077,
}


def test_normalize_currency_fixes_mixed_ratios(monkeypatch):
    monkeypatch.setattr(pa, "fx_rate", lambda a, b: 1.2134)
    out = pa.normalize_currency(ROCHE)
    assert out["trailingPE"] == pytest.approx(20.9, abs=0.1)     # Yahoo showed 25.4
    assert out["forwardPE"] == pytest.approx(15.1, abs=0.1)      # Yahoo showed 18.3
    assert out["freeCashflow"] == pytest.approx(10.318e9 * 1.2134)
    assert "CHF→USD" in out["fx_converted"]
    assert ROCHE["trailingPE"] == 25.38                           # input not mutated


def test_tsm_style_usd_per_share_is_not_double_converted(monkeypatch):
    # Real Oct-2026 TSM: TWD totals, but Yahoo's EPS already in USD per ADR
    tsm = {"currency": "USD", "financialCurrency": "TWD", "currentPrice": 418.51,
           "trailingEps": 15.43, "trailingPE": 27.12, "netIncomeToCommon": 2.5293e12,
           "sharesOutstanding": 5.19e9, "freeCashflow": 1.203e12}
    monkeypatch.setattr(pa, "fx_rate", lambda a, b: 0.0307)
    out = pa.normalize_currency(tsm)
    assert out["trailingPE"] == pytest.approx(27.12, abs=0.1)   # was 883 with blanket conversion
    assert out["freeCashflow"] == pytest.approx(1.203e12 * 0.0307)  # totals still converted


def test_roche_per_share_detected_as_chf(monkeypatch):
    monkeypatch.setattr(pa, "fx_rate", lambda a, b: 1.2134)
    info = {**ROCHE, "netIncomeToCommon": 12.341e9}
    assert pa.normalize_currency(info)["trailingPE"] == pytest.approx(20.9, abs=0.1)


def test_same_currency_untouched():
    info = {"currency": "USD", "financialCurrency": "USD", "trailingPE": 20}
    assert pa.normalize_currency(info) is info


def test_missing_fx_rate_warns(monkeypatch):
    monkeypatch.setattr(pa, "fx_rate", lambda a, b: None)
    assert "fx_warning" in pa.normalize_currency(ROCHE)


def test_subject_rule():
    assert pa._is_subject("Roche presents interim phase III data showing…", "roche")
    assert pa._is_subject("Shares jump after FDA nod", "roche")             # untagged → ok
    assert not pa._is_subject("GSK's two FDA approvals in one week, Roche's $15B bet", "roche")
    assert not pa._is_subject("Jefferies raises Zymeworks stock price target on Roche deal",
                              "roche")


def test_forward_revenue_growth_preferred_over_trailing_fx_distorted():
    ests = pd.DataFrame({"growth": [0.0288, 0.0462]}, index=["0y", "+1y"])
    # +1y picked over 0y, then the analyst-optimism haircut applies
    from analyst_signals import debias_growth
    assert pa.forward_revenue_growth("X", ests) == pytest.approx(debias_growth(0.0462, "+1y"))
    info = {**ROCHE, "revenueGrowthFwd": 0.0462}
    assert pa._sv_growth_rate(info) > 0                           # trailing −1.2% gave 0


def test_dcf_scenarios_do_not_collapse_at_zero_growth():
    info = {**ROCHE, "revenueGrowth": -0.05, "revenueGrowthFwd": None}
    v = pa.valuation_signal("RHHBY", info)
    assert v["buy_low"] is not None and v["buy_high"] is not None
    assert v["buy_high"] > v["buy_low"]                          # was identical $41.36


def test_roman_numeral_phase_iii_scores(monkeypatch):
    now = datetime.now()
    monkeypatch.setattr(pa, "news_items", lambda t: [
        ("Roche presents interim phase III data showing sefaxersen significantly reduced "
         "proteinuria", "", now)])
    monkeypatch.setattr(pa, "_safe_info", lambda t: {"longName": "Roche Holding AG"})
    monkeypatch.setattr(pa, "_safe_attr", lambda t, a: None)
    sig = pa.fetch_catalyst_signals(["RHHBY"]).get("RHHBY")
    assert sig and sig["catalyst_type"] == "Clinical Trial" and sig["score"] > 0
