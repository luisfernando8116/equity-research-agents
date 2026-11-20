"""Offline tests for scan.py trading-activity signals."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scan  # noqa: E402


def _history(heavy_selling: bool):
    idx = pd.bdate_range("2026-05-01", periods=130)
    close = np.linspace(100, 120, 130)
    vol = np.full(130, 1_000_000.0)
    if heavy_selling:
        close[-20:] = np.linspace(120, 90, 20) + np.tile([0, 1.5], 10)   # mostly down days
        vol[-20:] = 1_000_000.0
        down = np.diff(close[-21:]) < 0
        vol[-20:][down] = 3_000_000.0                                    # heavy down-day volume
    return pd.DataFrame({"Close": close, "Volume": vol}, index=idx)


def _patch(monkeypatch, insider_value=0.0, short=0.02, heavy=False):
    it = pd.DataFrame({"Start Date": [pd.Timestamp.now() - pd.Timedelta(days=10)],
                       "Text": ["Purchase at price 50.00 per share."],
                       "Value": [insider_value]})
    monkeypatch.setattr(scan.pa, "_safe_attr", lambda t, a: it if a == "insider_transactions" else None)
    monkeypatch.setattr(scan.pa, "_safe_info", lambda t: {"shortPercentOfFloat": short, "shortRatio": 2.0})
    monkeypatch.setattr(scan.pa, "_call_with_timeout", lambda fn: _history(heavy))


def test_all_three_signals_fire(monkeypatch):
    _patch(monkeypatch, insider_value=250_000, short=0.15, heavy=True)
    a = scan.activity_signals("X")
    assert "250,000" in a["insider_buy"]
    assert "15%" in a["short_warn"]
    assert "staging" in a["selling_pressure"]


def test_quiet_stock_has_no_flags(monkeypatch):
    _patch(monkeypatch, insider_value=10_000, short=0.02, heavy=False)
    a = scan.activity_signals("X")
    assert a == {"insider_buy": None, "short_warn": None, "selling_pressure": None}
    assert "no notable" in scan._activity_text(a)
