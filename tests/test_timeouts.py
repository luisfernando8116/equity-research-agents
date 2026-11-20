"""A stalled Yahoo call must not freeze the run (Oct-2026: a run hung for 6 days)."""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import portfolio_analyzer_advanced as pa  # noqa: E402


def test_call_with_timeout_gives_up_quickly():
    t0 = time.time()
    with pytest.raises(pa.YahooTimeout):
        pa._call_with_timeout(lambda: time.sleep(30), timeout=0.5)
    assert time.time() - t0 < 3


def test_call_with_timeout_passes_value_and_errors_through():
    assert pa._call_with_timeout(lambda: 42, timeout=2) == 42
    with pytest.raises(ValueError):
        pa._call_with_timeout(lambda: (_ for _ in ()).throw(ValueError("x")), timeout=2)


def test_safe_info_returns_empty_on_hang(monkeypatch):
    monkeypatch.setattr(pa, "YF_CALL_TIMEOUT", 0.5)
    monkeypatch.setattr(pa, "_yf_cache_load", lambda t: None)

    class _Hang:
        def __init__(self, *a, **k): pass
        @property
        def info(self):
            time.sleep(30)

    monkeypatch.setattr(pa.yf, "Ticker", _Hang)
    # default arg captured at def time → patch the module constant used inside
    orig = pa._call_with_timeout
    monkeypatch.setattr(pa, "_call_with_timeout", lambda fn, timeout=0.5: orig(fn, timeout))
    t0 = time.time()
    assert pa._safe_info("ZZZZ") == {}
    assert time.time() - t0 < 5
