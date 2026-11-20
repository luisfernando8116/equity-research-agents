"""
Offline tests for analyst_signals.py and its wiring into the factor screen / DCF.
Frames mirror yfinance's real column names (checked against MSFT, Oct 2026).
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyst_signals as an  # noqa: E402
import portfolio_analyzer_advanced as pa  # noqa: E402

_IDX = ["0q", "+1q", "0y", "+1y"]


def _trend(cur_1y, old_1y):
    return pd.DataFrame({"current": [1, 1, 1, cur_1y], "7daysAgo": [1, 1, 1, cur_1y],
                         "30daysAgo": [1, 1, 1, old_1y], "60daysAgo": [1, 1, 1, old_1y],
                         "90daysAgo": [1, 1, 1, old_1y]}, index=_IDX)


def _revs(up, down):
    # yfinance spells the last column 'downLast7Days' — case-insensitive lookup required
    return pd.DataFrame({"upLast7days": [0] * 4, "upLast30days": [0, 0, up, up],
                         "downLast30days": [0, 0, down, down], "downLast7Days": [0] * 4},
                        index=_IDX)


def _hist(beats):
    est = [1.0] * len(beats)
    act = [1.1 if b else 0.9 for b in beats]
    return pd.DataFrame({"epsActual": act, "epsEstimate": est})


def test_msft_like_inputs():
    m = an.revision_metrics(_trend(23.70, 22.61), _revs(4, 1), _hist([1, 1, 1, 1]))
    assert m["trend_90d"] == pytest.approx(23.70 / 22.61 - 1)
    assert m["breadth"] == pytest.approx((8 - 2) / 10)
    assert m["surprise_hit"] == 1.0 and m["n_quarters"] == 4
    assert 0 < an.revision_score(m) <= 1


def test_downgrades_score_negative():
    m = an.revision_metrics(_trend(9.0, 10.0), _revs(0, 5), _hist([0, 0, 1, 0]))
    assert an.revision_score(m) < -0.5


def test_trend_is_clipped():
    m = an.revision_metrics(_trend(30.0, 10.0), None, None)   # +200% revision
    assert an.revision_score(m) == pytest.approx(1.0)


def test_no_coverage_is_none_not_neutral_number():
    assert an.revision_score(an.revision_metrics(None, None, None)) is None
    assert an.revision_score(an.revision_metrics(pd.DataFrame(), pd.DataFrame(), None)) is None


def test_debias_shrinks_positive_more_at_longer_horizon():
    assert an.debias_growth(0.20, "+1y") < an.debias_growth(0.20, "0y") < 0.20
    assert an.debias_growth(-0.10, "+1y") == -0.10       # documented bias is upward only
    assert an.debias_growth(None) is None


def test_forward_revenue_growth_is_debiased():
    df = pd.DataFrame({"growth": [0.05, 0.06, 0.10, 0.12]}, index=_IDX)
    assert pa.forward_revenue_growth("X", ests=df) == pytest.approx(an.debias_growth(0.12, "+1y"))


def test_consensus_flags():
    c = an.consensus_context({"currentPrice": 100, "targetMeanPrice": 160,
                              "recommendationMean": 1.3, "numberOfAnalystOpinions": 25})
    assert c["upside"] == pytest.approx(0.60)
    assert any("crowded" in f for f in c["flags"])
    assert any("stretched" in f for f in c["flags"])
    quiet = an.consensus_context({"currentPrice": 100, "targetMeanPrice": 110,
                                  "recommendationMean": 2.4, "numberOfAnalystOpinions": 6})
    assert quiet["flags"] == []


def test_fetch_uses_injected_fetchers_and_skips_uncovered():
    frames = {"A": (_trend(11, 10), _revs(3, 0), _hist([1, 1, 0, 1])),
              "B": (None, None, None)}
    attr = {"eps_trend": 0, "eps_revisions": 1, "earnings_history": 2}
    out = an.fetch_revision_signals(["A", "B"], lambda t, a: frames[t][attr[a]],
                                    lambda t: {"currentPrice": 50, "targetMeanPrice": 60})
    assert set(out) == {"A"} and out["A"]["context"]["upside"] == pytest.approx(0.2)


def test_snapshot_roundtrip(tmp_path):
    sig = {"A": {"score": 0.4, "metrics": {"trend_90d": 0.05}}}
    p = an.save_snapshot(sig, {"A": 123.0}, asof=date(2026, 10, 24), directory=tmp_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert p.name == "2026-10-24.json"
    assert data["tickers"]["A"]["price"] == 123.0 and "haircut" in data
    assert an.save_snapshot({}, directory=tmp_path) is None


def test_factor_blend_sums_to_one():
    assert sum(pa.FACTOR_BLEND.values()) == pytest.approx(1.0)


def test_revisions_move_the_screen():
    # Identical price paths → momentum ties; no fundamentals → quality/value tie.
    # Revisions are then the only thing separating the names.
    rng = np.random.default_rng(1)
    path = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    idx = pd.bdate_range("2024-01-01", periods=300)
    px = pd.DataFrame({f"T{i}": path for i in range(10)}, index=idx)
    revs = {"T7": {"score": 0.9}, "T2": {"score": 0.5}, "T5": {"score": 0.1},
            "T0": {"score": -0.8}}
    top = pa.multi_factor_screen(px, {}, top_n=2, force_custom=False, revision_signals=revs)
    assert top == ["T7", "T2"]
    bottom = pa.multi_factor_screen(px, {}, top_n=10, force_custom=False,
                                    revision_signals=revs)[-1]
    assert bottom == "T0"                       # downgrades rank below uncovered (neutral)


def test_implied_eps_growth_is_haircut():
    info = {"forwardEps": 1.15, "trailingEps": 1.0, "industry": "Software"}
    assert pa._sv_growth_rate(info) == pytest.approx(an.debias_growth(0.15, "+1y"))
