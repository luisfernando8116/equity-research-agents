"""Lot-level tax tracking (Oct 2026: position-level dating called GH's 2026 lots long-term)."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import portfolio_tracker as pt  # noqa: E402
import portfolio_analyzer_advanced as pa  # noqa: E402


def _d(days_ago):
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def _state(trades):
    return {"trades": [{"ticker": "GH", "action": a, "shares": s, "price": p, "date": d}
                       for a, s, p, d in trades]}


def test_lt_date_is_day_after_anniversary():
    assert pt._lt_date("2024-04-01") == "2025-04-02"
    assert pt._lt_date("2024-02-29") == "2025-03-01"           # leap-day purchase


def test_mixed_lots_flagged_and_split():
    pos = pt._compute_positions(_state([("BUY", 10, 20.0, _d(900)), ("BUY", 3, 130.0, _d(120))]))["GH"]
    assert pos["tax_status"] == "MIXED" and pos["lt_shares"] == 10 and pos["st_shares"] == 3


def test_sells_relieve_oldest_lots_first():
    lots = pt._open_lots(_state([("BUY", 10, 20.0, _d(900)), ("BUY", 3, 130.0, _d(120)),
                                 ("SELL", 4, 150.0, _d(10))]))["GH"]
    assert lots[0]["shares"] == 6 and lots[1]["shares"] == 3


def test_new_listing_held_but_unmodelled_is_hold_not_sell():
    profile = {"name": "T", "capital": 0, "existing_shares": {"NEW": 3, "A": 10}}
    plan = pa.build_personalized_plan(profile, {"A": 1.0}, {"NEW": 147.0, "A": 100.0},
                                      too_new={"NEW"})
    rows = {p["Ticker"]: p for p in plan["positions"]}
    assert rows["NEW"]["Action"] == "HOLD — too new to model"


def test_tax_note_recommends_highest_cost_long_term_lot():
    pos = pt._compute_positions(_state([("BUY", 10, 20.0, _d(900)), ("BUY", 5, 49.81, _d(470)),
                                        ("BUY", 2, 130.0, _d(130))]))["GH"]
    note = pa._tax_note("GH", pos, 169.08)
    assert "SHORT-term" in note and "@ $49.81" in note
