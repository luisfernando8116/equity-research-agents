"""
Portfolio Tracker — Companion to portfolio_analyzer_advanced.py
================================================================
Records actual trades and capital additions, tracks cost basis and unrealized
P&L, and surfaces next-action signals (hold / add capital / rebalance).

Supports multiple investor profiles. State files:
  portfolio_state_<profile>.json   (e.g. portfolio_state_fresh.json) — git-ignored
Default profile: "default"

Usage
-----
  python portfolio_tracker.py [-p PROFILE] status
  python portfolio_tracker.py [-p PROFILE] log buy  MSFT 5  420.00  ["note"]
  python portfolio_tracker.py [-p PROFILE] log buy  SPY  10 500.00  --date 2024-04-15  ["note"]
  python portfolio_tracker.py [-p PROFILE] log sell MSFT 2  450.00
  python portfolio_tracker.py [-p PROFILE] add-capital 5000          ["note"]
  python portfolio_tracker.py [-p PROFILE] set-target HighConviction [weights.json]
  python portfolio_tracker.py [-p PROFILE] next-action
  python portfolio_tracker.py [-p PROFILE] history [TICKER]
  python portfolio_tracker.py [-p PROFILE] export
  python portfolio_tracker.py profiles     ← list all tracked profiles

Examples
--------
  # A fresh portfolio
  python portfolio_tracker.py -p fresh add-capital 10000 "Initial deployment"
  python portfolio_tracker.py -p fresh log buy MSFT 5 420.00

  # An existing portfolio (purchase dates give accurate short/long-term tax status)
  python portfolio_tracker.py -p existing log buy SPY 10 450.00 --date 2024-04-01 "Existing position"
  python portfolio_tracker.py -p existing status
"""

import sys
import subprocess

REQUIRED = {"yfinance": "yfinance", "pandas": "pandas"}
for mod, pkg in REQUIRED.items():
    try:
        __import__(mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import json
import uuid
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import yfinance as yf
import pandas as pd

warnings.filterwarnings("ignore")

# ─── STATE FILE ───────────────────────────────────────────────────────────────

_PROFILE_NAME = "default"   # overridden by -p flag in main()

def _state_file(profile: str = "") -> Path:
    name = profile or _PROFILE_NAME
    if name == "default":
        return Path(__file__).parent / "portfolio_state.json"
    return Path(__file__).parent / f"portfolio_state_{name}.json"

STATE_FILE = _state_file()   # resolved after arg parsing; kept for back-compat

_EMPTY_STATE = {
    "version":          2,
    "target_scenario":  "",
    "target_weights":   {},
    "initial_investment": 15000,
    "dca_interval_days":  30,
    "drift_alert_pct":    0.04,
    "trades":           [],
    "capital_events":   [],
}


def _load(profile: str = "") -> dict:
    sf = _state_file(profile)
    if sf.exists():
        with open(sf, encoding="utf-8") as f:
            state = json.load(f)
        for k, v in _EMPTY_STATE.items():
            state.setdefault(k, v)
        return state
    return dict(_EMPTY_STATE)


def _save(state: dict, profile: str = "") -> None:
    sf = _state_file(profile)
    with open(sf, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print(f"  ✓ State saved → {sf}")


# ─── PRICE FETCH ──────────────────────────────────────────────────────────────

def _current_prices(tickers: list) -> dict:
    """Fetch latest price for each ticker via yfinance fast_info."""
    def _get(t):
        try:
            return t, yf.Ticker(t).fast_info.last_price
        except Exception:
            return t, None
    with ThreadPoolExecutor(max_workers=10) as pool:
        return dict(pool.map(_get, tickers))


# ─── PORTFOLIO CALCULATIONS ───────────────────────────────────────────────────

def _lt_date(bought: str) -> str:
    """First day a sale counts as LONG-term: the day after the 1-year anniversary
    (IRS: held 'more than one year')."""
    d = datetime.strptime(bought, "%Y-%m-%d")
    try:
        anniv = d.replace(year=d.year + 1)
    except ValueError:                       # Feb 29 purchase
        anniv = d.replace(year=d.year + 1, day=28)
    return (anniv + timedelta(days=1)).strftime("%Y-%m-%d")


def _holding_period(first_bought: str) -> tuple[int, str]:
    """Returns (days_held, 'LT'|'ST') for ONE purchase date."""
    try:
        bought = datetime.strptime(first_bought, "%Y-%m-%d")
        days   = (datetime.now() - bought).days
        lt     = datetime.now().strftime("%Y-%m-%d") >= _lt_date(first_bought)
        return days, "LT" if lt else "ST"
    except Exception:
        return 0, "?"


def _open_lots(state: dict) -> dict:
    """
    Open tax lots per ticker, FIFO-relieved by sells (the IRS default unless you
    designate specific lots with your broker).
    Returns {ticker: [{date, shares, price, cost, lt_date, term}]}.
    Position-level holding periods are wrong for mixed lots: GH bought 2024 + 2026
    would show all shares long-term when the 2026 lots are short-term.
    """
    lots: dict = {}
    for tr in sorted(state["trades"], key=lambda x: x.get("date", "")):
        t, sh, px = tr["ticker"], float(tr["shares"]), float(tr["price"])
        if tr["action"].upper() == "BUY":
            lots.setdefault(t, []).append({"date": tr.get("date", ""), "shares": sh, "price": px})
        elif tr["action"].upper() == "SELL":
            remaining = sh
            for lot in lots.get(t, []):
                take = min(lot["shares"], remaining)
                lot["shares"] -= take
                remaining -= take
                if remaining <= 1e-9:
                    break
    out = {}
    for t, ls in lots.items():
        open_ls = []
        for lot in ls:
            if lot["shares"] > 1e-6:
                term = _holding_period(lot["date"])[1] if lot["date"] else "?"
                open_ls.append({**lot, "cost": round(lot["shares"] * lot["price"], 2),
                                "lt_date": _lt_date(lot["date"]) if lot["date"] else "?",
                                "term": term})
        if open_ls:
            out[t] = open_ls
    return out


def _compute_positions(state: dict) -> dict:
    """
    Derives current positions from trade log.
    Returns {ticker: {shares, cost_basis, avg_price, first_bought, days_held, tax_status}}
    """
    positions: dict = {}
    for trade in state["trades"]:
        t      = trade["ticker"]
        shares = float(trade["shares"])
        price  = float(trade["price"])
        action = trade["action"].upper()
        date   = trade.get("date", "")
        if t not in positions:
            positions[t] = {"shares": 0.0, "total_cost": 0.0, "first_bought": date}
        if action == "BUY":
            positions[t]["shares"]     += shares
            positions[t]["total_cost"] += shares * price
            # Track the earliest buy date
            if date and (not positions[t]["first_bought"] or date < positions[t]["first_bought"]):
                positions[t]["first_bought"] = date
        elif action == "SELL":
            if positions[t]["shares"] > 0:
                ratio = min(shares / positions[t]["shares"], 1.0)
                positions[t]["total_cost"] -= positions[t]["total_cost"] * ratio
            positions[t]["shares"] = max(0.0, positions[t]["shares"] - shares)

    lots_by_ticker = _open_lots(state)
    result = {}
    for t, pos in positions.items():
        if pos["shares"] > 0.001:
            avg_px = pos["total_cost"] / pos["shares"] if pos["shares"] > 0 else 0.0
            days, _ = _holding_period(pos["first_bought"])
            lots   = lots_by_ticker.get(t, [])
            lt_sh  = sum(l["shares"] for l in lots if l["term"] == "LT")
            st_sh  = sum(l["shares"] for l in lots if l["term"] == "ST")
            tax_st = "LT" if st_sh == 0 else ("ST" if lt_sh == 0 else "MIXED")
            result[t] = {
                "shares":      round(pos["shares"], 4),
                "cost_basis":  round(pos["total_cost"], 2),
                "avg_price":   round(avg_px, 4),
                "first_bought": pos["first_bought"],
                "days_held":   days,
                "tax_status":  tax_st,          # per-lot aware (was earliest-lot only)
                "lt_shares":   round(lt_sh, 4),
                "st_shares":   round(st_sh, 4),
                "lots":        lots,
            }
    return result


def _total_capital(state: dict) -> float:
    return sum(float(e["amount"]) for e in state["capital_events"]
               if e["type"] == "deposit")


def _total_withdrawn(state: dict) -> float:
    return sum(float(e["amount"]) for e in state["capital_events"]
               if e["type"] == "withdrawal")


# ─── COMMANDS ────────────────────────────────────────────────────────────────

def cmd_status(state: dict) -> None:
    positions = _compute_positions(state)
    if not positions:
        print("  No positions recorded yet. Use: python portfolio_tracker.py log buy TICKER SHARES PRICE")
        return

    tickers       = list(positions.keys())
    prices        = _current_prices(tickers)
    total_cost    = sum(p["cost_basis"] for p in positions.values())
    total_mktval  = sum(
        positions[t]["shares"] * (prices.get(t) or positions[t]["avg_price"])
        for t in tickers
    )
    total_unrealised = total_mktval - total_cost
    total_cap    = _total_capital(state)
    target_w     = state.get("target_weights", {})
    scenario     = state.get("target_scenario", "—")

    print("=" * 110)
    print(f"PORTFOLIO STATUS  —  Target: {scenario}")
    print(f"Capital deployed: ${total_cap:,.2f}  |  "
          f"Market value: ${total_mktval:,.2f}  |  "
          f"Unrealised P&L: ${total_unrealised:+,.2f} "
          f"({total_unrealised / total_cost * 100:+.1f}%)")
    print("=" * 110)
    print(f"\n  {'Ticker':<8} {'Shares':>8}  {'Avg Cost':>10}  "
          f"{'Cur Price':>10}  {'Mkt Val':>10}  {'Unreal P&L':>12}  "
          f"{'Wt%':>5}  {'Target%':>7}  {'Drift':>7}  {'Held':>6}  {'Tax':>4}")
    print("  " + "─" * 106)

    for t in sorted(tickers, key=lambda x: -(positions[x]["shares"]
                                              * (prices.get(x) or positions[x]["avg_price"]))):
        pos   = positions[t]
        px    = prices.get(t) or pos["avg_price"]
        mval  = pos["shares"] * px
        pl    = mval - pos["cost_basis"]
        wt    = mval / total_mktval if total_mktval > 0 else 0.0
        tgt_w = target_w.get(t, 0.0)
        drift = wt - tgt_w
        pl_str    = f"${pl:+,.2f}"
        drift_str = f"{drift:+.1%}" if tgt_w else "  n/a"
        alert     = " ⚠" if abs(drift) >= state.get("drift_alert_pct", 0.04) and tgt_w else "   "
        days_h    = pos.get("days_held", 0)
        held_str  = f"{days_h}d" if days_h else "—"
        tax_st    = pos.get("tax_status", "?")
        print(f"  {t:<8} {pos['shares']:>8.3f}  ${pos['avg_price']:>9,.2f}  "
              f"${px:>9,.2f}  ${mval:>9,.2f}  {pl_str:>12}  "
              f"{wt:>5.1%}  {tgt_w:>7.1%}  {drift_str:>7}{alert}  "
              f"{held_str:>6}  {tax_st:>4}")

    print("  " + "─" * 106)
    print(f"  {'TOTAL':<8} {'':>8}  {'':>10}  {'':>10}  "
          f"${total_mktval:>9,.2f}  ${total_unrealised:>+10,.2f}")
    print(f"  Tax: LT = long-term (held ≥ 1yr, lower rate)  |  ST = short-term")

    # Drift summary
    drifters = [
        (t, positions[t], prices.get(t) or positions[t]["avg_price"],
         (positions[t]["shares"] * (prices.get(t) or positions[t]["avg_price"]))
         / total_mktval - target_w.get(t, 0.0))
        for t in tickers
        if target_w.get(t, 0.0) and
        abs((positions[t]["shares"] * (prices.get(t) or positions[t]["avg_price"]))
            / total_mktval - target_w.get(t, 0.0)) >= state.get("drift_alert_pct", 0.04)
    ]
    if drifters:
        print(f"\n  ⚠  Rebalance alerts (drift ≥ {state.get('drift_alert_pct', 0.04):.0%}):")
        for t, pos, px, drift in sorted(drifters, key=lambda x: -abs(x[3])):
            action = "TRIM" if drift > 0 else "ADD"
            print(f"     {t:<7}  drift {drift:+.1%}  → {action}")

    # Last capital event
    if state["capital_events"]:
        last = state["capital_events"][-1]
        print(f"\n  Last capital event: {last['type'].upper()} ${float(last['amount']):,.2f}"
              f"  on {last['date']}"
              + (f"  ({last['note']})" if last.get("note") else ""))


def cmd_log_trade(state: dict, action: str, ticker: str,
                  shares: float, price: float, note: str = "",
                  trade_date: str = "") -> None:
    ticker = ticker.upper()
    # Validate optional date
    date = datetime.now().strftime("%Y-%m-%d")
    if trade_date:
        try:
            datetime.strptime(trade_date, "%Y-%m-%d")
            date = trade_date
        except ValueError:
            print(f"  ⚠ Invalid --date format '{trade_date}' — use YYYY-MM-DD. Using today.")
    days, tax_st = _holding_period(date)
    trade  = {
        "id":     str(uuid.uuid4())[:8],
        "date":   date,
        "ticker": ticker,
        "shares": round(shares, 4),
        "price":  round(price, 4),
        "action": action.upper(),
        "note":   note,
    }
    state["trades"].append(trade)
    cost = shares * price
    date_note = f"  (bought {date}, held {days}d → {tax_st})" if trade_date else ""
    print(f"  ✓ Logged {action.upper()} {shares} × {ticker} @ ${price:.2f} = ${cost:,.2f}{date_note}")


def cmd_add_capital(state: dict, amount: float, note: str = "") -> None:
    event = {
        "date":   datetime.now().strftime("%Y-%m-%d"),
        "amount": round(amount, 2),
        "type":   "deposit",
        "note":   note,
    }
    state["capital_events"].append(event)
    total = _total_capital(state)
    print(f"  ✓ Recorded capital addition: ${amount:,.2f}  (total deployed: ${total:,.2f})")


def cmd_set_target(state: dict, scenario: str,
                   weights_file: str | None = None) -> None:
    """
    Set the target scenario name. Optionally load weights from a JSON file
    exported by the analyzer (format: {"TICKER": weight_float, ...}).
    If no weights file, checks portfolio_state.json for a prior export.
    """
    state["target_scenario"] = scenario
    if weights_file:
        p = Path(weights_file)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                state["target_weights"] = json.load(f)
            print(f"  ✓ Loaded {len(state['target_weights'])} target weights from {weights_file}")
        else:
            print(f"  ⚠ Weights file not found: {weights_file}")
    print(f"  ✓ Target scenario set to: {scenario}")


def cmd_next_action(state: dict) -> None:
    """
    Analyses the current portfolio state and recommends the next action:
    HOLD / ADD CAPITAL (DCA) / REBALANCE / BOTH
    """
    positions    = _compute_positions(state)
    target_w     = state.get("target_weights", {})
    dca_interval = int(state.get("dca_interval_days", 30))

    # Last capital event date
    last_cap_date = None
    if state["capital_events"]:
        try:
            last_cap_date = datetime.strptime(
                state["capital_events"][-1]["date"], "%Y-%m-%d")
        except Exception:
            pass

    days_since_last = (
        (datetime.now() - last_cap_date).days
        if last_cap_date else None
    )

    tickers    = list(positions.keys())
    prices     = _current_prices(tickers) if tickers else {}
    total_mktval = sum(
        positions[t]["shares"] * (prices.get(t) or positions[t]["avg_price"])
        for t in tickers
    )

    # Drift check
    drifters = []
    if target_w and total_mktval > 0:
        for t in tickers:
            px    = prices.get(t) or positions[t]["avg_price"]
            wt    = positions[t]["shares"] * px / total_mktval
            tgt   = target_w.get(t, 0.0)
            drift = wt - tgt
            if abs(drift) >= state.get("drift_alert_pct", 0.04):
                drifters.append((t, drift))

    # Also flag target tickers with zero position
    missing = [t for t in target_w if t not in positions and target_w[t] >= 0.02]

    print("=" * 70)
    print("NEXT ACTION RECOMMENDATION")
    print("=" * 70)

    actions = []

    if days_since_last is not None and days_since_last >= dca_interval:
        actions.append("ADD CAPITAL")
        print(f"\n  ► ADD CAPITAL")
        print(f"    {days_since_last} days since last capital event "
              f"(DCA interval: {dca_interval} days).")
        total_cap = _total_capital(state)
        print(f"    Total deployed so far: ${total_cap:,.2f}")
        if state.get("initial_investment"):
            per_tranche = state["initial_investment"] / 3
            print(f"    Suggested next tranche: ${per_tranche:,.0f}")
    elif days_since_last is None and not positions:
        actions.append("INITIAL DEPLOYMENT")
        print(f"\n  ► INITIAL DEPLOYMENT")
        print(f"    No trades recorded yet. Run the analyzer, pick a scenario,")
        print(f"    then log your first purchases with: "
              f"python portfolio_tracker.py log buy TICKER SHARES PRICE")
    else:
        remaining = dca_interval - (days_since_last or 0)
        print(f"\n  ✓ Capital: no new tranche needed "
              f"(next in ~{remaining} days)")

    if drifters:
        actions.append("REBALANCE")
        print(f"\n  ► REBALANCE ({len(drifters)} position(s) drifted ≥ "
              f"{state.get('drift_alert_pct', 0.04):.0%}):")
        for t, drift in sorted(drifters, key=lambda x: -abs(x[1])):
            act = "TRIM" if drift > 0 else "ADD"
            print(f"    {t:<7}  current drift {drift:+.1%}  → {act}")
    else:
        print(f"\n  ✓ Drift: all positions within ±"
              f"{state.get('drift_alert_pct', 0.04):.0%} of target")

    if missing:
        print(f"\n  ► MISSING POSITIONS ({len(missing)} target tickers not yet held):")
        for t in missing:
            print(f"    {t:<7}  target {target_w[t]:.1%} — not yet bought")

    if not actions:
        print("\n  ✓ HOLD — portfolio is on track. No action needed today.")

    if state["target_scenario"]:
        print(f"\n  Target scenario: {state['target_scenario']}")
    print(f"  Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


def cmd_history(state: dict, ticker: str | None = None) -> None:
    trades = state["trades"]
    if ticker:
        trades = [t for t in trades if t["ticker"].upper() == ticker.upper()]
    if not trades:
        print("  No trades found.")
        return

    print(f"\n  {'Date':<12} {'Action':<6} {'Ticker':<8} {'Shares':>8}  "
          f"{'Price':>10}  {'Total':>10}  Note")
    print("  " + "─" * 72)
    total_spent = total_received = 0.0
    for t in trades:
        cost = float(t["shares"]) * float(t["price"])
        sign = "+" if t["action"] == "SELL" else "-"
        if t["action"] == "BUY":
            total_spent += cost
        else:
            total_received += cost
        print(f"  {t['date']:<12} {t['action']:<6} {t['ticker']:<8} "
              f"{float(t['shares']):>8.3f}  ${float(t['price']):>9,.2f}  "
              f"{sign}${cost:>8,.2f}  {t.get('note','')}")
    print("  " + "─" * 72)
    print(f"  Total spent: ${total_spent:,.2f}  |  "
          f"Total received: ${total_received:,.2f}  |  "
          f"Net: ${total_received - total_spent:+,.2f}")


def cmd_export(state: dict, profile: str = "") -> None:
    """
    Exports current holdings as JSON (loaded by the analyzer as CURRENT_HOLDINGS)
    and as an Excel file for records.
    """
    positions = _compute_positions(state)
    tickers   = list(positions.keys())
    prices    = _current_prices(tickers) if tickers else {}
    pfx       = f"_{profile}" if profile and profile != "default" else ""

    # JSON for CURRENT_HOLDINGS integration
    holdings_json = {
        t: round(positions[t]["shares"] * (prices.get(t) or positions[t]["avg_price"]), 2)
        for t in tickers
    }
    out_json = Path(__file__).parent / f"current_holdings_export{pfx}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(holdings_json, f, indent=2)
    print(f"  ✓ Holdings JSON → {out_json}")

    if state.get("target_weights"):
        out_tw = Path(__file__).parent / f"target_weights_export{pfx}.json"
        with open(out_tw, "w", encoding="utf-8") as f:
            json.dump(state["target_weights"], f, indent=2)
        print(f"  ✓ Target weights JSON → {out_tw}")

    rows = []
    total_mktval = sum(
        positions[t]["shares"] * (prices.get(t) or positions[t]["avg_price"])
        for t in tickers
    )
    for t in tickers:
        px   = prices.get(t) or positions[t]["avg_price"]
        mval = positions[t]["shares"] * px
        pl   = mval - positions[t]["cost_basis"]
        rows.append({
            "Ticker":             t,
            "Shares":             positions[t]["shares"],
            "Avg Cost ($)":       positions[t]["avg_price"],
            "Current Price ($)":  round(px, 2),
            "Market Value ($)":   round(mval, 2),
            "Cost Basis ($)":     positions[t]["cost_basis"],
            "Unrealised P&L ($)": round(pl, 2),
            "Weight (%)":         f"{mval / total_mktval:.1%}" if total_mktval else "—",
            "Target Weight (%)":  f"{state['target_weights'].get(t, 0):.1%}",
        })
    df  = pd.DataFrame(rows)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_xl = Path(__file__).parent / f"portfolio_snapshot{pfx}_{ts}.xlsx"
    df.to_excel(out_xl, index=False)
    print(f"  ✓ Excel snapshot → {out_xl}")


def cmd_profiles() -> None:
    """List all tracked investor profiles (state files in this directory)."""
    base = Path(__file__).parent
    files = list(base.glob("portfolio_state*.json"))
    if not files:
        print("  No portfolio state files found.")
        return
    print(f"\n  {'Profile':<20}  {'Trades':>6}  {'Capital Events':>14}  State File")
    print("  " + "─" * 70)
    for sf in sorted(files):
        try:
            with open(sf, encoding="utf-8") as f:
                s = json.load(f)
            # Derive profile name from filename
            name = sf.stem.replace("portfolio_state_", "").replace("portfolio_state", "default")
            ntrades = len(s.get("trades", []))
            nevents = len(s.get("capital_events", []))
            scenario = s.get("target_scenario", "—")
            print(f"  {name:<20}  {ntrades:>6}  {nevents:>14}  {sf.name}  [{scenario}]")
        except Exception:
            print(f"  (unreadable) {sf.name}")


# ─── CLI ENTRY POINT ─────────────────────────────────────────────────────────

def _usage():
    print(__doc__)


def main():
    global _PROFILE_NAME, STATE_FILE
    args = sys.argv[1:]

    # Parse optional -p / --profile flag (must come before the command)
    profile = "default"
    if args and args[0] in ("-p", "--profile") and len(args) > 1:
        profile = args[1].lower()
        args    = args[2:]
    _PROFILE_NAME = profile
    STATE_FILE    = _state_file(profile)

    if not args or args[0] in ("-h", "--help", "help"):
        _usage()
        return

    cmd   = args[0].lower()
    state = _load(profile)

    if profile != "default":
        print(f"  Profile: {profile}  |  State: {STATE_FILE.name}")

    if cmd == "status":
        cmd_status(state)

    elif cmd == "log":
        if len(args) < 5:
            print("Usage: python portfolio_tracker.py log buy|sell TICKER SHARES PRICE [--date YYYY-MM-DD] [note]")
            return
        action = args[1]
        ticker = args[2]
        try:
            shares = float(args[3])
            price  = float(args[4])
        except ValueError:
            print("SHARES and PRICE must be numbers.")
            return
        # Extract optional --date flag from remaining args
        rest       = args[5:]
        trade_date = ""
        note_parts = []
        i = 0
        while i < len(rest):
            if rest[i] == "--date" and i + 1 < len(rest):
                trade_date = rest[i + 1]
                i += 2
            else:
                note_parts.append(rest[i])
                i += 1
        note = " ".join(note_parts)
        cmd_log_trade(state, action, ticker, shares, price, note, trade_date)
        _save(state, profile)

    elif cmd == "add-capital":
        if len(args) < 2:
            print("Usage: python portfolio_tracker.py [-p PROFILE] add-capital AMOUNT [note]")
            return
        try:
            amount = float(args[1])
        except ValueError:
            print("AMOUNT must be a number.")
            return
        note = " ".join(args[2:]) if len(args) > 2 else ""
        cmd_add_capital(state, amount, note)
        _save(state, profile)

    elif cmd == "set-target":
        if len(args) < 2:
            print("Usage: python portfolio_tracker.py [-p PROFILE] set-target SCENARIO [weights.json]")
            return
        scenario     = args[1]
        weights_file = args[2] if len(args) > 2 else None
        cmd_set_target(state, scenario, weights_file)
        _save(state, profile)

    elif cmd == "next-action":
        cmd_next_action(state)

    elif cmd == "history":
        ticker = args[1].upper() if len(args) > 1 else None
        cmd_history(state, ticker)

    elif cmd == "export":
        cmd_export(state, profile)

    elif cmd == "profiles":
        cmd_profiles()

    else:
        print(f"Unknown command: {cmd}")
        _usage()


if __name__ == "__main__":
    main()
