"""
FDA / Catalyst Event Study — Pre/Post Approval Price Analysis
=============================================================
Downloads price history for one or more (ticker, event_date) pairs and
computes cumulative returns at standard windows relative to the event.
Useful for anchoring Black-Litterman expected-return views to empirical comps.

Usage
-----
  python fda_event_study.py --gh-comps
      Run the pre-built Guardant Health diagnostic-oncology comp set.

  python fda_event_study.py EXAS 2014-08-11
      Single ticker + event date (YYYY-MM-DD).

  python fda_event_study.py EXAS 2014-08-11 "Cologuard approval" NTRA 2021-03-15 "Signatera MRD"
      Multiple events. Optional label after each date.

Output
------
  Console summary table + calibration suggestion for GH BL view
  event_study_<ts>.png   (indexed price chart + return heatmap)
  event_study_<ts>.html  (full interactive report)
  Both saved to Financial_analysis/outputs/

Notes on comp dates
-------------------
  Always verify event dates against SEC filings or FDA press releases before
  using them for investment decisions. Approximate dates are flagged with (*).
"""

import sys
import subprocess

REQUIRED = {
    "yfinance": "yfinance",
    "pandas":   "pandas",
    "numpy":    "numpy",
    "matplotlib": "matplotlib",
}
for mod, pkg in REQUIRED.items():
    try:
        __import__(mod)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── CONFIG ────────────────────────────────────────────────────────────────────

# Trading-day offsets relative to event day (negative = before, positive = after)
WINDOWS_PRE  = [-252, -180, -90, -60, -30, -10, -5, -1]
WINDOWS_POST = [+1,   +5,  +10, +30, +60, +90, +180, +252]
ALL_WINDOWS  = WINDOWS_PRE + WINDOWS_POST

BENCHMARK    = "IBB"    # iShares Biotechnology ETF — closest sector for diagnostic oncology
CHART_RANGE  = 252      # trading days before and after event to plot

# Pre-built RVMD therapeutic oncology comps
# Focus: small-to-mid cap oncology drugs with strong survival benefit in high-unmet-need cancers.
# These calibrate the post-approval commercial ramp trajectory — different from diagnostic comps.
# (*) = approximate date — verify against FDA press release before use.
RVMD_COMPS = [
    # All dates verified against FDA records (FDA novel-approvals page / openFDA drugsfda), Oct 2026.
    # RVMD own approval — Rasonque (daraxonrasib), FDA 8/26/2026
    ("RVMD", "2026-08-26", "Rasonque (daraxonrasib) — FDA approval, metastatic pancreatic cancer"),
    # Exelixis: Cabometyx tablet NDA208692, advanced RCC
    ("EXEL", "2016-04-25", "Cabometyx (cabozantinib) — FDA approval, advanced renal cell carcinoma"),
    # Karyopharm: Xpovio NDA212306, relapsed/refractory multiple myeloma
    ("KPTI", "2019-07-03", "Xpovio (selinexor) — FDA approval, relapsed/refractory multiple myeloma"),
    # Alnylam: Onpattro NDA210922 — first RNAi drug, hATTR amyloidosis
    ("ALNY", "2018-08-10", "Onpattro (patisiran) — FDA approval, hATTR amyloidosis"),
    # Acadia: Nuplazid NDA207318, Parkinson's disease psychosis
    ("ACAD", "2016-04-29", "Nuplazid (pimavanserin) — FDA approval, Parkinson's disease psychosis"),
]

# Pre-built GH diagnostic oncology comps
# Dates verified against public FDA press releases where noted.
# (*) = approximate — confirm before use.
# EXAS (Exact Sciences / Cologuard) removed — delisted from Yahoo Finance as of 2026.
GH_COMPS = [
    # Verified against openFDA device PMA database, Oct 2026.
    # GH Shield: PMA decision 2024-07-26, publicly announced Mon 2024-07-29 (market event date)
    ("GH",   "2024-07-29", "Shield ctDNA — FDA PMA approval (colorectal screening)"),
    # Natera: Signatera CDx PMA decision 2026-05-15 (<1yr of post-event data available)
    ("NTRA", "2026-05-15", "Signatera CDx — FDA PMA approval"),
    # REMOVED (no FDA event on record — earlier entries were unverified and wrong):
    #   GRAL "2024-09-26 Galleri approval" — Galleri has no PMA approval
    #   VCYT "2019-10-02 Afirma clearance" — Afirma is a lab-developed test, no PMA
    #   NTRA "2021-03-23" / "2022-03-10" — dates unverified
    # The diagnostic comp set is now too thin to calibrate a GH view on its own.
]

# ─── DATA ──────────────────────────────────────────────────────────────────────

def fetch_prices(ticker: str, event_date: str,
                 days_buffer: int = CHART_RANGE + 30) -> pd.Series | None:
    """Download adjusted close prices covering event ± days_buffer calendar days."""
    ev    = pd.Timestamp(event_date)
    start = (ev - timedelta(days=int(days_buffer * 1.5))).strftime("%Y-%m-%d")
    end   = (ev + timedelta(days=int(days_buffer * 1.5))).strftime("%Y-%m-%d")
    try:
        hist = yf.download(ticker, start=start, end=end,
                           auto_adjust=True, progress=False)
        if hist.empty:
            # Distinguish delisted vs. just no data in that date window
            recent = yf.download(ticker, period="5d", progress=False)
            if recent.empty:
                print(f"    ✗ {ticker}: ticker unavailable on Yahoo Finance "
                      f"(delisted or acquired) — remove from comp list")
            else:
                print(f"    ✗ {ticker}: no price data around {event_date} "
                      f"(ticker exists but insufficient history for that date — verify event date)")
            return None
        close = hist["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        close.index = pd.DatetimeIndex(close.index).tz_localize(None)
        return close
    except Exception as e:
        print(f"    ✗ {ticker}: fetch error — {e}")
        return None


def event_returns(prices: pd.Series, event_date: str,
                  windows: list[int]) -> dict[int, float | None]:
    """
    Compute cumulative return from event day to each window (in trading days).
    window < 0 → return from that day up to event day (pre-event run-up)
    window > 0 → return from event day to that day (post-event move)
    """
    ev = pd.Timestamp(event_date)
    idx = prices.index
    # Find closest trading day on or after event date
    future = idx[idx >= ev]
    if future.empty:
        return {w: None for w in windows}
    event_idx = future[0]
    p0 = float(prices.loc[event_idx])
    pos = list(idx).index(event_idx)

    result = {}
    for w in windows:
        target_pos = pos + w
        if 0 <= target_pos < len(idx):
            p_w = float(prices.iloc[target_pos])
            if w < 0:
                # Pre-event: return FROM day w TO event day
                result[w] = (p0 / p_w) - 1.0
            else:
                # Post-event: return FROM event day TO day w
                result[w] = (p_w / p0) - 1.0
        else:
            result[w] = None
    return result


def fetch_benchmark(event_date: str, days_buffer: int = CHART_RANGE + 30) -> pd.Series | None:
    return fetch_prices(BENCHMARK, event_date, days_buffer)


# ─── ANALYSIS ──────────────────────────────────────────────────────────────────

def run_event_study(events: list[tuple[str, str, str]]) -> list[dict]:
    """
    events: list of (ticker, event_date, label)
    Returns list of result dicts.
    """
    results = []
    for ticker, event_date, label in events:
        print(f"  Fetching {ticker} around {event_date}  [{label[:60]}]")
        prices = fetch_prices(ticker, event_date)
        if prices is None or len(prices) < 20:
            if prices is not None:
                print(f"    ✗ {ticker}: only {len(prices)} trading days around {event_date} "
                      f"— need ≥20 (check event date)")
            print(f"    → Skipping {ticker} {event_date}")
            continue
        bench  = fetch_benchmark(event_date)
        rets   = event_returns(prices, event_date, ALL_WINDOWS)
        b_rets = event_returns(bench, event_date, ALL_WINDOWS) if bench is not None else {}

        results.append({
            "ticker":     ticker,
            "event_date": event_date,
            "label":      label,
            "prices":     prices,
            "bench":      bench,
            "returns":    rets,
            "bench_rets": b_rets,
        })
    return results


# ─── CALIBRATION ───────────────────────────────────────────────────────────────

def calibrate_view(results: list[dict], anchor_ticker: str,
                   haircut: float = 0.80, label: str = "") -> dict:
    """
    Derive a suggested BL expected_return from comp median post-event returns.
    anchor_ticker is excluded from the median (it's the stock being calibrated).
    haircut: scale factor applied to comp median (default 0.80 = 20% haircut for
             partial pricing-in; use 1.0 if the catalyst is genuinely unpriced).
    """
    post_windows = [w for w in WINDOWS_POST if w <= 252]
    medians = {}
    for w in post_windows:
        vals = [r["returns"].get(w) for r in results
                if r["returns"].get(w) is not None
                and r["ticker"] != anchor_ticker]
        if vals:
            medians[w] = float(np.median(vals))

    own = next((r for r in results if r["ticker"] == anchor_ticker), None)
    comp_1yr = medians.get(252)
    comp_90d = medians.get(90)
    suggested = comp_1yr * haircut if comp_1yr is not None else None

    return {
        "anchor":           anchor_ticker,
        "label":            label,
        "haircut":          haircut,
        "comp_medians":     medians,
        "comp_1yr":         comp_1yr,
        "comp_90d":         comp_90d,
        "own_post":         own,
        "suggested_return": round(suggested, 2) if suggested is not None else None,
    }


# Keep backward-compatible alias used by existing HTML generator
def calibrate_gh_view(results: list[dict]) -> dict:
    return calibrate_view(results, anchor_ticker="GH",
                          haircut=0.80, label="GH — next MCED catalyst")


# ─── CHART ─────────────────────────────────────────────────────────────────────

def plot_event_study(results: list[dict], ts: str) -> str:
    """
    Two-panel chart:
      Top:    Indexed price (100 = event day) for each comp + benchmark
      Bottom: Return heatmap across pre/post windows
    """
    fig = plt.figure(figsize=(16, 10), facecolor="#1a1a2e")
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 2], hspace=0.35)
    ax_price  = fig.add_subplot(gs[0])
    ax_heat   = fig.add_subplot(gs[1])

    COLORS = ["#00BCD4", "#FF9800", "#4CAF50", "#F44336",
              "#9C27B0", "#FFEB3B", "#E91E63", "#03A9F4"]
    DARK_BG = "#1a1a2e"

    for ax in [ax_price, ax_heat]:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#aaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # ── Price panel ──────────────────────────────────────────────────────────
    ax_price.set_title("Pre/Post Event Indexed Price  (100 = Event Day)",
                        color="#eee", fontsize=11, pad=8)
    ax_price.axvline(0, color="#fff", lw=1.5, ls="--", alpha=0.6, label="Event Day")
    ax_price.axvspan(-CHART_RANGE, 0, alpha=0.04, color="#4CAF50")
    ax_price.axvspan(0, CHART_RANGE, alpha=0.04, color="#F44336")

    bench_plotted = False
    for i, r in enumerate(results):
        prices = r["prices"]
        ev     = pd.Timestamp(r["event_date"])
        idx    = prices.index
        future = idx[idx >= ev]
        if future.empty:
            continue
        ev_day   = future[0]
        ev_pos   = list(idx).index(ev_day)
        p0       = float(prices.iloc[ev_pos])
        # Slice ±CHART_RANGE trading days
        lo = max(0, ev_pos - CHART_RANGE)
        hi = min(len(prices), ev_pos + CHART_RANGE + 1)
        sub = prices.iloc[lo:hi]
        days_rel = [j - ev_pos for j in range(lo, hi)]
        indexed  = (sub.values / p0) * 100

        lbl = f"{r['ticker']} — {r['label'][:45]}"
        ax_price.plot(days_rel, indexed, color=COLORS[i % len(COLORS)],
                      lw=1.8, label=lbl, alpha=0.9)

        # Plot benchmark once (use first result's bench)
        if not bench_plotted and r.get("bench") is not None:
            bench_prices = r["bench"]
            b_idx = bench_prices.index
            b_future = b_idx[b_idx >= ev]
            if not b_future.empty:
                b_ev_day = b_future[0]
                b_ev_pos = list(b_idx).index(b_ev_day)
                b_p0     = float(bench_prices.iloc[b_ev_pos])
                b_lo = max(0, b_ev_pos - CHART_RANGE)
                b_hi = min(len(bench_prices), b_ev_pos + CHART_RANGE + 1)
                b_sub    = bench_prices.iloc[b_lo:b_hi]
                b_days   = [j - b_ev_pos for j in range(b_lo, b_hi)]
                b_indexed = (b_sub.values / b_p0) * 100
                ax_price.plot(b_days, b_indexed, color="#555", lw=1.2,
                              ls="--", label=f"{BENCHMARK} (benchmark)", alpha=0.7)
                bench_plotted = True

    ax_price.axhline(100, color="#555", lw=0.8, ls=":")
    ax_price.set_xlabel("Trading Days Relative to Event", color="#aaa", fontsize=8)
    ax_price.set_ylabel("Indexed Price (100 = Event Day)", color="#aaa", fontsize=8)
    ax_price.legend(fontsize=7, loc="upper left",
                    facecolor="#1a1a2e", edgecolor="#333", labelcolor="#eee")
    ax_price.set_xlim(-CHART_RANGE, CHART_RANGE)

    # ── Return heatmap ───────────────────────────────────────────────────────
    display_windows = [-252, -180, -90, -60, -30, -10, +10, +30, +60, +90, +180, +252]
    col_labels = [f"{'+' if w > 0 else ''}{w}d" for w in display_windows]
    row_labels  = [f"{r['ticker']} {r['event_date']}" for r in results]
    data = np.full((len(results), len(display_windows)), np.nan)

    for i, r in enumerate(results):
        for j, w in enumerate(display_windows):
            v = r["returns"].get(w)
            if v is not None:
                data[i, j] = v

    vmax = max(0.5, np.nanmax(np.abs(data))) if not np.all(np.isnan(data)) else 0.5
    im = ax_heat.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax_heat.set_xticks(range(len(display_windows)))
    ax_heat.set_xticklabels(col_labels, fontsize=8, color="#aaa")
    ax_heat.set_yticks(range(len(results)))
    ax_heat.set_yticklabels(row_labels, fontsize=8, color="#eee")
    ax_heat.set_title("Cumulative Returns at Each Window  (green = gain, red = loss)",
                       color="#eee", fontsize=10, pad=6)

    for i in range(len(results)):
        for j in range(len(display_windows)):
            v = data[i, j]
            if not np.isnan(v):
                ax_heat.text(j, i, f"{v:+.0%}", ha="center", va="center",
                             fontsize=7.5, color="#000" if abs(v) < vmax * 0.6 else "#eee",
                             fontweight="bold")

    # Divider between pre and post columns
    pre_count = sum(1 for w in display_windows if w < 0)
    ax_heat.axvline(pre_count - 0.5, color="#fff", lw=1.5, alpha=0.4)
    ax_heat.text(pre_count - 0.5 - 0.5, -0.7, "← PRE",
                 ha="right", va="bottom", fontsize=7, color="#4CAF50",
                 transform=ax_heat.get_xaxis_transform())
    ax_heat.text(pre_count - 0.5 + 0.5, -0.7, "POST →",
                 ha="left", va="bottom", fontsize=7, color="#F44336",
                 transform=ax_heat.get_xaxis_transform())

    fig.colorbar(im, ax=ax_heat, orientation="vertical",
                 fraction=0.02, pad=0.01).ax.tick_params(colors="#aaa", labelsize=7)

    fig.suptitle("FDA / Catalyst Event Study — Diagnostic Oncology Comps",
                 color="#eee", fontsize=13, y=0.99)

    out = str(OUTPUT_DIR / f"event_study_{ts}.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    return out


# ─── HTML REPORT ───────────────────────────────────────────────────────────────

_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#1a1a2e;color:#eee;font-family:'Segoe UI',sans-serif;font-size:14px;padding:24px}
h1{font-size:1.6rem;margin-bottom:6px;color:#eee}
h2{font-size:1.1rem;margin:28px 0 10px;color:#90caf9;border-bottom:1px solid #333;padding-bottom:4px}
p{color:#aaa;margin-bottom:10px;line-height:1.55}
table{width:100%;border-collapse:collapse;margin-bottom:20px;font-size:0.82rem}
th{background:#16213e;color:#90caf9;padding:8px 10px;text-align:left;white-space:nowrap}
td{padding:7px 10px;border-bottom:1px solid #1e2a3a}
tr:hover td{background:#1e2a3a}
.num{text-align:right;font-variant-numeric:tabular-nums}
.good{color:#4caf50;font-weight:600}
.bad{color:#f44336;font-weight:600}
.neutral{color:#9e9e9e}
.hero{background:#16213e;border-radius:10px;padding:22px 28px;margin-bottom:24px}
.hero p{margin-top:6px}
.callout{background:#0d2137;border-left:4px solid #00BCD4;border-radius:6px;
         padding:16px 20px;margin:16px 0}
.callout h3{color:#00BCD4;margin-bottom:6px;font-size:1rem}
img{max-width:100%;border-radius:8px;margin:12px 0}
"""

def fmt_ret(v) -> str:
    if v is None:
        return '<span style="color:#555">—</span>'
    cls = "good" if v > 0 else ("bad" if v < 0 else "neutral")
    return f'<span class="{cls}">{v:+.1%}</span>'


def generate_html(results: list[dict], cal: dict,
                  chart_path: str, ts: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [f"<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>",
             f"<title>FDA Event Study</title>",
             f"<style>{_CSS}</style></head><body>"]

    parts.append(f"""
<div class="hero">
  <h1>FDA / Catalyst Event Study</h1>
  <p>Pre/post approval cumulative returns for diagnostic oncology comps.
  Calibrates the Guardant Health (GH) Black-Litterman expected-return view.</p>
  <p style="color:#555;font-size:0.8rem">Generated {now} &nbsp;·&nbsp; Benchmark: {BENCHMARK}</p>
</div>""")

    # Chart
    parts.append(f'<img src="{Path(chart_path).name}" alt="Event Study Chart">')

    # Per-event return tables
    parts.append("<h2>Cumulative Returns by Window</h2>")
    display_windows = [-252, -180, -90, -60, -30, -10, +10, +30, +60, +90, +180, +252]
    col_labels = [f"{'Pre ' if w < 0 else 'Post '}{abs(w)}d" for w in display_windows]

    parts.append("<table><thead><tr><th>Ticker</th><th>Event Date</th><th>Label</th>")
    for lbl in col_labels:
        parts.append(f"<th class='num'>{lbl}</th>")
    parts.append("<th class='num'>Excess vs IBB (1yr post)</th></tr></thead><tbody>")

    for r in results:
        excess = None
        own_1yr   = r["returns"].get(252)
        bench_1yr = r["bench_rets"].get(252)
        if own_1yr is not None and bench_1yr is not None:
            excess = own_1yr - bench_1yr

        parts.append(f"<tr><td><strong>{r['ticker']}</strong></td>"
                     f"<td>{r['event_date']}</td>"
                     f"<td style='color:#aaa;font-size:0.8rem'>{r['label']}</td>")
        for w in display_windows:
            parts.append(f"<td class='num'>{fmt_ret(r['returns'].get(w))}</td>")
        parts.append(f"<td class='num'>{fmt_ret(excess)}</td></tr>")

    # Median row
    parts.append("<tr style='background:#0d2137'><td colspan='3'><strong>Median (excl. GH)</strong></td>")
    for w in display_windows:
        vals = [r["returns"].get(w) for r in results
                if r["returns"].get(w) is not None and r["ticker"] != "GH"]
        med = float(np.median(vals)) if vals else None
        parts.append(f"<td class='num'>{fmt_ret(med)}</td>")
    # median excess
    exc_vals = []
    for r in results:
        if r["ticker"] == "GH":
            continue
        own_1yr   = r["returns"].get(252)
        bench_1yr = r["bench_rets"].get(252)
        if own_1yr is not None and bench_1yr is not None:
            exc_vals.append(own_1yr - bench_1yr)
    med_exc = float(np.median(exc_vals)) if exc_vals else None
    parts.append(f"<td class='num'>{fmt_ret(med_exc)}</td></tr>")
    parts.append("</tbody></table>")

    # Calibration callout
    parts.append("<h2>GH View Calibration</h2>")
    cal_medians = cal.get("comp_medians", {})
    suggested   = cal.get("suggested_return")
    comp_1yr    = cal.get("comp_1yr")
    comp_90d    = cal.get("comp_90d")

    interp_lines = []
    if comp_90d is not None:
        interp_lines.append(
            f"Comp median at <strong>+90 days post-approval: {comp_90d:+.1%}</strong>.")
    if comp_1yr is not None:
        interp_lines.append(
            f"Comp median at <strong>+1 year post-approval: {comp_1yr:+.1%}</strong>.")
    if suggested is not None:
        interp_lines.append(
            f"Suggested GH <code>expected_return</code> (comp median × 0.80 haircut "
            f"for partial pricing-in): <strong style='color:#00BCD4'>{suggested:+.2f}</strong> "
            f"({suggested:.0%} annualised).")

    parts.append(f"""
<div class="callout">
  <h3>Calibration Output</h3>
  {''.join(f'<p>{l}</p>' for l in interp_lines)}
  <p style="margin-top:10px;font-size:0.82rem;color:#aaa">
  The 0.80 haircut assumes some of the comp median return is already reflected in GH's
  current price given the existing Shield approval. If you believe the next catalyst
  (expanded MCED panel or CMS national coverage) is <em>not</em> yet priced in,
  use the raw comp median instead (remove the haircut — set multiplier to 1.0).
  </p>
</div>""")

    # Window median table
    parts.append("<h2>Comp Median Returns by Window (excl. GH own event)</h2>")
    parts.append("<table><thead><tr>")
    for w in display_windows:
        lbl = f"{'Pre ' if w < 0 else 'Post '}{abs(w)}d"
        parts.append(f"<th class='num'>{lbl}</th>")
    parts.append("</tr></thead><tbody><tr>")
    for w in display_windows:
        vals = [r["returns"].get(w) for r in results
                if r["returns"].get(w) is not None and r["ticker"] != "GH"]
        med = float(np.median(vals)) if vals else None
        parts.append(f"<td class='num'>{fmt_ret(med)}</td>")
    parts.append("</tr></tbody></table>")

    # How to use section
    parts.append("""<h2>How to Use This Output</h2>
<p>1. Look at the <strong>Post 252d</strong> column (one year after approval).
   The median across comps is your empirical anchor for GH's expected return.</p>
<p>2. Apply a haircut if the catalyst is partially priced in (GH already has one
   approval). The calibration above uses 20% as a default haircut — adjust to taste.</p>
<p>3. Plug the suggested value into the profile's <code>bl_views</code>
   in <code>profiles_local.py</code> and re-run the analyzer.</p>
<p>4. To refine: add more comps by running<br>
   <code>python fda_event_study.py TICKER YYYY-MM-DD "label" ...</code></p>
<p style="color:#555;font-size:0.8rem;margin-top:16px">
Disclaimer: Past comp behavior does not guarantee future results.
This tool is for calibrating model inputs, not financial advice.</p>""")

    parts.append("</body></html>")
    return "\n".join(parts)


# ─── CONSOLE SUMMARY ───────────────────────────────────────────────────────────

def print_summary(results: list[dict], cal: dict) -> None:
    anchor  = cal.get("anchor", "GH")
    label   = cal.get("label", anchor)
    print("\n" + "=" * 72)
    print("EVENT STUDY RESULTS")
    print("=" * 72)

    key_windows = [-90, -30, +30, +90, +252]
    header = f"{'Ticker':<6} {'Event Date':<12} {'−90d':>7} {'−30d':>7} {'+30d':>7} {'+90d':>7} {'+252d':>8}  Label"
    print(header)
    print("-" * 72)

    def fret(v):
        if v is None:
            return "   n/a"
        return f"{v:+.0%}".rjust(6)

    for r in results:
        rets = r["returns"]
        cols = "  ".join(fret(rets.get(w)) for w in key_windows)
        print(f"{r['ticker']:<6} {r['event_date']:<12} {cols}  {r['label'][:38]}")

    print("-" * 72)
    med_vals = []
    for w in key_windows:
        vals = [r["returns"].get(w) for r in results
                if r["returns"].get(w) is not None and r["ticker"] != anchor]
        med_vals.append(float(np.median(vals)) if vals else None)
    cols = "  ".join(fret(v) for v in med_vals)
    print(f"{'MEDIAN':<6} {('excl. ' + anchor):<12} {cols}")

    print("\n" + "-" * 72)
    print(f"VIEW CALIBRATION  —  {label}")
    print("-" * 72)
    c1yr  = cal.get("comp_1yr")
    c90d  = cal.get("comp_90d")
    sugg  = cal.get("suggested_return")
    hcut  = cal.get("haircut", 0.80)
    if c90d is not None:  print(f"  Comp median  +90d post:   {c90d:+.1%}")
    if c1yr is not None:  print(f"  Comp median +252d post:   {c1yr:+.1%}")
    if sugg is not None:
        print(f"  Haircut applied: {hcut:.0%}  (partial pricing-in adjustment)")
        print(f"  Suggested expected_return: {sugg:+.2f}  ({sugg:.0%}/yr)")
        print(f"\n  --> In portfolio_analyzer_advanced.py, update bl_views for {anchor}:")
        print(f'      "ticker": "{anchor}",  "expected_return": {sugg},  "confidence": 0.70')
    print("=" * 72)


# ─── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str]) -> tuple[list[tuple[str, str, str]], str, float]:
    """
    Returns (events, anchor_ticker, haircut).

    Modes:
      --gh-comps          diagnostic oncology comps, anchor=GH, haircut=0.80
      --rvmd-comps        therapeutic oncology comps, anchor=RVMD, haircut=1.0
                          (daraxonrasib approval was NOT priced in — no haircut)
      TICKER DATE [label] ...   custom events, anchor=first ticker, haircut=0.80
    """
    if not argv or argv[0] == "--gh-comps":
        return GH_COMPS, "GH", 0.80
    if argv[0] == "--rvmd-comps":
        return RVMD_COMPS, "RVMD", 1.0

    events = []
    i = 0
    while i < len(argv):
        ticker = argv[i].upper()
        i += 1
        if i >= len(argv):
            print(f"⚠ Missing date for {ticker} — skipping.")
            break
        date_str = argv[i]
        i += 1
        # Optional label (next arg that isn't a ticker-like or date-like)
        label = ""
        if i < len(argv) and not (len(argv[i]) <= 5 and argv[i].isalpha()):
            # Heuristic: if next arg is not a short alpha string and not a date, treat as label
            if not (argv[i][0].isdigit() and len(argv[i]) == 10):
                label = argv[i]
                i += 1
        if not label:
            label = f"{ticker} catalyst"
        try:
            pd.Timestamp(date_str)
        except Exception:
            print(f"⚠ Invalid date '{date_str}' for {ticker} — skipping.")
            continue
        events.append((ticker, date_str, label))
    anchor = events[0][0] if events else "GH"
    return events, anchor, 0.80


def main():
    events, anchor, haircut = parse_args(sys.argv[1:])
    if not events:
        print("No events to study. Run with --gh-comps, --rvmd-comps, or TICKER DATE pairs.")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "--rvmd-comps" if anchor == "RVMD" else "--gh-comps"
    print(f"\nFDA / Catalyst Event Study — {len(events)} event(s)  [anchor: {anchor}]\n")

    results = run_event_study(events)
    if not results:
        print("No usable data returned. Check tickers and dates.")
        sys.exit(1)

    anchor_label = next((r["label"] for r in results if r["ticker"] == anchor), anchor)
    cal = calibrate_view(results, anchor_ticker=anchor,
                         haircut=haircut, label=anchor_label)

    print_summary(results, cal)

    chart_path = plot_event_study(results, ts)
    print(f"\n  Chart saved: {chart_path}")

    html_path  = str(OUTPUT_DIR / f"event_study_{ts}.html")
    html       = generate_html(results, cal, chart_path, ts)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML saved:  {html_path}")
    print(f"\nRun with custom comps:")
    print(f"  python fda_event_study.py EXAS 2014-08-11 \"Cologuard approval\"")


if __name__ == "__main__":
    main()
