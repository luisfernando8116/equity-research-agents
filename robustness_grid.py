"""
Robustness test — does the walk-forward edge survive when the settings change?

Re-runs the rolling walk-forward (point-in-time S&P membership, costs) over many
variations of the strategy's knobs and reports the SPREAD of results, not the best.

    python robustness_grid.py                          # AggressiveGrowth, 60 variations
    python robustness_grid.py --scenario HighConviction --n 80

What is varied (revisions and catalyst factors have no history, so they are neutral
in every walk-forward and are not varied; news-based settings such as structural-risk
severity and exit-signal levels can't be replayed and are judged by the decision log):
    momentum / quality / value weights · position cap · screen size · objective ·
    industry cap · binary-event cap · early-exit drop that overrides the 1-year hold

Read the output as:
  * share of variations beating SPY, and the 10th–90th percentile alpha:
    an edge that only exists at today's exact settings is probably luck;
  * Deflated Sharpe Ratio (Bailey & López de Prado, 2014): probability that the
    BEST variation's information ratio is real after accounting for how many
    variations were tried. Below ~0.95 = not distinguishable from luck.
Use the result to CHECK the settings, never to pick the best-looking variation.

Results are appended to outputs/robustness_<scenario>.csv as they finish, so an
interrupted run resumes where it left off.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import portfolio_analyzer_advanced as pa

EULER_GAMMA = 0.5772156649

GRID = {
    "momentum":     [0.25, 0.35, 0.45],
    "quality":      [0.15, 0.25, 0.35],
    "value":        [0.10, 0.20, 0.30],
    "max_weight":   [0.10, 0.15, 0.20, 0.25],
    "top_n":        [50, 100, 150],
    # equal_weight = top 15 screened names at 1/N (caps still applied); in run15 it beat
    # SPY by ~3%/yr while both optimizers trailed by ~5% — the question this run settles
    "objective":    ["max_return", "max_sharpe", "equal_weight"],
    "industry_cap": [0.20, 0.25, 0.35, 1.00],     # 1.00 = off
    "binary_cap":   [0.10, 0.15, 0.25],
    "early_exit":   [0.15, 0.25, 0.35, 1.00],     # drop that overrides the 1-yr hold; 1.00 = never
}

# Verdict rule — fixed BEFORE seeing results, so the outcome can't be cherry-picked.
# These cut-offs are judgment calls too, stated up front.
ROBUST_SHARE   = 0.70    # ≥70% of variations beat SPY → ROBUST
FRAGILE_SHARE  = 0.50    # <50% beat SPY → FRAGILE (between = MIXED)
SENSITIVE_PP   = 0.05    # a setting whose values move median alpha >5 pp/yr is SENSITIVE:
                         # use its middle value, never the best-looking one


# ── Statistics ────────────────────────────────────────────────────────────────

def sharpe(x: np.ndarray) -> float:
    """Per-period (not annualised) Sharpe of a return series."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return float(x.mean() / sd) if sd > 0 else 0.0


def expected_max_sharpe(sr_var: float, n_trials: int) -> float:
    """Expected maximum Sharpe among n_trials unskilled strategies whose Sharpe
    estimates have variance sr_var (Bailey & López de Prado 2014, eq. for SR0)."""
    if n_trials < 2 or sr_var <= 0:
        return 0.0
    return math.sqrt(sr_var) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
                                + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(returns: np.ndarray, sr_benchmark: float) -> float:
    """Probability that the true Sharpe exceeds sr_benchmark, correcting for
    sample length, skew and fat tails (Probabilistic Sharpe Ratio with SR0 = the
    expected max under multiple testing → Deflated Sharpe Ratio)."""
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    t = len(x)
    if t < 3:
        return float("nan")
    sr = sharpe(x)
    z = (x - x.mean()) / x.std(ddof=1)
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))                  # non-excess (normal = 3)
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denom <= 0:
        return float("nan")
    return float(norm.cdf((sr - sr_benchmark) * math.sqrt(t - 1) / math.sqrt(denom)))


def summarise(rows: list[dict], excess: dict[str, np.ndarray], baseline_id: str) -> str:
    alphas = np.array([r["alpha"] for r in rows if r["alpha"] is not None], dtype=float)
    if not len(alphas):
        return "No completed variations."
    srs = {k: sharpe(v) for k, v in excess.items()}
    best_id = max(srs, key=srs.get)
    sr0 = expected_max_sharpe(float(np.var(list(srs.values()), ddof=1)) if len(srs) > 1 else 0.0,
                              len(srs))
    base = next((r for r in rows if r["id"] == baseline_id), None)
    lines = [
        f"Variations completed: {len(alphas)}",
        f"Alpha vs SPY:  median {np.median(alphas):+.1%}   "
        f"10th–90th pct {np.percentile(alphas, 10):+.1%} … {np.percentile(alphas, 90):+.1%}",
        f"Beat SPY in {np.mean(alphas > 0):.0%} of variations",
    ]
    if base and base["alpha"] is not None:
        rank = float(np.mean(alphas <= base["alpha"]))
        lines.append(f"Current settings: alpha {base['alpha']:+.1%}  "
                     f"(better than {rank:.0%} of variations)")
        if baseline_id in excess:
            lines.append(f"  Deflated Sharpe (current, vs {len(srs)} trials): "
                         f"{deflated_sharpe(excess[baseline_id], sr0):.2f}")
    best = next(r for r in rows if r["id"] == best_id)
    lines.append(f"Best information ratio: {best_id}  alpha {best['alpha']:+.1%}")
    lines.append(f"  Deflated Sharpe (best, vs {len(srs)} trials): "
                 f"{deflated_sharpe(excess[best_id], sr0):.2f}   (≥0.95 ≈ real edge)")
    at = [float(r["after_tax"]) for r in rows if r.get("after_tax") not in (None, "", "None")]
    if at:
        lines.append(f"After tax (median across variations): {np.median(at):+.1%}/yr")

    # Sensitivity: median alpha for each value of each setting
    lines.append("Sensitivity (median alpha by setting value):")
    sensitive = []
    for knob in GRID:
        by_val: dict[str, list] = {}
        for r in rows:
            if r.get(knob) is not None and r["alpha"] is not None:
                by_val.setdefault(str(r[knob]), []).append(r["alpha"])
        meds = {v: float(np.median(a)) for v, a in by_val.items()}
        if len(meds) < 2:
            continue
        spread = max(meds.values()) - min(meds.values())
        flag = "  ← SENSITIVE" if spread > SENSITIVE_PP else ""
        if flag:
            sensitive.append(knob)
        lines.append(f"  {knob:<13} " + "  ".join(f"{v}: {m:+.1%}" for v, m in sorted(meds.items()))
                     + flag)

    def _verdict(a):
        s = float(np.mean(np.asarray(a) > 0))
        return ("ROBUST" if s >= ROBUST_SHARE else
                "FRAGILE" if s < FRAGILE_SHARE else "MIXED"), s

    verdict, share = _verdict(alphas)
    lines.append(f"\nVERDICT (rule fixed in advance): {verdict} — {share:.0%} of variations beat "
                 f"SPY (robust ≥{ROBUST_SHARE:.0%}, fragile <{FRAGILE_SHARE:.0%})")
    objs = sorted({str(r.get("objective")) for r in rows if r.get("objective")})
    if len(objs) > 1:
        for o in objs:
            a = [r["alpha"] for r in rows if str(r.get("objective")) == o and r["alpha"] is not None]
            if a:
                v, s = _verdict(a)
                lines.append(f"  {o:<13} {v:<8} {s:.0%} beat SPY  (median alpha "
                             f"{np.median(a):+.1%}, n={len(a)})")
    if sensitive:
        lines.append(f"Sensitive settings: {', '.join(sensitive)} — use their middle values, "
                     f"not the best-looking ones.")
    lines.append("Never adopt the best variation: with this many trials, the top one is "
                 "expected to look good by luck (see its Deflated Sharpe).")
    return "\n".join(lines)


# ── Grid ──────────────────────────────────────────────────────────────────────

def config_id(c: dict) -> str:
    return (f"m{float(c['momentum']):.2f}-q{float(c['quality']):.2f}-v{float(c['value']):.2f}-"
            f"cap{float(c['max_weight']):.2f}-n{int(c['top_n'])}-{c['objective']}-"
            f"ind{float(c['industry_cap']):.2f}-bin{float(c['binary_cap']):.2f}-"
            f"ex{float(c['early_exit']):.2f}")


def sample_configs(baseline: dict, n: int, seed: int) -> list[dict]:
    keys = list(GRID)
    full = [dict(zip(keys, vals)) for vals in itertools.product(*GRID.values())]
    rng = random.Random(seed)
    rng.shuffle(full)
    # Equal numbers per objective, so each gets its own fair verdict
    by_obj: dict[str, list] = {o: [] for o in GRID["objective"]}
    for c in full:
        if config_id(c) != config_id(baseline):
            by_obj[c["objective"]].append(c)
    out = [baseline]
    for i in itertools.count():
        if len(out) >= n or not any(i < len(v) for v in by_obj.values()):
            break
        out += [v[i] for v in by_obj.values() if i < len(v)]
    return out[:max(n, 1)]


def factor_blend(c: dict, base: dict | None = None) -> dict:
    """Varied weights for momentum/quality/value; revisions + catalyst keep their
    current share (they're neutral in the walk-forward anyway)."""
    base = base or pa.FACTOR_BLEND
    fixed = {k: base[k] for k in ("revisions", "catalyst")}
    room = 1.0 - sum(fixed.values())
    s = c["momentum"] + c["quality"] + c["value"]
    return {"momentum": room * c["momentum"] / s, "quality": room * c["quality"] / s,
            "value": room * c["value"] / s, **fixed}


# ── Data (same steps as the main run; .info is disk-cached) ───────────────────

def load_data() -> tuple[dict, dict | None]:
    """Same backtest setup as the main run (10-year history, survivorship,
    point-in-time SEC fundamentals, group caps). Falls back to today's
    fundamentals — with the look-ahead warning — when SEC EDGAR is off."""
    print("Loading universe and backtest data (same setup as the main run)...")
    bt = pa.prepare_backtest(pa.fetch_universe())
    fund = None if bt["pit"] is not None else pa.fetch_fundamentals(bt["universe"])
    return bt, fund


# ── Main ──────────────────────────────────────────────────────────────────────

def _keep_awake():
    """Ask Windows not to sleep while this process runs (released automatically on
    exit; power settings are not changed). A sleeping PC stretched run15's backtest
    from minutes to ~20 hours."""
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except Exception:
        pass


def main():
    _keep_awake()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--scenario", default="AggressiveGrowth", choices=list(pa.PORTFOLIOS))
    ap.add_argument("--n", type=int, default=60, help="variations to run (incl. current)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    pf = pa.PORTFOLIOS[args.scenario]
    fb = pa.FACTOR_BLEND
    baseline = {"momentum": fb["momentum"], "quality": fb["quality"], "value": fb["value"],
                "max_weight": pf["max_weight"], "top_n": pf["screen_top_n"],
                "objective": pf["objective"], "industry_cap": pa.INDUSTRY_MAX_WEIGHT,
                "binary_cap": pa.BINARY_EVENT_MAX, "early_exit": pa.WF_EARLY_EXIT_DRAWDOWN}
    baseline_id = config_id(baseline)
    configs = sample_configs(baseline, args.n, args.seed)

    out_dir = Path(pa.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"robustness_{args.scenario}.csv"
    ret_path = out_dir / f"robustness_{args.scenario}_excess.pkl"
    fields = ["id", *GRID, "alpha", "oos_annualized", "oos_sharpe", "oos_max_dd",
              "hit_rate", "avg_turnover", "after_tax", "early_exits", "seconds"]
    rows, excess = [], {}
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["alpha"] = float(r["alpha"]) if r["alpha"] not in ("", "None") else None
                rows.append(r)
    if ret_path.exists():
        excess = pd.read_pickle(ret_path)
    done = {r["id"] for r in rows}
    todo = [c for c in configs if config_id(c) not in done]
    print(f"{args.scenario}: {len(configs)} variations, {len(done)} already done, "
          f"{len(todo)} to run. Current settings = {baseline_id}")

    bench = None
    if todo:
        bt, fund = load_data()
        ew = pa._wf(bt, "equal_weight", 1.0, 0.0, pa.TOP_N_FACTORS, fund)
        bench = {"1/N top-15": ew.get("oos_annualized"), "SPY": ew.get("spy_annualized"),
                 "RSP": ew.get("rsp_annualized")}
        saved_blend = dict(pa.FACTOR_BLEND)
        new_file = not csv_path.exists()
        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                if new_file:
                    w.writeheader()
                for i, c in enumerate(todo, 1):
                    blend = factor_blend(c, saved_blend)
                    pa.FACTOR_BLEND.clear()
                    pa.FACTOR_BLEND.update(blend)
                    t0 = time.time()
                    groups = pa.build_group_caps(bt["infos"],
                                                 industry_cap=float(c["industry_cap"]),
                                                 binary_cap=float(c["binary_cap"]))
                    wf = pa._wf(bt, c["objective"], float(c["max_weight"]), pf["min_weight"],
                                int(c["top_n"]), fund, group_caps=groups,
                                early_exit=float(c["early_exit"]))
                    at = pa.after_tax_replay(wf.get("schedule") or [], bt["prices"]) or {}
                    secs = time.time() - t0
                    row = {"id": config_id(c), **c, "seconds": round(secs, 1),
                           "after_tax": at.get("after_tax_annualized"),
                           "early_exits": wf.get("early_exits"),
                           **{k: wf.get(k) for k in ("alpha", "oos_annualized", "oos_sharpe",
                                                     "oos_max_dd", "hit_rate", "avg_turnover")}}
                    w.writerow(row)
                    fh.flush()
                    rows.append(row)
                    if wf.get("daily") is not None:
                        excess[row["id"]] = (wf["daily"] - wf["spy_daily"]).dropna().values
                        pd.to_pickle(excess, ret_path)
                    eta = secs * (len(todo) - i) / 60
                    a = row["alpha"]
                    print(f"  [{i}/{len(todo)}] {row['id']:<52} alpha "
                          f"{(f'{a:+.1%}' if a is not None else 'n/a'):>7}  "
                          f"({secs:.0f}s, ~{eta:.0f} min left)")
        finally:
            pa.FACTOR_BLEND.clear()
            pa.FACTOR_BLEND.update(saved_blend)

    summary = summarise(rows, excess, baseline_id)
    if bench:
        summary += ("\nBenchmarks over the same test period: " + "  ·  ".join(
            f"{k} {v:+.1%}/yr" for k, v in bench.items() if v is not None)
            + "\n  (alpha above is vs SPY; a variation that can't beat 1/N adds nothing "
              "over picking the top 15 screened names equally)")
    (out_dir / f"robustness_{args.scenario}.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
