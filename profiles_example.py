"""
Example investor profiles — FICTIONAL demo data, safe to commit.

Copy this file to profiles_local.py (git-ignored) and replace the values with your
own; the analyzer loads profiles_local.py when it exists.

existing_shares: shares currently held  (leave {} for a fresh investor)
capital:         new cash ready to deploy RIGHT NOW
pending_capital: {"amount": X, "days": N} — cash arriving in ~N days
monthly_contribution: recurring income allocation (0 = not set yet)
"""

# Values-based exclusions — Yahoo industry names and/or tickers never recommended.
# Example only; leave both lists empty to screen the whole universe.
VALUE_EXCLUSIONS = {
    "industries": ["Tobacco"],
    "tickers": [],
}

# Taxable-account replay inputs (federal + state). Example values only.
TAX_OVERRIDES = {
    "state_name":   "State",
    "state_rate":   0.05,
    "fed_ordinary": 0.22,
    "fed_lt":       0.15,
}

INVESTOR_PROFILES: dict = {
    "fresh": {
        "name":                 "Example — Fresh Portfolio",
        "capital":              10_000,
        "existing_shares":      {},
        "pending_capital":      {},
        "monthly_contribution": 0,
        "note":                 "Example: new investor deploying $10k with a 30% SPY core.",
        "forced_min_weight":    {"SPY": 0.30},
        "bl_views":             [],
        "cash_reserves":        [],
        "portfolio_split": {"Conviction": 0.00, "Upside / Growth": 0.65, "Bluechip Core": 0.35},
    },
    "existing": {
        "name":                 "Example — Existing Portfolio",
        "capital":              5_000,
        "existing_shares":      {"SPY": 10, "MSFT": 5},
        "pending_capital":      {},
        "keep_existing":        True,   # keep holdings the model doesn't pick (no forced sales)
        "no_short_term_sales":  True,   # routine trims use long-term lots only
        "fractional_shares":    False,
        "show_fractional_view": True,
        "sell_intent":          {},     # {"TICKER": "reason"} → shown as EVALUATE SELL
        "monthly_contribution": 250,
        "note":                 "Example: existing holdings plus $5k new cash and $250/month.",
        # Conviction views (Black-Litterman): annualised expected return + confidence 0–1
        "bl_views":             [],
        "cash_reserves":        [],
        "portfolio_split": {"Conviction": 0.10, "Upside / Growth": 0.60, "Bluechip Core": 0.30},
        "forced_min_weight":    {},
    },
}
