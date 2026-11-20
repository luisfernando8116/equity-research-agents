"""
Biotech Pipeline Intelligence Module
=====================================
Finds early-stage clinical movers BEFORE they appear in S&P 500/NASDAQ-100 screening.

Three intelligence layers:
  1. Curated mechanism-based biotech universe — organised by therapeutic area
  2. ClinicalTrials.gov catalyst proximity scoring — how close is the next readout?
  3. Mechanism validation propagation — positive peer data boosts the class

Scores are exported as catalyst_signals-compatible dicts, injected into the
main portfolio_analyzer_advanced.py screening pipeline via CUSTOM_TICKERS +
the fetch_catalyst_signals hook.

Standalone usage
----------------
  python biotech_pipeline.py                    # score all tracked companies
  python biotech_pipeline.py --top 20           # show top 20 by pipeline score
  python biotech_pipeline.py --area RAS         # filter by therapeutic area
  python biotech_pipeline.py --ticker RVMD      # single-ticker deep-dive
  python biotech_pipeline.py --export           # write scores to pipeline_scores.json

Integration with portfolio_analyzer_advanced.py
------------------------------------------------
  Add to CUSTOM_TICKERS any ticker that scores above PIPELINE_SCORE_THRESHOLD.
  The module is called automatically when RUN_BIOTECH_PIPELINE = True (set in
  portfolio_analyzer_advanced.py config).
"""

import sys
import subprocess
import json
import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

REQUIRED = {"yfinance": "yfinance", "pandas": "pandas",
            "numpy": "numpy", "requests": "requests"}
for mod, pkg in REQUIRED.items():
    try:
        __import__(mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── BIOTECH UNIVERSE ─────────────────────────────────────────────────────────
# Organised by therapeutic mechanism.
# Add tickers here as you develop conviction in a mechanism area.
# Format: "Area": {ticker: "brief description"}

#
# Names verified against yfinance (Oct 2026). Descriptions state only the company's
# broad focus — check specific programs on ClinicalTrials.gov before acting.
# Removed Oct 2026: KLDI (KLDiscovery — software, not biotech), RDW (Redwire —
# aerospace), FUSN / SAGE / TERN / NUVL (no longer trade — acquired or delisted).

BIOTECH_UNIVERSE: dict[str, dict[str, str]] = {
    # ── Targeted oncology ────────────────────────────────────────────────────
    "Targeted Oncology": {
        "RVMD": "Revolution Medicines — RAS(ON) inhibitors; Rasonque (daraxonrasib) FDA-approved 8/26/2026, metastatic pancreatic cancer",
        "ELVN": "Enliven Therapeutics — small-molecule kinase inhibitors, oncology",
        "BNTX": "BioNTech — mRNA and oncology platforms",
        "ARVN": "Arvinas — protein degraders; oncology",
        "STTK": "Shattuck Labs — fusion proteins, oncology/immunology",
    },

    # ── Multi-cancer early detection / liquid biopsy ──────────────────────────
    "MCED Diagnostics": {
        "GH":   "Guardant Health — liquid biopsy; Shield blood test FDA PMA (colorectal) Jul 2024",
        "GRAL": "Grail — Galleri multi-cancer early detection (no FDA PMA on record as of Oct 2026)",
        "NTRA": "Natera — cfDNA testing; Signatera CDx FDA PMA May 2026",
        "VCYT": "Veracyte — genomic diagnostics (lab-developed tests)",
    },

    # ── Immunology / autoimmune ──────────────────────────────────────────────
    "Immunology": {
        "KYMR": "Kymera Therapeutics — oral protein degraders, immunology",
        "IMVT": "Immunovant — FcRn inhibitors, autoimmune disease",
        "ARQT": "Arcutis Biotherapeutics — dermatology",
        "RXRX": "Recursion Pharmaceuticals — AI-driven drug discovery",
    },

    # ── Rare / genetic disease ───────────────────────────────────────────────
    "Rare Disease": {
        "SRPT": "Sarepta Therapeutics — Duchenne and limb-girdle muscular dystrophy",
        "BMRN": "BioMarin Pharmaceutical — rare genetic diseases",
        "PTCT": "PTC Therapeutics — rare neurological and metabolic diseases",
        "KRYS": "Krystal Biotech — redosable gene therapy, rare skin/eye disease",
    },

    # ── RNA medicines ────────────────────────────────────────────────────────
    "RNA Medicines": {
        "ALNY": "Alnylam Pharmaceuticals — RNAi therapeutics",
        "IONS": "Ionis Pharmaceuticals — antisense oligonucleotides",
    },

    # ── Neuroscience / CNS ───────────────────────────────────────────────────
    "Neuroscience": {
        "PRAX": "Praxis Precision Medicines — epilepsy and movement disorders",
        "ACAD": "Acadia Pharmaceuticals — CNS disorders",
        "AVXL": "Anavex Life Sciences — CNS / neurodegeneration",
    },

    # ── Gene editing ─────────────────────────────────────────────────────────
    "Gene Editing": {
        "CRSP": "CRISPR Therapeutics — CRISPR gene editing; Casgevy approved",
        "EDIT": "Editas Medicine — CRISPR gene editing",
        "BEAM": "Beam Therapeutics — base editing",
        "NTLA": "Intellia Therapeutics — in vivo CRISPR gene editing",
    },

    # ── Gene & cell therapy ──────────────────────────────────────────────────
    "Gene & Cell Therapy": {
        "PGEN": "Precigen — gene and cell therapy",
    },

    # ── Metabolic / obesity ──────────────────────────────────────────────────
    "Metabolic & Obesity": {
        "GPCR": "Structure Therapeutics — oral small-molecule GLP-1 agonists",
    },
}

# Tickers whose success de-risks a SAME-MECHANISM peer group.
# Only genuine shared-technology classes — RVMD→KYMR/BNTX and GPCR→ELVN/TERN were
# removed: those companies don't share the validator's mechanism.
MECHANISM_VALIDATORS: dict[str, list[str]] = {
    "CRSP": ["EDIT", "BEAM", "NTLA"],  # gene editing
    "GRAL": ["GH", "NTRA", "VCYT"],    # cfDNA / liquid biopsy diagnostics
    "ALNY": ["IONS"],                   # oligonucleotide therapeutics
}

# How much a mechanism validation event boosts the class (0-1 score boost)
# A peer's 3-month price move is only a proxy for clinical validation, so its
# weight is kept small (0.25 previously pushed GH/NTRA/VCYT straight to 1.000).
VALIDATION_BOOST = 0.10

# Minimum pipeline score to auto-suggest adding to CUSTOM_TICKERS
PIPELINE_SCORE_THRESHOLD = 0.72   # raised — 0.60 produced too many candidates for a 15-stock portfolio

# ClinicalTrials.gov API endpoint
CT_API = "https://clinicaltrials.gov/api/v2/studies"

# ─── CLINICALTRIALS.GOV FETCH ─────────────────────────────────────────────────

_LEGAL_WORDS = {"inc", "corp", "corporation", "co", "company", "ltd", "llc", "plc", "se",
                "nv", "ag", "sa", "holding", "holdings", "group", "and", "the"}


def name_key(name: str) -> str:
    """Normalised two-word company key: 'Scholar Rock Holding Corporation' and
    'Scholar Rock, Inc.' both → 'scholar rock'. Used to match lead sponsors."""
    words = [w for w in re.sub(r"[^a-z0-9 ]", " ", (name or "").lower()).split()
             if w not in _LEGAL_WORDS]
    return " ".join(words[:2])


def fetch_trials(description: str, ticker: str) -> list[dict]:
    """
    Query ClinicalTrials.gov API v2 for Phase 2/3 trials the company LEADS.
    `description` is the BIOTECH_UNIVERSE string; the company name is the part
    before " — ". Trials where the company is only a collaborator (e.g. Guardant
    supplying a test for AstraZeneca's osimertinib trial) are excluded — they are
    not the company's own catalysts. Phase filtering is done in Python.
    """
    ACTIVE_STATUSES = {
        "RECRUITING", "ACTIVE_NOT_RECRUITING",
        "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING",
    }
    TARGET_PHASES = {"PHASE2", "PHASE3", "PHASE2_PHASE3"}

    company = description.split(" — ")[0].strip()
    key     = name_key(company)
    results = []
    if not key:
        return results

    for query in [key]:
        try:
            params = {
                "query.spons": query,
                "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING,"
                                        "ENROLLING_BY_INVITATION,NOT_YET_RECRUITING",
                "pageSize": 30,
                "format": "json",
            }
            resp = requests.get(CT_API, params=params, timeout=12)
            if resp.status_code != 200:
                continue
            data = resp.json()
            studies = data.get("studies", [])
            for s in studies:
                pm      = s.get("protocolSection", {})
                lead    = (pm.get("sponsorCollaboratorsModule", {})
                             .get("leadSponsor", {}).get("name", ""))
                if name_key(lead) != key:
                    continue          # collaborator only — not this company's catalyst
                id_m    = pm.get("identificationModule", {})
                stat_m  = pm.get("statusModule", {})
                des_m   = pm.get("designModule", {})
                cond_m  = pm.get("conditionsModule", {})
                phases  = des_m.get("phases", [])
                status  = stat_m.get("overallStatus", "")
                # Filter: only Phase 2 / Phase 3, only active/recruiting
                if not any(p in TARGET_PHASES for p in phases):
                    continue
                if status.upper() not in ACTIVE_STATUSES:
                    continue
                title   = id_m.get("briefTitle", "")
                pcd_raw = stat_m.get("primaryCompletionDateStruct", {}).get("date", "")
                conds   = ", ".join(cond_m.get("conditions", [])[:3])
                phase_str = "/".join(phases)    # e.g. "PHASE1/PHASE2" — keep the full label
                days_away = None
                if pcd_raw:
                    try:
                        pcd = pd.Timestamp(pcd_raw)
                        days_away = (pcd - pd.Timestamp.now()).days
                    except Exception:
                        pass
                results.append({
                    "nct_id":    id_m.get("nctId", ""),
                    "title":     title[:120],
                    "phase":     phase_str,
                    "status":    status,
                    "condition": conds,
                    "pcd":       pcd_raw,
                    "days_away": days_away,
                })
            if results:
                break
        except Exception:
            continue
    return results


# ─── UNMET NEED SCORING ───────────────────────────────────────────────────────

# High-value indications — conditions where FDA approval is a major re-rating event
HIGH_UNMET_NEED: dict[str, float] = {
    "pancreatic":       1.0,   # PDAC — median survival ~12mo, very high unmet need
    "glioblastoma":     1.0,   # GBM — median survival ~15mo
    "mesothelioma":     0.9,
    "ovarian":          0.85,
    "lung":             0.80,
    "sickle cell":      0.85,
    "huntington":       0.90,
    "als":              0.95,
    "duchenne":         0.90,
    "alzheimer":        0.85,
    "colorectal":       0.70,
    "breast":           0.65,
    "prostate":         0.60,
    "leukemia":         0.75,
    "lymphoma":         0.70,
    "melanoma":         0.60,
    "renal":            0.65,
    "bladder":          0.65,
    "thyroid":          0.50,
    "parkinson":        0.70,
    "depression":       0.55,
    "epilepsy":         0.65,
    "obesity":          0.60,
    "nash":             0.70,
    "mash":             0.70,
}

def unmet_need_score(trials: list[dict]) -> float:
    """Return max unmet-need score across all trials for this company."""
    best = 0.3  # default if no match
    for t in trials:
        cond_lower = t["condition"].lower() + " " + t["title"].lower()
        for keyword, score in HIGH_UNMET_NEED.items():
            if keyword in cond_lower:
                best = max(best, score)
    return best


# ─── PIPELINE SCORING ─────────────────────────────────────────────────────────

def phase_score(phase: str) -> float:
    """Convert trial phase string to 0-1 score. Combined Phase 1/2 trials are
    early-stage and score between Phase 1 and Phase 2."""
    p = phase.upper()
    if "3" in p:                 return 1.0
    if "2" in p and "1" in p:    return 0.45
    if "2" in p:                 return 0.65
    if "1" in p:                 return 0.25
    return 0.10


def proximity_score(days_away: int | None) -> float:
    """
    How close is the primary completion date?
    Closer = higher score (more imminent catalyst).
    """
    if days_away is None:
        return 0.30
    if days_away < 0:      return 0.10   # past — may have read out
    if days_away < 60:     return 1.00   # imminent (< 2 months)
    if days_away < 120:    return 0.85
    if days_away < 180:    return 0.70
    if days_away < 270:    return 0.55
    if days_away < 365:    return 0.40
    if days_away < 548:    return 0.30
    return 0.20


def compute_pipeline_score(ticker: str, desc: str,
                            trials: list[dict],
                            price_data: dict,
                            validation_boost: float = 0.0) -> dict:
    """
    Pipeline score (0-1) = clinical quality × pre-discovery multiplier
      clinical quality: 40% best trial phase + 30% catalyst proximity + 30% unmet need
      multiplier:       0.35 + 0.65 × pre_discovery
    The multiplier makes "hasn't run yet" a gate, not a 20% add-on: a stock that
    already re-rated on its data (PRAX at +375%) keeps at most ~55% of its
    clinical score, so it no longer qualifies on trial quality alone.
    Plus: small mechanism validation boost from same-class peers.
    """
    # ── Trial signals ────────────────────────────────────────────────────────
    if trials:
        best_phase   = max(phase_score(t["phase"]) for t in trials)
        best_prox    = max(proximity_score(t["days_away"]) for t in trials)
        need_score   = unmet_need_score(trials)
        n_trials     = min(len(trials) / 3, 1.0)   # more trials = slightly higher
    else:
        best_phase = 0.10
        best_prox  = 0.20
        need_score = 0.30
        n_trials   = 0.0

    # ── Pre-discovery score — THIS is the key signal ─────────────────────────
    # We want PRAX at $53 (flat/quiet stock with trial data starting to emerge),
    # NOT PRAX at $260 (stock already re-rated on the news).
    #
    # A stock scores HIGH here when:
    #   - 12-month return is modest (<50%) — hasn't been discovered yet
    #   - 3-month momentum is starting to accelerate — early signal emerging
    #   - 52-week range position is low (<40%) — still near the base
    #   - Market cap is small-to-mid ($300M–$5B) — pre-institutional discovery zone
    #
    # A stock scores LOW here when:
    #   - It has already run >150% in 12 months (PRAX at $260, RVMD at $210)
    #   - It is near its 52-week high
    px_data      = price_data.get(ticker, {})
    ret_12m      = float(px_data.get("ret_12m",  0.0))
    range_pos    = float(px_data.get("range_pos", 0.5))
    ret_3m       = float(px_data.get("ret_3m",    0.0))
    mkt_cap      = float(px_data.get("market_cap", 0.0) or 0.0)
    fallen_angel = bool(px_data.get("fallen_angel", False))
    dd_3yr       = float(px_data.get("dd_3yr", 0.0))

    # Late-cycle hard penalty — stock already had its run
    if ret_12m > 1.50:
        # +150% YTD and above: linearly reduce score. At +400% = 0.0
        run_factor = max(0.0, 1.0 - (ret_12m - 1.50) / 2.50)
    elif ret_12m > 0.50:
        # +50–150% YTD: mild discount — partially discovered
        run_factor = 1.0 - (ret_12m - 0.50) / 2.0
    else:
        run_factor = 1.0   # flat or modest gain — full credit

    # Range position penalty — stock near its 52wk high has already re-rated
    if range_pos > 0.85:
        range_factor = max(0.0, 1.0 - (range_pos - 0.85) * 5.0)
    elif range_pos > 0.65:
        range_factor = 1.0 - (range_pos - 0.65) * 0.5
    else:
        range_factor = 1.0   # in lower half of range — full credit

    # Early acceleration bonus — flat stock starting to move (the PRAX-at-$53 signal)
    # ret_12m < 30% but ret_3m > 5% = something is changing, not yet priced in
    if ret_12m < 0.30 and ret_3m > 0.05:
        accel_bonus = min(ret_3m * 2.0, 0.30)   # up to +0.30 bonus
    else:
        accel_bonus = 0.0

    # Market cap discovery zone: $300M–$5B = small enough to be undiscovered,
    # large enough to have institutional-quality clinical programs
    if 300_000_000 < mkt_cap < 5_000_000_000:
        cap_bonus = 0.15
    elif 5_000_000_000 <= mkt_cap < 15_000_000_000:
        cap_bonus = 0.05   # mid-cap — partially discovered
    else:
        cap_bonus = 0.0    # large-cap already institutional, or micro-cap too risky

    # Falling knife: still trending down (12m < −30% AND last 3m negative).
    # That's distress, not quiet accumulation — the PRAX-at-$53 pattern is a
    # flat base that has *started* to turn up.
    downtrend = ret_12m < -0.30 and ret_3m < 0.0

    # Components sum to at most 1.0 so the score keeps discriminating
    # (a constant +0.2 used to saturate every flat or falling stock at 1.00)
    pre_discovery = float(np.clip(
        0.40 * run_factor + 0.25 * range_factor
        + min(accel_bonus, 0.25) + min(cap_bonus, 0.10)
        - (0.30 if downtrend else 0.0),
        0.0, 1.0
    ))

    # ── Fallen angel assessment ───────────────────────────────────────────────
    # A prior blow-up (>65% off 3yr high) DOES NOT automatically disqualify a
    # stock. History shows 30-40% of fallen biotech stocks recover — often
    # through a new drug in a different indication.
    #
    # The key question: are the ACTIVE trials in NEW indications, or are they
    # trying to rescue the same failed thesis?
    #
    # We apply a MODERATE flag penalty, not elimination — the stock is kept in
    # the list but ranked lower, surfacing it for human review rather than
    # auto-adding it to the optimizer.
    if fallen_angel:
        # Check if any active trials are for genuinely different conditions
        # from what likely caused the blow-up (rough heuristic: >1 distinct condition)
        distinct_conditions = len(set(
            t["condition"].split(",")[0].strip().lower()
            for t in trials if t.get("condition")
        ))
        if distinct_conditions >= 2:
            # Multiple distinct indications in pipeline — may have a new thesis
            fallen_penalty = 0.30   # flag but don't eliminate: new pipeline possible
        else:
            # Single indication — likely trying to rescue the same failing thesis
            fallen_penalty = 0.55   # heavier discount: same-indication retry unlikely to work
    elif dd_3yr < -0.45:
        fallen_penalty = 0.20   # moderate prior drawdown — light flag
    else:
        fallen_penalty = 0.0

    # ── Composite ────────────────────────────────────────────────────────────
    clinical  = 0.40 * best_phase + 0.30 * best_prox + 0.30 * need_score
    raw_score = clinical * (0.35 + 0.65 * pre_discovery)

    discounted = raw_score * (1.0 - fallen_penalty)

    # Mechanism validation boost applied after fallen-angel discount
    final_score = min(1.0, discounted + validation_boost * (1.0 - fallen_penalty))

    return {
        "ticker":           ticker,
        "description":      desc,
        "pipeline_score":   round(final_score, 3),
        "trial_phase":      round(best_phase, 2),
        "catalyst_days":    min((t["days_away"] for t in trials
                                  if t["days_away"] is not None), default=None),
        "proximity":        round(best_prox, 2),
        "unmet_need":       round(need_score, 2),
        "pre_discovery":    round(pre_discovery, 2),
        "ret_12m":          round(ret_12m, 2),
        "range_pos":        round(range_pos, 2),
        "fallen_angel":     fallen_angel,
        "dd_3yr":           round(dd_3yr, 2),
        "validation_boost": round(validation_boost, 2),
        "n_trials":         len(trials),
        # nearest upcoming readouts first (past/undated last)
        "trials":           sorted(trials, key=lambda t: (t.get("days_away") is None
                                                          or t["days_away"] < 0,
                                                          t.get("days_away") or 0))[:3],
        "suggest_add":      final_score >= PIPELINE_SCORE_THRESHOLD,
    }


# ─── CT UNIVERSE DISCOVERY ────────────────────────────────────────────────────

# Known ticker mappings for common biotech sponsors
# ClinicalTrials.gov uses company legal names; map to US exchange tickers
_SPONSOR_TO_TICKER: dict[str, str] = {
    "revolution medicines": "RVMD",
    "kymera therapeutics": "KYMR",
    "praxis precision medicine": "PRAX",
    "fusion pharmaceuticals": "FUSN",
    "arvinas": "ARVN",
    "beam therapeutics": "BEAM",
    "intellia therapeutics": "NTLA",
    "crispr therapeutics": "CRSP",
    "editas medicine": "EDIT",
    "alnylam pharmaceuticals": "ALNY",
    "ionis pharmaceuticals": "IONS",
    "acadia pharmaceuticals": "ACAD",
    "arcutis biotherapeutics": "ARQT",
    "recursion pharmaceuticals": "RXRX",
    "veracyte": "VCYT",
    "natera": "NTRA",
    "guardant health": "GH",
    "sage therapeutics": "SAGE",
    "structure therapeutics": "GPCR",
    "terns pharmaceuticals": "TERN",
    "biontech": "BNTX",
    "immunovant": "IMVT",
    "krystal biotech": "KRYS",
    "shattuck labs": "STTK",
    "sarepta therapeutics": "SRPT",   # Duchenne muscular dystrophy exon-skipping therapies
    "sarepta therapeutics, inc.": "SRPT",
}


def discover_pipeline_from_ct(
    conditions: list[str] | None = None,
    days_window: int = 540,
    max_results: int = 50,
) -> list[dict]:
    """
    Query ClinicalTrials.gov for Phase 3 trials in high-unmet-need indications
    completing within `days_window` days. Returns sponsor names + known tickers.

    This is how we find 'PRAX at $53' — companies with active Phase 3 trials
    that haven't had their stock re-rating yet.
    """
    if conditions is None:
        # Focus on highest-unmet-need oncology and rare disease indications
        conditions = [
            "pancreatic cancer", "glioblastoma", "ALS", "Huntington",
            "Duchenne muscular dystrophy", "sickle cell disease",
            "triple negative breast cancer", "ovarian cancer",
            "acute myeloid leukemia", "Alzheimer",
        ]

    cutoff_date = (datetime.now() + timedelta(days=days_window)).strftime("%Y-%m-%d")
    today_str   = datetime.now().strftime("%Y-%m-%d")
    discovered  = []

    for condition in conditions:
        try:
            params = {
                "query.cond":       condition,
                "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING,ENROLLING_BY_INVITATION",
                "pageSize":         20,
                "format":           "json",
            }
            resp = requests.get(CT_API, params=params, timeout=12)
            if resp.status_code != 200:
                continue
            studies = resp.json().get("studies", [])
            for s in studies:
                pm     = s.get("protocolSection", {})
                des_m  = pm.get("designModule", {})
                stat_m = pm.get("statusModule", {})
                id_m   = pm.get("identificationModule", {})
                spons_m = pm.get("sponsorCollaboratorsModule", {})
                phases = des_m.get("phases", [])
                if not any("PHASE3" in p for p in phases):
                    continue
                pcd_raw = stat_m.get("primaryCompletionDateStruct", {}).get("date", "")
                if not pcd_raw:
                    continue
                try:
                    pcd = pd.Timestamp(pcd_raw)
                    days_away = (pcd - pd.Timestamp.now()).days
                    if days_away < 0 or days_away > days_window:
                        continue
                except Exception:
                    continue
                # Get lead sponsor
                lead = spons_m.get("leadSponsor", {}).get("name", "")
                if not lead:
                    continue
                # Map to ticker if known
                ticker = _SPONSOR_TO_TICKER.get(lead.lower().split(",")[0].strip(), None)
                discovered.append({
                    "sponsor":    lead,
                    "ticker":     ticker,
                    "condition":  condition,
                    "title":      id_m.get("briefTitle", "")[:80],
                    "days_away":  days_away,
                    "pcd":        pcd_raw,
                    "nct_id":     id_m.get("nctId", ""),
                })
        except Exception:
            continue

    # Deduplicate by sponsor, keep soonest trial
    seen: dict[str, dict] = {}
    for d in discovered:
        key = d["sponsor"].lower()
        if key not in seen or d["days_away"] < seen[key]["days_away"]:
            seen[key] = d
    return sorted(seen.values(), key=lambda x: x["days_away"])


# ─── PRICE DATA FETCH ─────────────────────────────────────────────────────────

def fetch_price_signals(tickers: list[str]) -> dict[str, dict]:
    """
    Fetch 12m return, 3m return, 52-week range position, market cap, and
    fallen-angel flag per ticker.

    Fallen angel detection uses 3-year price history:
    A stock that collapsed >65% from its 3-year high is flagged — it's a
    prior blow-up (trial failure, FDA rejection, revenue collapse), not an
    undiscovered early-stage company.
    """
    results = {}
    def _get(t):
        try:
            # 3-year history to detect fallen angels
            hist3 = yf.download(t, period="3y", auto_adjust=True, progress=False)
            if hist3.empty or len(hist3) < 20:
                return t, {}
            close3 = hist3["Close"]
            if isinstance(close3, pd.DataFrame):
                close3 = close3.iloc[:, 0]
            close3 = close3.dropna()

            # 12-month and 3-month windows within the 3yr history
            close   = close3.iloc[-252:] if len(close3) >= 252 else close3
            ret_12m = float(close.iloc[-1] / close.iloc[0] - 1)
            idx_3m  = max(0, len(close) - 63)
            ret_3m  = float(close.iloc[-1] / close.iloc[idx_3m] - 1)
            high_52 = float(close.max())
            low_52  = float(close.min())
            rng     = high_52 - low_52
            range_pos = float((close.iloc[-1] - low_52) / (rng + 1e-8))

            # Fallen angel: stock is down >65% from its 3-year high
            high_3yr = float(close3.max())
            cur_px   = float(close3.iloc[-1])
            dd_3yr   = (cur_px - high_3yr) / (high_3yr + 1e-8)   # negative = drawdown
            fallen_angel = dd_3yr < -0.65   # >65% off 3-year high

            # Market cap
            mkt_cap = 0.0
            try:
                info    = yf.Ticker(t).fast_info
                mkt_cap = float(getattr(info, "market_cap", None) or 0)
            except Exception:
                pass

            return t, {
                "ret_12m":      ret_12m,
                "ret_3m":       ret_3m,
                "range_pos":    float(np.clip(range_pos, 0, 1)),
                "price":        cur_px,
                "market_cap":   mkt_cap,
                "dd_3yr":       round(dd_3yr, 3),
                "fallen_angel": fallen_angel,
            }
        except Exception:
            return t, {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        for t, d in pool.map(_get, tickers):
            results[t] = d
    return results


# ─── MECHANISM VALIDATION ─────────────────────────────────────────────────────

def compute_validation_boosts(tickers: list[str], price_data: dict) -> dict[str, float]:
    """
    For each ticker that is a MECHANISM_VALIDATORS key:
    if it has strong recent price momentum (likely positive data), boost its class peers.
    """
    boosts: dict[str, float] = {}
    for validator, beneficiaries in MECHANISM_VALIDATORS.items():
        if validator not in price_data:
            continue
        vdata = price_data.get(validator, {})
        ret_3m = vdata.get("ret_3m", 0.0)
        # Strong 3-month momentum in the validator → class gets a small boost
        if ret_3m > 0.25:   # >25% in 3 months — 15% is ordinary biotech noise
            boost = min(VALIDATION_BOOST, ret_3m * 0.2)
            for b in beneficiaries:
                boosts[b] = max(boosts.get(b, 0.0), boost)
    return boosts


# ─── MAIN SCORING RUN ─────────────────────────────────────────────────────────

def score_pipeline(area_filter: str | None = None,
                   verbose: bool = True) -> list[dict]:
    """Score all tickers in BIOTECH_UNIVERSE. Returns sorted list of score dicts."""
    # Flatten universe — optionally filtered by area
    flat: dict[str, tuple[str, str]] = {}   # ticker → (area, desc)
    for area, companies in BIOTECH_UNIVERSE.items():
        if area_filter and area_filter.lower() not in area.lower():
            continue
        for t, desc in companies.items():
            flat[t] = (area, desc)

    all_tickers = list(flat.keys())
    # Also fetch price data for validators not in the filtered set
    validators_needed = [v for v in MECHANISM_VALIDATORS if v not in flat]
    price_tickers = list(set(all_tickers + validators_needed))

    if verbose:
        print(f"\nBiotech Pipeline Intelligence — {len(all_tickers)} companies")
        print(f"  Fetching price signals ({len(price_tickers)} tickers)...")
    price_data = fetch_price_signals(price_tickers)

    if verbose:
        print(f"  Computing mechanism validation boosts...")
    boosts = compute_validation_boosts(all_tickers, price_data)

    if verbose:
        print(f"  Querying ClinicalTrials.gov for active Phase 2/3 trials...")

    scored = []
    def _score(args):
        t, (area, desc) = args
        trials = fetch_trials(desc, t)
        boost  = boosts.get(t, 0.0)
        result = compute_pipeline_score(t, desc, trials, price_data, boost)
        result["area"] = area
        return result

    with ThreadPoolExecutor(max_workers=6) as pool:
        for result in pool.map(_score, flat.items()):
            scored.append(result)

    scored.sort(key=lambda x: -x["pipeline_score"])
    return scored


# ─── DISPLAY ──────────────────────────────────────────────────────────────────

def print_scores(scores: list[dict], top_n: int = 999) -> None:
    print(f"\n{'='*74}")
    print(f"BIOTECH PIPELINE SCORES  —  {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'='*74}")
    print(f"{'Ticker':<6} {'Score':>6} {'Phase':>6} {'Prox':>6} {'Need':>6} "
          f"{'PreDisc':>7} {'12mRet':>7} {'Boost':>6} {'Add?':>5}  Area / Description")
    print("-" * 80)

    for r in scores[:top_n]:
        is_fa     = r.get("fallen_angel", False)
        fa_flag   = " ⚠ review" if is_fa else ""
        add_flag  = "? RVW" if is_fa else ("★ YES" if r["suggest_add"] else "  —  ")
        phase_s   = f"{r['trial_phase']:.2f}"
        prox_s    = f"{r['proximity']:.2f}"
        need_s    = f"{r['unmet_need']:.2f}"
        disc_s    = f"{r.get('pre_discovery', 0):.2f}"
        ret12_s   = f"{r.get('ret_12m', 0):+.0%}"
        boost_s   = f"+{r['validation_boost']:.2f}" if r["validation_boost"] > 0 else "   — "
        print(f"{r['ticker']:<6} {r['pipeline_score']:>6.3f} {phase_s:>6} "
              f"{prox_s:>6} {need_s:>6} {disc_s:>7} {ret12_s:>7} {boost_s:>6} {add_flag}"
              f"{fa_flag}  {r['area'][:20]} — {r['description'][:30]}")

        if r.get("trials"):
            # Show the NEAREST upcoming readout — the one driving the proximity score
            upcoming = [t for t in r["trials"] if (t.get("days_away") or -1) >= 0]
            best = sorted(upcoming or r["trials"],
                          key=lambda t: t.get("days_away") if t.get("days_away") is not None else 9999)[:1]
            for tr in best:
                d = tr.get("days_away")
                d_str = f"{d}d away" if d is not None else "date TBD"
                print(f"       {tr['phase']:<8} {tr['condition'][:35]:<35}  "
                      f"completion: {d_str}")

    candidates = [r for r in scores if r["suggest_add"]]
    print(f"\n{'='*74}")
    print(f"AUTO-ADD CANDIDATES (score >= {PIPELINE_SCORE_THRESHOLD}):")
    if candidates:
        tickers = list(dict.fromkeys(r["ticker"] for r in candidates))  # dedup, preserve order
        print(f"  {', '.join(tickers)}")
        base = ["GH", "RVMD"]
        merged = list(dict.fromkeys(base + tickers))
        print(f"\n  Add these to CUSTOM_TICKERS in portfolio_analyzer_advanced.py:")
        print(f'  CUSTOM_TICKERS = {json.dumps(merged)}')
    else:
        print("  None above threshold — review individual scores above.")
    print(f"{'='*74}")


# ─── EXPORT ───────────────────────────────────────────────────────────────────

def export_scores(scores: list[dict]) -> str:
    path = str(OUTPUT_DIR / "pipeline_scores.json")
    export = [{k: v for k, v in s.items() if k != "trials"} for s in scores]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"generated": datetime.now().isoformat(),
                   "threshold": PIPELINE_SCORE_THRESHOLD,
                   "scores": export}, f, indent=2, default=str)
    return path


def as_catalyst_signals(scores: list[dict]) -> dict:
    """
    Convert pipeline scores to the catalyst_signals dict format used by
    portfolio_analyzer_advanced.py — allows pipeline scores to influence
    the multi-factor screen catalyst component.
    """
    out = {}
    for r in scores:
        if r["pipeline_score"] < 0.40:
            continue
        score_normalised = (r["pipeline_score"] - 0.40) / 0.60   # 0.40-1.0 → 0-1
        out[r["ticker"]] = {
            "signal":        "POSITIVE" if r["pipeline_score"] >= PIPELINE_SCORE_THRESHOLD else "NEUTRAL",
            "score":         round(score_normalised * 2 - 1, 2),   # → -1…+1 scale
            "catalyst_type": "Pipeline Score",
            "headline":      (f"Phase {'3' if r['trial_phase']>=0.9 else '2'} "
                              f"trial — {r.get('trials', [{}])[0].get('condition', 'oncology') if r.get('trials') else r['area']} "
                              f"({r['catalyst_days']}d to completion)"
                              if r.get("catalyst_days") else
                              f"Pipeline score {r['pipeline_score']:.2f} — {r['area']}"),
            "days_ago":      0,
        }
    return out


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Biotech Pipeline Intelligence")
    parser.add_argument("--top",      type=int, default=999, help="Show top N by score")
    parser.add_argument("--area",     type=str, default=None, help="Filter by therapeutic area")
    parser.add_argument("--ticker",   type=str, default=None, help="Single-ticker deep-dive")
    parser.add_argument("--export",   action="store_true",    help="Export scores to JSON")
    parser.add_argument("--discover", action="store_true",
                        help="Query ClinicalTrials.gov for pre-breakout Phase 3 sponsors")
    args = parser.parse_args()

    if args.ticker:
        # Find ticker in universe
        found = None
        for area, companies in BIOTECH_UNIVERSE.items():
            if args.ticker.upper() in companies:
                found = (area, companies[args.ticker.upper()])
                break
        if not found:
            print(f"⚠ {args.ticker.upper()} not found in BIOTECH_UNIVERSE — "
                  f"add it under the appropriate therapeutic area first.")
            sys.exit(1)
        area, desc = found
        price_data = fetch_price_signals([args.ticker.upper()])
        boosts     = compute_validation_boosts([args.ticker.upper()], price_data)
        trials     = fetch_trials(desc, args.ticker.upper())
        result     = compute_pipeline_score(
            args.ticker.upper(), desc, trials, price_data,
            boosts.get(args.ticker.upper(), 0.0))
        result["area"] = area
        print_scores([result])
        if trials:
            print(f"\nActive Phase 2/3 trials ({len(trials)}):")
            for t in trials:
                d = t.get("days_away")
                print(f"  [{t['phase']:<8}] {t['condition'][:40]:<40}  "
                      f"completion: {t['pcd']} ({d}d)" if d else
                      f"  [{t['phase']:<8}] {t['condition'][:40]}")
        return

    if args.discover:
        print("\nClinicalTrials.gov Phase 3 Discovery — pre-breakout sponsors")
        print("Searching for companies with Phase 3 completions in next 18 months...\n")
        discovered = discover_pipeline_from_ct(days_window=540)
        if not discovered:
            print("  No results found — check network / API availability.")
        else:
            print(f"{'Sponsor':<40} {'Ticker':>7} {'Days':>6}  Condition / Title")
            print("-" * 80)
            for d in discovered[:30]:
                tk = d["ticker"] or "  ?"
                in_universe = any(
                    tk in companies
                    for companies in BIOTECH_UNIVERSE.values()
                ) if tk != "  ?" else False
                flag = "  [tracked]" if in_universe else "  ← NEW"
                print(f"{d['sponsor'][:38]:<40} {tk:>7} {d['days_away']:>6}d  "
                      f"{d['condition'][:30]}{flag}")
            unknown = [d for d in discovered if not d["ticker"]]
            if unknown:
                print(f"\n  {len(unknown)} sponsor(s) without a mapped ticker — "
                      f"look them up and add to _SPONSOR_TO_TICKER:")
                for d in unknown[:10]:
                    print(f"    '{d['sponsor'].lower()}': '???',  # {d['condition']}")
        return

    scores = score_pipeline(area_filter=args.area)
    print_scores(scores, top_n=args.top)

    if args.export:
        path = export_scores(scores)
        print(f"\n  Scores exported: {path}")


if __name__ == "__main__":
    main()
