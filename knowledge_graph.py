"""
knowledge_graph.py — Financial Knowledge Graph for Portfolio Analyzer
======================================================================

Models structural relationships between companies: supply chains, competitors,
clinical rivals, and strategic partners. Propagates event-driven catalyst
signals across related tickers to capture second-order effects invisible to
single-stock factor screening.

Examples of what this catches:
  - NVDA earnings beat  → positive signal for TSM (their foundry)
                        → negative signal for AMD (competitor)
  - GH FDA approval     → negative signal for EXAS (clinical rival)
                        → positive signal for ILMN (sequencing partner)
  - TSM capacity cut    → negative signal for NVDA, AMD, AAPL (customers)

Usage:
    from knowledge_graph import FinancialKnowledgeGraph
    kg = FinancialKnowledgeGraph()
    kg.build(tickers)
    enhanced = kg.propagate_signals(catalyst_signals)   # dict of enhanced signals
    html     = kg.to_html_section(portfolio_tickers, enhanced)
"""

import json
import os
import sys
import subprocess
import time
from datetime import datetime

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import networkx as nx
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "networkx"])
    import networkx as nx

# ── Relationship catalogue ────────────────────────────────────────────────────
# base_weight: default propagation weight for this relationship type
#   positive → source's positive event is good for target
#   negative → source's positive event is bad for target (competitive)
# reciprocal: True → edge automatically added in both directions

RELATIONSHIP_TYPES: dict[str, dict] = {
    "COMPETITOR":       {"base_weight": -0.35, "reciprocal": True},
    "SUPPLY_CUSTOMER":  {"base_weight": +0.40, "reciprocal": False},
    "SUPPLY_VENDOR":    {"base_weight": +0.30, "reciprocal": False},
    "CLINICAL_RIVAL":   {"base_weight": -0.40, "reciprocal": True},
    "PARTNER":          {"base_weight": +0.30, "reciprocal": True},
    "SECTOR_PEER":      {"base_weight": -0.10, "reciprocal": True},
    "ACQUISITION":      {"base_weight": +0.60, "reciprocal": False},
}

# ── Hardcoded structural relationships ───────────────────────────────────────
# (source, target, rel_type, weight_override, description)
# These encode well-known structural relationships unlikely to change quickly.

KNOWN_RELATIONSHIPS: list[tuple] = [
    # ── Semiconductors / AI infrastructure ───────────────────────────────────
    ("NVDA", "AMD",    "COMPETITOR",      -0.40, "GPU/AI accelerator direct rival"),
    ("NVDA", "INTC",   "COMPETITOR",      -0.25, "Data center / AI silicon overlap"),
    ("NVDA", "TSM",    "SUPPLY_VENDOR",   +0.45, "TSMC primary fab for all NVDA chips"),
    ("NVDA", "GOOGL",  "SUPPLY_CUSTOMER", +0.20, "Google is a major Blackwell/H100 buyer"),
    ("NVDA", "MSFT",   "SUPPLY_CUSTOMER", +0.20, "Azure buys NVDA GPU at scale"),
    ("NVDA", "AMZN",   "SUPPLY_CUSTOMER", +0.18, "AWS buys NVDA GPU at scale"),
    ("NVDA", "META",   "SUPPLY_CUSTOMER", +0.15, "Meta AI cluster uses NVDA"),
    ("NVDA", "TSLA",   "SUPPLY_CUSTOMER", +0.10, "NVDA AI chips in Tesla FSD"),
    ("NVDA", "AVGO",   "COMPETITOR",      -0.20, "Broadcom custom AI ASIC vs NVDA GPU"),
    # AMD
    ("AMD",  "TSM",    "SUPPLY_VENDOR",   +0.40, "TSMC primary fab for AMD CPUs/GPUs"),
    ("AMD",  "INTC",   "COMPETITOR",      -0.35, "x86 CPU direct rival"),
    ("AMD",  "GOOGL",  "SUPPLY_CUSTOMER", +0.10, "Google uses AMD EPYC in some cloud nodes"),
    ("AMD",  "AMZN",   "SUPPLY_CUSTOMER", +0.12, "AWS Graviton competes; also uses AMD"),
    # TSMC
    ("TSM",  "NVDA",   "SUPPLY_CUSTOMER", +0.45, "NVIDIA is TSMC largest revenue driver"),
    ("TSM",  "AMD",    "SUPPLY_CUSTOMER", +0.30, "AMD designs fabbed by TSMC"),
    ("TSM",  "AAPL",   "SUPPLY_CUSTOMER", +0.35, "Apple A-series SoC fabbed at TSMC"),
    ("TSM",  "QCOM",   "SUPPLY_CUSTOMER", +0.20, "Qualcomm mobile chips at TSMC"),
    ("TSM",  "AVGO",   "SUPPLY_CUSTOMER", +0.15, "Broadcom networking chips at TSMC"),
    ("TSM",  "INTC",   "COMPETITOR",      -0.30, "Intel IDM foundry (IFS) competes with TSMC"),
    # Broadcom / AVGO
    ("AVGO", "AAPL",   "SUPPLY_CUSTOMER", +0.40, "AAPL wireless chips: AVGO's largest customer"),
    ("AVGO", "GOOGL",  "SUPPLY_CUSTOMER", +0.30, "Google TPU networking chips from AVGO"),
    ("AVGO", "META",   "SUPPLY_CUSTOMER", +0.20, "Meta custom AI chip uses AVGO networking"),
    # ASML
    ("ASML", "TSM",    "SUPPLY_CUSTOMER", +0.50, "TSMC largest buyer of ASML EUV tools"),
    ("ASML", "INTC",   "SUPPLY_CUSTOMER", +0.30, "Intel buying ASML EUV for IFS"),
    ("ASML", "SMSN",   "SUPPLY_CUSTOMER", +0.25, "Samsung buys ASML EUV"),
    # Intel
    ("INTC", "AMD",    "COMPETITOR",      -0.35, "x86 CPU competition — AMD gaining share"),
    ("INTC", "NVDA",   "COMPETITOR",      -0.25, "Data center AI accelerator overlap"),
    ("INTC", "TSM",    "COMPETITOR",      -0.20, "IFS vs TSMC foundry competition"),
    # ── Cloud / Hyperscalers / AI ─────────────────────────────────────────────
    ("GOOGL", "MSFT",  "COMPETITOR",      -0.30, "GCP vs Azure; Search vs Copilot; AI"),
    ("GOOGL", "AMZN",  "COMPETITOR",      -0.25, "GCP vs AWS; advertising competition"),
    ("GOOGL", "META",  "COMPETITOR",      -0.30, "Digital advertising rivalry"),
    ("GOOGL", "NVDA",  "SUPPLY_VENDOR",   +0.15, "Google buys NVDA GPUs for AI infra"),
    ("MSFT",  "AMZN",  "COMPETITOR",      -0.22, "Azure vs AWS cloud rivalry"),
    ("MSFT",  "GOOGL", "COMPETITOR",      -0.25, "Cloud + AI + productivity competition"),
    ("MSFT",  "NVDA",  "SUPPLY_VENDOR",   +0.15, "Azure buys NVDA GPU at scale"),
    ("AMZN",  "GOOGL", "COMPETITOR",      -0.20, "AWS vs GCP; advertising overlap"),
    ("AMZN",  "MSFT",  "COMPETITOR",      -0.22, "AWS vs Azure — primary cloud rivalry"),
    ("META",  "GOOGL", "COMPETITOR",      -0.30, "Digital advertising rivalry"),
    ("META",  "SNAP",  "COMPETITOR",      -0.35, "Social / ads competition"),
    ("META",  "PINS",  "COMPETITOR",      -0.25, "Social / ads competition"),
    # ── Biotech / Liquid Biopsy ───────────────────────────────────────────────
    ("GH",   "EXAS",   "CLINICAL_RIVAL",  -0.45, "Exact Sciences: cancer screening rival"),
    ("GH",   "ILMN",   "PARTNER",         +0.30, "Illumina sequencing platform powers GH"),
    ("GH",   "GRAIL",  "CLINICAL_RIVAL",  -0.40, "Galleri multi-cancer test: direct rival"),
    ("GH",   "NVAX",   "SECTOR_PEER",     -0.05, "Biotech sector peer"),
    ("ILMN", "EXAS",   "SUPPLY_CUSTOMER", +0.20, "Exact Sciences uses Illumina sequencing"),
    ("ILMN", "GH",     "SUPPLY_CUSTOMER", +0.25, "Guardant uses Illumina NGS platform"),
    # ── Ride-share / Mobility ─────────────────────────────────────────────────
    ("UBER",  "LYFT",  "COMPETITOR",      -0.55, "Direct US ride-share head-to-head"),
    ("UBER",  "DASH",  "COMPETITOR",      -0.20, "Food delivery overlap (Uber Eats)"),
    ("UBER",  "GOOGL", "PARTNER",         +0.10, "Google Maps embedded in Uber app"),
    ("LYFT",  "UBER",  "COMPETITOR",      -0.55, "Uber dominant; Lyft US-only challenger"),
    # ── EV / Autonomous ───────────────────────────────────────────────────────
    ("TSLA",  "GM",    "COMPETITOR",      -0.25, "EV market share rivals"),
    ("TSLA",  "F",     "COMPETITOR",      -0.20, "EV + ICE vehicle competition"),
    ("TSLA",  "RIVN",  "COMPETITOR",      -0.30, "EV pickup/van segment"),
    ("TSLA",  "NVDA",  "SUPPLY_VENDOR",   +0.12, "NVDA chips in Tesla FSD hardware"),
    ("TSLA",  "GOOGL", "COMPETITOR",      -0.15, "Waymo vs Tesla FSD AV competition"),
    # ── Quantum Computing ──────────────────────────────────────────────────────
    ("IONQ",  "QUBT",  "COMPETITOR",      -0.30, "Trapped-ion vs photonic QC peers"),
    ("IONQ",  "RGTI",  "COMPETITOR",      -0.25, "Quantum hardware peers"),
    ("IONQ",  "QBTS",  "COMPETITOR",      -0.20, "Quantum computing peers"),
    ("QUBT",  "RGTI",  "COMPETITOR",      -0.30, "Quantum hardware peers"),
    ("QUBT",  "QBTS",  "COMPETITOR",      -0.25, "Quantum hardware peers"),
    # ── Space ──────────────────────────────────────────────────────────────────
    ("RKLB",  "ASTS",  "SECTOR_PEER",     -0.10, "Space infrastructure sector peers"),
    ("RKLB",  "SPCE",  "SECTOR_PEER",     -0.10, "Launch / space access peers"),
    ("ASTS",  "VSAT",  "COMPETITOR",      -0.20, "Satellite broadband rivals"),
    # ── Nuclear / Power ────────────────────────────────────────────────────────
    ("CEG",   "NNE",   "SECTOR_PEER",     -0.10, "Nuclear power peers"),
    ("CEG",   "VST",   "COMPETITOR",      -0.15, "Power generation competition"),
    ("CCJ",   "NNE",   "SUPPLY_CUSTOMER", +0.20, "Cameco uranium -> small modular reactors"),
    ("NNE",   "SMR",   "COMPETITOR",      -0.25, "Small modular reactor rivals"),
    ("OKLO",  "NNE",   "COMPETITOR",      -0.25, "Micro-reactor / SMR peers"),
    # ── Robotics / AI Hardware ─────────────────────────────────────────────────
    ("PATH",  "FANUC", "COMPETITOR",      -0.15, "Industrial automation competition"),
    ("AXON",  "SOUN",  "SECTOR_PEER",     -0.05, "AI applications peers"),
    # ── Digital Assets ─────────────────────────────────────────────────────────
    ("MSTR",  "COIN",  "SECTOR_PEER",     -0.10, "Crypto-adjacent equity peers"),
    ("COIN",  "MARA",  "SECTOR_PEER",     -0.10, "Crypto infrastructure peers"),
    ("MARA",  "CLSK",  "COMPETITOR",      -0.30, "Bitcoin mining rivals"),
    # ── Streaming / Entertainment ──────────────────────────────────────────────
    ("NFLX",  "DIS",   "COMPETITOR",      -0.25, "Streaming market competition"),
    ("NFLX",  "PARA",  "COMPETITOR",      -0.20, "Streaming peers"),
    ("NFLX",  "AMZN",  "COMPETITOR",      -0.20, "Amazon Prime Video vs Netflix"),
    # ── Financial ─────────────────────────────────────────────────────────────
    ("GS",    "MS",    "COMPETITOR",      -0.30, "Investment banking / trading rivals"),
    ("JPM",   "BAC",   "COMPETITOR",      -0.25, "Commercial/consumer banking rivals"),
    ("V",     "MA",    "COMPETITOR",      -0.35, "Payment network direct competition"),
    ("PYPL",  "SQ",    "COMPETITOR",      -0.30, "Digital payments rivalry"),
    ("PYPL",  "V",     "COMPETITOR",      -0.20, "Digital vs card network"),
]

# ── Event-type propagation multipliers ───────────────────────────────────────
# How strongly each event type propagates through each relationship type.
# A competitor's positive event propagates negatively (handled by edge sign).

EVENT_PROPAGATION: dict[str, dict] = {
    "FDA Approval":      {"COMPETITOR": 0.80, "CLINICAL_RIVAL": 0.90, "PARTNER": 0.60,
                          "SUPPLY_CUSTOMER": 0.25, "SUPPLY_VENDOR": 0.15, "SECTOR_PEER": 0.20},
    "FDA Rejection":     {"COMPETITOR": 0.90, "CLINICAL_RIVAL": 1.00, "PARTNER": 0.50,
                          "SUPPLY_CUSTOMER": 0.15, "SUPPLY_VENDOR": 0.10, "SECTOR_PEER": 0.20},
    "Clinical Trial":    {"COMPETITOR": 0.65, "CLINICAL_RIVAL": 0.80, "PARTNER": 0.50,
                          "SUPPLY_CUSTOMER": 0.10, "SUPPLY_VENDOR": 0.10, "SECTOR_PEER": 0.10},
    "Trial Failure":     {"COMPETITOR": 0.85, "CLINICAL_RIVAL": 1.00, "PARTNER": 0.40,
                          "SUPPLY_CUSTOMER": 0.10, "SUPPLY_VENDOR": 0.05, "SECTOR_PEER": 0.10},
    "Earnings Beat":     {"COMPETITOR": 0.30, "CLINICAL_RIVAL": 0.20, "PARTNER": 0.30,
                          "SUPPLY_CUSTOMER": 0.50, "SUPPLY_VENDOR": 0.40, "SECTOR_PEER": 0.15},
    "Earnings Miss":     {"COMPETITOR": 0.30, "CLINICAL_RIVAL": 0.20, "PARTNER": 0.30,
                          "SUPPLY_CUSTOMER": 0.50, "SUPPLY_VENDOR": 0.40, "SECTOR_PEER": 0.15},
    "Guidance Raise":    {"COMPETITOR": 0.20, "CLINICAL_RIVAL": 0.10, "PARTNER": 0.30,
                          "SUPPLY_CUSTOMER": 0.60, "SUPPLY_VENDOR": 0.50, "SECTOR_PEER": 0.10},
    "Guidance Cut":      {"COMPETITOR": 0.20, "CLINICAL_RIVAL": 0.10, "PARTNER": 0.30,
                          "SUPPLY_CUSTOMER": 0.60, "SUPPLY_VENDOR": 0.50, "SECTOR_PEER": 0.10},
    "Analyst Upgrade":   {"COMPETITOR": 0.15, "CLINICAL_RIVAL": 0.10, "PARTNER": 0.20,
                          "SUPPLY_CUSTOMER": 0.15, "SUPPLY_VENDOR": 0.15, "SECTOR_PEER": 0.10},
    "M&A":               {"COMPETITOR": 0.45, "CLINICAL_RIVAL": 0.40, "PARTNER": 0.40,
                          "SUPPLY_CUSTOMER": 0.25, "SUPPLY_VENDOR": 0.25, "SECTOR_PEER": 0.25},
    "Buyback":           {"COMPETITOR": 0.10, "CLINICAL_RIVAL": 0.05, "PARTNER": 0.10,
                          "SUPPLY_CUSTOMER": 0.10, "SUPPLY_VENDOR": 0.10, "SECTOR_PEER": 0.05},
    "default":           {"COMPETITOR": 0.20, "CLINICAL_RIVAL": 0.20, "PARTNER": 0.20,
                          "SUPPLY_CUSTOMER": 0.20, "SUPPLY_VENDOR": 0.20, "SECTOR_PEER": 0.08},
}


class FinancialKnowledgeGraph:
    """
    Directed knowledge graph of financial relationships between tickers.

    Build order (layered, each layer adds on top of the previous):
      1. build_hardcoded()      — structural supply chains / known rivals (~200 edges)
      2. build_from_sectors()   — auto-detect sector/industry peers from yfinance data
      3. build_from_trials()    — clinical rivals from ClinicalTrials.gov indication data

    Primary public methods:
      build(...)                — full build pipeline with disk cache
      propagate_signals(...)    — enhance catalyst dict with KG-derived signals
      to_html_section(...)      — HTML snippet for investor report
    """

    VERSION = "1.0"

    def __init__(self, graph_dir: str = None):
        self.graph_dir  = graph_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "kg_data"
        )
        self.graph_path = os.path.join(self.graph_dir, "kg_graph.json")
        self.meta_path  = os.path.join(self.graph_dir, "kg_meta.json")
        self.G: nx.DiGraph = nx.DiGraph()
        self._built = False

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        os.makedirs(self.graph_dir, exist_ok=True)
        data = nx.node_link_data(self.G, edges="links")
        with open(self.graph_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        meta = {
            "version":  self.VERSION,
            "built_at": datetime.now().isoformat(),
            "nodes":    self.G.number_of_nodes(),
            "edges":    self.G.number_of_edges(),
        }
        with open(self.meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        self._built = True

    def load(self) -> bool:
        try:
            with open(self.graph_path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.G = nx.node_link_graph(data, edges="links")
            self._built = True
            return True
        except Exception:
            return False

    def is_fresh(self, max_age_hours: float = 23.0) -> bool:
        try:
            age = time.time() - os.path.getmtime(self.graph_path)
            return age < max_age_hours * 3600
        except Exception:
            return False

    # ── Node / edge helpers ───────────────────────────────────────────────────

    def _ensure_node(self, ticker: str, **attrs) -> None:
        if ticker not in self.G:
            self.G.add_node(ticker, node_type="company", **attrs)
        else:
            self.G.nodes[ticker].update(attrs)

    def add_relationship(self, source: str, target: str, rel_type: str,
                         weight: float = None, **metadata) -> None:
        self._ensure_node(source)
        self._ensure_node(target)
        base_w = RELATIONSHIP_TYPES.get(rel_type, {}).get("base_weight", 0.0)
        w = weight if weight is not None else base_w
        # Keep higher-magnitude edge (don't downgrade a hardcoded relationship)
        existing = self.G.get_edge_data(source, target)
        if existing is None or abs(w) > abs(existing.get("weight", 0)):
            self.G.add_edge(source, target, rel_type=rel_type, weight=w, **metadata)
        # Reciprocal
        if RELATIONSHIP_TYPES.get(rel_type, {}).get("reciprocal", False):
            existing_r = self.G.get_edge_data(target, source)
            if existing_r is None or abs(w) > abs(existing_r.get("weight", 0)):
                self.G.add_edge(target, source, rel_type=rel_type, weight=w, **metadata)

    # ── Graph builders ────────────────────────────────────────────────────────

    def build_hardcoded(self) -> int:
        """Add well-known structural relationships from KNOWN_RELATIONSHIPS."""
        before = self.G.number_of_edges()
        for src, tgt, rel, wt, desc in KNOWN_RELATIONSHIPS:
            self.add_relationship(src, tgt, rel, wt, description=desc, rel_source="hardcoded")
        return self.G.number_of_edges() - before

    def build_from_sectors(self, sector_map: dict,
                           industry_map: dict = None) -> int:
        """
        Auto-detect competitors from same GICS industry.
        Tickers sharing the same sub-industry (≤8 peers) → COMPETITOR.
        Larger groups or same broad sector only → SECTOR_PEER (weak signal).
        """
        industry_map = industry_map or {}
        before = self.G.number_of_edges()
        by_industry: dict[str, list] = {}
        for t, sector in sector_map.items():
            if not sector or sector == "Unknown":
                continue
            self._ensure_node(t, sector=sector)
            ind = industry_map.get(t) or sector
            by_industry.setdefault(ind, []).append(t)
        for ind, group in by_industry.items():
            group = list(set(group))
            if len(group) < 2:
                continue
            rel = "COMPETITOR" if len(group) <= 8 else "SECTOR_PEER"
            wt  = -0.28 if rel == "COMPETITOR" else -0.08
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    if not self.G.has_edge(a, b):
                        self.add_relationship(a, b, rel, wt,
                                              industry=ind, rel_source="auto_sector")
        return self.G.number_of_edges() - before

    def build_from_trials(self, trials_data: dict) -> int:
        """
        Mark tickers with drugs targeting the same indication as CLINICAL_RIVAL.
        trials_data: {ticker: {"trials": [{"indication": str}, ...]}}
        """
        before = self.G.number_of_edges()
        ind_tickers: dict[str, list] = {}
        for ticker, td in (trials_data or {}).items():
            for trial in td.get("trials", []):
                ind = (trial.get("indication") or "").strip().lower()
                if len(ind) > 3:
                    ind_tickers.setdefault(ind, []).append(ticker)
        for ind, group in ind_tickers.items():
            group = list(set(group))
            if len(group) < 2:
                continue
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    if not self.G.has_edge(a, b):
                        self.add_relationship(a, b, "CLINICAL_RIVAL", -0.35,
                                              indication=ind, rel_source="clinicaltrials")
        return self.G.number_of_edges() - before

    def build(self, tickers: list, sector_map: dict = None,
              industry_map: dict = None, trials_data: dict = None,
              force: bool = False) -> None:
        """
        Full build pipeline. Loads from disk if fresh and force=False; rebuilds otherwise.

        Layer order:
          1. Hardcoded structural relationships (always applied)
          2. Sector-based auto-detection (requires sector_map)
          3. Clinical trial rivals (requires trials_data)
        """
        if not force and self.is_fresh() and self.load():
            # Patch in any new universe tickers missing from the cached graph
            for t in tickers:
                self._ensure_node(t)
            return

        self.G = nx.DiGraph()
        for t in tickers:
            self._ensure_node(t)

        n_hard   = self.build_hardcoded()
        n_sector = self.build_from_sectors(sector_map or {}, industry_map or {})
        n_trial  = self.build_from_trials(trials_data or {})
        self._built = True
        self.save()

        print(f"      KG built: {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges "
              f"({n_hard} hardcoded, {n_sector} sector peers, {n_trial} clinical rivals)")

    def enhance_with_sectors(self, sector_map: dict,
                             industry_map: dict = None) -> None:
        """Add sector peers to an already-built graph (called after sector_map is available)."""
        n = self.build_from_sectors(sector_map, industry_map)
        if n > 0:
            self.save()
            print(f"      KG enhanced: +{n} sector-peer edges ({self.G.number_of_edges()} total)")

    # ── Signal propagation ────────────────────────────────────────────────────

    def propagate_signals(self, catalyst_signals: dict,
                          max_hops: int = 2,
                          min_signal: float = 0.05) -> dict:
        """
        Propagate direct catalyst signals through the knowledge graph.

        For each source ticker with a direct catalyst signal, find all reachable
        neighbors (up to max_hops away) and compute an attenuated signal for each.

        Parameters
        ----------
        catalyst_signals : {ticker: {"score": float, "catalyst_type": str, ...}}
        max_hops         : propagation depth (2 = direct neighbors + their neighbors)
        min_signal       : discard propagated signals below this magnitude

        Returns
        -------
        Enhanced copy of catalyst_signals with additional KG-derived entries.
        KG-derived entries include "kg_source" and "kg_hops" keys.
        """
        if not self._built and not self.load():
            return catalyst_signals

        enhanced = dict(catalyst_signals)

        for source, sig in catalyst_signals.items():
            score = float(sig.get("score", 0.0))
            if abs(score) < 0.05 or source not in self.G:
                continue
            event_type  = sig.get("catalyst_type", "default")
            multipliers = _get_multipliers(event_type)
            self._propagate(source, score, sig, event_type, multipliers,
                            enhanced, hop=1, max_hops=max_hops,
                            min_signal=min_signal, visited={source})

        n_new = sum(1 for t, v in enhanced.items()
                    if v.get("kg_source") and t not in catalyst_signals)
        if n_new:
            print(f"      KG propagation: {n_new} new second-order signal(s) derived")
        return enhanced

    def _propagate(self, source: str, score: float, source_sig: dict,
                   event_type: str, multipliers: dict, enhanced: dict,
                   hop: int, max_hops: int, min_signal: float,
                   visited: set) -> None:
        if hop > max_hops or source not in self.G:
            return
        decay = 0.5 ** (hop - 1)   # 1.0 at hop 1, 0.50 at hop 2

        for neighbor, edge_data in self.G[source].items():
            if neighbor in visited:
                continue
            rel_type = edge_data.get("rel_type", "SECTOR_PEER")
            edge_wt  = float(edge_data.get("weight", 0.0))
            ev_mult  = multipliers.get(rel_type,
                                       multipliers.get("SECTOR_PEER", 0.08))

            # edge_wt is negative for competitors — their positive event = bad for rival
            propagated = score * edge_wt * ev_mult * decay

            if abs(propagated) < min_signal:
                continue

            existing_score = enhanced.get(neighbor, {}).get("score", 0.0)
            if abs(propagated) > abs(existing_score):
                direction = "positive" if propagated > 0 else "negative"
                src_event = source_sig.get("headline", event_type)[:60]
                enhanced[neighbor] = {
                    "signal":        "POSITIVE" if propagated > 0 else "NEGATIVE",
                    "score":         round(propagated, 3),
                    "catalyst_type": f"KG {rel_type.replace('_',' ').title()} ({source})",
                    "headline":      (f"{source} {event_type} propagated "
                                      f"as {direction} via {rel_type} — {src_event}"),
                    "days_ago":      source_sig.get("days_ago", 0),
                    "kg_source":     source,
                    "kg_hops":       hop,
                }

            if hop < max_hops:
                self._propagate(
                    neighbor, propagated, source_sig, event_type, multipliers,
                    enhanced, hop + 1, max_hops, min_signal,
                    visited | {neighbor},
                )

    # ── Query helpers ─────────────────────────────────────────────────────────

    def get_relationships(self, ticker: str) -> list[dict]:
        if ticker not in self.G:
            return []
        return [
            {
                "ticker":      n,
                "rel_type":    d.get("rel_type", ""),
                "weight":      d.get("weight", 0.0),
                "description": d.get("description", ""),
                "source":      d.get("rel_source", ""),
            }
            for n, d in self.G[ticker].items()
        ]

    def get_competitors(self, ticker: str) -> list[dict]:
        return [r for r in self.get_relationships(ticker)
                if r["rel_type"] in ("COMPETITOR", "CLINICAL_RIVAL")]

    def get_supply_chain(self, ticker: str) -> dict:
        rels = self.get_relationships(ticker)
        return {
            "customers": [r for r in rels if r["rel_type"] == "SUPPLY_CUSTOMER"],
            "vendors":   [r for r in rels if r["rel_type"] == "SUPPLY_VENDOR"],
        }

    def summary(self) -> dict:
        rel_counts = {}
        for _, _, d in self.G.edges(data=True):
            rt = d.get("rel_type", "OTHER")
            rel_counts[rt] = rel_counts.get(rt, 0) + 1
        return {
            "nodes":      self.G.number_of_nodes(),
            "edges":      self.G.number_of_edges(),
            "rel_counts": rel_counts,
        }

    # ── HTML report section ───────────────────────────────────────────────────

    def to_html_section(self, tickers: list, catalyst_signals: dict = None) -> str:
        """
        Return an HTML block showing KG relationships for the given tickers.
        Highlights any tickers that received KG-propagated signals.
        """
        if not self._built and not self.load():
            return ""

        catalyst_signals = catalyst_signals or {}
        rel_tickers = [t for t in tickers if t in self.G]
        if not rel_tickers:
            return ""

        def esc(s: str) -> str:
            return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        smry = self.summary()
        parts = [
            "<h2>Knowledge Graph — Relationship Map</h2>",
            "<p>Structural relationships used to propagate catalyst signals across "
            "related tickers. A competitor's FDA approval is a negative signal for rivals; "
            "a key customer's earnings beat is a positive signal for their supplier.</p>",
            f'<p style="font-size:0.8rem;color:var(--muted)">'
            f'Graph: {smry["nodes"]} companies &nbsp;|&nbsp; '
            f'{smry["edges"]} relationships &nbsp;|&nbsp; '
            f'Competitors: {smry["rel_counts"].get("COMPETITOR",0)} &nbsp;|&nbsp; '
            f'Supply chain: {smry["rel_counts"].get("SUPPLY_CUSTOMER",0)+smry["rel_counts"].get("SUPPLY_VENDOR",0)} &nbsp;|&nbsp; '
            f'Clinical rivals: {smry["rel_counts"].get("CLINICAL_RIVAL",0)} &nbsp;|&nbsp; '
            f'Partners: {smry["rel_counts"].get("PARTNER",0)}</p>',
        ]

        for ticker in sorted(rel_tickers):
            rels = sorted(self.get_relationships(ticker),
                          key=lambda r: abs(r["weight"]), reverse=True)[:12]
            if not rels:
                continue

            sig = catalyst_signals.get(ticker, {})
            kg_badge = ""
            if sig.get("kg_source"):
                kg_badge = (
                    f' <span style="background:var(--accent);color:#000;'
                    f'font-size:0.7rem;padding:1px 5px;border-radius:3px">'
                    f'KG signal from {esc(sig["kg_source"])}'
                    f'</span>'
                )

            parts.append(
                f'<h3 style="margin:16px 0 4px">'
                f'{esc(ticker)}{kg_badge}</h3>'
            )
            parts.append(
                '<table><thead><tr>'
                '<th>Related Ticker</th><th>Relationship</th>'
                '<th class="num">Propagation Weight</th><th>Description</th>'
                '</tr></thead><tbody>'
            )
            for r in rels:
                color = "good" if r["weight"] > 0 else "bad"
                src_badge = (
                    f' <span style="font-size:0.7rem;color:var(--muted)">'
                    f'[{esc(r["source"])}]</span>'
                    if r.get("source") else ""  # "source" key set by get_relationships()
                )
                parts.append(
                    f'<tr>'
                    f'<td><strong>{esc(r["ticker"])}</strong></td>'
                    f'<td>{esc(r["rel_type"].replace("_"," ").title())}{src_badge}</td>'
                    f'<td class="num {color}">{r["weight"]:+.2f}</td>'
                    f'<td style="font-size:0.8rem">{esc(r.get("description",""))}</td>'
                    f'</tr>'
                )
            parts.append("</tbody></table>")

        return "\n".join(parts)


# ── Module helpers ────────────────────────────────────────────────────────────

def _get_multipliers(event_type: str) -> dict:
    """Return propagation multipliers for the given event type, falling back to default."""
    for key in EVENT_PROPAGATION:
        if key.lower() in event_type.lower() or event_type.lower() in key.lower():
            return EVENT_PROPAGATION[key]
    return EVENT_PROPAGATION["default"]
