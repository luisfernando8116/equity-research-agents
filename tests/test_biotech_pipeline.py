"""Offline tests for biotech_pipeline.py scoring logic. Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import biotech_pipeline as bp  # noqa: E402


def test_name_key_matches_legal_variants():
    assert bp.name_key("Scholar Rock Holding Corporation") == bp.name_key("Scholar Rock, Inc.")
    assert bp.name_key("Revolution Medicines, Inc.") == "revolution medicines"
    assert bp.name_key("Eli Lilly and Company") == "eli lilly"


def test_phase_1_2_scores_below_phase_2():
    assert bp.phase_score("PHASE1/PHASE2") < bp.phase_score("PHASE2") < bp.phase_score("PHASE3")


_PHASE3 = [{"phase": "PHASE3", "days_away": 200, "condition": "Pancreatic Cancer",
            "title": "x"}]


def _score(ret_12m, ret_3m, range_pos, mcap=2e9, dd_3yr=-0.1):
    px = {"T": {"ret_12m": ret_12m, "ret_3m": ret_3m, "range_pos": range_pos,
                "market_cap": mcap, "dd_3yr": dd_3yr, "fallen_angel": dd_3yr < -0.65}}
    return bp.compute_pipeline_score("T", "Test Co — x", _PHASE3, px)


def test_pre_discovery_discriminates_flat_from_falling():
    quiet_turn = _score(ret_12m=0.05, ret_3m=0.10, range_pos=0.35)    # PRAX-at-$53 shape
    falling    = _score(ret_12m=-0.49, ret_3m=-0.08, range_pos=0.10)  # ALNY-like
    late_run   = _score(ret_12m=3.90, ret_3m=0.20, range_pos=0.95)    # PRAX-at-$260
    assert quiet_turn["pre_discovery"] > falling["pre_discovery"]
    assert quiet_turn["pre_discovery"] > late_run["pre_discovery"]
    assert falling["pre_discovery"] < 1.0                              # used to saturate


def test_late_run_cannot_qualify_on_clinical_quality_alone():
    late  = _score(ret_12m=3.75, ret_3m=0.20, range_pos=0.95)   # PRAX at $260
    quiet = _score(ret_12m=0.05, ret_3m=0.10, range_pos=0.35)   # PRAX at $53
    assert late["pipeline_score"] < bp.PIPELINE_SCORE_THRESHOLD
    assert quiet["pipeline_score"] >= bp.PIPELINE_SCORE_THRESHOLD


def test_validation_boost_is_small():
    assert bp.VALIDATION_BOOST <= 0.10
