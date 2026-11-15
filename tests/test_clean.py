import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pressure import clean, config as C, load  # noqa: E402


@pytest.fixture(scope="module")
def tables():
    raw = load.balls()
    m = clean.build_matches(raw)
    return raw, m, clean.build_balls(raw, m)


def test_no_duplicate_ball_keys(tables):
    _, _, b = tables
    assert not b.duplicated(["match_id", "innings", "innings_seq"]).any()


def test_only_two_innings(tables):
    assert set(tables[2].innings.unique()) == {1, 2}


def test_legal_balls_never_exceed_120(tables):
    b = tables[2]
    assert b.groupby(["match_id", "innings"]).is_legal.sum().max() <= C.BALLS_PER_INNINGS


def test_most_full_innings_have_120_legal_balls(tables):
    g = tables[2].groupby(["match_id", "innings"]).is_legal.sum()
    assert (g == 120).mean() > 0.5 and g.median() == 120


def test_total_runs_is_bat_plus_extras(tables):
    b = tables[2]
    assert (b.total_runs == b.bat_runs + b.extra_runs).all()


def test_wide_is_not_a_ball_faced_or_legal(tables):
    w = tables[2][tables[2].is_wide]
    assert not w.batter_faced.any() and not w.is_legal.any()


def test_run_out_not_credited_to_bowler(tables):
    b = tables[2]
    assert not b[b.dismissal_type == "RunOut"].bowler_wicket.any()
    assert b[b.dismissal_type == "Bowled"].bowler_wicket.all()


def test_league_tags(tables):
    m = tables[1]
    assert set(m.league.unique()) == {"IPL", "T20I", "OTHER"}
    assert m[m.league == "IPL"].comp_name.str.startswith("IPL").all()
    assert m[m.league == "OTHER"].comp_name.str.contains("Emerging|Rising").all()


def test_usable_matches_are_clean(tables):
    m = tables[1]
    u = m[m.usable]
    assert (u.n_innings >= 2).all() and not u.is_dls.any() and not u.no_result.any()
    assert u.league.isin(C.RANK_LEAGUES).all()


def test_first_innings_scores_are_plausible(tables):
    b = tables[2]
    tot = b[b.usable & (b.innings == 1)].groupby("match_id").total_runs.sum()
    assert 130 < tot.mean() < 190
