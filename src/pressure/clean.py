"""Step 1: raw CSVs -> clean parquet tables in data/interim/.

Outputs
  matches.parquet : one row per match, with league tag + quality flags
  balls.parquet   : one row per delivery (innings 1-2 only), with run / ball / wicket rules applied
  players.parquet : player_id -> name
"""
import numpy as np
import pandas as pd
from . import config as C
from . import load


def tag_league(comp_name: str, match_format: str) -> str:
    if str(comp_name).startswith("IPL"):
        return "IPL"
    if match_format == "T20I":
        return "T20I"
    return "OTHER"


def build_matches(balls_raw: pd.DataFrame) -> pd.DataFrame:
    m = load.matches().merge(
        load.competitions()[["comp_id", "comp_name", "match_format", "season"]], on="comp_id", how="left"
    )
    m["league"] = [tag_league(a, b) for a, b in zip(m.comp_name, m.match_format)]
    inn = balls_raw.groupby("match_id").innings.agg(["nunique", "max"])
    m["has_balls"] = m.match_id.isin(inn.index)
    m["n_innings"] = m.match_id.map(inn["nunique"]).fillna(0).astype(int)
    m["has_super_over"] = m.match_id.map(inn["max"]).fillna(0) > 2
    m["no_result"] = m.result.fillna("").str.contains("|".join(C.NO_RESULT_TEXT), case=False)
    # rain-affected: revised overs/target given, or result decided by D/L
    m["is_dls"] = (
        m.revised_target.notna() | m.first_inn_revised_overs.notna() | m.second_inn_revised_overs.notna()
        | m.result.fillna("").str.contains(r"D/L|DLS", case=False)
    )
    m["usable"] = (
        m.has_balls & (m.n_innings >= 2) & ~m.no_result & ~m.is_dls & m.league.isin(C.RANK_LEAGUES)
    )
    keep = ["match_id", "comp_id", "comp_name", "season", "match_format", "league", "match_date", "stage",
            "venue_id", "venue_country", "home_team_id", "away_team_id", "toss_win_team_id", "toss_opted",
            "win_team_id", "result", "knock_out", "has_balls", "n_innings", "has_super_over", "no_result",
            "is_dls", "usable"]
    return m[keep]


def build_balls(balls_raw: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    b = balls_raw[balls_raw.innings <= 2].copy()  # innings 3-4 are Super Overs -> out of scope
    b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

    ex = b.extras_type.fillna("None")
    b["bat_runs"] = b.runs.astype(int)
    b["extra_runs"] = (b.extras1_runs + b.extras_2_runs).astype(int)
    b["total_runs"] = b.bat_runs + b.extra_runs
    b["is_wide"] = ex.eq("wide")
    b["is_noball"] = ex.isin(C.NOBALL_EXTRAS)
    b["is_legal"] = ~ex.isin(C.ILLEGAL_EXTRAS)
    b["batter_faced"] = ~b.is_wide & ~ex.eq("Penalty")  # a wide is not a ball faced by the batter
    b["bowler_runs"] = np.where(ex.isin(C.NOT_BOWLER_EXTRAS), b.bat_runs, b.total_runs)
    b["is_dot_bowler"] = b.is_legal & (b.bowler_runs == 0)
    b["is_dot_batter"] = b.batter_faced & (b.bat_runs == 0)
    b["is_boundary"] = b.is_four | b.is_six
    b["is_wicket"] = b.is_wicket.astype(bool)
    b["bowler_wicket"] = b.is_wicket & b.dismissal_type.isin(C.BOWLER_WICKETS)
    b["over0"] = b.overid - 1  # 0-based over (0..19)
    b["innings_seq"] = b.groupby(["match_id", "innings"]).cumcount()  # row order within innings
    b["legal_balls_before"] = b.groupby(["match_id", "innings"]).is_legal.cumsum() - b.is_legal.astype(int)

    mm = matches[["match_id", "league", "comp_name", "season", "match_date", "usable", "win_team_id"]]
    b = b.merge(mm, on="match_id", how="left")
    cols = ["match_id", "league", "comp_name", "season", "match_date", "usable", "win_team_id", "innings",
            "innings_seq", "over0", "over_ball_num", "legal_balls_before", "batting_team_id", "bowling_team_id",
            "striker_id", "striker_name", "non_striker_id", "non_striker_name", "bowler_id", "bowler_name",
            "batting_order", "batting_type", "bowling_type",
            "pace_or_spin", "bat_runs", "extra_runs", "total_runs", "bowler_runs", "is_wide", "is_noball",
            "is_legal", "batter_faced", "is_four", "is_six", "is_boundary", "is_dot_batter", "is_dot_bowler",
            "is_wicket", "bowler_wicket", "dismissal_type", "dismissed_player_id", "extras_type"]
    return b[cols]


def build_players(balls_raw: pd.DataFrame) -> pd.DataFrame:
    parts = [
        balls_raw[["striker_id", "striker_name"]].set_axis(["player_id", "player_name"], axis=1),
        balls_raw[["non_striker_id", "non_striker_name"]].set_axis(["player_id", "player_name"], axis=1),
        balls_raw[["bowler_id", "bowler_name"]].set_axis(["player_id", "player_name"], axis=1),
    ]
    return pd.concat(parts).dropna().drop_duplicates("player_id").sort_values("player_id").reset_index(drop=True)


def run() -> dict:
    C.INTERIM.mkdir(parents=True, exist_ok=True)
    raw = load.balls()
    matches = build_matches(raw)
    balls = build_balls(raw, matches)
    players = build_players(raw)
    matches.to_parquet(C.INTERIM / "matches.parquet", index=False)
    balls.to_parquet(C.INTERIM / "balls.parquet", index=False)
    players.to_parquet(C.INTERIM / "players.parquet", index=False)
    u = balls[balls.usable]
    summary = {
        "matches_total": len(matches), "matches_usable": int(matches.usable.sum()),
        "balls_raw": len(raw), "balls_clean": len(balls), "balls_usable": len(u),
        "usable_balls_by_league": u.league.value_counts().to_dict(),
        "dropped_super_over_balls": int((raw.innings > 2).sum()),
    }
    return summary
