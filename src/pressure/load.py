"""Read the raw CSVs exactly as provided (no cleaning here)."""
import pandas as pd
from . import config as C


def _read(name: str, **kw) -> pd.DataFrame:
    path = C.RAW / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Put {name}.csv in {C.RAW}")
    return pd.read_csv(path, **kw)


def balls() -> pd.DataFrame:
    return _read("ball_by_ball")


def matches() -> pd.DataFrame:
    return _read("matches", parse_dates=["match_date"])


def competitions() -> pd.DataFrame:
    return _read("competitions")


def teams() -> pd.DataFrame:
    return _read("teams")


def mapping() -> pd.DataFrame:
    return _read("competition_team_player_mapping")


def playing_eleven() -> pd.DataFrame:
    return _read("playing_eleven")
