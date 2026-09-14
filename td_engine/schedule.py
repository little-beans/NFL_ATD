from __future__ import annotations
import pandas as pd


def load_schedules_pandas(seasons):
    import nflreadpy as nfl
    x = nfl.load_schedules(seasons=seasons)
    return x.to_pandas() if hasattr(x, "to_pandas") else pd.DataFrame(x)


def schedule_to_team_rows(schedules: pd.DataFrame) -> pd.DataFrame:
    """Convert home/away schedule rows into one row per team-game."""
    s=schedules.copy()
    if "game_type" in s.columns:
        s=s[s["game_type"].fillna("REG").eq("REG")].copy()
    needed=["season","week","game_id","home_team","away_team"]
    miss=[c for c in needed if c not in s.columns]
    if miss: raise ValueError(f"Schedule missing: {miss}")

    def col(name, default=None):
        return s[name] if name in s.columns else default

    home=pd.DataFrame({
        "season":s["season"], "week":s["week"], "game_id":s["game_id"],
        "posteam":s["home_team"], "defteam":s["away_team"], "is_home":1,
        "total_line":col("total_line"), "team_spread":col("spread_line"),
        "rest_days":col("home_rest"), "gameday":col("gameday"), "gametime":col("gametime"),
    })
    away=pd.DataFrame({
        "season":s["season"], "week":s["week"], "game_id":s["game_id"],
        "posteam":s["away_team"], "defteam":s["home_team"], "is_home":0,
        "total_line":col("total_line"),
        # nflverse spread_line is expressed for the home team; invert for away.
        "team_spread":(-s["spread_line"] if "spread_line" in s.columns else None),
        "rest_days":col("away_rest"), "gameday":col("gameday"), "gametime":col("gametime"),
    })
    out=pd.concat([home,away], ignore_index=True)
    out["total_line"]=pd.to_numeric(out["total_line"], errors="coerce")
    out["team_spread"]=pd.to_numeric(out["team_spread"], errors="coerce")
    # Positive team_spread means favored by that many points in nflverse schedule convention.
    out["implied_team_points"] = out["total_line"] / 2.0 + out["team_spread"] / 2.0
    return out


def load_week_team_schedule(season:int, week:int) -> pd.DataFrame:
    s=load_schedules_pandas([season])
    out=schedule_to_team_rows(s)
    return out[(out["season"].eq(season)) & (out["week"].eq(week))].copy()
