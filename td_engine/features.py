import numpy as np
import pandas as pd
from .config import RED_ZONE_YARDLINE, GOAL_LINE_YARDLINE


def _first_present(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_player_game_features(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per player-game from nflverse-style play-by-play.

    touches = rush attempts + receptions
    red-zone opportunities = rush attempts + targets inside opponent 20
    goal-line opportunities = rush attempts + targets inside opponent 5
    """
    df = pbp.copy()

    needed = ["game_id", "posteam", "yardline_100"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for c in ["rush_attempt", "complete_pass", "touchdown"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    rid = _first_present(df, ["rusher_player_id", "rusher_id"])
    rname = _first_present(df, ["rusher_player_name", "rusher_name"])
    recid = _first_present(df, ["receiver_player_id", "receiver_id"])
    recname = _first_present(df, ["receiver_player_name", "receiver_name"])

    if rid is None or recid is None:
        raise ValueError("Could not find rusher/receiver player ID columns.")

    rush = df[df["rush_attempt"].eq(1) & df[rid].notna()].copy()
    rush["player_id"] = rush[rid]
    rush["player_name"] = rush[rname] if rname else rush[rid]
    rush["rush_att"] = 1
    rush["targets"] = 0
    rush["receptions"] = 0
    rush["touches"] = 1
    rush["rz_opp"] = (rush["yardline_100"] <= RED_ZONE_YARDLINE).astype(int)
    rush["gl_opp"] = (rush["yardline_100"] <= GOAL_LINE_YARDLINE).astype(int)
    rush["td"] = rush["touchdown"].astype(int)

    rec = df[df[recid].notna()].copy()
    rec["player_id"] = rec[recid]
    rec["player_name"] = rec[recname] if recname else rec[recid]
    rec["rush_att"] = 0
    rec["targets"] = 1
    rec["receptions"] = rec["complete_pass"].astype(int)
    rec["touches"] = rec["receptions"]
    rec["rz_opp"] = (rec["yardline_100"] <= RED_ZONE_YARDLINE).astype(int)
    rec["gl_opp"] = (rec["yardline_100"] <= GOAL_LINE_YARDLINE).astype(int)

    if "pass_touchdown" in rec.columns:
        rec["td"] = pd.to_numeric(rec["pass_touchdown"], errors="coerce").fillna(0).astype(int)
    else:
        rec["td"] = rec["touchdown"].astype(int)

    cols = [
        "game_id", "posteam", "player_id", "player_name",
        "rush_att", "targets", "receptions", "touches",
        "rz_opp", "gl_opp", "td"
    ]

    long = pd.concat([rush[cols], rec[cols]], ignore_index=True)

    p = (
        long.groupby(["game_id", "posteam", "player_id", "player_name"], as_index=False)
        [["rush_att", "targets", "receptions", "touches", "rz_opp", "gl_opp", "td"]]
        .sum()
    )

    p["scored_td"] = (p["td"] > 0).astype(int)

    team = (
        p.groupby(["game_id", "posteam"], as_index=False)
        .agg(
            team_touches=("touches", "sum"),
            team_targets=("targets", "sum"),
            team_rz_opp=("rz_opp", "sum"),
            team_gl_opp=("gl_opp", "sum"),
        )
    )

    out = p.merge(team, on=["game_id", "posteam"], how="left")

    def safe_share(num, den):
        return np.where(den > 0, num / den, 0.0)

    out["red_zone_share"] = safe_share(out["rz_opp"], out["team_rz_opp"])
    out["volume_share"] = safe_share(out["touches"], out["team_touches"])
    out["goal_line_share"] = safe_share(out["gl_opp"], out["team_gl_opp"])
    out["target_share"] = safe_share(out["targets"], out["team_targets"])

    return out
