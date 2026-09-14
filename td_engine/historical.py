from __future__ import annotations

import numpy as np
import pandas as pd


SKILL_POSITIONS = {"RB", "WR", "TE", "FB", "HB"}


def _coalesce(df: pd.DataFrame, columns, default=np.nan):
    for col in columns:
        if col in df.columns:
            return df[col]
    return pd.Series(default, index=df.index)


def build_player_game_table(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Convert nflverse play-by-play into a historical player-game opportunity table.

    Each row describes what happened IN that game. This is an outcome table, not a
    pregame feature table. Use ``build_rolling_pregame_features`` afterward.
    """
    df = pbp.copy()

    required = ["game_id", "posteam", "yardline_100"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"PBP is missing required columns: {missing}")

    # Standard game metadata.
    season = _coalesce(df, ["season"])
    week = _coalesce(df, ["week"])
    season_type = _coalesce(df, ["season_type"], "REG")
    opponent = _coalesce(df, ["defteam"])
    game_date = _coalesce(df, ["game_date"])

    # Normalize booleans/numerics used below.
    numeric_flags = [
        "rush_attempt", "complete_pass", "touchdown", "pass_touchdown",
        "rush_touchdown", "two_point_conv_result", "qb_scramble",
    ]
    for c in numeric_flags:
        if c in df.columns and c != "two_point_conv_result":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Exclude two-point tries from TD-model opportunity accounting.
    if "two_point_conv_result" in df.columns:
        is_two_point = df["two_point_conv_result"].notna()
    elif "two_point_attempt" in df.columns:
        is_two_point = pd.to_numeric(df["two_point_attempt"], errors="coerce").fillna(0).eq(1)
    else:
        is_two_point = pd.Series(False, index=df.index)

    df = df.loc[~is_two_point].copy()

    # Re-resolve metadata after filtering.
    meta_cols = [c for c in ["game_id", "season", "week", "season_type", "game_date", "posteam", "defteam"] if c in df.columns]

    rid = next((c for c in ["rusher_player_id", "rusher_id"] if c in df.columns), None)
    rname = next((c for c in ["rusher_player_name", "rusher_name"] if c in df.columns), None)
    recid = next((c for c in ["receiver_player_id", "receiver_id"] if c in df.columns), None)
    recname = next((c for c in ["receiver_player_name", "receiver_name"] if c in df.columns), None)

    if rid is None or recid is None:
        raise ValueError("Could not identify rusher/receiver player ID columns in PBP.")

    rush_mask = pd.to_numeric(df.get("rush_attempt", 0), errors="coerce").fillna(0).eq(1) & df[rid].notna()
    rush = df.loc[rush_mask].copy()
    rush["player_id"] = rush[rid]
    rush["player_name"] = rush[rname] if rname else rush[rid]
    rush["rush_att"] = 1
    rush["targets"] = 0
    rush["receptions"] = 0
    rush["touches"] = 1
    rush["rz_opp"] = (pd.to_numeric(rush["yardline_100"], errors="coerce") <= 20).astype(int)
    rush["inside10_opp"] = (pd.to_numeric(rush["yardline_100"], errors="coerce") <= 10).astype(int)
    rush["inside5_opp"] = (pd.to_numeric(rush["yardline_100"], errors="coerce") <= 5).astype(int)
    rush["rz_carry"] = rush["rz_opp"]
    rush["inside10_carry"] = rush["inside10_opp"]
    rush["inside5_carry"] = rush["inside5_opp"]
    rush["rz_target"] = 0
    rush["inside10_target"] = 0
    rush["inside5_target"] = 0
    if "rush_touchdown" in rush.columns:
        rush["td"] = rush["rush_touchdown"].astype(int)
    else:
        rush["td"] = pd.to_numeric(rush.get("touchdown", 0), errors="coerce").fillna(0).astype(int)

    target_mask = df[recid].notna()
    rec = df.loc[target_mask].copy()
    rec["player_id"] = rec[recid]
    rec["player_name"] = rec[recname] if recname else rec[recid]
    rec["rush_att"] = 0
    rec["targets"] = 1
    rec["receptions"] = pd.to_numeric(rec.get("complete_pass", 0), errors="coerce").fillna(0).astype(int)
    rec["touches"] = rec["receptions"]
    rec["rz_opp"] = (pd.to_numeric(rec["yardline_100"], errors="coerce") <= 20).astype(int)
    rec["inside10_opp"] = (pd.to_numeric(rec["yardline_100"], errors="coerce") <= 10).astype(int)
    rec["inside5_opp"] = (pd.to_numeric(rec["yardline_100"], errors="coerce") <= 5).astype(int)
    rec["rz_carry"] = 0
    rec["inside10_carry"] = 0
    rec["inside5_carry"] = 0
    rec["rz_target"] = rec["rz_opp"]
    rec["inside10_target"] = rec["inside10_opp"]
    rec["inside5_target"] = rec["inside5_opp"]
    if "pass_touchdown" in rec.columns:
        rec["td"] = rec["pass_touchdown"].astype(int)
    else:
        rec["td"] = pd.to_numeric(rec.get("touchdown", 0), errors="coerce").fillna(0).astype(int)

    event_cols = [
        "game_id", "posteam", "player_id", "player_name",
        "rush_att", "targets", "receptions", "touches", "td",
        "rz_opp", "inside10_opp", "inside5_opp",
        "rz_carry", "inside10_carry", "inside5_carry",
        "rz_target", "inside10_target", "inside5_target",
    ]
    for c in ["season", "week", "season_type", "game_date", "defteam"]:
        if c in df.columns:
            event_cols.insert(1, c)

    events = pd.concat([rush[event_cols], rec[event_cols]], ignore_index=True)

    group_cols = [c for c in ["season", "week", "season_type", "game_date", "game_id", "posteam", "defteam", "player_id", "player_name"] if c in events.columns]
    value_cols = [
        "rush_att", "targets", "receptions", "touches", "td",
        "rz_opp", "inside10_opp", "inside5_opp",
        "rz_carry", "inside10_carry", "inside5_carry",
        "rz_target", "inside10_target", "inside5_target",
    ]

    player = events.groupby(group_cols, as_index=False)[value_cols].sum()
    player["scored_td"] = (player["td"] > 0).astype(int)

    team_group = [c for c in ["season", "week", "game_id", "posteam"] if c in player.columns]
    team = (
        player.groupby(team_group, as_index=False)
        .agg(
            team_touches=("touches", "sum"),
            team_rush_att=("rush_att", "sum"),
            team_targets=("targets", "sum"),
            team_rz_opp=("rz_opp", "sum"),
            team_inside10_opp=("inside10_opp", "sum"),
            team_inside5_opp=("inside5_opp", "sum"),
            team_rz_carry=("rz_carry", "sum"),
            team_inside10_carry=("inside10_carry", "sum"),
            team_inside5_carry=("inside5_carry", "sum"),
            team_rz_target=("rz_target", "sum"),
            team_inside10_target=("inside10_target", "sum"),
            team_inside5_target=("inside5_target", "sum"),
            team_tds=("td", "sum"),
        )
    )
    out = player.merge(team, on=team_group, how="left")

    def safe(num, den):
        return np.divide(num, den, out=np.zeros(len(out), dtype=float), where=np.asarray(den) > 0)

    out["red_zone_share_game"] = safe(out["rz_opp"], out["team_rz_opp"])
    out["inside10_share_game"] = safe(out["inside10_opp"], out["team_inside10_opp"])
    out["goal_line_share_game"] = safe(out["inside5_opp"], out["team_inside5_opp"])
    out["volume_share_game"] = safe(out["touches"], out["team_touches"])
    out["carry_share_game"] = safe(out["rush_att"], out["team_rush_att"])
    out["target_share_game"] = safe(out["targets"], out["team_targets"])
    out["rz_carry_share_game"] = safe(out["rz_carry"], out["team_rz_carry"])
    out["inside5_carry_share_game"] = safe(out["inside5_carry"], out["team_inside5_carry"])
    out["rz_target_share_game"] = safe(out["rz_target"], out["team_rz_target"])

    sort_cols = [c for c in ["season", "week", "game_date", "game_id", "player_id"] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def merge_weekly_player_metadata(player_games: pd.DataFrame, player_stats: pd.DataFrame | None) -> pd.DataFrame:
    """Attach position and other weekly metadata when nflverse weekly stats are available."""
    if player_stats is None or player_stats.empty:
        return player_games.copy()

    stats = player_stats.copy()
    # Keep only columns that are safe/descriptive or useful as same-game outcomes.
    preferred = [
        "player_id", "season", "week", "season_type", "position", "position_group",
        "team", "opponent_team", "snap_pct", "offense_snaps",
    ]
    cols = [c for c in preferred if c in stats.columns]
    if "player_id" not in cols or "season" not in cols or "week" not in cols:
        return player_games.copy()

    stats = stats[cols].drop_duplicates(["player_id", "season", "week"])
    merged = player_games.merge(stats, on=["player_id", "season", "week"], how="left", suffixes=("", "_stats"))
    return merged
