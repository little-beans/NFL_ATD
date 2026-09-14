from __future__ import annotations
import pandas as pd


def _first_present(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_ids(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _extract_roster_columns(d: pd.DataFrame) -> pd.DataFrame:
    if d is None or d.empty:
        return pd.DataFrame(columns=[
            "player_id","current_team","current_position","roster_status"
        ])

    id_col = _first_present(d, ["gsis_id","player_id","nflverse_id"])
    team_col = _first_present(d, ["team","team_abbr","club_code"])
    pos_col = _first_present(d, ["position","position_group"])
    status_col = _first_present(d, ["status","roster_status"])

    if id_col is None or team_col is None:
        return pd.DataFrame(columns=[
            "player_id","current_team","current_position","roster_status"
        ])

    out = pd.DataFrame({
        "player_id": _normalize_ids(d[id_col]),
        "current_team": d[team_col].astype("string").str.strip(),
        "current_position": d[pos_col].astype("string").str.strip() if pos_col else "",
        "roster_status": d[status_col].astype("string").str.strip() if status_col else "",
    })

    return out[
        out["player_id"].notna()
        & out["current_team"].notna()
        & out["current_team"].ne("")
    ].copy()


def build_current_roster(
    weekly_rosters: pd.DataFrame,
    season_rosters: pd.DataFrame,
    season: int,
    week: int,
) -> pd.DataFrame:
    """
    Resolve current team/position with priority:

      1. exact requested weekly roster row
      2. latest available weekly roster row <= requested week
      3. season roster
      4. caller may later retain historical team as last resort

    Adds:
      roster_source
      roster_source_week
    """
    pieces = []

    weekly = weekly_rosters.copy() if weekly_rosters is not None else pd.DataFrame()
    if not weekly.empty and "season" in weekly.columns:
        weekly = weekly[pd.to_numeric(weekly["season"], errors="coerce").eq(season)]

    week_col = _first_present(weekly, ["week","game_week"]) if not weekly.empty else None

    exact = pd.DataFrame()
    latest = pd.DataFrame()

    if not weekly.empty and week_col:
        weekly["_week_num"] = pd.to_numeric(weekly[week_col], errors="coerce")

        exact_raw = weekly[weekly["_week_num"].eq(week)].copy()
        exact = _extract_roster_columns(exact_raw)
        if not exact.empty:
            exact["roster_source"] = "weekly_exact"
            exact["roster_source_week"] = week
            exact = exact.drop_duplicates("player_id", keep="last")
            pieces.append(exact)

        prior_raw = weekly[
            weekly["_week_num"].notna()
            & weekly["_week_num"].le(week)
        ].copy()

        if not prior_raw.empty:
            id_col = _first_present(prior_raw, ["gsis_id","player_id","nflverse_id"])
            if id_col:
                prior_raw["_pid"] = _normalize_ids(prior_raw[id_col])
                prior_raw = prior_raw.sort_values(["_pid","_week_num"])
                prior_raw = prior_raw.drop_duplicates("_pid", keep="last")

                latest = _extract_roster_columns(prior_raw)
                latest["roster_source"] = "weekly_latest"
                # map source week using normalized id
                wk_map = (
                    prior_raw[["_pid","_week_num"]]
                    .drop_duplicates("_pid", keep="last")
                    .set_index("_pid")["_week_num"]
                )
                latest["roster_source_week"] = latest["player_id"].map(wk_map)
                latest = latest.drop_duplicates("player_id", keep="last")
                pieces.append(latest)

    season_df = season_rosters.copy() if season_rosters is not None else pd.DataFrame()
    if not season_df.empty and "season" in season_df.columns:
        season_df = season_df[
            pd.to_numeric(season_df["season"], errors="coerce").eq(season)
        ]

    season_norm = _extract_roster_columns(season_df)
    if not season_norm.empty:
        season_norm["roster_source"] = "season_roster"
        season_norm["roster_source_week"] = pd.NA
        season_norm = season_norm.drop_duplicates("player_id", keep="last")
        pieces.append(season_norm)

    if not pieces:
        return pd.DataFrame(columns=[
            "player_id","current_team","current_position","roster_status",
            "roster_source","roster_source_week"
        ])

    # Priority by concatenation order: exact -> latest -> season.
    combined = pd.concat(pieces, ignore_index=True)
    combined["_priority"] = combined["roster_source"].map({
        "weekly_exact": 0,
        "weekly_latest": 1,
        "season_roster": 2,
    }).fillna(9)

    combined = (
        combined.sort_values(["player_id","_priority"])
        .drop_duplicates("player_id", keep="first")
        .drop(columns="_priority")
        .reset_index(drop=True)
    )

    return combined


def apply_current_roster(rows: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()

    if "player_id" not in out.columns:
        raise KeyError("Rows require player_id before current-roster mapping.")

    out["player_id"] = _normalize_ids(out["player_id"])

    out["historical_team"] = out["posteam"] if "posteam" in out.columns else ""
    out["historical_position"] = out["position"] if "position" in out.columns else ""

    use_cols = [
        c for c in [
            "player_id","current_team","current_position","roster_status",
            "roster_source","roster_source_week"
        ]
        if c in roster.columns
    ]

    use = roster[use_cols].drop_duplicates("player_id")
    out = out.merge(use, on="player_id", how="left")

    out["roster_matched"] = out["current_team"].notna() & out["current_team"].ne("")
    out["team_changed"] = (
        out["roster_matched"]
        & out["historical_team"].fillna("").astype(str)
        .ne(out["current_team"].fillna("").astype(str))
    )

    out["posteam"] = out["current_team"].where(
        out["roster_matched"],
        out["historical_team"]
    )

    if "position" not in out.columns:
        out["position"] = out["current_position"]
    else:
        good_pos = out["current_position"].fillna("").ne("")
        out.loc[good_pos, "position"] = out.loc[good_pos, "current_position"]

    out["roster_source"] = out["roster_source"].fillna("historical_fallback")

    return out
