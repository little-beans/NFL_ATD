from __future__ import annotations
import pandas as pd


def _first_present(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_weekly_roster(rosters: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """
    Normalize nflverse weekly roster data to:
      player_id, current_team, current_position, roster_status

    Uses exact season/week where available.
    """
    if rosters is None or rosters.empty:
        return pd.DataFrame(columns=[
            "player_id","current_team","current_position","roster_status"
        ])

    d = rosters.copy()

    if "season" in d.columns:
        d = d[pd.to_numeric(d["season"], errors="coerce").eq(season)]

    week_col = _first_present(d, ["week","game_week"])
    if week_col:
        d = d[pd.to_numeric(d[week_col], errors="coerce").eq(week)]

    id_col = _first_present(d, ["gsis_id","player_id","nflverse_id"])
    team_col = _first_present(d, ["team","team_abbr","club_code"])
    pos_col = _first_present(d, ["position","position_group"])
    status_col = _first_present(d, ["status","roster_status"])

    if id_col is None:
        raise KeyError("Weekly roster data has no recognized player ID column.")
    if team_col is None:
        raise KeyError("Weekly roster data has no recognized team column.")

    out = pd.DataFrame({
        "player_id": d[id_col],
        "current_team": d[team_col],
        "current_position": d[pos_col] if pos_col else "",
        "roster_status": d[status_col] if status_col else "",
    })

    out = out[out["player_id"].notna() & out["current_team"].notna()]
    out["player_id"] = out["player_id"].astype(str)
    out["current_team"] = out["current_team"].astype(str)

    # One current roster row per player. Prefer the last supplied row.
    out = out.drop_duplicates("player_id", keep="last").reset_index(drop=True)
    return out


def apply_current_roster(rows: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """
    Make weekly roster team/position the source of truth.
    Preserve historical team for diagnostics.
    """
    out = rows.copy()

    if "player_id" not in out.columns:
        raise KeyError("Rows require player_id before current-roster mapping.")

    out["player_id"] = out["player_id"].astype(str)

    if "posteam" in out.columns:
        out["historical_team"] = out["posteam"]
    else:
        out["historical_team"] = ""

    if "position" in out.columns:
        out["historical_position"] = out["position"]
    else:
        out["historical_position"] = ""

    use = roster[[
        "player_id","current_team","current_position","roster_status"
    ]].drop_duplicates("player_id")

    out = out.merge(use, on="player_id", how="left")

    out["roster_matched"] = out["current_team"].notna()
    out["team_changed"] = (
        out["roster_matched"]
        & out["historical_team"].fillna("").astype(str).ne(out["current_team"].astype(str))
    )

    out["posteam"] = out["current_team"].where(
        out["roster_matched"], out["historical_team"]
    )

    if "position" not in out.columns:
        out["position"] = out["current_position"]
    else:
        out["position"] = out["current_position"].where(
            out["current_position"].fillna("").ne(""),
            out["position"]
        )

    return out
