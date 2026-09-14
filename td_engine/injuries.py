from __future__ import annotations
import pandas as pd
import numpy as np

OUT_STATUSES = {"Out","OUT","Injured Reserve","IR","PUP","NFI"}
DOUBTFUL_STATUSES = {"Doubtful","DOUBTFUL"}
QUESTIONABLE_STATUSES = {"Questionable","QUESTIONABLE"}

def normalize_injury_status(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    aliases = {
        "O":"Out",
        "D":"Doubtful",
        "Q":"Questionable",
        "IR":"Injured Reserve",
        "PUP":"PUP",
        "NFI":"NFI",
    }
    return aliases.get(s, s)

def status_availability_multiplier(status: str) -> float:
    s = normalize_injury_status(status)
    if s in OUT_STATUSES:
        return 0.0
    if s in DOUBTFUL_STATUSES:
        return 0.15
    if s in QUESTIONABLE_STATUSES:
        return 0.80
    return 1.0

def prepare_injury_table(injuries: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """
    Prefer exact requested-week injury rows.
    If none exist yet, fall back to latest available report in the season
    for each player and mark source_week_fallback=True.
    """
    cols = [
        "player_id","team","position","full_name","report_status",
        "practice_status","availability_multiplier",
        "source_week","source_week_fallback"
    ]
    if injuries is None or injuries.empty:
        return pd.DataFrame(columns=cols)

    d = injuries.copy()
    if "season" in d.columns:
        d = d[pd.to_numeric(d["season"], errors="coerce").eq(season)]

    if d.empty:
        return pd.DataFrame(columns=cols)

    exact = d.copy()
    if "week" in exact.columns:
        exact = exact[pd.to_numeric(exact["week"], errors="coerce").eq(week)]

    use_fallback = exact.empty
    if use_fallback:
        work = d.copy()
        if "week" in work.columns:
            work["_week_num"] = pd.to_numeric(work["week"], errors="coerce")
            work = work[work["_week_num"].notna() & (work["_week_num"] <= week)]
        else:
            work["_week_num"] = np.nan
    else:
        work = exact.copy()
        work["_week_num"] = week

    if work.empty:
        return pd.DataFrame(columns=cols)

    if "gsis_id" in work.columns:
        work["player_id"] = work["gsis_id"]
    elif "player_id" not in work.columns:
        raise KeyError("Injury data must contain gsis_id or player_id.")

    for c in ["report_status","practice_status","team","position","full_name"]:
        if c not in work.columns:
            work[c] = ""

    if "date_modified" in work.columns:
        work["_modified"] = pd.to_datetime(work["date_modified"], errors="coerce", utc=True)
        work = work.sort_values(["player_id","_week_num","_modified"])
    else:
        work = work.sort_values(["player_id","_week_num"])

    # latest report for each player
    work = work.drop_duplicates("player_id", keep="last")

    work["report_status"] = work["report_status"].map(normalize_injury_status)
    work["availability_multiplier"] = work["report_status"].map(status_availability_multiplier)
    work["source_week"] = work["_week_num"]
    work["source_week_fallback"] = bool(use_fallback)

    keep = [c for c in cols if c in work.columns]
    return work[keep].reset_index(drop=True)
