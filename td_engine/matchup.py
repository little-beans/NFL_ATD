from __future__ import annotations
import numpy as np
import pandas as pd

DEFENSE_FEATURES = [
    "def_td_allowed_pg",
    "def_rz_td_rate",
    "def_inside10_td_rate",
    "def_inside5_td_rate",
    "def_rb_td_allowed_pg",
    "def_wr_td_allowed_pg",
    "def_te_td_allowed_pg",
]


def _safe_div(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def build_defense_game_table(player_games: pd.DataFrame) -> pd.DataFrame:
    """One row per defense-game from the player-game table."""
    req = ["season", "week", "game_id", "posteam", "defteam", "td"]
    missing = [c for c in req if c not in player_games.columns]
    if missing:
        raise ValueError(f"Missing columns needed for defense features: {missing}")

    d = player_games.copy()
    if "season_type" in d.columns:
        d = d[d["season_type"].fillna("REG").eq("REG")].copy()

    # Team opportunity denominators are duplicated on player rows, so take first.
    group = ["season", "week", "game_id", "posteam", "defteam"]
    agg = {
        "td": "sum",
        "team_rz_opp": "first",
        "team_inside10_opp": "first",
        "team_inside5_opp": "first",
    }
    for c in ["game_date"]:
        if c in d.columns:
            agg[c] = "first"

    g = d.groupby(group, as_index=False).agg(agg).rename(columns={"td":"td_allowed"})

    # Position-level TDs allowed.
    if "position" in d.columns:
        pos = d.assign(position=d["position"].fillna("UNK").astype(str).str.upper())
        for label, positions in {
            "rb_td_allowed": {"RB","FB","HB"},
            "wr_td_allowed": {"WR"},
            "te_td_allowed": {"TE"},
        }.items():
            x = (pos[pos["position"].isin(positions)]
                 .groupby(group)["td"].sum().rename(label))
            g = g.merge(x, on=group, how="left")
    for c in ["rb_td_allowed","wr_td_allowed","te_td_allowed"]:
        if c not in g.columns: g[c] = 0.0
        g[c] = g[c].fillna(0.0)

    g["rz_td_rate_game"] = _safe_div(g["td_allowed"], g["team_rz_opp"])
    g["inside10_td_rate_game"] = _safe_div(g["td_allowed"], g["team_inside10_opp"])
    g["inside5_td_rate_game"] = _safe_div(g["td_allowed"], g["team_inside5_opp"])

    sort_cols = [c for c in ["defteam","season","week","game_date","game_id"] if c in g.columns]
    return g.sort_values(sort_cols).reset_index(drop=True)


def build_defense_pregame_features(player_games: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe defense features for every historical game (shifted by 1)."""
    g = build_defense_game_table(player_games)
    g = g.sort_values([c for c in ["defteam","season","week","game_date","game_id"] if c in g.columns]).copy()

    source_map = {
        "def_td_allowed_pg": "td_allowed",
        "def_rz_td_rate": "rz_td_rate_game",
        "def_inside10_td_rate": "inside10_td_rate_game",
        "def_inside5_td_rate": "inside5_td_rate_game",
        "def_rb_td_allowed_pg": "rb_td_allowed",
        "def_wr_td_allowed_pg": "wr_td_allowed",
        "def_te_td_allowed_pg": "te_td_allowed",
    }

    for target, source in source_map.items():
        # Current-season last-5 prior.
        l5 = (g.groupby(["defteam","season"], sort=False)[source]
                .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean()))
        # Career prior for early season fallback.
        career = (g.groupby("defteam", sort=False)[source]
                    .transform(lambda s: s.shift(1).expanding(min_periods=1).mean()))
        g[target] = l5.fillna(career)
        # League fallback only for a defense's first observed game.
        league_prior = g[source].shift(1).expanding(min_periods=1).mean()
        g[target] = g[target].fillna(league_prior).fillna(g[source].mean())

    keep = ["season","week","game_id","posteam","defteam"] + DEFENSE_FEATURES
    return g[keep].copy()


def latest_defense_priors(player_games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Defense priors entering an upcoming week."""
    g = build_defense_game_table(player_games)
    hist = g[(g["season"] < season) | ((g["season"] == season) & (g["week"] < week))].copy()
    if hist.empty:
        raise ValueError(f"No defense history before {season} Week {week}")

    rows=[]
    for team, x in hist.groupby("defteam", sort=False):
        x=x.sort_values([c for c in ["season","week","game_date","game_id"] if c in x.columns])
        sx=x[x["season"].eq(season)].tail(5)
        cx=x.tail(12)
        base=sx if not sx.empty else cx
        def mean(col):
            v=base[col].mean()
            if pd.isna(v): v=cx[col].mean()
            return float(v) if not pd.isna(v) else np.nan
        rows.append({
            "defteam":team,
            "def_td_allowed_pg":mean("td_allowed"),
            "def_rz_td_rate":mean("rz_td_rate_game"),
            "def_inside10_td_rate":mean("inside10_td_rate_game"),
            "def_inside5_td_rate":mean("inside5_td_rate_game"),
            "def_rb_td_allowed_pg":mean("rb_td_allowed"),
            "def_wr_td_allowed_pg":mean("wr_td_allowed"),
            "def_te_td_allowed_pg":mean("te_td_allowed"),
        })
    out=pd.DataFrame(rows)
    for c in DEFENSE_FEATURES:
        out[c]=out[c].fillna(out[c].median())
    return out
