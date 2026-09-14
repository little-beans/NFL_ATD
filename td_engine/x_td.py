from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["yardline_100", "down", "ydstogo", "air_yards"]
CATEGORICAL_FEATURES = ["opp_type"]


class ExpectedTDModel:
    """Auditable play-level expected touchdown model.

    Each rushing attempt or target receives a TD probability (xTD). The first
    version intentionally stays simple so we can inspect and backtest it before
    adding tracking/coverage variables.
    """

    def __init__(self):
        prep = ColumnTransformer(
            [
                (
                    "num",
                    Pipeline([
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]),
                    NUMERIC_FEATURES,
                ),
                (
                    "cat",
                    Pipeline([
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )
        self.model = Pipeline([
            ("prep", prep),
            ("clf", LogisticRegression(max_iter=1000)),
        ])
        self.fitted = False

    def fit(self, opportunities: pd.DataFrame):
        d = opportunities.copy()
        if d.empty:
            raise ValueError("No xTD opportunities supplied for training.")
        y = d["touchdown"].astype(int)
        if y.nunique() < 2:
            raise ValueError("xTD training data must contain both TD and non-TD plays.")
        self.model.fit(d[NUMERIC_FEATURES + CATEGORICAL_FEATURES], y)
        self.fitted = True
        return self

    def predict_opportunities(self, opportunities: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("ExpectedTDModel has not been fit.")
        out = opportunities.copy()
        out["xTD"] = self.model.predict_proba(
            out[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        )[:, 1]
        return out


def build_xtd_opportunities(pbp: pd.DataFrame) -> pd.DataFrame:
    """Convert nflverse-style PBP into one row per rush attempt or target."""
    df = pbp.copy()

    required = ["game_id", "posteam", "yardline_100"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required PBP columns for xTD: {missing}")

    defaults = {
        "rush_attempt": 0,
        "complete_pass": 0,
        "touchdown": 0,
        "down": np.nan,
        "ydstogo": np.nan,
        "air_yards": np.nan,
    }
    for c, default in defaults.items():
        if c not in df.columns:
            df[c] = default

    rid = "rusher_player_id" if "rusher_player_id" in df.columns else "rusher_id"
    recid = "receiver_player_id" if "receiver_player_id" in df.columns else "receiver_id"
    rname = "rusher_player_name" if "rusher_player_name" in df.columns else None
    recname = "receiver_player_name" if "receiver_player_name" in df.columns else None

    for c in [rid, recid]:
        if c not in df.columns:
            raise ValueError(f"Missing player id column: {c}")

    common = [c for c in [
        "season", "week", "game_id", "posteam", "defteam",
        "yardline_100", "down", "ydstogo", "air_yards"
    ] if c in df.columns]

    rush = df[df["rush_attempt"].fillna(0).eq(1) & df[rid].notna()].copy()
    rush["player_id"] = rush[rid]
    rush["player_name"] = rush[rname] if rname else rush[rid]
    rush["opp_type"] = "rush"
    rush["touchdown"] = pd.to_numeric(rush["touchdown"], errors="coerce").fillna(0).astype(int)
    # air yards do not exist for rushes; use 0 rather than median target air yards.
    rush["air_yards"] = 0.0

    rec = df[df[recid].notna()].copy()
    rec["player_id"] = rec[recid]
    rec["player_name"] = rec[recname] if recname else rec[recid]
    rec["opp_type"] = "target"
    if "pass_touchdown" in rec.columns:
        rec["touchdown"] = pd.to_numeric(rec["pass_touchdown"], errors="coerce").fillna(0).astype(int)
    else:
        rec["touchdown"] = pd.to_numeric(rec["touchdown"], errors="coerce").fillna(0).astype(int)

    cols = common + ["player_id", "player_name", "opp_type", "touchdown"]
    out = pd.concat([rush[cols], rec[cols]], ignore_index=True)
    out = out[out["yardline_100"].notna()].copy()
    out["yardline_100"] = pd.to_numeric(out["yardline_100"], errors="coerce")
    out = out[out["yardline_100"].between(0, 100, inclusive="both")].copy()
    return out


def add_leakage_safe_xtd_priors(player_games: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative and recent xTD priors that exclude the current game."""
    d = player_games.copy()
    order = [c for c in ["season", "week", "game_id"] if c in d.columns]
    d = d.sort_values(["player_id"] + order).copy()

    g = d.groupby("player_id", group_keys=False)
    d["expected_tds_prior"] = g["expected_tds"].cumsum() - d["expected_tds"]
    d["actual_tds_prior"] = g["actual_tds"].cumsum() - d["actual_tds"]
    d["xtd_opps_prior"] = g["xtd_opps"].cumsum() - d["xtd_opps"]

    # Recent form excludes current game via shift(1).
    d["expected_tds_l5"] = g["expected_tds"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
    d["actual_tds_l5"] = g["actual_tds"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
    d["td_debt_l5"] = d["expected_tds_l5"].fillna(0) - d["actual_tds_l5"].fillna(0)
    d["td_debt"] = d["expected_tds_prior"].fillna(0) - d["actual_tds_prior"].fillna(0)
    return d


def add_td_debt_tags(df: pd.DataFrame, debt_col: str = "td_debt") -> pd.DataFrame:
    out = df.copy()
    debt = out[debt_col].fillna(0.0)
    recent = out.get("td_debt_l5", pd.Series(0.0, index=out.index)).fillna(0.0)

    # Tags are context only; they do not modify calibrated TD probability.
    out["td_debt_tag"] = np.select(
        [
            (debt >= 2.0) | (recent >= 1.25),
            (debt >= 1.0) | (recent >= 0.75),
            (debt <= -2.0) | (recent <= -1.25),
        ],
        ["DUE++", "DUE", "REGRESSION_RISK"],
        default="",
    )
    return out
