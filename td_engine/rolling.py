from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_WINDOWS = (3, 5, 8)

COUNT_COLS = [
    "rush_att", "targets", "receptions", "touches", "td",
    "rz_opp", "inside10_opp", "inside5_opp",
    "rz_carry", "inside10_carry", "inside5_carry",
    "rz_target", "inside10_target", "inside5_target",
]

SHARE_COLS = [
    "red_zone_share_game", "inside10_share_game", "goal_line_share_game",
    "volume_share_game", "carry_share_game", "target_share_game",
    "rz_carry_share_game", "inside5_carry_share_game", "rz_target_share_game",
]


def _expanding_prior(series: pd.Series, min_periods=1):
    return series.shift(1).expanding(min_periods=min_periods).mean()


def _rolling_prior(series: pd.Series, window: int, min_periods=1):
    return series.shift(1).rolling(window=window, min_periods=min_periods).mean()


def _rolling_sum_prior(series: pd.Series, window: int, min_periods=1):
    return series.shift(1).rolling(window=window, min_periods=min_periods).sum()


def build_rolling_pregame_features(
    player_games: pd.DataFrame,
    windows=DEFAULT_WINDOWS,
    include_postseason: bool = False,
) -> pd.DataFrame:
    """
    Create leakage-safe pregame features from historical player-game outcomes.

    CRITICAL INVARIANT: every historical statistic is shifted by one game before
    rolling/expanding aggregation. Therefore a row for Week N cannot contain Week N
    outcomes.

    Seasons are allowed to carry history forward so Week 1 can use prior-season
    information. Season-specific priors are also included.
    """
    df = player_games.copy()

    if "season_type" in df.columns and not include_postseason:
        df = df[df["season_type"].fillna("REG").eq("REG")].copy()

    sort_cols = [c for c in ["player_id", "season", "week", "game_date", "game_id"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Group across career for rolling form; seasonal group for current-season priors.
    career = df.groupby("player_id", group_keys=False, sort=False)
    season_grp = df.groupby(["player_id", "season"], group_keys=False, sort=False)

    # Career games played before this game.
    df["career_games_before"] = career.cumcount()
    df["season_games_before"] = season_grp.cumcount()

    available_counts = [c for c in COUNT_COLS if c in df.columns]
    available_shares = [c for c in SHARE_COLS if c in df.columns]

    for col in available_counts + available_shares:
        # Career-to-date prior mean.
        df[f"pregame_{col}_career"] = career[col].transform(_expanding_prior)
        # Current-season-to-date prior mean.
        df[f"pregame_{col}_season"] = season_grp[col].transform(_expanding_prior)

        for w in windows:
            df[f"pregame_{col}_l{w}"] = career[col].transform(lambda s, w=w: _rolling_prior(s, w))

    # Rolling sums are useful for rare red-zone/TD events and TD debt.
    sum_cols = [c for c in ["td", "rz_opp", "inside10_opp", "inside5_opp", "touches", "targets", "rush_att"] if c in df.columns]
    for col in sum_cols:
        for w in windows:
            df[f"pregame_{col}_sum_l{w}"] = career[col].transform(lambda s, w=w: _rolling_sum_prior(s, w))

    # Smoothed pregame TD efficiency based ONLY on prior touches and TDs.
    prior_tds = career["td"].transform(lambda s: s.shift(1).cumsum()).fillna(0) if "td" in df.columns else 0
    prior_touches = career["touches"].transform(lambda s: s.shift(1).cumsum()).fillna(0) if "touches" in df.columns else 0
    df["prior_tds"] = prior_tds
    df["prior_touches"] = prior_touches
    prior_rate = 0.045
    prior_n = 20.0
    df["pregame_td_rate_smoothed"] = (df["prior_tds"] + prior_rate * prior_n) / (df["prior_touches"] + prior_n)
    df["td_efficiency_score"] = 1.0 / (1.0 + np.exp(-35.0 * (df["pregame_td_rate_smoothed"] - 0.045)))

    # Baseline engine aliases. Prefer L5 form, then season prior, then career prior.
    def choose(base_col, neutral=0.0):
        candidates = [f"pregame_{base_col}_l5", f"pregame_{base_col}_season", f"pregame_{base_col}_career"]
        present = [c for c in candidates if c in df.columns]
        if not present:
            return pd.Series(neutral, index=df.index)
        s = df[present[0]].copy()
        for c in present[1:]:
            s = s.fillna(df[c])
        return s.fillna(neutral)

    df["red_zone_share"] = choose("red_zone_share_game")
    df["volume_share"] = choose("volume_share_game")
    df["goal_line_share"] = choose("goal_line_share_game")
    df["target_share"] = choose("target_share_game")

    # Stabilized role features for V2 modeling.
    df["inside10_share"] = choose("inside10_share_game")
    df["carry_share"] = choose("carry_share_game")
    df["rz_carry_share"] = choose("rz_carry_share_game")
    df["inside5_carry_share"] = choose("inside5_carry_share_game")
    df["rz_target_share"] = choose("rz_target_share_game")
    df["touches_per_game"] = choose("touches")
    df["rush_att_per_game"] = choose("rush_att")
    df["targets_per_game"] = choose("targets")

    return df


def validate_no_same_game_leakage(features: pd.DataFrame) -> None:
    """Raise if obvious same-game values leaked into core pregame aliases."""
    checks = [
        ("target_share", "target_share_game"),
        ("goal_line_share", "goal_line_share_game"),
        ("red_zone_share", "red_zone_share_game"),
    ]
    # Equality can legitimately occur by chance, so we only detect the impossible
    # structural case where an entire populated column exactly equals current-game data.
    for pre, current in checks:
        if pre in features.columns and current in features.columns:
            mask = features[pre].notna() & features[current].notna()
            if mask.any() and features.loc[mask, pre].equals(features.loc[mask, current]):
                raise AssertionError(f"Potential leakage: {pre} exactly matches {current} on all populated rows")
