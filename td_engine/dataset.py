from __future__ import annotations

from pathlib import Path
import pandas as pd

from .historical import build_player_game_table, merge_weekly_player_metadata
from .rolling import build_rolling_pregame_features, validate_no_same_game_leakage
from .score import baseline_score


def build_training_dataset(
    pbp: pd.DataFrame,
    player_stats: pd.DataFrame | None = None,
    min_prior_games: int = 1,
    skill_positions_only: bool = True,
) -> pd.DataFrame:
    games = build_player_game_table(pbp)
    games = merge_weekly_player_metadata(games, player_stats)
    features = build_rolling_pregame_features(games)
    validate_no_same_game_leakage(features)

    if skill_positions_only and "position" in features.columns:
        features = features[features["position"].isin(["RB", "WR", "TE", "FB", "HB"])].copy()

    features = features[features["career_games_before"] >= min_prior_games].copy()
    features = baseline_score(features)

    # The historical target remains the outcome from the current game.
    keep_first = [
        "season", "week", "season_type", "game_date", "game_id", "posteam", "defteam",
        "player_id", "player_name", "position", "scored_td", "td",
        "baseline_score", "career_games_before", "season_games_before",
    ]
    first = [c for c in keep_first if c in features.columns]
    rest = [c for c in features.columns if c not in first]
    return features[first + rest].reset_index(drop=True)


def save_dataset(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
