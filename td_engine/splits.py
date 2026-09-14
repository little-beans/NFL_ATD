from __future__ import annotations

import pandas as pd


def chronological_season_split(
    df: pd.DataFrame,
    train_through: int,
    calibrate_season: int,
    test_season: int,
):
    """Strict season-based train/calibrate/test split."""
    train = df[df["season"] <= train_through].copy()
    calibration = df[df["season"] == calibrate_season].copy()
    test = df[df["season"] == test_season].copy()
    return train, calibration, test


def walk_forward_folds(df: pd.DataFrame, first_test_season: int):
    """Yield (train, test, season) folds where training always precedes testing."""
    seasons = sorted(int(s) for s in df["season"].dropna().unique())
    for season in seasons:
        if season < first_test_season:
            continue
        train = df[df["season"] < season].copy()
        test = df[df["season"] == season].copy()
        if not train.empty and not test.empty:
            yield train, test, season
