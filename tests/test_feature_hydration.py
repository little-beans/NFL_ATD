import pandas as pd
from td_engine.feature_hydration import hydrate_model_features


def test_hydrates_latest_prior_row_only():
    weekly = pd.DataFrame({
        "player_id":["p1"],
        "existing":[1.0],
    })

    history = pd.DataFrame({
        "player_id":["p1","p1","p1"],
        "season":[2025,2026,2026],
        "week":[18,1,2],
        "f1":[10.0,20.0,999.0],
        "f2":[11.0,21.0,999.0],
    })

    out, diag = hydrate_model_features(
        weekly,
        history,
        required_features=["f1","f2"],
        target_season=2026,
        target_week=2,
    )

    assert out.loc[0,"f1"] == 20.0
    assert out.loc[0,"f2"] == 21.0
    assert not diag["still_missing"]
