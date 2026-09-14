import pandas as pd

from td_engine.projected_roles import (
    latest_pregame_role_state,
    attach_projected_roles,
)


def test_projected_roles_use_latest_prior_week():
    hist = pd.DataFrame({
        "player_id":["p1","p1","p1"],
        "season":[2025,2026,2026],
        "week":[18,1,2],
        "red_zone_share":[0.2,0.4,0.9],
        "volume_share":[0.3,0.5,0.9],
        "goal_line_share":[0.1,0.6,0.9],
        "target_share":[0.2,0.3,0.9],
        "td_efficiency_score":[0.5,0.6,0.9],
    })

    state = latest_pregame_role_state(hist,2026,2)
    assert state.iloc[0]["volume_share"] == 0.5

    weekly = pd.DataFrame({
        "player_id":["p1"],
        "volume_share":[0.0],
    })

    out, diag = attach_projected_roles(weekly,state)

    assert out.loc[0,"volume_share"] == 0.5
    assert out.loc[0,"projected_goal_line_share"] == 0.6
    assert diag["matched_players"] == 1
