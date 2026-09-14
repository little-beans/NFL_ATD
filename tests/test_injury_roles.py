import pandas as pd

from td_engine.injuries import status_availability_multiplier
from td_engine.role_reallocation import apply_role_reallocation


def test_status_multipliers():
    assert status_availability_multiplier("Out") == 0.0
    assert status_availability_multiplier("Doubtful") == 0.15
    assert status_availability_multiplier("Questionable") == 0.80
    assert status_availability_multiplier("") == 1.0


def test_out_player_role_is_redistributed():
    rows = pd.DataFrame(
        {
            "posteam": ["AAA", "AAA", "AAA"],
            "player_id": ["rb1", "rb2", "wr1"],
            "position": ["RB", "RB", "WR"],
            "availability_multiplier": [0.0, 1.0, 1.0],
            "volume_share": [0.55, 0.15, 0.10],
            "target_share": [0.08, 0.04, 0.30],
            "red_zone_share": [0.60, 0.15, 0.10],
            "goal_line_share": [0.75, 0.10, 0.02],
        }
    )

    out = apply_role_reallocation(rows)
    rb1 = out[out.player_id == "rb1"].iloc[0]
    rb2 = out[out.player_id == "rb2"].iloc[0]

    assert rb1["adjusted_volume_share"] == 0
    assert rb2["adjusted_volume_share"] > 0.15
    assert rb2["adjusted_goal_line_share"] > 0.10
    assert rb2["injury_role_boost"] > 0
