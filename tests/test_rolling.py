import pandas as pd

from td_engine.rolling import build_rolling_pregame_features


def synthetic_games():
    return pd.DataFrame({
        "player_id": ["A", "A", "A", "A"],
        "player_name": ["Alpha"] * 4,
        "season": [2025, 2025, 2025, 2025],
        "week": [1, 2, 3, 4],
        "game_id": ["g1", "g2", "g3", "g4"],
        "season_type": ["REG"] * 4,
        "touches": [10, 20, 30, 40],
        "rush_att": [8, 15, 20, 30],
        "targets": [3, 6, 8, 5],
        "receptions": [2, 5, 6, 4],
        "td": [0, 1, 0, 2],
        "rz_opp": [1, 2, 3, 4],
        "inside10_opp": [0, 1, 2, 3],
        "inside5_opp": [0, 1, 1, 2],
        "rz_carry": [1, 1, 2, 3],
        "inside10_carry": [0, 1, 1, 2],
        "inside5_carry": [0, 1, 1, 1],
        "rz_target": [0, 1, 1, 1],
        "inside10_target": [0, 0, 1, 1],
        "inside5_target": [0, 0, 0, 1],
        "red_zone_share_game": [.1, .2, .3, .4],
        "inside10_share_game": [0, .2, .3, .4],
        "goal_line_share_game": [0, .5, .25, .5],
        "volume_share_game": [.2, .3, .4, .5],
        "carry_share_game": [.2, .3, .4, .5],
        "target_share_game": [.1, .2, .3, .2],
        "rz_carry_share_game": [.1, .2, .3, .4],
        "inside5_carry_share_game": [0, .5, .5, .5],
        "rz_target_share_game": [0, .2, .3, .4],
    })


def test_week_two_uses_only_week_one():
    out = build_rolling_pregame_features(synthetic_games())
    w2 = out[out.week == 2].iloc[0]
    assert w2["pregame_touches_l3"] == 10
    assert w2["red_zone_share"] == .1
    assert w2["prior_tds"] == 0
    assert w2["prior_touches"] == 10


def test_week_four_does_not_use_week_four():
    out = build_rolling_pregame_features(synthetic_games())
    w4 = out[out.week == 4].iloc[0]
    assert w4["pregame_touches_l3"] == 20  # mean of 10,20,30
    assert round(w4["prior_touches"], 6) == 60
    assert round(w4["prior_tds"], 6) == 1
