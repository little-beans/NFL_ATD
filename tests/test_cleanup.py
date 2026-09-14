import pandas as pd

from td_engine.current_roster import apply_current_roster
from td_engine.role_reallocation import apply_role_reallocation
from td_engine.probability_guardrails import add_probability_diagnostics


def test_current_roster_overrides_historical_team():
    rows = pd.DataFrame({
        "player_id":["p1"],
        "posteam":["OLD"],
        "position":["WR"],
    })
    roster = pd.DataFrame({
        "player_id":["p1"],
        "current_team":["NEW"],
        "current_position":["WR"],
        "roster_status":["ACT"],
    })
    out = apply_current_roster(rows, roster)
    assert out.loc[0,"posteam"] == "NEW"
    assert bool(out.loc[0,"team_changed"])


def test_role_boost_uses_absolute_delta():
    rows = pd.DataFrame({
        "posteam":["AAA","AAA"],
        "player_id":["rb1","rb2"],
        "position":["RB","RB"],
        "availability_multiplier":[0.0,1.0],
        "volume_share":[0.60,0.20],
        "target_share":[0.05,0.05],
        "red_zone_share":[0.70,0.15],
        "goal_line_share":[0.80,0.10],
    })
    out = apply_role_reallocation(rows)
    rb2 = out[out.player_id=="rb2"].iloc[0]
    assert rb2["injury_role_boost"] > 0
    assert rb2["injury_role_boost"] < 1


def test_probability_guardrail():
    df = pd.DataFrame({"td_probability":[0.0,0.5,1.0]})
    out = add_probability_diagnostics(df)
    assert out.loc[0,"td_probability"] == 0.01
    assert out.loc[2,"td_probability"] == 0.85
