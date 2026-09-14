import pandas as pd

from td_engine.unified import finalize_weekly_table


def test_unified_keeps_current_roster_and_adjusts_roles():
    rows = pd.DataFrame({
        "player_id":["p1","p2"],
        "player_name":["RB One","RB Two"],
        "posteam":["OLD","OLD"],
        "position":["RB","RB"],
        "volume_share":[0.60,0.20],
        "target_share":[0.05,0.05],
        "red_zone_share":[0.70,0.15],
        "goal_line_share":[0.80,0.10],
        "td_probability":[0.50,0.20],
        "td_efficiency_score":[0.5,0.5],
    })

    roster = pd.DataFrame({
        "player_id":["p1","p2"],
        "current_team":["NEW","NEW"],
        "current_position":["RB","RB"],
        "roster_status":["ACT","ACT"],
        "roster_source":["weekly_latest","weekly_latest"],
        "roster_source_week":[1,1],
    })

    injuries = pd.DataFrame({
        "player_id":["p1"],
        "report_status":["Out"],
        "practice_status":["DNP"],
        "availability_multiplier":[0.0],
    })

    out = finalize_weekly_table(rows, roster, injuries)

    assert set(out["posteam"]) == {"NEW"}
    rb1 = out[out.player_id=="p1"].iloc[0]
    rb2 = out[out.player_id=="p2"].iloc[0]

    assert rb1["availability_multiplier"] == 0
    assert rb2["adjusted_volume_share"] > rb2["volume_share"]
