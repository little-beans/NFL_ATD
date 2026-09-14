import pandas as pd
from td_engine.current_roster import build_current_roster


def test_roster_priority_exact_then_latest_then_season():
    weekly = pd.DataFrame({
        "season":[2026,2026,2026],
        "week":[1,2,1],
        "gsis_id":["p1","p1","p2"],
        "team":["AAA","BBB","CCC"],
        "position":["WR","WR","RB"],
    })

    season = pd.DataFrame({
        "season":[2026,2026,2026],
        "gsis_id":["p1","p2","p3"],
        "team":["ZZZ","YYY","DDD"],
        "position":["WR","RB","TE"],
    })

    out = build_current_roster(weekly, season, 2026, 2).set_index("player_id")

    assert out.loc["p1","current_team"] == "BBB"
    assert out.loc["p1","roster_source"] == "weekly_exact"

    assert out.loc["p2","current_team"] == "CCC"
    assert out.loc["p2","roster_source"] == "weekly_latest"

    assert out.loc["p3","current_team"] == "DDD"
    assert out.loc["p3","roster_source"] == "season_roster"
