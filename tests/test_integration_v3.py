import pandas as pd
from td_engine.xtd_integration import select_latest_xtd_prior


def test_xtd_fallback_builds_cumulative_priors():
    d = pd.DataFrame({
        "player_id":["p1","p1","p1"],
        "season":[2025,2026,2026],
        "week":[18,1,2],
        "expected_tds":[0.4,0.7,5.0],
        "scored_td":[0,1,1],
        "expected_tds_prior":[0.0,0.0,0.0],
        "actual_tds_prior":[0.0,0.0,0.0],
    })

    out = select_latest_xtd_prior(d, 2026, 2)
    row = out.iloc[0]

    assert abs(row["expected_tds_prior"] - 1.1) < 1e-9
    assert row["actual_tds_prior"] == 1.0
    assert abs(row["td_debt"] - 0.1) < 1e-9
