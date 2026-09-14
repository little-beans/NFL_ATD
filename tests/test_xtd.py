import pandas as pd

from td_engine.x_td import add_leakage_safe_xtd_priors, add_td_debt_tags


def test_xtd_prior_excludes_current_game():
    d = pd.DataFrame({
        "player_id": ["p1", "p1", "p1"],
        "season": [2026, 2026, 2026],
        "week": [1, 2, 3],
        "game_id": ["g1", "g2", "g3"],
        "expected_tds": [0.4, 0.8, 0.2],
        "actual_tds": [0, 1, 0],
        "xtd_opps": [3, 5, 2],
    })
    out = add_leakage_safe_xtd_priors(d)
    w2 = out[out.week == 2].iloc[0]
    w3 = out[out.week == 3].iloc[0]
    assert abs(w2.expected_tds_prior - 0.4) < 1e-9
    assert w2.actual_tds_prior == 0
    assert abs(w3.expected_tds_prior - 1.2) < 1e-9
    assert w3.actual_tds_prior == 1


def test_td_debt_tagging():
    d = pd.DataFrame({
        "td_debt": [2.1, 1.1, -2.2, 0.1],
        "td_debt_l5": [0.0, 0.0, 0.0, 0.0],
    })
    out = add_td_debt_tags(d)
    assert list(out.td_debt_tag) == ["DUE++", "DUE", "REGRESSION_RISK", ""]
