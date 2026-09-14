import pandas as pd

from td_engine.value_quality import add_value_quality


def test_one_book_low_role_large_edge_is_review_only():
    best = pd.DataFrame({
        "player_id": ["p1"],
        "edge_probability_points": [11.0],
        "expected_roi_pct": [150.0],
        "projected_red_zone_share": [0.0],
        "projected_goal_line_share": [0.0],
        "expected_tds_l5": [0.1],
    })

    all_prices = pd.DataFrame({
        "player_id": ["p1"],
        "book_key": ["book1"],
        "sportsbook_odds": [1300],
        "sportsbook_implied_probability": [0.0714],
    })

    out = add_value_quality(best, all_prices)

    assert out.iloc[0]["value_quality"] == "REVIEW_ONLY"
    assert "THIN_MARKET" in out.iloc[0]["quality_flags"]
    assert "LOW_ROLE_SUPPORT" in out.iloc[0]["quality_flags"]


def test_two_book_supported_edge_can_be_actionable():
    best = pd.DataFrame({
        "player_id": ["p1"],
        "edge_probability_points": [5.0],
        "expected_roi_pct": [15.0],
        "projected_red_zone_share": [0.25],
        "projected_goal_line_share": [0.20],
        "expected_tds_l5": [2.0],
    })

    all_prices = pd.DataFrame({
        "player_id": ["p1", "p1"],
        "book_key": ["book1", "book2"],
        "sportsbook_odds": [150, 145],
        "sportsbook_implied_probability": [0.40, 0.408],
    })

    out = add_value_quality(best, all_prices)

    assert out.iloc[0]["value_quality"] == "ACTIONABLE"
