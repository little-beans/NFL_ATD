import pandas as pd

from td_engine.value_quality import add_value_quality


def test_two_books_default_gate_is_actionable():
    best = pd.DataFrame({
        "player_id": ["p1"],
        "edge_probability_points": [2.9],
        "expected_roi_pct": [6.3],
        "projected_red_zone_share": [0.22],
        "projected_goal_line_share": [0.12],
        "expected_tds_l5": [3.84],
    })

    all_prices = pd.DataFrame({
        "player_id": ["p1", "p1"],
        "book_key": ["book1", "book2"],
        "sportsbook_odds": [120, 115],
        "sportsbook_implied_probability": [0.4545, 0.4651],
    })

    out = add_value_quality(
        best,
        all_prices,
        min_books_actionable=2,
        min_role_support=35,
        max_dispersion_pp=7.5,
        min_edge_pp=2.0,
        min_roi_pct=0.0,
    )

    row = out.iloc[0]

    assert row["books_available"] == 2
    assert row["market_depth"] == "MODERATE"
    assert bool(row["passes_market_depth"])
    assert bool(row["passes_edge"])
    assert bool(row["passes_role_support"])
    assert bool(row["passes_market_dispersion"])
    assert row["value_quality"] == "ACTIONABLE"
    assert row["quality_flags"] == ""


def test_one_book_stays_review_only():
    best = pd.DataFrame({
        "player_id": ["p1"],
        "edge_probability_points": [6.0],
        "expected_roi_pct": [20.0],
        "projected_red_zone_share": [0.30],
        "projected_goal_line_share": [0.25],
        "expected_tds_l5": [3.0],
    })

    all_prices = pd.DataFrame({
        "player_id": ["p1"],
        "book_key": ["book1"],
        "sportsbook_odds": [200],
        "sportsbook_implied_probability": [0.3333],
    })

    out = add_value_quality(best, all_prices)
    row = out.iloc[0]

    assert row["books_available"] == 1
    assert not bool(row["passes_market_depth"])
    assert row["value_quality"] == "REVIEW_ONLY"
    assert "THIN_MARKET_1_BOOK" in row["quality_flags"]
