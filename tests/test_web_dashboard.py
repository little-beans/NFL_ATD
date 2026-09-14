import pandas as pd

from scripts.build_web_dashboard import build_dashboard, classify_tier


def test_actionable_tier():
    row = pd.Series({
        "edge_probability_points": 5.0,
        "expected_roi_pct": 12.0,
        "books_available": 3,
        "role_support_score": 80,
        "market_dispersion_pp": 1.0,
    })
    assert classify_tier(row) == "ACTIONABLE"


def test_dashboard_contains_player():
    df = pd.DataFrame({
        "player_name": ["A.Player"],
        "posteam": ["BUF"],
        "position": ["WR"],
        "model_probability": [0.50],
        "model_fair_odds": [100],
        "book_title": ["DraftKings"],
        "sportsbook_odds": [130],
        "sportsbook_implied_probability": [0.4348],
        "edge_probability_points": [6.5],
        "expected_roi_pct": [15.0],
        "books_available": [3],
        "role_support_score": [80],
        "market_dispersion_pp": [2.0],
        "expected_tds_l5": [2.1],
        "td_debt_tag": [""],
        "quality_flags": [""],
    })

    page = build_dashboard(df, 2026, 2, "test time")

    assert "A.Player" in page
    assert "Week 2 Value Board" in page
    assert "ACTIONABLE" in page
