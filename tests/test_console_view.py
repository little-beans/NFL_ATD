import pandas as pd

from td_engine.console_view import split_value_sections, build_display_rows


def test_watchlist_classification():
    d = pd.DataFrame({
        "edge_probability_points":[2.9],
        "expected_roi_pct":[6.4],
        "books_available":[2],
        "role_support_score":[76],
        "market_dispersion_pp":[0],
    })

    sections = split_value_sections(d)

    assert len(sections["watchlist"]) == 1
    assert len(sections["actionable"]) == 0


def test_actionable_requires_three_books():
    d = pd.DataFrame({
        "edge_probability_points":[5.0,5.0],
        "expected_roi_pct":[12.0,12.0],
        "books_available":[2,3],
        "role_support_score":[80,80],
        "market_dispersion_pp":[1,1],
    })

    sections = split_value_sections(d)

    assert len(sections["actionable"]) == 1
    assert len(sections["review"]) == 1


def test_compact_display_formats_percentages():
    d = pd.DataFrame({
        "player_name":["A.Player"],
        "posteam":["BUF"],
        "position":["WR"],
        "model_probability":[0.48],
        "model_fair_odds":[108],
        "book_title":["DraftKings"],
        "sportsbook_odds":[145],
        "sportsbook_implied_probability":[0.408],
        "edge_probability_points":[7.2],
        "expected_roi_pct":[17.7],
        "books_available":[3],
        "role_support_score":[95],
        "expected_tds_l5":[2.14],
        "td_debt_tag":[""],
    })

    out = build_display_rows(d)

    assert out.iloc[0]["Model"] == "48.0%"
    assert out.iloc[0]["Odds"] == "+145"
    assert out.iloc[0]["Edge"] == "+7.2pp"
