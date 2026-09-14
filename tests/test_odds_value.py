import math
import pandas as pd

from td_engine.odds_value import (
    american_to_implied_probability,
    expected_value_per_unit,
    player_match_key,
    extract_atd_prices,
    match_prices_to_model,
    build_value_board,
    best_price_board,
)


def test_american_implied_probability():
    assert abs(american_to_implied_probability(-200) - (2/3)) < 1e-9
    assert abs(american_to_implied_probability(150) - 0.4) < 1e-9


def test_ev_positive_when_model_beats_price():
    # Model 60% at +100 => +0.20 units EV
    assert abs(expected_value_per_unit(0.60, 100) - 0.20) < 1e-9


def test_player_match_key():
    assert player_match_key("Bijan Robinson") == ("b", "robinson")
    assert player_match_key("Bi.Robinson") == ("b", "robinson")
    assert player_match_key("J.Smith-Njigba") == ("j", "smithnjigba")


def test_extract_yes_player_prop_shape():
    event = {
        "id": "e1",
        "home_team": "Atlanta Falcons",
        "away_team": "Carolina Panthers",
        "commence_time": "2026-09-20T17:00:00Z",
        "bookmakers": [{
            "key": "fanduel",
            "title": "FanDuel",
            "markets": [{
                "key": "player_anytime_td",
                "outcomes": [
                    {
                        "name": "Yes",
                        "description": "Bijan Robinson",
                        "price": -130,
                    },
                    {
                        "name": "No",
                        "description": "Bijan Robinson",
                        "price": 100,
                    },
                ],
            }],
        }],
    }

    out = extract_atd_prices(event)
    assert len(out) == 1
    assert out.iloc[0]["sportsbook_player_name"] == "Bijan Robinson"
    assert out.iloc[0]["sportsbook_odds"] == -130


def test_match_and_best_price():
    model = pd.DataFrame({
        "player_id": ["p1"],
        "player_name": ["Bi.Robinson"],
        "posteam": ["ATL"],
        "position": ["RB"],
        "td_probability": [0.664],
        "fair_american_odds": [-198],
    })

    prices = pd.DataFrame([
        {
            "event_id": "e1",
            "home_team": "Atlanta Falcons",
            "away_team": "Carolina Panthers",
            "book_key": "fanduel",
            "book_title": "FanDuel",
            "sportsbook_player_name": "Bijan Robinson",
            "player_first_initial": "b",
            "player_surname_key": "robinson",
            "sportsbook_odds": -130,
            "sportsbook_implied_probability": american_to_implied_probability(-130),
        },
        {
            "event_id": "e1",
            "home_team": "Atlanta Falcons",
            "away_team": "Carolina Panthers",
            "book_key": "draftkings",
            "book_title": "DraftKings",
            "sportsbook_player_name": "Bijan Robinson",
            "player_first_initial": "b",
            "player_surname_key": "robinson",
            "sportsbook_odds": -120,
            "sportsbook_implied_probability": american_to_implied_probability(-120),
        },
    ])

    matched, unmatched = match_prices_to_model(model, prices)
    assert len(matched) == 2
    assert unmatched.empty

    value = build_value_board(matched)
    best = best_price_board(value)

    assert best.iloc[0]["book_key"] == "draftkings"
    assert best.iloc[0]["sportsbook_odds"] == -120
    assert best.iloc[0]["edge_probability_points"] > 0
    assert best.iloc[0]["expected_roi_pct"] > 0
