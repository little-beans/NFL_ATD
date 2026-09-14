from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd


NFL_TEAM_NAMES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "LA": "Los Angeles Rams",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}

FULL_TEAM_TO_ABBR = {v: k for k, v in NFL_TEAM_NAMES.items()}


def american_to_implied_probability(odds) -> float:
    """Convert American odds to raw implied probability."""
    if odds is None or pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds == 0:
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def american_profit_per_unit(odds) -> float:
    """
    Net profit on a 1-unit winning wager at American odds.
    Example: +150 -> 1.50 profit; -200 -> 0.50 profit.
    """
    if odds is None or pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return odds / 100.0
    if odds < 0:
        return 100.0 / abs(odds)
    return np.nan


def expected_value_per_unit(model_probability, american_odds) -> float:
    """
    Expected net profit for a 1-unit wager.
    EV = p(win)*profit - p(loss)*1
    """
    if pd.isna(model_probability) or pd.isna(american_odds):
        return np.nan
    p = float(model_probability)
    profit = american_profit_per_unit(american_odds)
    if pd.isna(profit):
        return np.nan
    return p * profit - (1.0 - p)


def expected_roi_pct(model_probability, american_odds) -> float:
    ev = expected_value_per_unit(model_probability, american_odds)
    return np.nan if pd.isna(ev) else 100.0 * ev


def _ascii(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    return "".join(c for c in value if not unicodedata.combining(c))


def _clean_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _ascii(value).lower())


def player_match_key(name: str) -> tuple[str, str]:
    """
    Create a conservative player key: first initial + normalized last token.

    Works with:
      "Bijan Robinson" -> ("b", "robinson")
      "Bi.Robinson" -> ("b", "robinson")
      "J.Smith-Njigba" -> ("j", "smithnjigba")

    Team/event context should also be used when matching.
    """
    raw = str(name or "").strip()
    if not raw:
        return "", ""

    # Convert dots to spaces while preserving hyphenated surnames.
    parts = [p for p in re.split(r"[\s.]+", raw) if p]
    if not parts:
        return "", ""

    first = _clean_token(parts[0])
    last = _clean_token(parts[-1])

    return (first[:1], last)


def extract_atd_prices(event: dict) -> pd.DataFrame:
    """
    Flatten The Odds API player_anytime_td market for one event.

    Expected player-prop shape commonly uses:
      outcome.name == "Yes"
      outcome.description == "<Player Name>"

    A fallback also supports player name directly in outcome.name.
    """
    rows = []

    event_id = event.get("id")
    home_team = event.get("home_team")
    away_team = event.get("away_team")
    commence_time = event.get("commence_time")

    for book in event.get("bookmakers", []) or []:
        book_key = book.get("key")
        book_title = book.get("title", book_key)
        book_updated = book.get("last_update")

        for market in book.get("markets", []) or []:
            if market.get("key") != "player_anytime_td":
                continue

            market_updated = market.get("last_update", book_updated)

            for outcome in market.get("outcomes", []) or []:
                outcome_name = str(outcome.get("name", "") or "")
                description = str(outcome.get("description", "") or "")

                # For Yes/No player props, use the player description and keep Yes.
                if description:
                    if outcome_name.lower() not in {"yes", "over"}:
                        continue
                    player_name = description
                else:
                    # Some feeds can expose the player directly as the outcome name.
                    if outcome_name.lower() in {"no", "under", "yes", "over", ""}:
                        continue
                    player_name = outcome_name

                price = outcome.get("price")
                if price is None:
                    continue

                first_initial, surname = player_match_key(player_name)

                rows.append({
                    "event_id": event_id,
                    "commence_time": commence_time,
                    "home_team": home_team,
                    "away_team": away_team,
                    "book_key": book_key,
                    "book_title": book_title,
                    "book_last_update": market_updated,
                    "sportsbook_player_name": player_name,
                    "player_first_initial": first_initial,
                    "player_surname_key": surname,
                    "sportsbook_odds": float(price),
                    "sportsbook_implied_probability": american_to_implied_probability(price),
                })

    return pd.DataFrame(rows)


def add_model_match_keys(model_rows: pd.DataFrame) -> pd.DataFrame:
    out = model_rows.copy()

    keys = out["player_name"].map(player_match_key)
    out["player_first_initial"] = keys.map(lambda x: x[0])
    out["player_surname_key"] = keys.map(lambda x: x[1])

    out["team_full_name"] = out["posteam"].map(NFL_TEAM_NAMES)
    return out


def attach_event_team(price_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each sportsbook row to the two possible event team abbreviations.

    Player name matching is later constrained to a model player's current team.
    """
    if price_rows.empty:
        return price_rows.copy()

    d = price_rows.copy()
    d["home_abbr"] = d["home_team"].map(FULL_TEAM_TO_ABBR)
    d["away_abbr"] = d["away_team"].map(FULL_TEAM_TO_ABBR)
    return d


def match_prices_to_model(
    model_rows: pd.DataFrame,
    price_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match sportsbook ATD player prices to model players using:
      - first initial
      - normalized surname
      - player's current team participating in the event

    Returns:
      matched price rows
      unmatched sportsbook rows
    """
    model = add_model_match_keys(model_rows)
    prices = attach_event_team(price_rows)

    if prices.empty:
        return pd.DataFrame(), prices

    matched = []
    unmatched = []

    for _, r in prices.iterrows():
        candidates = model[
            model["player_first_initial"].eq(r["player_first_initial"])
            & model["player_surname_key"].eq(r["player_surname_key"])
            & (
                model["posteam"].eq(r.get("home_abbr"))
                | model["posteam"].eq(r.get("away_abbr"))
            )
        ]

        if len(candidates) != 1:
            x = r.to_dict()
            x["match_candidate_count"] = len(candidates)
            unmatched.append(x)
            continue

        player = candidates.iloc[0]

        x = r.to_dict()
        x["player_id"] = player.get("player_id")
        x["player_name"] = player.get("player_name")
        x["posteam"] = player.get("posteam")
        x["position"] = player.get("position")
        x["model_probability"] = player.get("td_probability")
        if pd.isna(x["model_probability"]) and "td_probability_pct" in player.index:
            x["model_probability"] = player.get("td_probability_pct") / 100.0

        for c in [
            "model_fair_odds",
            "fair_american_odds",
            "projected_goal_line_share",
            "projected_red_zone_share",
            "expected_tds_l5",
            "td_debt_l5",
            "td_debt_tag",
            "injury_role_boost",
            "calibration_method",
        ]:
            if c in player.index:
                x[c] = player.get(c)

        matched.append(x)

    return pd.DataFrame(matched), pd.DataFrame(unmatched)


def classify_value(edge_pp, ev_roi_pct) -> str:
    """
    Conservative screening labels. These are review tiers, not bet commands.
    """
    if pd.isna(edge_pp) or pd.isna(ev_roi_pct):
        return "NO_DATA"

    if edge_pp < 2.0 or ev_roi_pct <= 0:
        return "NO_EDGE"
    if edge_pp < 4.0:
        return "SMALL_EDGE"
    if edge_pp < 7.0:
        return "VALUE"
    if edge_pp < 10.0:
        return "STRONG_VALUE"
    return "REVIEW_LARGE_DISAGREEMENT"


def build_value_board(matched_prices: pd.DataFrame) -> pd.DataFrame:
    if matched_prices.empty:
        return matched_prices.copy()

    d = matched_prices.copy()

    d["model_probability"] = pd.to_numeric(
        d["model_probability"], errors="coerce"
    )

    d["sportsbook_implied_probability"] = pd.to_numeric(
        d["sportsbook_implied_probability"], errors="coerce"
    )

    d["edge_probability_points"] = 100.0 * (
        d["model_probability"] - d["sportsbook_implied_probability"]
    )

    d["expected_value_per_unit"] = [
        expected_value_per_unit(p, o)
        for p, o in zip(d["model_probability"], d["sportsbook_odds"])
    ]

    d["expected_roi_pct"] = 100.0 * d["expected_value_per_unit"]

    # Consensus is a median of raw implied probabilities across books.
    # It is NOT labeled no-vig because ATD prices do not reliably provide
    # a complementary No price for every player/book.
    consensus = (
        d.groupby("player_id", dropna=False)["sportsbook_implied_probability"]
        .median()
        .rename("consensus_market_implied_probability")
    )
    d = d.merge(consensus, on="player_id", how="left")

    d["consensus_edge_probability_points"] = 100.0 * (
        d["model_probability"] - d["consensus_market_implied_probability"]
    )

    d["value_rating"] = [
        classify_value(edge, roi)
        for edge, roi in zip(
            d["edge_probability_points"],
            d["expected_roi_pct"],
        )
    ]

    return d


def best_price_board(value_rows: pd.DataFrame) -> pd.DataFrame:
    """
    One row per player using the most favorable available American price.
    For American odds, numerically larger is always more favorable:
      +200 > +150 > -110 > -150.
    """
    if value_rows.empty:
        return value_rows.copy()

    d = value_rows.copy()
    d["sportsbook_odds"] = pd.to_numeric(d["sportsbook_odds"], errors="coerce")

    best_idx = (
        d.sort_values("sportsbook_odds", ascending=False)
        .groupby("player_id", dropna=False)
        .head(1)
        .index
    )

    best = d.loc[best_idx].copy()

    book_count = (
        d.groupby("player_id", dropna=False)["book_key"]
        .nunique()
        .rename("books_available")
    )

    best = best.merge(book_count, on="player_id", how="left")

    best = best.sort_values(
        ["expected_roi_pct", "edge_probability_points"],
        ascending=[False, False],
    )

    return best.reset_index(drop=True)
