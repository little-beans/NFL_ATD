from __future__ import annotations

import math
import shutil
from typing import Iterable

import numpy as np
import pandas as pd


def _pct(value, digits=1):
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):.{digits}f}%"
    except Exception:
        return "-"


def _odds(value):
    try:
        if pd.isna(value):
            return "-"
        x = int(round(float(value)))
        return f"+{x}" if x > 0 else str(x)
    except Exception:
        return "-"


def _num(value, digits=1):
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def _short_book(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    text = str(value)
    replacements = {
        "DraftKings": "DK",
        "FanDuel": "FD",
        "BetMGM": "MGM",
        "Caesars": "CZR",
        "BetOnline.ag": "BOL",
    }
    return replacements.get(text, text[:8])


def _tag(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    value = str(value).strip()
    return value if value.lower() != "nan" else ""


def build_display_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert full value rows into a compact console-friendly view.
    Assumes model_probability is decimal in source data.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Player","Tm","Pos","Model","Fair","Book","Odds",
                "Mkt","Edge","ROI","Books","Role","xTD5","Tag"
            ]
        )

    rows = []

    for _, r in df.iterrows():
        model_p = pd.to_numeric(
            pd.Series([r.get("model_probability")]),
            errors="coerce"
        ).iloc[0]

        market_p = pd.to_numeric(
            pd.Series([r.get("sportsbook_implied_probability")]),
            errors="coerce"
        ).iloc[0]

        rows.append({
            "Player": str(r.get("player_name", "")),
            "Tm": str(r.get("posteam", "")),
            "Pos": str(r.get("position", "")),
            "Model": _pct(model_p * 100 if pd.notna(model_p) else np.nan),
            "Fair": _odds(r.get("model_fair_odds")),
            "Book": _short_book(r.get("book_title")),
            "Odds": _odds(r.get("sportsbook_odds")),
            "Mkt": _pct(market_p * 100 if pd.notna(market_p) else np.nan),
            "Edge": (
                f"{float(r.get('edge_probability_points')):+.1f}pp"
                if pd.notna(r.get("edge_probability_points"))
                else "-"
            ),
            "ROI": (
                f"{float(r.get('expected_roi_pct')):+.1f}%"
                if pd.notna(r.get("expected_roi_pct"))
                else "-"
            ),
            "Books": int(r.get("books_available", 0) or 0),
            "Role": _num(r.get("role_support_score"), 0),
            "xTD5": _num(r.get("expected_tds_l5"), 2),
            "Tag": _tag(r.get("td_debt_tag")),
        })

    return pd.DataFrame(rows)


def _terminal_width(default=160):
    try:
        return shutil.get_terminal_size((default, 30)).columns
    except Exception:
        return default


def print_table(df: pd.DataFrame, max_rows: int = 12) -> None:
    if df.empty:
        print("  None")
        return

    display = build_display_rows(df.head(max_rows))

    # Keep the important columns even on narrower terminals.
    width = _terminal_width()

    if width < 120:
        cols = [
            "Player","Tm","Pos","Model","Odds","Edge","ROI","Books","Role"
        ]
    elif width < 155:
        cols = [
            "Player","Tm","Pos","Model","Fair","Book","Odds",
            "Edge","ROI","Books","Role"
        ]
    else:
        cols = [
            "Player","Tm","Pos","Model","Fair","Book","Odds",
            "Mkt","Edge","ROI","Books","Role","xTD5","Tag"
        ]

    print(
        display[cols].to_string(
            index=False,
            justify="left",
        )
    )


def split_value_sections(
    board: pd.DataFrame,
    *,
    watch_min_edge: float = 2.0,
    actionable_min_edge: float = 4.0,
    strong_min_edge: float = 7.0,
    actionable_min_books: int = 3,
) -> dict[str, pd.DataFrame]:
    """
    Console-only interpretation layer.

    Does not alter the underlying model or saved value calculations.
    """
    if board.empty:
        empty = board.copy()
        return {
            "strong": empty,
            "actionable": empty,
            "watchlist": empty,
            "review": empty,
        }

    d = board.copy()

    edge = pd.to_numeric(
        d["edge_probability_points"], errors="coerce"
    ).fillna(-999)

    roi = pd.to_numeric(
        d["expected_roi_pct"], errors="coerce"
    ).fillna(-999)

    books = pd.to_numeric(
        d["books_available"], errors="coerce"
    ).fillna(0)

    role = pd.to_numeric(
        d["role_support_score"], errors="coerce"
    ).fillna(0)

    dispersion = pd.to_numeric(
        d["market_dispersion_pp"], errors="coerce"
    ).fillna(0)

    base_ok = (
        roi.gt(0)
        & role.ge(35)
        & dispersion.le(7.5)
    )

    strong_mask = (
        base_ok
        & edge.ge(strong_min_edge)
        & books.ge(actionable_min_books)
    )

    actionable_mask = (
        base_ok
        & edge.ge(actionable_min_edge)
        & edge.lt(strong_min_edge)
        & books.ge(actionable_min_books)
    )

    watch_mask = (
        base_ok
        & edge.ge(watch_min_edge)
        & edge.lt(actionable_min_edge)
        & books.ge(2)
    )

    already = strong_mask | actionable_mask | watch_mask

    review_mask = (
        edge.ge(watch_min_edge)
        & roi.gt(0)
        & ~already
    )

    def order(x):
        if x.empty:
            return x
        return x.sort_values(
            ["edge_probability_points","expected_roi_pct"],
            ascending=[False,False]
        )

    return {
        "strong": order(d[strong_mask].copy()),
        "actionable": order(d[actionable_mask].copy()),
        "watchlist": order(d[watch_mask].copy()),
        "review": order(d[review_mask].copy()),
    }


def print_value_console(
    board: pd.DataFrame,
    matched_count: int,
    unmatched_count: int,
    *,
    top_n: int = 10,
) -> None:
    sections = split_value_sections(board)

    print()
    print("=" * 78)
    print(" NFL ANYTIME TD VALUE BOARD")
    print("=" * 78)
    print(
        f" Matched odds: {matched_count}   "
        f"Unmatched: {unmatched_count}   "
        f"Players priced: {len(board)}"
    )
    print()

    print(
        f" STRONG VALUE: {len(sections['strong'])}   "
        f"ACTIONABLE: {len(sections['actionable'])}   "
        f"WATCHLIST: {len(sections['watchlist'])}   "
        f"REVIEW: {len(sections['review'])}"
    )

    print()
    print(">>> STRONG VALUE  | edge >= 7pp, 3+ books")
    print("-" * 78)
    print_table(sections["strong"], top_n)

    print()
    print(">>> ACTIONABLE    | edge 4-7pp, 3+ books")
    print("-" * 78)
    print_table(sections["actionable"], top_n)

    print()
    print(">>> WATCHLIST     | edge 2-4pp, 2+ books")
    print("-" * 78)
    print_table(sections["watchlist"], top_n)

    print()
    print(">>> REVIEW ONLY   | positive edge but thin/low-quality market")
    print("-" * 78)
    print_table(sections["review"], top_n)

    print()
    print("Legend:")
    print("  Model = model TD probability")
    print("  Fair  = model-implied American odds")
    print("  Odds  = best sportsbook price")
    print("  Edge  = model probability minus sportsbook implied probability")
    print("  ROI   = model-estimated expected return at that price")
    print("  Role  = current scoring-opportunity support score (0-100)")
    print("  xTD5  = expected TDs over recent 5-game window")
    print("=" * 78)
