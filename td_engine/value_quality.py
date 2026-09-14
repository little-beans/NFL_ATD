from __future__ import annotations

import numpy as np
import pandas as pd


def add_market_quality_metrics(all_prices: pd.DataFrame) -> pd.DataFrame:
    if all_prices.empty:
        return all_prices.copy()

    d = all_prices.copy()

    stats = (
        d.groupby("player_id", dropna=False)
        .agg(
            books_available=("book_key", "nunique"),
            market_odds_min=("sportsbook_odds", "min"),
            market_odds_max=("sportsbook_odds", "max"),
            market_implied_mean=("sportsbook_implied_probability", "mean"),
            market_implied_median=("sportsbook_implied_probability", "median"),
            market_implied_std=("sportsbook_implied_probability", "std"),
        )
        .reset_index()
    )

    stats["market_implied_std"] = stats["market_implied_std"].fillna(0.0)
    stats["market_dispersion_pp"] = 100.0 * stats["market_implied_std"]

    return d.merge(stats, on="player_id", how="left")


def _num(value, default=0.0):
    x = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return default if pd.isna(x) else float(x)


def role_support_score(row: pd.Series) -> float:
    rz = _num(row.get("projected_red_zone_share"), 0.0)
    gl = _num(row.get("projected_goal_line_share"), 0.0)
    xtd5 = _num(row.get("expected_tds_l5"), 0.0)

    rz_component = min(max(rz / 0.30, 0.0), 1.0)
    gl_component = min(max(gl / 0.20, 0.0), 1.0)
    xtd_component = min(max(xtd5 / 2.5, 0.0), 1.0)

    return 100.0 * (
        0.35 * rz_component
        + 0.35 * gl_component
        + 0.30 * xtd_component
    )


def classify_market_depth(books_available: int) -> str:
    n = int(books_available or 0)
    if n >= 4:
        return "DEEP"
    if n >= 2:
        return "MODERATE"
    if n == 1:
        return "THIN"
    return "NONE"


def add_value_quality(
    best_rows: pd.DataFrame,
    all_prices: pd.DataFrame,
    *,
    min_books_actionable: int = 2,
    min_role_support: float = 35.0,
    max_dispersion_pp: float = 7.5,
    min_edge_pp: float = 2.0,
    min_roi_pct: float = 0.0,
) -> pd.DataFrame:
    if best_rows.empty:
        return best_rows.copy()

    allq = add_market_quality_metrics(all_prices)

    market_cols = [
        "player_id",
        "books_available",
        "market_odds_min",
        "market_odds_max",
        "market_implied_mean",
        "market_implied_median",
        "market_dispersion_pp",
    ]
    market = allq[market_cols].drop_duplicates("player_id")

    out = (
        best_rows.drop(
            columns=[
                c for c in market_cols
                if c != "player_id" and c in best_rows.columns
            ],
            errors="ignore",
        )
        .merge(market, on="player_id", how="left")
    )

    out["books_available"] = (
        pd.to_numeric(out["books_available"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    out["market_dispersion_pp"] = (
        pd.to_numeric(out["market_dispersion_pp"], errors="coerce")
        .fillna(0.0)
    )
    out["edge_probability_points"] = pd.to_numeric(
        out["edge_probability_points"], errors="coerce"
    )
    out["expected_roi_pct"] = pd.to_numeric(
        out["expected_roi_pct"], errors="coerce"
    )

    out["market_depth"] = out["books_available"].apply(classify_market_depth)
    out["role_support_score"] = out.apply(role_support_score, axis=1)

    # Recompute all gates from the final merged metrics.
    out["passes_edge"] = (
        out["edge_probability_points"].ge(float(min_edge_pp))
        & out["expected_roi_pct"].ge(float(min_roi_pct))
    )
    out["passes_market_depth"] = (
        out["books_available"].ge(int(min_books_actionable))
    )
    out["passes_role_support"] = (
        out["role_support_score"].ge(float(min_role_support))
    )
    out["passes_market_dispersion"] = (
        out["market_dispersion_pp"].le(float(max_dispersion_pp))
    )

    actionable_mask = (
        out["passes_edge"]
        & out["passes_market_depth"]
        & out["passes_role_support"]
        & out["passes_market_dispersion"]
    )

    review_mask = out["passes_edge"] & ~actionable_mask

    out["value_quality"] = "NO_EDGE"
    out.loc[review_mask, "value_quality"] = "REVIEW_ONLY"
    out.loc[actionable_mask, "value_quality"] = "ACTIONABLE"

    flags = []

    for _, r in out.iterrows():
        reasons = []

        if not bool(r["passes_edge"]):
            reasons.append("EDGE_BELOW_THRESHOLD")

        if not bool(r["passes_market_depth"]):
            reasons.append(
                f"THIN_MARKET_{int(r['books_available'])}_BOOK"
            )

        if not bool(r["passes_role_support"]):
            reasons.append("LOW_ROLE_SUPPORT")

        if not bool(r["passes_market_dispersion"]):
            reasons.append("HIGH_BOOK_DISAGREEMENT")

        if (
            _num(r.get("edge_probability_points"), 0.0) >= 10.0
            and int(r.get("books_available", 0) or 0) <= 1
        ):
            reasons.append("LARGE_EDGE_ONE_BOOK")

        flags.append("|".join(reasons))

    out["quality_flags"] = flags

    # Store the thresholds used so CSVs are auditable.
    out["quality_min_books"] = int(min_books_actionable)
    out["quality_min_edge_pp"] = float(min_edge_pp)
    out["quality_min_roi_pct"] = float(min_roi_pct)
    out["quality_min_role_support"] = float(min_role_support)
    out["quality_max_dispersion_pp"] = float(max_dispersion_pp)

    return out


def sort_value_board(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    rank = {
        "ACTIONABLE": 0,
        "REVIEW_ONLY": 1,
        "NO_EDGE": 2,
    }

    out = df.copy()
    out["_quality_rank"] = out["value_quality"].map(rank).fillna(9)

    out = out.sort_values(
        [
            "_quality_rank",
            "expected_roi_pct",
            "edge_probability_points",
            "role_support_score",
        ],
        ascending=[True, False, False, False],
    )

    return out.drop(columns="_quality_rank").reset_index(drop=True)
