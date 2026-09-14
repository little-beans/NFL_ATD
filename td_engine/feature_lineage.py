from __future__ import annotations

import pandas as pd


CURRENT_WEEK_ALIAS_MAP = {
    # canonical model feature -> preferred current-week candidates
    "volume_share": [
        "adjusted_volume_share",
        "volume_share",
    ],
    "target_share": [
        "adjusted_target_share",
        "target_share",
    ],
    "red_zone_share": [
        "adjusted_red_zone_share",
        "red_zone_share",
    ],
    "goal_line_share": [
        "adjusted_goal_line_share",
        "goal_line_share",
    ],
    "team_implied_points": [
        "team_implied_points",
        "implied_team_points",
    ],
    "game_total": [
        "game_total",
        "total",
    ],
    "spread": [
        "team_spread",
        "spread",
    ],
}


def apply_current_week_aliases(
    rows: pd.DataFrame,
    required_features: list[str],
) -> tuple[pd.DataFrame, dict]:
    """
    Populate model features from current-week columns where an explicit,
    safe alias is known.

    This runs BEFORE historical feature hydration.
    """
    out = rows.copy()
    applied = {}

    for feature in required_features:
        candidates = CURRENT_WEEK_ALIAS_MAP.get(feature, [feature])

        chosen = None
        for c in candidates:
            if c in out.columns:
                chosen = c
                break

        if chosen is None:
            continue

        if feature not in out.columns:
            out[feature] = out[chosen]
        elif chosen != feature:
            # current-week alias should win when available
            current = pd.to_numeric(out[chosen], errors="coerce")
            base = pd.to_numeric(out[feature], errors="coerce")
            out[feature] = current.where(current.notna(), base)

        applied[feature] = chosen

    return out, applied


def build_feature_lineage(
    rows_before_hydration: pd.DataFrame,
    rows_after_hydration: pd.DataFrame,
    required_features: list[str],
    current_aliases: dict,
) -> pd.DataFrame:
    """
    Summarize where each model feature came from.
    """
    records = []

    before_cols = set(rows_before_hydration.columns)
    after_cols = set(rows_after_hydration.columns)

    for feature in required_features:
        if feature in current_aliases:
            source = f"current_week:{current_aliases[feature]}"
        elif feature in before_cols:
            source = "current_week:direct"
        elif feature in after_cols:
            source = "historical_fallback"
        else:
            source = "missing"

        nonnull = (
            rows_after_hydration[feature].notna().sum()
            if feature in rows_after_hydration.columns
            else 0
        )

        records.append({
            "feature": feature,
            "source": source,
            "nonnull_rows": int(nonnull),
            "total_rows": int(len(rows_after_hydration)),
        })

    return pd.DataFrame(records)


def lineage_summary(lineage: pd.DataFrame) -> dict:
    if lineage.empty:
        return {
            "total_features": 0,
            "current_week_features": 0,
            "historical_features": 0,
            "missing_features": 0,
            "historical_fraction": 0.0,
        }

    current = lineage["source"].str.startswith("current_week").sum()
    historical = lineage["source"].eq("historical_fallback").sum()
    missing = lineage["source"].eq("missing").sum()
    total = len(lineage)

    return {
        "total_features": int(total),
        "current_week_features": int(current),
        "historical_features": int(historical),
        "missing_features": int(missing),
        "historical_fraction": float(historical / total) if total else 0.0,
    }
