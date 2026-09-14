from __future__ import annotations

import pandas as pd


ROLE_FEATURES = [
    "red_zone_share",
    "volume_share",
    "goal_line_share",
    "target_share",
    "td_efficiency_score",
]


def latest_pregame_role_state(
    history: pd.DataFrame,
    target_season: int,
    target_week: int,
) -> pd.DataFrame:
    """
    Return the latest leakage-safe pregame role state strictly before
    target_season/target_week, one row per player.

    These are projections/priors for the upcoming week, not realized
    target-week role shares.
    """
    if history.empty:
        return pd.DataFrame(columns=["player_id"] + ROLE_FEATURES)

    if "player_id" not in history.columns:
        raise KeyError("Historical role data requires player_id.")

    h = history.copy()

    if "season" in h.columns and "week" in h.columns:
        s = pd.to_numeric(h["season"], errors="coerce")
        w = pd.to_numeric(h["week"], errors="coerce")
        h = h[
            s.lt(target_season)
            | (s.eq(target_season) & w.lt(target_week))
        ].copy()

    if h.empty:
        return pd.DataFrame(columns=["player_id"] + ROLE_FEATURES)

    sort_cols = [c for c in ["season","week"] if c in h.columns]
    if sort_cols:
        h = h.sort_values(["player_id"] + sort_cols)

    keep = ["player_id"] + [c for c in ROLE_FEATURES if c in h.columns]
    latest = (
        h[keep]
        .drop_duplicates("player_id", keep="last")
        .reset_index(drop=True)
    )

    return latest


def attach_projected_roles(
    weekly_rows: pd.DataFrame,
    role_state: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Replace placeholder upcoming-week role fields with the latest
    leakage-safe pregame role priors.

    Adds:
      projected_red_zone_share
      projected_volume_share
      projected_goal_line_share
      projected_target_share
      projected_td_efficiency_score

    Canonical model columns are also replaced by these projected values.
    """
    out = weekly_rows.copy()

    if "player_id" not in out.columns:
        raise KeyError("Weekly rows require player_id.")

    use_cols = ["player_id"] + [
        c for c in ROLE_FEATURES if c in role_state.columns
    ]

    use = role_state[use_cols].drop_duplicates("player_id", keep="last")

    renamed = {
        c: f"projected_{c}"
        for c in ROLE_FEATURES
        if c in use.columns
    }
    use = use.rename(columns=renamed)

    out = out.merge(use, on="player_id", how="left")

    matched_any = pd.Series(False, index=out.index)

    for feature in ROLE_FEATURES:
        pcol = f"projected_{feature}"
        if pcol not in out.columns:
            continue

        projected = pd.to_numeric(out[pcol], errors="coerce")
        matched_any |= projected.notna()

        # Projected pregame role should be model input.
        out[feature] = projected.where(projected.notna(), out.get(feature))

    diagnostics = {
        "matched_players": int(matched_any.sum()),
        "total_players": int(len(out)),
    }

    for feature in ROLE_FEATURES:
        pcol = f"projected_{feature}"
        if pcol in out.columns:
            diagnostics[f"{feature}_nonnull"] = int(out[pcol].notna().sum())
            diagnostics[f"{feature}_mean"] = float(
                pd.to_numeric(out[pcol], errors="coerce").mean()
            )
            diagnostics[f"{feature}_max"] = float(
                pd.to_numeric(out[pcol], errors="coerce").max()
            )

    return out, diagnostics
