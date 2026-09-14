from __future__ import annotations
import numpy as np
import pandas as pd


ROLE_WEIGHTS = {
    "volume_share": 0.30,
    "target_share": 0.20,
    "red_zone_share": 0.25,
    "goal_line_share": 0.25,
}

POSITION_GROUPS = {
    "RB": {"RB","FB"},
    "WR": {"WR"},
    "TE": {"TE"},
}


def _position_group(position: str) -> str:
    p = str(position or "").upper()
    for group, vals in POSITION_GROUPS.items():
        if p in vals:
            return group
    return "OTHER"


def _replacement_weights(group: pd.DataFrame, role_col: str) -> pd.Series:
    base = pd.to_numeric(group[role_col], errors="coerce").fillna(0).clip(lower=0)
    return base + 0.03


def reallocate_team_roles(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Redistribute unavailable role to teammates.

    Important V2 change:
    injury_role_boost is now an ABSOLUTE weighted opportunity delta,
    not the average relative percentage change. This prevents tiny
    0 -> 0.01 changes from creating exaggerated ROLE_BOOST tags.
    """
    d = team_df.copy()

    if "availability_multiplier" not in d.columns:
        d["availability_multiplier"] = 1.0

    d["availability_multiplier"] = (
        pd.to_numeric(d["availability_multiplier"], errors="coerce")
        .fillna(1.0).clip(0,1)
    )

    if "position" not in d.columns:
        d["position"] = ""

    d["_pos_group"] = d["position"].map(_position_group)

    delta_cols = []

    for role_col in ROLE_WEIGHTS:
        if role_col not in d.columns:
            d[role_col] = 0.0

        base = pd.to_numeric(d[role_col], errors="coerce").fillna(0).clip(lower=0)
        retained = base * d["availability_multiplier"]
        adjusted = retained.copy()

        unavailable = d[(d["availability_multiplier"] < 1) & (base > 0)]
        total_lost = float((base - retained).sum())
        remaining = total_lost

        eligible = d["availability_multiplier"] > 0

        # Keep most lost work within the same position group.
        for idx, row in unavailable.iterrows():
            lost = float(base.loc[idx] - retained.loc[idx])
            if lost <= 0:
                continue

            same_group_mask = (
                eligible
                & d["_pos_group"].eq(row["_pos_group"])
                & d.index.to_series().ne(idx)
            )
            candidates = d.loc[same_group_mask]

            pool = lost * (0.80 if not candidates.empty else 0.0)
            if pool > 0:
                w = _replacement_weights(candidates, role_col)
                w = w / w.sum()
                adjusted.loc[candidates.index] += pool * w
                remaining -= pool

        # Spill remainder across active skill players.
        active_skill = eligible & d["_pos_group"].isin(["RB","WR","TE"])
        candidates = d.loc[active_skill]
        if remaining > 0 and not candidates.empty:
            w = _replacement_weights(candidates, role_col)
            w = w / w.sum()
            adjusted.loc[candidates.index] += remaining * w

        adjusted = adjusted.clip(lower=0)

        if adjusted.sum() > 1.25:
            adjusted *= 1.25 / adjusted.sum()

        d[f"adjusted_{role_col}"] = adjusted
        d[f"delta_{role_col}"] = adjusted - base
        delta_cols.append(f"delta_{role_col}")

    # Weighted absolute opportunity delta.
    score = pd.Series(0.0, index=d.index)
    for role_col, weight in ROLE_WEIGHTS.items():
        score += d[f"delta_{role_col}"] * weight

    d["injury_role_boost"] = score

    d["injury_context_tag"] = np.select(
        [
            d["availability_multiplier"].eq(0),
            d["injury_role_boost"].ge(0.075),
            d["injury_role_boost"].ge(0.030),
            d["availability_multiplier"].lt(1),
        ],
        ["OUT","MAJOR_ROLE_BOOST","ROLE_BOOST","LIMITED"],
        default="",
    )

    return d.drop(columns=["_pos_group"])


def apply_role_reallocation(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()

    team_col = "posteam" if "posteam" in rows.columns else "team"
    if team_col not in rows.columns:
        raise KeyError("Role reallocation requires posteam or team.")

    return pd.concat(
        [reallocate_team_roles(g) for _, g in rows.groupby(team_col, sort=False, dropna=False)],
        ignore_index=True,
    )
