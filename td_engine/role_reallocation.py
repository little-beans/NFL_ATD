from __future__ import annotations

import numpy as np
import pandas as pd


ROLE_COLS = {
    "touch_share": "volume_share",
    "target_share": "target_share",
    "rz_share": "red_zone_share",
    "goal_line_share": "goal_line_share",
}

POSITION_GROUPS = {
    "RB": {"RB", "FB"},
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
    d = team_df.copy()

    if "availability_multiplier" not in d.columns:
        d["availability_multiplier"] = 1.0

    d["availability_multiplier"] = (
        pd.to_numeric(d["availability_multiplier"], errors="coerce")
        .fillna(1.0)
        .clip(0, 1)
    )

    if "position" not in d.columns:
        d["position"] = ""

    d["_pos_group"] = d["position"].map(_position_group)

    boost_components = []

    for _, role_col in ROLE_COLS.items():
        if role_col not in d.columns:
            d[role_col] = 0.0

        base = pd.to_numeric(d[role_col], errors="coerce").fillna(0).clip(lower=0)

        retained = base * d["availability_multiplier"]
        lost = float((base - retained).sum())

        adjusted = retained.copy()

        if lost > 0:
            eligible = d["availability_multiplier"] > 0
            unavailable = d[(d["availability_multiplier"] < 1) & (base > 0)]
            remaining = lost

            for idx, row in unavailable.iterrows():
                individual_lost = float(base.loc[idx] - retained.loc[idx])
                if individual_lost <= 0:
                    continue

                same_group = (
                    eligible
                    & d["_pos_group"].eq(row["_pos_group"])
                    & d.index.to_series().ne(idx)
                )
                candidates = d.loc[same_group]

                same_group_pool = individual_lost * (0.80 if not candidates.empty else 0.0)

                if same_group_pool > 0:
                    w = _replacement_weights(candidates, role_col)
                    w = w / w.sum()
                    adjusted.loc[candidates.index] += same_group_pool * w
                    remaining -= same_group_pool

            active_skill = eligible & d["_pos_group"].isin(["RB", "WR", "TE"])
            candidates = d.loc[active_skill]

            if remaining > 0 and not candidates.empty:
                w = _replacement_weights(candidates, role_col)
                w = w / w.sum()
                adjusted.loc[candidates.index] += remaining * w

        adjusted = adjusted.clip(lower=0)

        if adjusted.sum() > 1.25:
            adjusted *= 1.25 / adjusted.sum()

        out_col = f"adjusted_{role_col}"
        d[out_col] = adjusted

        denom = base.replace(0, np.nan)

        rel_boost = ((adjusted - base) / denom).replace(
            [np.inf, -np.inf], np.nan
        )

        # pandas fillna requires a scalar/dict/Series, not ndarray.
        fallback = pd.Series(
            np.where(adjusted > base, 1.0, 0.0),
            index=d.index,
            dtype=float,
        )

        rel_boost = rel_boost.fillna(fallback)
        boost_components.append(rel_boost)

    if boost_components:
        stacked = pd.concat(boost_components, axis=1)
        d["injury_role_boost"] = stacked.mean(axis=1).clip(-1, 3)
    else:
        d["injury_role_boost"] = 0.0

    d["injury_context_tag"] = np.select(
        [
            d["availability_multiplier"].eq(0),
            d["injury_role_boost"].ge(0.50),
            d["injury_role_boost"].ge(0.20),
            d["availability_multiplier"].lt(1),
        ],
        [
            "OUT",
            "MAJOR_ROLE_BOOST",
            "ROLE_BOOST",
            "LIMITED",
        ],
        default="",
    )

    return d.drop(columns=["_pos_group"])


def apply_role_reallocation(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()

    team_col = "posteam" if "posteam" in rows.columns else "team"
    if team_col not in rows.columns:
        raise KeyError("Role reallocation requires posteam or team.")

    pieces = []
    for _, g in rows.groupby(team_col, sort=False, dropna=False):
        pieces.append(reallocate_team_roles(g))

    return pd.concat(pieces, ignore_index=True)
