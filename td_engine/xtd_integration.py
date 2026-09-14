from __future__ import annotations

import numpy as np
import pandas as pd


PRIOR_FIELDS = [
    "expected_tds_prior",
    "actual_tds_prior",
    "td_debt",
    "expected_tds_l5",
    "actual_tds_l5",
    "td_debt_l5",
    "td_debt_tag",
]


def _first_present(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _before_target(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    d = df.copy()
    if "season" not in d.columns or "week" not in d.columns:
        return d

    d["season"] = pd.to_numeric(d["season"], errors="coerce")
    d["week"] = pd.to_numeric(d["week"], errors="coerce")

    mask = (
        d["season"].lt(season)
        | (d["season"].eq(season) & d["week"].lt(week))
    )
    return d[mask].copy()


def _build_priors_from_game_level(
    xtd_history: pd.DataFrame,
    target_season: int,
    target_week: int,
) -> pd.DataFrame:
    """
    Fallback for artifacts where expected_tds_prior/actual_tds_prior were
    not populated but game-level xTD and TD outcomes exist.

    Supported game-level xTD candidates:
      expected_tds, x_td, xtd, game_expected_tds

    Supported actual TD candidates:
      actual_tds, touchdowns, total_tds, scored_td
    """
    d = _before_target(xtd_history, target_season, target_week)

    if d.empty or "player_id" not in d.columns:
        return pd.DataFrame(columns=["player_id"] + PRIOR_FIELDS)

    exp_col = _first_present(
        d,
        ["expected_tds", "x_td", "xtd", "game_expected_tds", "expected_td"],
    )
    act_col = _first_present(
        d,
        ["actual_tds", "touchdowns", "total_tds", "scored_td", "td"],
    )

    if exp_col is None:
        return pd.DataFrame(columns=["player_id"] + PRIOR_FIELDS)

    d["_exp"] = pd.to_numeric(d[exp_col], errors="coerce").fillna(0)

    if act_col is not None:
        d["_act"] = pd.to_numeric(d[act_col], errors="coerce").fillna(0)
    else:
        d["_act"] = 0.0

    sort_cols = [c for c in ["season", "week"] if c in d.columns]
    if sort_cols:
        d = d.sort_values(["player_id"] + sort_cols)
    else:
        d = d.sort_values(["player_id"])

    rows = []
    for pid, g in d.groupby("player_id", sort=False):
        g = g.copy()

        exp_total = float(g["_exp"].sum())
        act_total = float(g["_act"].sum())

        l5 = g.tail(5)
        exp_l5 = float(l5["_exp"].sum())
        act_l5 = float(l5["_act"].sum())

        debt = exp_total - act_total
        debt_l5 = exp_l5 - act_l5

        if debt_l5 >= 1.25:
            tag = "DUE++"
        elif debt_l5 >= 0.65:
            tag = "DUE"
        elif debt_l5 <= -0.75:
            tag = "REGRESSION_RISK"
        else:
            tag = ""

        rows.append({
            "player_id": pid,
            "expected_tds_prior": exp_total,
            "actual_tds_prior": act_total,
            "td_debt": debt,
            "expected_tds_l5": exp_l5,
            "actual_tds_l5": act_l5,
            "td_debt_l5": debt_l5,
            "td_debt_tag": tag,
        })

    return pd.DataFrame(rows)


def select_latest_xtd_prior(
    xtd_history: pd.DataFrame,
    target_season: int,
    target_week: int,
) -> pd.DataFrame:
    """
    Prefer stored pregame prior fields when they are genuinely populated.
    Otherwise reconstruct priors from game-level xTD.
    """
    if xtd_history is None or xtd_history.empty:
        return pd.DataFrame(columns=["player_id"] + PRIOR_FIELDS)

    if "player_id" not in xtd_history.columns:
        raise KeyError("xTD history requires player_id.")

    d = _before_target(xtd_history, target_season, target_week)

    if d.empty:
        return pd.DataFrame(columns=["player_id"] + PRIOR_FIELDS)

    has_prior_fields = any(c in d.columns for c in PRIOR_FIELDS)

    if has_prior_fields:
        sort_cols = [c for c in ["season", "week"] if c in d.columns]
        if sort_cols:
            d = d.sort_values(["player_id"] + sort_cols)

        latest = d.drop_duplicates("player_id", keep="last")
        keep = ["player_id"] + [c for c in PRIOR_FIELDS if c in latest.columns]
        latest = latest[keep].copy()

        numeric_prior_cols = [
            c for c in [
                "expected_tds_prior",
                "actual_tds_prior",
                "expected_tds_l5",
                "actual_tds_l5",
            ]
            if c in latest.columns
        ]

        # Detect a broken "all-zero prior" artifact.
        prior_signal = 0.0
        for c in numeric_prior_cols:
            prior_signal += pd.to_numeric(latest[c], errors="coerce").fillna(0).abs().sum()

        if prior_signal > 0:
            return latest.reset_index(drop=True)

    return _build_priors_from_game_level(
        xtd_history,
        target_season=target_season,
        target_week=target_week,
    )


def attach_xtd_with_diagnostics(
    rows: pd.DataFrame,
    latest_xtd: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    out = rows.copy()

    if latest_xtd is None or latest_xtd.empty:
        out["xtd_matched"] = False
        return out, {
            "matched": 0,
            "total": len(out),
            "missing": len(out),
            "nonzero_expected_tds": 0,
        }

    keep = ["player_id"] + [c for c in PRIOR_FIELDS if c in latest_xtd.columns]
    use = latest_xtd[keep].drop_duplicates("player_id", keep="last")

    overlap = [c for c in PRIOR_FIELDS if c in out.columns]
    if overlap:
        out = out.drop(columns=overlap)

    out = out.merge(use, on="player_id", how="left")

    signal_cols = [
        c for c in [
            "expected_tds_prior",
            "actual_tds_prior",
            "expected_tds_l5",
            "actual_tds_l5",
        ]
        if c in out.columns
    ]

    if signal_cols:
        out["xtd_matched"] = out[signal_cols].notna().any(axis=1)
    else:
        out["xtd_matched"] = False

    nonzero = 0
    if "expected_tds_prior" in out.columns:
        nonzero = int(
            pd.to_numeric(out["expected_tds_prior"], errors="coerce")
            .fillna(0)
            .gt(0)
            .sum()
        )

    matched = int(out["xtd_matched"].sum())
    return out, {
        "matched": matched,
        "total": len(out),
        "missing": len(out) - matched,
        "nonzero_expected_tds": nonzero,
    }
