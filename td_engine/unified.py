from __future__ import annotations
import pandas as pd
import numpy as np

from .current_roster import apply_current_roster
from .role_reallocation import apply_role_reallocation
from .probability_guardrails import add_probability_diagnostics


def merge_injury_context(rows: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if injuries is None or injuries.empty:
        out["report_status"] = ""
        out["practice_status"] = ""
        out["availability_multiplier"] = 1.0
        return out

    keep = [
        c for c in [
            "player_id","report_status","practice_status",
            "availability_multiplier","source_week","source_week_fallback"
        ]
        if c in injuries.columns
    ]
    use = injuries[keep].drop_duplicates("player_id", keep="last")

    out = out.drop(
        columns=[c for c in keep if c != "player_id" and c in out.columns],
        errors="ignore"
    )
    out = out.merge(use, on="player_id", how="left")

    if "report_status" not in out.columns:
        out["report_status"] = ""
    else:
        out["report_status"] = out["report_status"].fillna("")

    if "practice_status" not in out.columns:
        out["practice_status"] = ""
    else:
        out["practice_status"] = out["practice_status"].fillna("")

    if "availability_multiplier" not in out.columns:
        out["availability_multiplier"] = 1.0
    else:
        out["availability_multiplier"] = (
            pd.to_numeric(out["availability_multiplier"], errors="coerce")
            .fillna(1.0)
            .clip(0,1)
        )

    return out


def merge_xtd_context(rows: pd.DataFrame, xtd: pd.DataFrame | None) -> pd.DataFrame:
    out = rows.copy()
    if xtd is None or xtd.empty or "player_id" not in xtd.columns:
        return out

    xtd_cols = [
        c for c in [
            "player_id",
            "expected_tds_prior","actual_tds_prior","td_debt",
            "expected_tds_l5","actual_tds_l5","td_debt_l5","td_debt_tag"
        ]
        if c in xtd.columns
    ]

    if len(xtd_cols) <= 1:
        return out

    use = xtd[xtd_cols].drop_duplicates("player_id", keep="last")

    overlap = [c for c in xtd_cols if c != "player_id" and c in out.columns]
    if overlap:
        out = out.drop(columns=overlap)

    return out.merge(use, on="player_id", how="left")


def recompute_role_score(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Transparent injury-adjusted role score.
    This remains a descriptive score, not a retrained probability model.
    """
    out = rows.copy()

    mapping = {
        "adjusted_red_zone_share": 0.35,
        "adjusted_volume_share": 0.25,
        "adjusted_goal_line_share": 0.15,
        "adjusted_target_share": 0.15,
    }

    score = pd.Series(0.0, index=out.index)
    total_weight = 0.0

    for col, w in mapping.items():
        if col in out.columns:
            score += pd.to_numeric(out[col], errors="coerce").fillna(0).clip(0,1) * w
            total_weight += w

    if "td_efficiency_score" in out.columns:
        score += (
            pd.to_numeric(out["td_efficiency_score"], errors="coerce")
            .fillna(0).clip(0,1) * 0.10
        )
        total_weight += 0.10

    if total_weight > 0:
        out["injury_adjusted_role_score"] = 100 * score / total_weight
    else:
        out["injury_adjusted_role_score"] = np.nan

    return out


def finalize_weekly_table(
    rows: pd.DataFrame,
    roster: pd.DataFrame,
    injuries: pd.DataFrame | None = None,
    xtd: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = rows.copy()

    out = apply_current_roster(out, roster)
    out = merge_injury_context(out, injuries)
    out = apply_role_reallocation(out)
    out = merge_xtd_context(out, xtd)
    out = recompute_role_score(out)

    if "td_probability" not in out.columns and "td_probability_pct" in out.columns:
        out["td_probability"] = (
            pd.to_numeric(out["td_probability_pct"], errors="coerce") / 100.0
        )

    out = add_probability_diagnostics(out, "td_probability")

    if "td_probability" in out.columns:
        out["td_probability_pct"] = 100 * out["td_probability"]

    if "td_probability" in out.columns:
        p = out["td_probability"].clip(1e-6, 1-1e-6)
        out["fair_american_odds"] = np.where(
            p >= 0.5,
            np.round(-100 * p / (1-p)),
            np.round(100 * (1-p) / p),
        )

    # Useful composite tags for the front-end.
    tags = []
    for _, r in out.iterrows():
        t = []
        for c in ["td_debt_tag","injury_context_tag"]:
            v = str(r.get(c, "") or "").strip()
            if v and v.lower() != "nan":
                t.append(v)
        if bool(r.get("probability_clipped", False)):
            t.append("CALIBRATION_GUARDRAIL")
        if bool(r.get("team_changed", False)):
            t.append("TEAM_UPDATED")
        tags.append("|".join(dict.fromkeys(t)))

    out["tags"] = tags
    return out
