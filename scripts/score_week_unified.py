import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from td_engine.unified import finalize_weekly_table
from td_engine.xtd_integration import (
    select_latest_xtd_prior,
    attach_xtd_with_diagnostics,
)
from td_engine.raw_probability import resolve_raw_probability
from td_engine.projected_roles import (
    latest_pregame_role_state,
    attach_projected_roles,
)


def _read_any(path: Path):
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _find_week_source(season: int, week: int) -> Path:
    candidates = [
        ROOT/"data"/"output"/f"{season}_week_{week}_td_matchup_rankings.parquet",
        ROOT/"data"/"output"/f"{season}_week_{week}_td_matchup_rankings.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Run score_week_matchup.py first.")


def _ensure_player_ids(rows):
    if "player_id" in rows.columns and rows["player_id"].notna().all():
        return rows

    hist = pd.read_parquet(ROOT/"data"/"processed"/"td_pregame_features.parquet")

    latest = (
        hist.sort_values(["season","week"])
        .drop_duplicates(["player_name","posteam"], keep="last")
        [["player_name","posteam","player_id"]]
        .drop_duplicates(["player_name","posteam"])
    )

    out = rows.merge(
        latest,
        on=["player_name","posteam"],
        how="left",
        suffixes=("","_hist"),
    )

    if "player_id_hist" in out.columns:
        if "player_id" not in out.columns:
            out["player_id"] = out["player_id_hist"]
        else:
            out["player_id"] = out["player_id"].fillna(out["player_id_hist"])
        out = out.drop(columns=["player_id_hist"])

    return out


def _load_xtd_history():
    for p in [
        ROOT/"data"/"processed"/"td_pregame_features_xtd.parquet",
        ROOT/"data"/"processed"/"xtd_player_games.parquet",
    ]:
        if p.exists():
            return pd.read_parquet(p), p
    return pd.DataFrame(), None


def _load_model_feature_history():
    p = ROOT/"data"/"processed"/"td_pregame_features_xtd.parquet"
    if p.exists():
        return pd.read_parquet(p), p

    p = ROOT/"data"/"processed"/"td_pregame_features.parquet"
    if p.exists():
        return pd.read_parquet(p), p

    raise FileNotFoundError("Missing pregame feature history.")


def _apply_raw_calibration(rows):
    out = rows.copy()

    model = joblib.load(ROOT/"data"/"processed"/"td_model.joblib")
    raw, diag = resolve_raw_probability(model, out)
    out["base_td_probability_raw"] = raw

    if not diag["ok"]:
        raise RuntimeError(
            "Raw probability resolution failed after projected-role build: "
            f"{diag['reason']}; missing={diag['missing_features']}"
        )

    cal_path = ROOT/"data"/"processed"/"td_sigmoid_calibrator.joblib"

    if cal_path.exists():
        calibrator = joblib.load(cal_path)
        calibrated = pd.Series(
            calibrator.predict(raw.to_numpy()),
            index=out.index,
            dtype=float,
        ).clip(0.01, 0.85)

        out["td_probability"] = calibrated
        out["td_probability_pct"] = 100 * calibrated
        out["calibration_method"] = "sigmoid_v2_projected_roles"
    else:
        out["td_probability"] = raw.clip(0.01,0.85)
        out["td_probability_pct"] = 100*out["td_probability"]
        out["calibration_method"] = "raw_base_projected_roles"

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    args = p.parse_args()

    source = _find_week_source(args.season,args.week)
    print(f"Loading weekly matchup rows: {source}")
    rows = _ensure_player_ids(_read_any(source))

    # 1) Build leakage-safe projected role priors BEFORE injuries.
    feature_history, feature_history_path = _load_model_feature_history()
    role_state = latest_pregame_role_state(
        feature_history,
        target_season=args.season,
        target_week=args.week,
    )
    rows, role_diag = attach_projected_roles(rows, role_state)

    print(f"Projected-role source: {feature_history_path}")
    print(
        f"Projected-role matches: "
        f"{role_diag['matched_players']}/{role_diag['total_players']}"
    )

    for feature in [
        "red_zone_share",
        "volume_share",
        "goal_line_share",
        "target_share",
        "td_efficiency_score",
    ]:
        n = role_diag.get(f"{feature}_nonnull")
        mean = role_diag.get(f"{feature}_mean")
        mx = role_diag.get(f"{feature}_max")
        if n is not None:
            print(
                f"  {feature}: non-null={n}, "
                f"mean={mean:.4f}, max={mx:.4f}"
            )

    # 2) Apply current roster + injury role redistribution.
    roster = pd.read_parquet(
        ROOT/"data"/"processed"/f"roster_{args.season}_week_{args.week}.parquet"
    )

    injury_path = ROOT/"data"/"processed"/f"injuries_{args.season}_week_{args.week}.parquet"
    injuries = pd.read_parquet(injury_path) if injury_path.exists() else pd.DataFrame()

    final = finalize_weekly_table(
        rows=rows,
        roster=roster,
        injuries=injuries,
        xtd=pd.DataFrame(),
    )

    # IMPORTANT:
    # The model was trained on the canonical role columns. After injury
    # redistribution, feed adjusted role projections into those canonical columns.
    alias_pairs = {
        "red_zone_share": "adjusted_red_zone_share",
        "volume_share": "adjusted_volume_share",
        "goal_line_share": "adjusted_goal_line_share",
        "target_share": "adjusted_target_share",
    }

    for canonical, adjusted in alias_pairs.items():
        if adjusted in final.columns:
            final[canonical] = pd.to_numeric(
                final[adjusted], errors="coerce"
            ).where(
                final[adjusted].notna(),
                final[canonical]
            )

    # 3) xTD context.
    xtd_history, xtd_path = _load_xtd_history()
    latest_xtd = select_latest_xtd_prior(
        xtd_history,
        target_season=args.season,
        target_week=args.week,
    )
    final, xtd_diag = attach_xtd_with_diagnostics(final, latest_xtd)

    print(f"xTD source: {xtd_path}")
    print(f"xTD matched: {xtd_diag['matched']}/{xtd_diag['total']}")
    print(
        f"xTD non-zero expected_tds_prior: "
        f"{xtd_diag['nonzero_expected_tds']}"
    )

    # 4) Raw base probability + sigmoid calibration.
    final = _apply_raw_calibration(final)

    pvals = final["td_probability"].clip(1e-6,1-1e-6)
    final["fair_american_odds"] = np.where(
        pvals >= 0.5,
        np.round(-100*pvals/(1-pvals)),
        np.round(100*(1-pvals)/pvals),
    )

    rankable = final[
        pd.to_numeric(final["availability_multiplier"],errors="coerce")
        .fillna(1.0)
        .gt(0)
    ].copy()

    rankable = rankable.sort_values(
        ["td_probability","injury_adjusted_role_score"],
        ascending=[False,False],
    )

    out_dir = ROOT/"data"/"output"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = out_dir/f"{args.season}_week_{args.week}_td_unified_full.csv"
    rank_path = out_dir/f"{args.season}_week_{args.week}_td_unified_rankings.csv"

    final.to_csv(full_path,index=False)
    rankable.to_csv(rank_path,index=False)

    show_cols = [
        c for c in [
            "player_name","posteam","position",
            "projected_red_zone_share",
            "projected_volume_share",
            "projected_goal_line_share",
            "projected_target_share",
            "base_td_probability_raw",
            "td_probability_pct",
            "fair_american_odds",
            "expected_tds_l5",
            "td_debt_l5",
            "td_debt_tag",
            "calibration_method",
        ]
        if c in rankable.columns
    ]

    print("\nTop 30 unified rankings:")
    print(rankable[show_cols].head(30).to_string(index=False))

    print(f"\nSaved full table: {full_path}")
    print(f"Saved rankings: {rank_path}")


if __name__ == "__main__":
    main()
