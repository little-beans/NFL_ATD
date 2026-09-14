import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from td_engine.feature_lineage import (
    apply_current_week_aliases,
    build_feature_lineage,
    lineage_summary,
)
from td_engine.feature_hydration import hydrate_model_features


def _read_any(path: Path):
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _get_required_features(model):
    for attr in ["features","feature_names","feature_columns","model_features"]:
        if hasattr(model, attr):
            value = getattr(model, attr)
            if value is not None:
                vals = list(value)
                if vals:
                    return vals

    for attr in ["base","base_model","estimator","classifier","model","clf","pipeline"]:
        if hasattr(model, attr):
            obj = getattr(model, attr)
            if obj is not None and hasattr(obj, "feature_names_in_"):
                vals = list(obj.feature_names_in_)
                if vals:
                    return vals

    return []


def _find_week_source(season, week):
    candidates = [
        ROOT/"data"/"output"/f"{season}_week_{week}_td_matchup_rankings.parquet",
        ROOT/"data"/"output"/f"{season}_week_{week}_td_matchup_rankings.csv",
        ROOT/"data"/"output"/f"{season}_week_{week}_td_unified_full.csv",
    ]
    return next((p for p in candidates if p.exists()), None)


def _ensure_player_ids(rows: pd.DataFrame) -> pd.DataFrame:
    if "player_id" in rows.columns and rows["player_id"].notna().all():
        return rows

    hist_path = ROOT/"data"/"processed"/"td_pregame_features.parquet"
    if not hist_path.exists():
        raise FileNotFoundError(
            "player_id missing from weekly source and historical feature file "
            "is unavailable for reconstruction."
        )

    hist = pd.read_parquet(hist_path)

    if not {"player_name","posteam","player_id"}.issubset(hist.columns):
        raise KeyError(
            "Historical feature file must contain player_name, posteam, player_id "
            "to reconstruct weekly IDs."
        )

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

    matched = int(out["player_id"].notna().sum()) if "player_id" in out.columns else 0
    print(f"Reconstructed player_id for {matched}/{len(out)} weekly rows")

    if matched == 0:
        raise KeyError(
            "Could not reconstruct any player_id values from player_name + posteam."
        )

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    args = p.parse_args()

    model = joblib.load(ROOT/"data"/"processed"/"td_model.joblib")
    features = _get_required_features(model)

    print(f"Model features: {len(features)}")
    for f in features:
        print(f"  - {f}")

    source = _find_week_source(args.season, args.week)
    if source is None:
        raise FileNotFoundError("No weekly ranking/unified source found.")

    rows = _ensure_player_ids(_read_any(source))

    hist_path = ROOT/"data"/"processed"/"td_pregame_features_xtd.parquet"
    if not hist_path.exists():
        hist_path = ROOT/"data"/"processed"/"td_pregame_features.parquet"

    hist = pd.read_parquet(hist_path)

    aliased, aliases = apply_current_week_aliases(rows, features)
    before = aliased.copy()

    hydrated, diag = hydrate_model_features(
        aliased,
        hist,
        required_features=features,
        target_season=args.season,
        target_week=args.week,
    )

    lineage = build_feature_lineage(
        before,
        hydrated,
        required_features=features,
        current_aliases=aliases,
    )

    summary = lineage_summary(lineage)

    print("\nFeature lineage:")
    print(lineage.to_string(index=False))

    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out = ROOT/"data"/"output"/f"{args.season}_week_{args.week}_feature_lineage.csv"
    lineage.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
