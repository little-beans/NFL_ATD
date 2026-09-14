import argparse
import sys
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from td_engine.raw_probability import resolve_raw_probability


def _read_any(path: Path):
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _find_week_source(season: int, week: int):
    candidates = [
        ROOT/"data"/"output"/f"{season}_week_{week}_td_matchup_rankings.parquet",
        ROOT/"data"/"output"/f"{season}_week_{week}_td_matchup_rankings.csv",
        ROOT/"data"/"output"/f"{season}_week_{week}_td_unified_full.csv",
        ROOT/"data"/"output"/f"{season}_week_{week}_td_cleaned.csv",
    ]
    return next((p for p in candidates if p.exists()), None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    args = p.parse_args()

    model_path = ROOT/"data"/"processed"/"td_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = joblib.load(model_path)
    print(f"Model file: {model_path}")
    print(f"Model type: {type(model)}")

    attrs = [
        "base","base_model","estimator","classifier","model","clf","pipeline",
        "features","feature_names","feature_columns","model_features","feature_names_in_"
    ]

    print("\nRelevant model attributes:")
    for a in attrs:
        if hasattr(model, a):
            v = getattr(model, a)
            if isinstance(v, (list, tuple)):
                print(f"  {a}: {len(v)} items")
                print(f"    {list(v)}")
            else:
                print(f"  {a}: {type(v)}")

    source = _find_week_source(args.season, args.week)
    if source is None:
        print("\nNo weekly source found.")
        return

    rows = _read_any(source)
    print(f"\nWeekly source: {source}")
    print(f"Weekly columns: {len(rows.columns)}")

    raw, diag = resolve_raw_probability(model, rows)

    print("\nRaw-probability resolver diagnostics:")
    for k, v in diag.items():
        if k == "features":
            print(f"  {k}: {len(v)}")
        else:
            print(f"  {k}: {v}")

    if diag["ok"]:
        print("\nRaw probability summary:")
        print(raw.describe().to_string())


if __name__ == "__main__":
    main()
