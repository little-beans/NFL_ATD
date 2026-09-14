from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pandas as pd
import joblib

from td_engine.model import TDProbabilityModel
from td_engine.splits import chronological_season_split


def parse_args():
    p = argparse.ArgumentParser(description="Train/calibrate/test the Anytime TD model chronologically.")
    p.add_argument("--data", default="data/processed/td_pregame_features.parquet")
    p.add_argument("--train-through", type=int, default=2024)
    p.add_argument("--calibrate", type=int, default=2025)
    p.add_argument("--test", type=int, default=2026)
    p.add_argument("--model-out", default="data/processed/td_model.joblib")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.data)
    train, cal, test = chronological_season_split(df, args.train_through, args.calibrate, args.test)

    print(f"Train: {len(train):,} rows through {args.train_through}")
    print(f"Calibration: {len(cal):,} rows from {args.calibrate}")
    print(f"Test: {len(test):,} rows from {args.test}")

    if train.empty or cal.empty:
        raise SystemExit("Train/calibration split is empty. Adjust season arguments.")

    model = TDProbabilityModel().fit_base(train).fit_calibrator(cal)
    print("Calibration metrics:", model.evaluate(cal))
    if not test.empty and test["scored_td"].nunique() > 1:
        print("Test metrics:", model.evaluate(test))

    joblib.dump(model, args.model_out)
    print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    main()
