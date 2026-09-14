import argparse
import sys
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from td_engine.calibration_v2 import SigmoidProbabilityCalibrator, probability_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/td_pregame_features.parquet")
    p.add_argument("--model", default="data/processed/td_model.joblib")
    p.add_argument("--calibrate", type=int, default=2025)
    p.add_argument("--test", type=int, default=2026)
    args = p.parse_args()

    df = pd.read_parquet(ROOT / args.features)
    model = joblib.load(ROOT / args.model)

    cal = df[df["season"].eq(args.calibrate)].copy()
    test = df[df["season"].eq(args.test)].copy()

    if cal.empty or test.empty:
        raise ValueError("Calibration or test split is empty.")

    # Use the base estimator's raw probability if available.
    if not hasattr(model, "base"):
        raise AttributeError(
            "Expected saved TD model to expose .base LogisticRegression."
        )

    feature_names = getattr(model, "features", None)
    if not feature_names:
        raise AttributeError("Saved TD model does not expose model.features.")

    raw_cal = model.base.predict_proba(cal[feature_names].fillna(0))[:, 1]
    raw_test = model.base.predict_proba(test[feature_names].fillna(0))[:, 1]

    calibrator = SigmoidProbabilityCalibrator()
    calibrator.fit(raw_cal, cal["scored_td"].astype(int))

    p_cal = calibrator.predict(raw_cal)
    p_test = calibrator.predict(raw_test)

    print("Calibration V2 metrics:", probability_metrics(cal["scored_td"], p_cal))
    print("Test V2 metrics:", probability_metrics(test["scored_td"], p_test))

    out = ROOT / "data" / "processed" / "td_sigmoid_calibrator.joblib"
    joblib.dump(calibrator, out)
    print(f"Saved calibrator to {out}")


if __name__ == "__main__":
    main()
