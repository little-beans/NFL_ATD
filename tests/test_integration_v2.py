import pandas as pd
from td_engine.xtd_integration import select_latest_xtd_prior, attach_xtd_with_diagnostics
from td_engine.calibration_v2 import SigmoidProbabilityCalibrator


def test_xtd_prior_is_strictly_before_target_week():
    d = pd.DataFrame({
        "player_id":["p1","p1","p1"],
        "season":[2025,2026,2026],
        "week":[18,1,2],
        "expected_tds_prior":[3.0,4.0,999.0],
        "actual_tds_prior":[2.0,3.0,999.0],
    })
    out = select_latest_xtd_prior(d, 2026, 2)
    row = out.iloc[0]
    assert row["expected_tds_prior"] == 4.0
    assert row["actual_tds_prior"] == 3.0


def test_xtd_missing_stays_missing_not_zero():
    rows = pd.DataFrame({"player_id":["p1","p2"]})
    xtd = pd.DataFrame({
        "player_id":["p1"],
        "expected_tds_prior":[2.5],
    })
    out, diag = attach_xtd_with_diagnostics(rows, xtd)
    assert pd.isna(out.loc[out.player_id=="p2","expected_tds_prior"]).all()
    assert diag["matched"] == 1


def test_sigmoid_calibrator_is_smooth():
    raw = [0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80]
    y =   [0,0,0,0,1,0,1,1]
    c = SigmoidProbabilityCalibrator().fit(raw, y)
    p = c.predict([0.25,0.35,0.45,0.55])
    assert len(set(round(float(x),6) for x in p)) == 4
