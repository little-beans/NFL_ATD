import numpy as np
import pandas as pd
from .config import (
    BASELINE_WEIGHTS,
    TD_EFFICIENCY_PRIOR_TOUCHES,
    TD_EFFICIENCY_PRIOR_RATE,
)


def add_smoothed_td_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    tds = out.get("prior_tds", pd.Series(0, index=out.index)).fillna(0).astype(float)
    touches = out.get("prior_touches", pd.Series(0, index=out.index)).fillna(0).astype(float)

    prior_success = TD_EFFICIENCY_PRIOR_RATE * TD_EFFICIENCY_PRIOR_TOUCHES
    rate = (tds + prior_success) / (touches + TD_EFFICIENCY_PRIOR_TOUCHES)

    out["td_rate_smoothed"] = rate
    out["td_efficiency_score"] = 1.0 / (1.0 + np.exp(-35.0 * (rate - 0.045)))
    return out


def baseline_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "td_efficiency_score" not in out.columns:
        out = add_smoothed_td_efficiency(out)

    missing = [k for k in BASELINE_WEIGHTS if k not in out.columns]
    if missing:
        raise ValueError(f"Missing score inputs: {missing}")

    weighted = sum(
        out[k].clip(0, 1) * w
        for k, w in BASELINE_WEIGHTS.items()
    )

    out["baseline_score"] = 100 * weighted
    return out


def add_td_debt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "expected_tds" not in out.columns or "actual_tds" not in out.columns:
        raise ValueError("Need expected_tds and actual_tds for TD debt.")

    out["td_debt"] = out["expected_tds"] - out["actual_tds"]

    out["due_tag"] = np.select(
        [
            out["td_debt"] >= 1.50,
            out["td_debt"] >= 0.75,
            out["td_debt"] <= -1.50,
        ],
        ["DUE++", "DUE", "REGRESSION_RISK"],
        default="",
    )
    return out
