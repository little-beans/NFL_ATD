from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


@dataclass
class ModelMetrics:
    brier: float
    logloss: float
    auc: float


class TDProbabilityModel:
    DEFAULT_FEATURES = [
        "red_zone_share",
        "volume_share",
        "goal_line_share",
        "target_share",
        "td_efficiency_score",
    ]

    def __init__(self, features=None):
        self.features = features or self.DEFAULT_FEATURES
        self.base = LogisticRegression(max_iter=500)
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self._base_fitted = False
        self._cal_fitted = False

    def fit_base(self, train: pd.DataFrame):
        X = train[self.features].fillna(0)
        y = train["scored_td"].astype(int)
        self.base.fit(X, y)
        self._base_fitted = True
        return self

    def fit_calibrator(self, calibration: pd.DataFrame):
        if not self._base_fitted:
            raise RuntimeError("fit_base() first.")

        raw = self.base.predict_proba(
            calibration[self.features].fillna(0)
        )[:, 1]

        y = calibration["scored_td"].astype(int).to_numpy()
        self.calibrator.fit(raw, y)
        self._cal_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._base_fitted:
            raise RuntimeError("Model has not been fit.")

        out = df.copy()

        raw = self.base.predict_proba(
            out[self.features].fillna(0)
        )[:, 1]

        out["td_probability_raw"] = raw
        out["td_probability"] = (
            self.calibrator.predict(raw)
            if self._cal_fitted
            else raw
        )

        out["fair_american_odds"] = out["td_probability"].apply(
            _prob_to_american
        )

        return out

    def evaluate(self, df: pd.DataFrame) -> ModelMetrics:
        pred = self.predict(df)["td_probability"].clip(1e-6, 1 - 1e-6)
        y = df["scored_td"].astype(int)

        return ModelMetrics(
            brier=brier_score_loss(y, pred),
            logloss=log_loss(y, pred),
            auc=roc_auc_score(y, pred) if y.nunique() > 1 else float("nan"),
        )


def _prob_to_american(p: float):
    if p <= 0 or p >= 1:
        return np.nan
    if p >= 0.5:
        return round(-100 * p / (1 - p))
    return round(100 * (1 - p) / p)
