from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


class SigmoidProbabilityCalibrator:
    """
    Platt/sigmoid calibration for RAW base-model probabilities.
    Never fit/apply this on already-isotonic-calibrated probabilities.
    """

    def __init__(self):
        self.model = LogisticRegression(max_iter=500)
        self.fitted = False

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw_probability, y):
        self.model.fit(self._logit(raw_probability), np.asarray(y, dtype=int))
        self.fitted = True
        return self

    def predict(self, raw_probability):
        if not self.fitted:
            raise RuntimeError("Calibrator not fitted.")
        return self.model.predict_proba(self._logit(raw_probability))[:, 1]


def probability_metrics(y, p):
    y = pd.Series(y).astype(int)
    p = pd.Series(p).astype(float).clip(1e-6, 1 - 1e-6)
    return {
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else float("nan"),
    }
