import pandas as pd
from sklearn.linear_model import LogisticRegression


class ExpectedTDModel:
    """
    Starter play-level xTD model.

    Later versions should add:
    down, distance, game state, air yards, personnel,
    defenders in box, route/coverage, offense/defense quality.
    """

    FEATURES = ["yardline_100", "is_rush", "is_target"]

    def __init__(self):
        self.model = LogisticRegression(max_iter=500)
        self.fitted = False

    def fit(self, opportunities: pd.DataFrame):
        d = opportunities.copy()
        self.model.fit(
            d[self.FEATURES].fillna(0),
            d["touchdown"].astype(int)
        )
        self.fitted = True
        return self

    def predict_opportunity(self, opportunities: pd.DataFrame):
        if not self.fitted:
            raise RuntimeError("xTD model not fitted.")

        out = opportunities.copy()
        out["xTD"] = self.model.predict_proba(
            out[self.FEATURES].fillna(0)
        )[:, 1]
        return out

    def aggregate_player(self, opportunities: pd.DataFrame):
        pred = self.predict_opportunity(opportunities)

        return (
            pred.groupby("player_id", as_index=False)
            .agg(
                expected_tds=("xTD", "sum"),
                actual_tds=("touchdown", "sum")
            )
        )
