import pandas as pd
from sklearn.linear_model import LogisticRegression

from td_engine.raw_probability import resolve_raw_probability


class Wrapper:
    def __init__(self, base, features):
        self.base = base
        self.features = features


def test_resolve_raw_probability_from_wrapper():
    X = pd.DataFrame({
        "a":[0,1,0,1,0,1],
        "b":[0,0,1,1,0,1],
    })
    y = [0,0,0,1,0,1]

    base = LogisticRegression().fit(X, y)
    model = Wrapper(base, ["a","b"])

    p, diag = resolve_raw_probability(model, X)

    assert diag["ok"]
    assert p.notna().all()


def test_missing_feature_is_reported():
    X = pd.DataFrame({"a":[0,1]})

    train = pd.DataFrame({"a":[0,1,0,1], "b":[0,0,1,1]})
    y = [0,0,0,1]

    base = LogisticRegression().fit(train, y)
    model = Wrapper(base, ["a","b"])

    _, diag = resolve_raw_probability(model, X)

    assert not diag["ok"]
    assert "b" in diag["missing_features"]
