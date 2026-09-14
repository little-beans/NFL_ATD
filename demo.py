import numpy as np
import pandas as pd

from td_engine.score import baseline_score, add_td_debt
from td_engine.model import TDProbabilityModel


rng = np.random.default_rng(7)

n = 5000

df = pd.DataFrame({
    "red_zone_share": rng.beta(1.7, 5.0, n),
    "volume_share": rng.beta(2.0, 4.0, n),
    "goal_line_share": rng.beta(1.3, 5.0, n),
    "target_share": rng.beta(1.5, 5.0, n),
    "prior_tds": rng.poisson(3.0, n),
    "prior_touches": rng.integers(20, 180, n),
})

df = baseline_score(df)

logit = (
    -3.2
    + 2.2 * df["red_zone_share"]
    + 1.5 * df["volume_share"]
    + 1.8 * df["goal_line_share"]
    + 0.9 * df["target_share"]
    + 0.8 * df["td_efficiency_score"]
)

p = 1 / (1 + np.exp(-logit))
df["scored_td"] = rng.binomial(1, p)

train = df.iloc[:3000]
cal = df.iloc[3000:4000]
test = df.iloc[4000:]

m = TDProbabilityModel()
m.fit_base(train)
m.fit_calibrator(cal)

pred = m.predict(test)

print("Evaluation:", m.evaluate(test))
print()

print(
    pred.sort_values("td_probability", ascending=False)
    [["baseline_score", "td_probability", "fair_american_odds"]]
    .head(10)
    .to_string(index=False)
)

debt = pd.DataFrame({
    "player": ["Player A", "Player B", "Player C"],
    "expected_tds": [4.0, 2.6, 1.2],
    "actual_tds": [1, 2, 3],
})

print()
print(add_td_debt(debt).to_string(index=False))
