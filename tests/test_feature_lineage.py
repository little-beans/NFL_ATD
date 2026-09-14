import pandas as pd

from td_engine.feature_lineage import (
    apply_current_week_aliases,
    build_feature_lineage,
    lineage_summary,
)


def test_adjusted_role_alias_wins():
    rows = pd.DataFrame({
        "adjusted_volume_share":[0.7],
        "volume_share":[0.2],
    })

    out, aliases = apply_current_week_aliases(
        rows,
        required_features=["volume_share"],
    )

    assert out.loc[0,"volume_share"] == 0.7
    assert aliases["volume_share"] == "adjusted_volume_share"


def test_lineage_counts_historical():
    before = pd.DataFrame({"a":[1]})
    after = pd.DataFrame({"a":[1],"b":[2]})

    lineage = build_feature_lineage(
        before,
        after,
        required_features=["a","b"],
        current_aliases={},
    )

    summary = lineage_summary(lineage)

    assert summary["current_week_features"] == 1
    assert summary["historical_features"] == 1
