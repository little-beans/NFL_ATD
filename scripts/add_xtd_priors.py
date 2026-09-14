import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _find_xtd_source() -> Path:
    candidates = [
        ROOT / "data" / "processed" / "xtd_player_games.parquet",
        ROOT / "data" / "processed" / "td_xtd_history.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Missing xTD player-game file. Expected one of:\n"
        f"  {candidates[0]}\n"
        f"  {candidates[1]}\n"
        "Run scripts/build_xtd_history.py first."
    )


def _collapse_one_player_game(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    additive = {
        "expected_tds",
        "actual_tds",
        "scored_td",
        "touchdowns",
        "total_tds",
        "x_td",
        "xtd",
        "game_expected_tds",
    }

    agg = {}
    for c in df.columns:
        if c in keys:
            continue
        agg[c] = "sum" if c in additive else "first"

    out = df.groupby(keys, dropna=False, as_index=False).agg(agg)

    if "scored_td" in out.columns:
        out["scored_td"] = (pd.to_numeric(out["scored_td"], errors="coerce").fillna(0) > 0).astype(int)

    return out


def main():
    base_path = ROOT / "data" / "processed" / "td_pregame_features.parquet"
    xtd_path = _find_xtd_source()

    print(f"Loading base features: {base_path}")
    base = pd.read_parquet(base_path)

    print(f"Loading xTD player-games: {xtd_path}")
    xtd = pd.read_parquet(xtd_path)

    keys = ["season", "week", "game_id", "player_id"]

    missing_base = [k for k in keys if k not in base.columns]
    missing_xtd = [k for k in keys if k not in xtd.columns]

    if missing_base:
        raise KeyError(f"Base feature file missing keys: {missing_base}")
    if missing_xtd:
        raise KeyError(f"xTD player-game file missing keys: {missing_xtd}")

    base_dupes = int(base.duplicated(keys, keep=False).sum())
    xtd_dupes = int(xtd.duplicated(keys, keep=False).sum())

    if base_dupes:
        print(f"Collapsing duplicate base player-games: {base_dupes:,} rows involved")
        base = _collapse_one_player_game(base, keys)

    if xtd_dupes:
        print(f"Collapsing duplicate xTD player-games: {xtd_dupes:,} rows involved")
        xtd = _collapse_one_player_game(xtd, keys)

    expected_col = None
    for c in ["expected_tds", "x_td", "xtd", "game_expected_tds", "expected_td"]:
        if c in xtd.columns:
            expected_col = c
            break

    actual_col = None
    for c in ["actual_tds", "touchdowns", "total_tds", "scored_td", "td"]:
        if c in xtd.columns:
            actual_col = c
            break

    if expected_col is None:
        raise KeyError(
            "Could not find a game-level xTD column in xtd_player_games.parquet."
        )

    xtd = xtd.sort_values(["player_id", "season", "week"]).copy()

    xtd["_expected_game"] = pd.to_numeric(
        xtd[expected_col], errors="coerce"
    ).fillna(0.0)

    if actual_col is not None:
        xtd["_actual_game"] = pd.to_numeric(
            xtd[actual_col], errors="coerce"
        ).fillna(0.0)
    else:
        xtd["_actual_game"] = 0.0

    # Pregame priors: cumulative values shifted one game.
    grp = xtd.groupby("player_id", sort=False)

    xtd["expected_tds_prior"] = (
        grp["_expected_game"].cumsum()
        - xtd["_expected_game"]
    )
    xtd["actual_tds_prior"] = (
        grp["_actual_game"].cumsum()
        - xtd["_actual_game"]
    )
    xtd["td_debt"] = (
        xtd["expected_tds_prior"] - xtd["actual_tds_prior"]
    )

    xtd["expected_tds_l5"] = (
        grp["_expected_game"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
        .fillna(0.0)
    )
    xtd["actual_tds_l5"] = (
        grp["_actual_game"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
        .fillna(0.0)
    )
    xtd["td_debt_l5"] = (
        xtd["expected_tds_l5"] - xtd["actual_tds_l5"]
    )

    def _tag(x):
        if x >= 1.25:
            return "DUE++"
        if x >= 0.65:
            return "DUE"
        if x <= -0.75:
            return "REGRESSION_RISK"
        return ""

    xtd["td_debt_tag"] = xtd["td_debt_l5"].map(_tag)

    prior_cols = keys + [
        "expected_tds_prior",
        "actual_tds_prior",
        "td_debt",
        "expected_tds_l5",
        "actual_tds_l5",
        "td_debt_l5",
        "td_debt_tag",
    ]

    prior = xtd[prior_cols].copy()

    existing = [c for c in prior_cols if c not in keys and c in base.columns]
    if existing:
        base = base.drop(columns=existing)

    merged = base.merge(
        prior,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    match_count = int(merged["expected_tds_prior"].notna().sum())
    nonzero_count = int(
        pd.to_numeric(merged["expected_tds_prior"], errors="coerce")
        .fillna(0)
        .gt(0)
        .sum()
    )

    out = ROOT / "data" / "processed" / "td_pregame_features_xtd.parquet"
    merged.to_parquet(out, index=False)

    print(f"xTD matched player-games: {match_count:,}/{len(merged):,}")
    print(f"Rows with non-zero expected_tds_prior: {nonzero_count:,}")
    print(f"Saved xTD-enriched feature file: {out}")


if __name__ == "__main__":
    main()
