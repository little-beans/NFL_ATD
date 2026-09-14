import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _collapse_base(base: pd.DataFrame, join_cols):
    """
    Collapse duplicate player-game rows in the base feature table.

    These duplicates can occur when a player appears under multiple
    team/role records within the same game. For pregame/model features,
    keep a single representative row per player-game rather than
    duplicating the game during the merge.
    """
    if not base.duplicated(join_cols, keep=False).any():
        return base

    print(
        f"Base duplicates found: "
        f"{base.duplicated(join_cols, keep=False).sum():,} rows; collapsing..."
    )

    # Prefer first non-null / first row for metadata and pregame features.
    # Sum only realized counting stats if present.
    sum_cols = {
        "td", "scored_td", "rush_att", "targets", "receptions",
        "touches", "rz_opp", "gl_opp", "inside10_opp",
        "inside5_opp", "rz_carries", "inside10_carries",
        "inside5_carries", "rz_targets", "inside10_targets",
        "inside5_targets",
    }

    agg = {}
    for c in base.columns:
        if c in join_cols:
            continue
        if c in sum_cols:
            # scored_td should remain binary after summation.
            agg[c] = "sum"
        else:
            agg[c] = "first"

    out = base.groupby(join_cols, as_index=False, dropna=False).agg(agg)

    if "scored_td" in out.columns:
        out["scored_td"] = (out["scored_td"] > 0).astype(int)

    return out


def _collapse_xtd(xtd: pd.DataFrame, join_cols):
    """
    Collapse xTD rows to exactly one player-game row.

    xTD and realized TD counts are additive across a player's opportunities
    within the same game. Prior/debt fields should not be summed repeatedly;
    use the first available value because they describe the state entering
    that game.
    """
    if not xtd.duplicated(join_cols, keep=False).any():
        return xtd

    print(
        f"xTD duplicates found: "
        f"{xtd.duplicated(join_cols, keep=False).sum():,} rows; collapsing..."
    )

    additive = {
        "expected_tds_game",
        "actual_tds_game",
        "xTD",
        "touchdown",
    }

    agg = {}
    for c in xtd.columns:
        if c in join_cols:
            continue
        if c in additive:
            agg[c] = "sum"
        else:
            agg[c] = "first"

    return xtd.groupby(join_cols, as_index=False, dropna=False).agg(agg)


def main():
    base_path = ROOT / "data" / "processed" / "td_pregame_features.parquet"
    xtd_path = ROOT / "data" / "processed" / "td_xtd_history.parquet"
    out_path = ROOT / "data" / "processed" / "td_pregame_features_xtd.parquet"

    if not base_path.exists():
        raise FileNotFoundError(f"Missing base feature file: {base_path}")
    if not xtd_path.exists():
        raise FileNotFoundError(
            f"Missing xTD history file: {xtd_path}\n"
            "Run scripts/build_xtd_history.py first."
        )

    print(f"Loading base features: {base_path}")
    base = pd.read_parquet(base_path)
    print(f"Loaded {len(base):,} base rows")

    print(f"Loading xTD history: {xtd_path}")
    xtd = pd.read_parquet(xtd_path)
    print(f"Loaded {len(xtd):,} xTD rows")

    join_cols = ["season", "week", "game_id", "player_id"]

    missing_base = [c for c in join_cols if c not in base.columns]
    missing_xtd = [c for c in join_cols if c not in xtd.columns]
    if missing_base:
        raise KeyError(f"Base file missing join columns: {missing_base}")
    if missing_xtd:
        raise KeyError(f"xTD file missing join columns: {missing_xtd}")

    base = _collapse_base(base, join_cols)
    xtd = _collapse_xtd(xtd, join_cols)

    print(f"After collapse: {len(base):,} base rows, {len(xtd):,} xTD rows")

    xtd_feature_cols = [
        c for c in [
            "expected_tds_prior",
            "actual_tds_prior",
            "td_debt",
            "expected_tds_l5",
            "actual_tds_l5",
            "td_debt_l5",
            "td_debt_tag",
            "expected_tds_game",
            "actual_tds_game",
        ]
        if c in xtd.columns
    ]

    keep = join_cols + xtd_feature_cols

    # Remove stale xTD columns from base if this script is rerun.
    overlap = [c for c in xtd_feature_cols if c in base.columns]
    if overlap:
        print(f"Dropping existing xTD columns before merge: {overlap}")
        base = base.drop(columns=overlap)

    out = base.merge(
        xtd[keep],
        on=join_cols,
        how="left",
        validate="one_to_one",
    )

    matched = out[xtd_feature_cols].notna().any(axis=1).sum() if xtd_feature_cols else 0
    print(f"Matched xTD features onto {matched:,} / {len(out):,} player-games")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(f"Saved {len(out):,} rows to {out_path}")


if __name__ == "__main__":
    main()
