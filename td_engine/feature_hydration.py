from __future__ import annotations
import pandas as pd


def hydrate_model_features(
    weekly_rows: pd.DataFrame,
    history: pd.DataFrame,
    required_features: list[str],
    target_season: int,
    target_week: int,
) -> tuple[pd.DataFrame, dict]:
    """
    Fill missing model feature columns in weekly rows from the most recent
    leakage-safe historical pregame row for each player strictly before
    target_season/target_week.

    Existing weekly columns always win.
    """
    out = weekly_rows.copy()

    if "player_id" not in out.columns:
        raise KeyError("weekly_rows requires player_id")
    if "player_id" not in history.columns:
        raise KeyError("history requires player_id")

    h = history.copy()

    if "season" in h.columns and "week" in h.columns:
        s = pd.to_numeric(h["season"], errors="coerce")
        w = pd.to_numeric(h["week"], errors="coerce")
        h = h[
            s.lt(target_season)
            | (s.eq(target_season) & w.lt(target_week))
        ].copy()

    if h.empty:
        return out, {
            "hydrated_features": [],
            "still_missing": list(required_features),
            "matched_players": 0,
        }

    sort_cols = [c for c in ["season","week"] if c in h.columns]
    if sort_cols:
        h = h.sort_values(["player_id"] + sort_cols)

    latest = h.drop_duplicates("player_id", keep="last")

    available = [
        c for c in required_features
        if c in latest.columns
    ]

    use = latest[["player_id"] + available].copy()
    use = use.drop_duplicates("player_id")

    before_cols = set(out.columns)
    out = out.merge(
        use,
        on="player_id",
        how="left",
        suffixes=("","__hist")
    )

    hydrated = []

    for c in available:
        hc = f"{c}__hist"

        if c not in before_cols:
            if hc in out.columns:
                out[c] = out[hc]
                hydrated.append(c)
        elif hc in out.columns:
            # Existing weekly values win; fill only missing values.
            original = out[c].copy()
            out[c] = out[c].where(out[c].notna(), out[hc])
            if out[c].notna().sum() > original.notna().sum():
                hydrated.append(c)

        if hc in out.columns:
            out = out.drop(columns=hc)

    still_missing = [c for c in required_features if c not in out.columns]

    matched_players = int(
        out["player_id"].isin(use["player_id"]).sum()
    )

    return out, {
        "hydrated_features": sorted(set(hydrated)),
        "still_missing": still_missing,
        "matched_players": matched_players,
    }
