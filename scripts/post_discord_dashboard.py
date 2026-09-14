from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        load_dotenv()
    except Exception:
        pass


def _num_series(df, name, default):
    if name not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return pd.to_numeric(df[name], errors="coerce").fillna(default)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    _load_env()

    webhook = os.getenv("DISCORD_ATD_WEBHOOK_URL", "").strip()
    dashboard_url = os.getenv("ATD_DASHBOARD_URL", "").strip()

    if not webhook:
        raise RuntimeError("DISCORD_ATD_WEBHOOK_URL is missing from .env")
    if not dashboard_url:
        raise RuntimeError("ATD_DASHBOARD_URL is missing from .env")

    board_path = (
        ROOT / "data" / "output" /
        f"{args.season}_week_{args.week}_atd_value_board.csv"
    )
    if not board_path.exists():
        raise FileNotFoundError(board_path)

    df = pd.read_csv(board_path)

    edge = _num_series(df, "edge_probability_points", -999)
    roi = _num_series(df, "expected_roi_pct", -999)
    books = _num_series(df, "books_available", 0)
    role = _num_series(df, "role_support_score", 0)
    dispersion = _num_series(df, "market_dispersion_pp", 0)

    base_ok = roi.gt(0) & role.ge(35) & dispersion.le(7.5)

    strong = int((base_ok & edge.ge(7) & books.ge(3)).sum())
    actionable = int((base_ok & edge.ge(4) & edge.lt(7) & books.ge(3)).sum())
    watchlist = int((base_ok & edge.ge(2) & edge.lt(4) & books.ge(2)).sum())

    payload = {
        "username": "NFL ATD Engine",
        "embeds": [
            {
                "title": f"NFL Anytime TD — {args.season} Week {args.week}",
                "description": (
                    f"**Strong Value:** {strong}\n"
                    f"**Actionable:** {actionable}\n"
                    f"**Watchlist:** {watchlist}\n\n"
                    f"[Open the live value dashboard]({dashboard_url})"
                ),
                "color": 7587839,
            }
        ],
    }

    response = requests.post(webhook, json=payload, timeout=20)
    response.raise_for_status()

    print("Discord notification sent.")
    print(f"Dashboard: {dashboard_url}")


if __name__ == "__main__":
    main()
