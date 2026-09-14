# NFL Anytime TD Engine — V2 Historical + Pregame Pipeline

V2 turns raw nflverse history into **leakage-safe player-game features** suitable for training an anytime-touchdown model.

## What V2 adds

- Real nflverse play-by-play downloader (`nflreadpy`)
- Local parquet caching
- Historical player-game opportunity table
- Red-zone, inside-10, and inside-5 opportunity splits
- Carry, target, touch, and team-share features
- Leakage-safe rolling pregame features (L3 / L5 / L8)
- Career and season-to-date priors
- Smoothed TD/touch efficiency based only on previous games
- Chronological train/calibration/test splits
- Scripts for dataset generation and model training
- Leakage smoke tests

## Install

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Build real historical data

From the project root:

```bash
python scripts/build_history.py --start 2021 --end 2026
```

This downloads/caches nflverse PBP and weekly player stats, then writes:

```text
data/processed/td_pregame_features.parquet
```

Force a refresh:

```bash
python scripts/build_history.py --start 2021 --end 2026 --force
```

## Train chronologically

```bash
python scripts/train_historical.py \
  --train-through 2024 \
  --calibrate 2025 \
  --test 2026
```

2026 stays untouched by training/calibration and acts as the live out-of-sample season.

## Leakage rule

For a Week N row, all rolling features are calculated as:

```python
series.shift(1).rolling(...)
```

So the Week N outcome never enters its own predictors.

Example:

```text
Week 1 touches = 10
Week 2 touches = 20
Week 3 touches = 30

Week 3 pregame L3 touches = mean(10, 20) = 15
```

not 20, which would leak Week 3.

## Core pregame model aliases

The original V1 scoring interface is preserved:

- `red_zone_share`
- `volume_share`
- `goal_line_share`
- `target_share`
- `td_efficiency_score`

They now come from prior-game rolling history rather than the current game.

V2 additionally exposes:

- `inside10_share`
- `carry_share`
- `rz_carry_share`
- `inside5_carry_share`
- `rz_target_share`
- `touches_per_game`
- `rush_att_per_game`
- `targets_per_game`

## Test

```bash
pytest -q
```

or without pytest:

```bash
python tests/smoke_test.py
```

## Next modeling layer

The next upgrade should add **this-week context** on top of historical role:

1. projected team points / game total
2. opponent red-zone and goal-line defense
3. expected touches / expected targets
4. teammate injury reallocation
5. weather
6. play-level xTD
7. book odds + no-vig edge

Those should be joined by player/game after the leakage-safe historical role table is built.
