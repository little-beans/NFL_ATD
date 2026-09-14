# V2 Architecture

```text
nflverse PBP + weekly player stats
              |
              v
       raw parquet cache
              |
              v
   player-game outcome table
              |
      +-------+--------+
      |                |
      v                v
 player role       team denominators
      |                |
      +-------+--------+
              v
   rolling pregame engine
   (shift BEFORE rolling)
              |
              v
  player-week feature table
              |
              v
 baseline opportunity score
              |
              v
 chronological model fitting
      train -> calibrate -> test
```

The processed dataset intentionally contains the current-game outcome (`scored_td`) as the target, while every `pregame_*` field is computed strictly from earlier games.
