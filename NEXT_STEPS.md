# V2 Roadmap

## Historical pregame features
Create rolling features where Week N uses only data available before Week N.

Add:
- red-zone opportunity share
- inside-10 and inside-5 shares
- touches/game
- carry share
- target share
- route participation
- snap share
- red-zone target share
- red-zone carry share

## Better xTD
Fit a play-level expected-TD model using:
- field position
- rush vs target
- down
- distance
- air yards
- score differential
- time remaining
- shotgun
- designed rush vs scramble
- offense quality
- defense quality

## Game environment
Add:
- team implied points
- spread
- game total
- projected play volume
- pace

## Opponent adjustment
Add shrunk opponent rates:
- TD rate allowed inside 20
- TD rate allowed inside 10
- TD rate allowed inside 5
- rush TD conversion allowed
- receiving TD conversion allowed

## Injuries
Redistribute touches and targets when teammates are inactive.

## Validation
Use walk-forward chronological validation.

Track:
- Brier score
- log loss
- ROC-AUC
- calibration curve
- hit rate by probability bucket

## Market layer
Convert book odds to implied probability and remove vig.

edge = model_probability - no_vig_probability
