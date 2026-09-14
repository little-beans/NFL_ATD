# Clean PowerShell Console Output Patch

This patch keeps the detailed CSV outputs but replaces the giant PowerShell
dataframe dump with a compact decision-oriented console board.

## Adds

- `td_engine/console_view.py`
- `tests/test_console_view.py`

## Replaces

- `scripts/build_atd_value_board.py`

## Default console sections

- STRONG VALUE: edge >= 7 pp, 3+ books
- ACTIONABLE: edge 4-7 pp, 3+ books
- WATCHLIST: edge 2-4 pp, 2+ books
- REVIEW ONLY: positive edge but insufficient market/role quality

These sections are console presentation only. The full underlying CSVs are
unchanged.

## Run

No new odds call is needed:

```powershell
python scripts/build_atd_value_board.py --season 2026 --week 2
```

Show only 5 rows per section:

```powershell
python scripts/build_atd_value_board.py --season 2026 --week 2 --top 5
```

Show diagnostic details too:

```powershell
python scripts/build_atd_value_board.py --season 2026 --week 2 --verbose
```
