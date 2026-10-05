# Isotherm

Research platform for Kalshi daily temperature markets: a read-only logger, a calibrated
temperature model, an evaluation against the NWS forecast and the market, and forward paper trading.

No API keys, no account, no orders. The logger records:

| What | Source | Why |
|---|---|---|
| Market prices (bid/ask/size/volume) for every open bracket | Kalshi public API | What the crowd believed, and when |
| Contract terms + settled results + official value | Kalshi public API | The ground truth you're scored against |
| Every ensemble member: GEFS (31), ECMWF IFS (51), ECMWF AIFS (51), ICON (40), GEM (21) | Open-Meteo | Raw material for a calibrated model |
| Deterministic NBM, HRRR, GFS, ECMWF | Open-Meteo | Extra predictors |
| NBM text (NBS, with its own std dev) and GFS MOS | Iowa Environmental Mesonet | **The professional baseline to beat** |
| NWS daily climate reports | Iowa Environmental Mesonet | Cross-check + years of history |

Markets and stations are discovered automatically from each contract's rules text, so new
cities appear on their own and a station change on Kalshi's side shows up in the data.

## Things learned while building this

- **Settlement source is The Weather Company** (weather.com/kalshi) reading the NWS station named
  in the rules (e.g. `CLINYC` = Central Park, `CLIMDW` = Chicago Midway, `CLIHOU` = Houston Hobby).
  In a 30-day check, it matched the NWS climate report on all 1,437 events where both exist.
- **Climate day = midnight to midnight local *standard* time**, all year. During DST that's
  1 AM–1 AM. All forecast max/mins here use that window (`timeutil.py`).
- **Fees are `quadratic`** (price-dependent), not a flat per-contract amount.
- **Brackets are 2°F wide**, with open-ended tails. `greater 75` = 76+, `less 68` = 67 or below,
  `between 70 71` = 70 or 71. Verified against 8,634 settled contracts.
- **GEFS/ECMWF ensembles are 3-hourly and AIFS is 6-hourly** natively, so the daily max from
  them runs cold (and min runs warm). That's expected; calibration in Phase 2 learns it.
- Individual models can be badly off for a station (e.g. GEM ran ~11°F cold vs. everything
  else for a Phoenix low). Raw ensemble vote-counting would read that as "edge."

## Usage

```powershell
cd C:\Users\iakog\project2026\isotherm
pip install -r requirements.txt          # requests, numpy, pandas, scipy
pip install -r requirements-dev.txt      # pytest; run: python -m pytest tests -q

python -m isotherm run                   # run whichever jobs are due (what a scheduler would call)
python -m isotherm run --force           # run every job now
python -m isotherm status                # health: row counts, last runs, Kalshi-vs-NWS check
python -m isotherm peek KXHIGHNY-26OCT05 # one event: models, NBS baseline, market, raw share
python -m isotherm backfill-cli 2015 2026      # historical observed highs/lows
python -m isotherm backfill-outcomes --days 30 # recently settled Kalshi results
python -m isotherm backfill-history --step all # Phase 2/3 history (resumable, hours)
python -m isotherm backfill-history --step candles --retry-empty  # re-fetch empty price histories
python -m isotherm evaluate              # Phase 2/3 report; freezes data/reports/blend.json once
python -m isotherm paper-report          # score forward paper predictions
```

Individual jobs: `discover`, `markets`, `forecasts`, `mos`, `settle`, `cli`, `paper`.

`run` is safe to call as often as you like. Each job has a minimum interval (`config.py`),
forecasts are deduplicated by model run time, and a lock file prevents overlapping runs.

## Schedule

The Windows scheduled task "Isotherm Logger" runs `python -m isotherm run` hourly at :07
(`scripts/install_task.ps1`, `scripts/uninstall_task.ps1`). It only runs while the user is logged
on and the PC is awake; gaps show up in `job_runs`, and missed paper decisions in `paper_decisions`.

## Data

`data/isotherm.db` (SQLite). Logs: `logs/isotherm.log`. Expect roughly 5–15 MB/day.
Backups: a verified copy is made daily in `data/backups/` (7 kept); copy one off the PC now and then.

Main tables: `markets`, `market_snapshots`, `ensemble_forecasts` (members as JSON),
`deterministic_forecasts`, `mos_forecasts`, `outcomes`, `cli_reports`, `job_runs`.
History for Phases 2/3: `hist_forecasts`, `market_candles`, `candles_done`.
Paper trading: `paper_predictions`, `paper_decisions`. Reports: `data/reports/`.

## Evaluation discipline

- **DEV** (target dates before 2026-04-01): the market+model blend is fit on the earlier 70% of
  dates and its trading threshold chosen on the latest 30%.
- **VALIDATION** (2026-04-01 onward): reported at DEV-chosen settings. It was inspected while the
  model was built, so it is *not* a clean holdout. Don't re-tune on it and report it again.
- **FORWARD** (`paper_predictions`): the only clean test. Paper trading uses the frozen
  `blend.json` and refuses one whose config hash doesn't match the code.
- Uncertainty is clustered by calendar date; blend accuracy is also scored per event.

## Roadmap

1. **Phase 1 (done):** read-only logger.
2. **Phase 2 (done):** station-level bias correction + calibrated Student-t spread → bracket
   probabilities. Beats NBS on lows; about even on highs. The market beats both.
3. **Phase 3 (in progress):** market+model blend, fee-aware backtest, then forward paper trading
   with frozen settings and a stopping rule written in advance.
4. **Phase 4:** only if 3 passes on forward data. Tiny stakes, pre-written stop rules.

`peek` shows a raw ensemble share next to market prices for sanity checking only.
It is uncalibrated and **not a trading signal**.
