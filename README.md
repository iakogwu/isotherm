# Isotherm

Research platform for Kalshi daily temperature markets. **Phase 1: read-only data logger.**

No API keys, no account, no orders. It only records:

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

## Things learned while building this (worth knowing before Phase 2)

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
pip install -r requirements.txt          # just `requests`

python -m isotherm run                   # run whichever jobs are due (what a scheduler would call)
python -m isotherm run --force           # run every job now
python -m isotherm status                # health: row counts, last runs, Kalshi-vs-NWS check
python -m isotherm peek KXHIGHNY-26OCT05 # one event: models, NBS baseline, market, raw share
python -m isotherm backfill-cli 2015 2026      # historical observed highs/lows
python -m isotherm backfill-outcomes --days 30 # recently settled Kalshi results
```

Individual jobs: `discover`, `markets`, `forecasts`, `mos`, `settle`, `cli`.

`run` is safe to call as often as you like. Each job has a minimum interval (`config.py`),
forecasts are deduplicated by model run time, and a lock file prevents overlapping runs.

## Schedule

Data only accumulates while something runs `python -m isotherm run` regularly, ideally hourly.
Gaps are fine, since the computer being off just leaves holes. No scheduler is installed yet.

## Data

`data/isotherm.db` (SQLite). Logs: `logs/isotherm.log`. Expect roughly 5–15 MB/day.

Main tables: `markets`, `market_snapshots`, `ensemble_forecasts` (members as JSON),
`deterministic_forecasts`, `mos_forecasts`, `outcomes`, `cli_reports`, `job_runs`.

## Roadmap

1. **Phase 1 (this):** collect for 8+ weeks. Don't trade.
2. **Phase 2:** station-level bias correction + calibrated spread → bracket probabilities.
   Must beat NBS on Brier/log score out-of-sample.
3. **Phase 3:** score the market as a forecaster; paper-trade at the ask, after fees, frozen params.
4. **Phase 4:** only if 2 and 3 pass. Tiny stakes, pre-written stop rules.

`peek` shows a raw ensemble share next to market prices for sanity checking only.
It is uncalibrated and **not a trading signal**.
