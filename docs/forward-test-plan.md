# Forward test plan

Written 2026-10-04, before any forward result existed. Committed to git so the date can't be
argued later. The hourly job applies this rule mechanically (`paper.check_looks`).

- **`python -m isotherm paper-status`:** collection health only (decisions recorded, missed or
  errored, completeness, settled dates). It shows no performance, so use it for routine checks.
- **`python -m isotherm paper-report`:** prints the recorded looks exactly as stored, and
  nothing before the first look.

## What is being tested

The frozen blend in `data/reports/blend.json`: config `c516ba3b45ab`, commit `ba30ac1`,
thresholds 0.08 for both horizons. It was fit on DEV data only.

**The primary question is whether the blend is more accurate than the market price on forward
data.** The validation run said yes, but by a small margin, and the validation period had already
been looked at. Profit is a separate, later question. On validation, taking the ask at the frozen
threshold lost about 1.4 cents per contract.

## Metric

- **Score:** the per-event log score of the blend minus that of the market mid. Brackets are
  renormalised to sum to 1. Only events where every bracket had a quote count.
- **Pooling:** both decision times (11 AM the day before, and midnight) are pooled.
- **Better means negative:** below zero means the blend is better.
- **Uncertainty:** a 90% confidence interval from resampling whole calendar dates, since weather
  on one day moves every city together.
- **Benchmark:** on validation this was −0.014, with an interval of about ±0.007 over 186 dates.

## Schedule and decisions

The count is of settled forward target dates, starting 2026-10-05. A date counts as settled only
when every prediction recorded for it has a final result (yes, no, or void/scalar, which are not
scored), and only as part of an unbroken run of settled dates from the start. So a look can never
freeze while some city's events are still unsettled. Each look happens at the
**first settled date** that has the required number of dates **and** at least 90% of expected
decisions recorded from 2026-10-05 through that date. The data fixes that point, not whoever
runs a command. Each look is computed once on data through that date, stored in the
`forward_looks` table, and never recomputed. Database triggers refuse any update or delete.

| Look | When (at the earliest) | Rule |
|---|---|---|
| Interim | 45 settled dates (around mid-November) | Futility only: **stop** if the blend is not ahead (estimate ≥ 0). Never declare success here. |
| Final | 90 settled dates, after the interim (around early January) | **Pass** if the 90% interval is entirely below 0, otherwise **fail**. |

There is exactly one success look. This keeps the false-pass rate at about 5%. At the validation
effect size, 90 dates gives a comfortable margin. If the true gain is half that size, the test
will often fail, and that's acceptable.

## Validity conditions

The period only counts if all of these hold. Otherwise extend it.

- **At least 90% of expected decisions recorded,** counted from 2026-10-05. Expected means
  48 series × 2 horizons per day. A PC that is off leaves no row, so this is measured against the
  expected count. Below 90%, the look waits until the share recovers; it is never taken early.
- **No change to the model, blend, thresholds, stations, eligible markets or paper decision
  logic** (`paper.py`, `evaluate.py`, `model.py`, `features.py`) during the period. Any such
  change restarts the clock at zero. A change to the config hash always counts. Changes that
  only add data collection, such as order books or trades, do not restart the clock.
- **No live orders of any kind.**

## After the final look

| Result | Next step |
|---|---|
| **Fail** | The blend's information does not hold up forward. Park the temperature model, or start a new hypothesis with a new frozen config and a fresh forward clock. |
| **Pass, paper P&L 90% interval below or spanning 0** | The information is real but not tradable by taking the ask. Study resting limit orders using the order books captured at each decision (`order_books`) plus the trade history (`market_trades`). Use a conservative queue model: the order joins the back of its price level and fills only after trades at that price exhaust the size ahead, with no credit for cancellations. Report optimistic and pessimistic fill bounds. A new strategy needs its own frozen config and forward period. |
| **Pass, paper P&L 90% interval above 0, at least 300 trades** | Eligible for Phase 4: a tiny live test (1 contract per qualifying event). This also needs the execution and risk controls from the 2026-10-04 review: exposure limits, daily loss cap, stale-data rejection, order reconciliation and a kill switch. |

## Known limits

- **Season:** autumn into winter only. A pass does not prove the edge holds in summer.
- **Quote timing:** forward quotes come from a snapshot up to 15 minutes after the decision time,
  usually about 7 minutes. The backtest used the last hourly candle before it.
- **Fills:** paper P&L assumes full fills at the quoted price, with no market impact.
