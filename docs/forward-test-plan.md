# Forward test plan

Written 2026-10-04, before any forward result existed. Committed to git so the date can't be
argued later. `python -m isotherm paper-report` applies this rule mechanically.

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

The count is of settled forward target dates, starting 2026-10-05.

| Look | When | Rule |
|---|---|---|
| Interim | 45 settled dates (around mid-November) | Futility only: **stop** if the blend is not ahead (estimate ≥ 0). Never declare success here. |
| Final | 90 settled dates (around early January) | **Pass** if the 90% interval is entirely below 0, otherwise **fail**. |

There is exactly one success look. This keeps the false-pass rate at about 5%. At the validation
effect size, 90 dates gives a comfortable margin. If the true gain is half that size, the test
will often fail, and that's acceptable.

## Validity conditions

The period only counts if all of these hold. Otherwise extend it.

- **At least 90% of expected decisions recorded.** Expected means 48 series × 2 horizons per day.
  A PC that is off leaves no row, so this is measured against the expected count, not the logged
  one.
- **No code, model, threshold, station or eligibility change** during the period. Any change
  produces a new config hash, and the clock restarts at zero.
- **No live orders of any kind.**

## After the final look

| Result | Next step |
|---|---|
| **Fail** | The blend's information does not hold up forward. Park the temperature model, or start a new hypothesis with a new frozen config and a fresh forward clock. |
| **Pass, paper P&L 90% interval below or spanning 0** | The information is real but not tradable by taking the ask. Use the logged trade history (`market_trades`) to test resting limit orders. A new strategy needs its own frozen config and forward period. |
| **Pass, paper P&L 90% interval above 0, at least 300 trades** | Eligible for Phase 4: a tiny live test (1 contract per qualifying event). This also needs the execution and risk controls from the 2026-10-04 review: exposure limits, daily loss cap, stale-data rejection, order reconciliation and a kill switch. |

## Known limits

- **Season:** autumn into winter only. A pass does not prove the edge holds in summer.
- **Quote timing:** forward quotes come from a snapshot up to 15 minutes after the decision time,
  usually about 7 minutes. The backtest used the last hourly candle before it.
- **Fills:** paper P&L assumes full fills at the quoted price, with no market impact.
