"""Invariants that, if broken, silently poison every result.  Run: python -m pytest tests -q"""
import math
from datetime import date

import numpy as np

from isotherm import evaluate, features, model
from isotherm.timeutil import climate_day_window, parse_event_date


def test_climate_day_is_local_standard_time_all_year():
    # New York: EST = UTC-5. Climate day starts 05:00Z in winter AND summer.
    for d in (date(2026, 1, 15), date(2026, 7, 15)):
        start, end = climate_day_window("America/New_York", d)
        assert (start.hour, (end - start).total_seconds()) == (5, 86400)
    # Phoenix has no DST: MST = UTC-7
    assert climate_day_window("America/Phoenix", date(2026, 7, 15))[0].hour == 7


def test_decision_times():
    d = date(2026, 10, 5)
    eve = features.decision_time("America/New_York", d, "eve")
    opn = features.decision_time("America/New_York", d, "open")
    assert eve.isoformat() == "2026-10-05T05:00:00+00:00"
    assert opn.isoformat() == "2026-10-04T16:00:00+00:00"   # 11:00 EST on D-1


def test_event_date_parsing():
    assert parse_event_date("KXHIGHNY-26OCT05") == date(2026, 10, 5)
    assert parse_event_date("HIGHNY-21AUG06") == date(2021, 8, 6)


def test_bracket_bounds_match_kalshi_wording():
    assert model.bracket_bounds("greater", 75, None) == (76, math.inf)    # "76 or above"
    assert model.bracket_bounds("less", None, 68) == (-math.inf, 67)      # "67 or below"
    assert model.bracket_bounds("between", 70, 71) == (70, 71)


def test_partition_probabilities_sum_to_one():
    bounds = [(-math.inf, 67), (68, 69), (70, 71), (72, 73), (74, 75), (76, math.inf)]
    for mu, sd in ((69.3, 1.8), (60.0, 3.0), (80.0, 0.7)):
        tot = sum(float(model.prob_in(mu, sd, lo, hi)) for lo, hi in bounds)
        assert abs(tot - 1) < 1e-9


def test_taker_fee():
    # ceil(0.07 * 10 * 0.5 * 0.5 * 100) / 100 = 0.18 per 10 contracts -> 0.018 each
    assert abs(float(evaluate.taker_fee(0.5)) - 0.018) < 1e-12
    assert float(evaluate.taker_fee(0.99)) > 0          # never free


def test_trade_rule_respects_fees_and_threshold():
    import pandas as pd
    df = pd.DataFrame({"yes_bid": [0.40], "yes_ask": [0.42], "p_model": [0.45], "y": [1],
                       "station_id": ["K"], "target_date": ["2026-01-01"]})
    assert evaluate.trades(df, 0.05).empty                 # 3c raw edge minus fee < 5c
    t = evaluate.trades(df.assign(p_model=0.60), 0.05)
    assert len(t) == 1 and t.iloc[0]["side"] == "yes"
    assert np.isclose(t.iloc[0]["pnl"], 1 - 0.42 - float(evaluate.taker_fee(0.42)))


def test_blend_recovers_known_weights():
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 20000
    lm, lq = rng.normal(0, 1.5, n), rng.normal(0, 1.5, n)
    p_true = 1 / (1 + np.exp(-(0.1 + 0.9 * lm + 0.3 * lq)))
    df = pd.DataFrame({"p_mkt": 1 / (1 + np.exp(-lm)), "p_model": 1 / (1 + np.exp(-lq)),
                       "y": (rng.random(n) < p_true).astype(int)})
    b = evaluate.fit_blend(df)
    assert abs(b[1] - 0.9) < 0.1 and abs(b[2] - 0.3) < 0.1
