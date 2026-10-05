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


def _candle_db():
    import sqlite3
    from isotherm import db
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(db.SCHEMA)
    db._migrate(conn)
    conn.execute("INSERT INTO stations(station_id, cli_code, tz) VALUES ('KNYC', 'CLINYC', 'America/New_York')")
    conn.execute("INSERT INTO series(series_ticker, kind, station_id) VALUES ('KXHIGHNY', 'high', 'KNYC')")
    for t in ("A", "B"):
        conn.execute("""INSERT INTO markets(ticker, event_ticker, series_ticker, target_date, result)
                        VALUES (?, 'KXHIGHNY-26SEP01', 'KXHIGHNY', '2026-09-01', 'no')""", (t,))
    return conn


def test_empty_candle_fetch_is_retried_only_on_request(monkeypatch):
    from isotherm import history
    calls = []

    def fake_get_json(url, params=None, pause=None):
        if url.endswith("/historical/cutoff"):
            return {"market_settled_ts": None}
        calls.append(url.split("/markets/")[1].split("/")[0])
        return {"candlesticks": []}   # transient empty response

    monkeypatch.setattr(history, "get_json", fake_get_json)
    conn = _candle_db()
    history.backfill_candles(conn, date(2026, 1, 1))
    assert sorted(calls) == ["A", "B"]
    assert conn.execute("SELECT COUNT(*) FROM candles_done WHERE n = 0").fetchall()[0][0] == 2

    calls.clear()
    history.backfill_candles(conn, date(2026, 1, 1))             # default: empties stay skipped
    assert calls == []
    history.backfill_candles(conn, date(2026, 1, 1), retry_empty=True)
    assert sorted(calls) == ["A", "B"]


def _paper_db():
    conn = _candle_db()   # KNYC high market tickers A, B for 2026-09-01
    conn.execute("UPDATE markets SET strike_type='greater', floor_strike=80 WHERE ticker='A'")
    conn.execute("UPDATE markets SET strike_type='less', cap_strike=81 WHERE ticker='B'")
    return conn


def _snap(conn, ticker, ts, bid=0.40, ask=0.45):
    conn.execute("INSERT INTO market_snapshots(ticker, ts, yes_bid, yes_ask, yes_bid_size, yes_ask_size) "
                 "VALUES (?,?,?,?,10,20)", (ticker, ts, bid, ask))


def test_paper_quote_must_be_near_decision():
    from datetime import datetime, timezone
    from isotherm import paper
    conn = _paper_db()
    dec = datetime(2026, 9, 1, 4, 0, tzinfo=timezone.utc)
    _snap(conn, "A", "2026-09-01T03:20:00+00:00", 0.10, 0.12)   # 40 min early: too stale
    _snap(conn, "A", "2026-09-01T04:07:00+00:00")                # 7 min late: the one to use
    _snap(conn, "B", "2026-09-01T06:07:00+00:00")                # 2 h late: must not be used
    q = paper._quotes_at(conn, "KNYC", date(2026, 9, 1), dec)
    assert list(q["ticker"]) == ["A"]
    assert q.iloc[0]["quote_ts"] == "2026-09-01T04:07:00+00:00" and q.iloc[0]["yes_ask_size"] == 20


def _fake_inputs(monkeypatch, paper):
    import pandas as pd
    monkeypatch.setattr(paper.features, "fetch_prev_runs", lambda *a, **k: None)
    monkeypatch.setattr(paper.history, "store_prev_runs", lambda *a, **k: None)
    monkeypatch.setattr(paper.history, "fetch_nbs", lambda *a, **k: 0)
    monkeypatch.setattr(paper.model, "build_dataset", lambda *a, **k: None)
    monkeypatch.setattr(paper.model, "fit_predict", lambda *a, **k: pd.DataFrame(
        [{"kind": "high", "mu": 80.0, "sigma": 2.0, "nbs": 80.0, "nbs_sd": 2.0}]))


def test_paper_records_missed_decision_instead_of_late_price(monkeypatch):
    from datetime import datetime, timezone
    from isotherm import paper
    _fake_inputs(monkeypatch, paper)
    conn = _paper_db()
    st = conn.execute("SELECT * FROM stations").fetchone()
    dec = datetime(2026, 9, 1, 4, 0, tzinfo=timezone.utc)
    _snap(conn, "A", "2026-09-01T06:07:00+00:00")   # the hourly run was two hours late
    assert paper._record(conn, st, date(2026, 9, 1), "eve", dec, ["high"], None) == 0
    assert conn.execute("SELECT status FROM paper_decisions").fetchall()[0][0] == "missed_quote"
    assert conn.execute("SELECT COUNT(*) FROM paper_predictions").fetchall()[0][0] == 0

    _snap(conn, "A", "2026-09-01T04:07:00+00:00")
    _snap(conn, "B", "2026-09-01T04:07:00+00:00")
    conn.execute("DELETE FROM paper_decisions")
    assert paper._record(conn, st, date(2026, 9, 1), "eve", dec, ["high"], None) == 2
    row = conn.execute("SELECT quote_ts, yes_bid_size, yes_ask_size FROM paper_predictions LIMIT 1").fetchone()
    assert tuple(row) == ("2026-09-01T04:07:00+00:00", 10.0, 20.0)
    assert conn.execute("SELECT status, contracts FROM paper_decisions").fetchall()[0][:] == ("recorded", 2)


def test_paper_one_station_failing_does_not_stop_others(monkeypatch):
    from datetime import datetime, timezone
    from isotherm import paper
    conn = _paper_db()
    conn.execute("INSERT INTO stations(station_id, cli_code, tz) VALUES ('KMDW', 'CLIMDW', 'America/Chicago')")
    conn.execute("INSERT INTO series(series_ticker, kind, station_id) VALUES ('KXHIGHCHI', 'high', 'KMDW')")
    conn.commit()   # run() rolls back a failed station's work; setup rows must survive that
    dec = datetime(2026, 9, 1, 4, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(paper, "ENABLED", True)
    monkeypatch.setattr(paper, "_load_blend", lambda: (None, "test"))
    monkeypatch.setattr(paper, "_due", lambda conn, now: [
        (s, date(2026, 9, 1), "eve", dec) for s in conn.execute("SELECT * FROM stations ORDER BY station_id")])

    def fake_record(conn, st, d, hz, dec, kinds, blend):
        if st["station_id"] == "KMDW":
            raise ConnectionError("network down")
        for k in kinds:
            paper._mark(conn, st, d, hz, k, dec, "recorded", contracts=1)
        return 1

    monkeypatch.setattr(paper, "_record", fake_record)
    paper.run(conn)
    got = dict(conn.execute("SELECT station_id, status FROM paper_decisions").fetchall())
    assert got == {"KMDW": "error", "KNYC": "recorded"}
    paper.run(conn)   # nothing is retried or double-recorded
    assert conn.execute("SELECT COUNT(*) FROM paper_decisions").fetchall()[0][0] == 2


def test_blend_file_is_frozen_and_version_checked(monkeypatch, tmp_path):
    import json
    from isotherm import paper
    monkeypatch.setattr(evaluate, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(evaluate, "BLEND_FILE", tmp_path / "blend.json")
    coefs = {"eve": {"b0": 0.1, "b_market": 0.95, "b_model": 0.2, "n_dev": 3000}}
    assert "Froze" in evaluate._freeze_blend(coefs, {"eve": 0.04})
    first = (tmp_path / "blend.json").read_text()
    assert "NOT changed" in evaluate._freeze_blend({"eve": {**coefs["eve"], "b_model": 9.9}}, {"eve": 0.0})
    assert (tmp_path / "blend.json").read_text() == first
    assert len(list(tmp_path.glob("blend_candidate_*.json"))) == 1

    blend, why = paper._load_blend()
    assert blend is not None and why == "ok"
    monkeypatch.setattr(evaluate.model, "SIGMA_SCALE", 1.0)   # code changed after the freeze
    blend, why = paper._load_blend()
    assert blend is None and "config" in why
    assert json.loads(first)["config_hash"] != evaluate.config_hash()


def _synthetic_contracts(seed=3):
    """Events of 5 mutually exclusive brackets per day and horizon; market and model are both
    noisy views of the true probabilities, so the blend should lean on both."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range("2025-04-01", "2026-05-31"):
        for hz in ("open", "eve"):
            truth = rng.dirichlet(np.ones(5) * 2)
            win = rng.choice(5, p=truth)
            ev = f"KXHIGHNY-{d:%y%b%d}".upper() + hz
            for i in range(5):
                mid = np.clip(truth[i] + rng.normal(0, 0.05), 0.02, 0.97)
                rows.append(dict(ticker=f"{ev}-B{i}", event_ticker=ev, target_date=d.date().isoformat(),
                                 horizon=hz, station_id="KNYC", y=int(i == win),
                                 yes_bid=round(mid - 0.01, 2), yes_ask=round(mid + 0.01, 2),
                                 p_model=float(np.clip(truth[i] + rng.normal(0, 0.05), 0.01, 0.99))))
    con = pd.DataFrame(rows)
    con["split"] = evaluate.split_of(con["target_date"])
    return con


def test_blend_threshold_is_chosen_on_later_dev_dates_than_the_fit():
    con = _synthetic_contracts()
    part = evaluate.dev_parts(con)
    fit_last = con.loc[part == "fit", "target_date"].max()
    tune_first = con.loc[part == "tune", "target_date"].min()
    assert fit_last < tune_first < evaluate.HOLDOUT_START
    assert part[con["split"] == "validation"].isna().all()
    share = con.loc[part == "tune", "target_date"].nunique() / con.loc[con["split"] == "dev", "target_date"].nunique()
    assert abs(share - evaluate.BLEND_TUNE_FRAC) < 0.01


def test_event_scores_renormalise_and_need_every_bracket_quoted():
    con = _synthetic_contracts()
    con["dev_part"] = evaluate.dev_parts(con)
    con["p_mkt"] = (con["yes_bid"] + con["yes_ask"]) / 2
    first = con["event_ticker"].iloc[0]
    con.loc[con.index[0], ["yes_bid", "yes_ask"]] = np.nan          # one bracket unquoted
    ev = evaluate.event_scores(con, ["p_mkt", "p_model"])
    assert first not in set(ev["event_ticker"])
    assert len(ev) == con["event_ticker"].nunique() - 1
    # 0.92 on the winner, 0.02 on each of 4 others: one event, log score -log(0.92)
    one = con[con["event_ticker"] == con["event_ticker"].iloc[10]].assign(p_mkt=lambda d: d["y"] * 0.9 + 0.02)
    e1 = evaluate.event_scores(one, ["p_mkt"])
    assert len(e1) == 1 and math.isclose(e1["ls_p_mkt"].iloc[0], -math.log(0.92), rel_tol=1e-9)


def test_phase3_end_to_end_on_synthetic_data(monkeypatch, tmp_path):
    import json
    monkeypatch.setattr(evaluate, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(evaluate, "BLEND_FILE", tmp_path / "blend.json")
    monkeypatch.setattr(evaluate, "MIN_BLEND_FIT", 500)
    text = evaluate.phase3(_synthetic_contracts())
    assert "Per event" in text and "DEV tune threshold sweep" in text and "holdout" not in text.lower()
    blend = json.loads((tmp_path / "blend.json").read_text())
    for hz in ("open", "eve"):
        c = blend["coefs"][hz]
        assert c["b_model_ci90"][0] < c["b_model"] < c["b_model_ci90"][1]
        assert c["b_model_ci90"][0] > 0          # the synthetic model does carry information
        assert c["fit_dates"][1] < "2026-04-01"


def test_traded_crps_matches_numeric_integral(monkeypatch):
    from scipy.integrate import quad
    from scipy.stats import t as student_t
    mu, sigma, y = 70.0, 2.5, 73.1
    s, v = sigma * model.SIGMA_SCALE, model.TAIL_DF
    F = lambda x: student_t.cdf((x - mu) / s, v)
    numeric = quad(lambda x: F(x) ** 2, -np.inf, y)[0] + quad(lambda x: (1 - F(x)) ** 2, y, np.inf)[0]
    assert math.isclose(float(model.crps_traded(mu, sigma, y)), numeric, rel_tol=1e-6)
    # the 80% interval used for coverage is the traded distribution's, not a Gaussian's
    assert math.isclose(student_t.cdf(model.central_z(0.8) / model.SIGMA_SCALE, v), 0.9, rel_tol=1e-9)


def test_trades_pages_through_and_retries_failures(monkeypatch):
    from isotherm import kalshi
    conn = _candle_db()   # settled markets A, B for 2026-09-01
    conn.execute("UPDATE markets SET close_time='2026-09-02T04:00:00Z'")
    conn.execute("UPDATE markets SET close_time='2026-07-01T04:00:00Z' WHERE ticker='B'")
    paths, fail = [], {"B"}

    def fake_get_json(url, params=None, pause=None):
        if url.endswith("/historical/cutoff"):
            return {"market_settled_ts": "2026-08-05T00:00:00Z"}
        paths.append((url.split("/trade-api/v2")[1], params["ticker"], params.get("cursor")))
        if params["ticker"] in fail:
            raise ConnectionError("down")
        page = params.get("cursor") or "p1"
        trade = {"trade_id": f"{params['ticker']}-{page}", "ticker": params["ticker"],
                 "created_time": "2026-09-01T12:00:00Z", "yes_price_dollars": "0.4200",
                 "count_fp": "5.00", "taker_side": "yes", "is_block_trade": False}
        return {"trades": [trade], "cursor": "p2" if page == "p1" else ""}

    monkeypatch.setattr(kalshi, "get_json", fake_get_json)
    assert kalshi.trades(conn, since="2026-01-01") == 2           # A: two pages; B failed
    assert ("/markets/trades", "A", "p2") in paths
    assert ("/historical/trades", "B", None) in paths              # closed before the cutoff
    assert [r[0] for r in conn.execute("SELECT ticker FROM trades_done")] == ["A"]
    row = conn.execute("SELECT yes_price, count, taker_side FROM market_trades LIMIT 1").fetchone()
    assert tuple(row) == (0.42, 5.0, "yes")

    fail.clear(); paths.clear()
    assert kalshi.trades(conn, since="2026-01-01") == 2           # only B is fetched again
    assert {p[1] for p in paths} == {"B"}


def test_books_only_near_a_decision(monkeypatch):
    import json
    from datetime import datetime, timezone
    from isotherm import kalshi
    conn = _candle_db()   # KNYC markets A, B for 2026-09-01
    conn.execute("UPDATE markets SET result=NULL")
    eve = features.decision_time("America/New_York", date(2026, 9, 1), "eve")   # 05:00Z
    assert kalshi._book_targets(conn, eve.replace(minute=7)) == ["A", "B"]
    assert kalshi._book_targets(conn, eve.replace(hour=eve.hour + 3)) == []    # window over

    class FakeNow(datetime):
        @classmethod
        def now(cls, tz=None):
            return eve.replace(minute=7)

    monkeypatch.setattr(kalshi, "datetime", FakeNow)
    monkeypatch.setattr(kalshi, "get_json", lambda url, params=None, pause=None: {
        "orderbook_fp": {"yes_dollars": [["0.4000", "12.00"]], "no_dollars": [["0.5500", "3.00"]]}})
    assert kalshi.books(conn) == 2
    yes, no = conn.execute("SELECT yes_levels, no_levels FROM order_books WHERE ticker='A'").fetchone()
    assert json.loads(yes) == [["0.4000", "12.00"]] and json.loads(no) == [["0.5500", "3.00"]]


def _forward_db(monkeypatch, start, n_dates, skip_decisions=0, sharp=False):
    """Forward paper data for n_dates from `start`; the first `skip_decisions` dates have no
    recorded decision (the PC was off) though markets still settle."""
    import pandas as pd
    from isotherm import paper
    monkeypatch.setattr(paper, "FORWARD_START", start)
    monkeypatch.setattr(paper, "_load_blend", lambda: ({"config_hash": "test"}, "ok"))
    con = _synthetic_contracts()
    dates = sorted(d for d in con["target_date"].unique() if d >= start)[:n_dates]
    con = con[con["target_date"].isin(dates)]
    conn = _candle_db()
    conn.execute("DELETE FROM markets")
    for r in con.itertuples():
        conn.execute("""INSERT INTO markets(ticker, event_ticker, series_ticker, target_date, result)
                        VALUES (?,?,'KXHIGHNY',?,?)""", (r.ticker, r.event_ticker, r.target_date,
                                                         "yes" if r.y else "no"))
        mid = (r.yes_bid + r.yes_ask) / 2
        conn.execute("""INSERT INTO paper_predictions(ticker, horizon, event_ticker, decision_ts,
                            model_version, p_model, yes_bid, yes_ask, action, created_at, p_trade)
                        VALUES (?,?,?,?,'test',?,?,?,'none','now',?)""",
                     (r.ticker, r.horizon, r.event_ticker, r.target_date, r.p_model, r.yes_bid,
                      r.yes_ask, 0.4 * mid + 0.6 * r.y if sharp else (mid + r.p_model) / 2))
    for d in dates[skip_decisions:]:
        for hz in ("open", "eve"):
            conn.execute("""INSERT INTO paper_decisions(station_id, target_date, horizon, kind,
                                decision_ts, status, created_at) VALUES ('KNYC',?,?,'high',?,'recorded','now')""",
                         (d, hz, d))
    conn.commit()
    return conn, dates


def test_forward_look_is_recorded_once_at_a_point_fixed_by_the_data(monkeypatch):
    import sqlite3
    import pytest
    from isotherm import paper
    conn, dates = _forward_db(monkeypatch, "2025-06-01", 60)
    assert paper.report(conn).startswith("No forward look recorded")    # hidden before the look
    assert paper.check_looks(conn) == 1
    look = conn.execute("SELECT * FROM forward_looks").fetchone()
    assert look["look"] == "interim" and look["through_date"] == dates[44]   # the 45th date
    assert look["verdict"].startswith("CONTINUE") and look["estimate"] < 0
    first = tuple(look)
    conn.execute("UPDATE paper_predictions SET p_trade = 0.5")           # later data can't change it
    assert paper.check_looks(conn) == 0
    assert tuple(conn.execute("SELECT * FROM forward_looks").fetchone()) == first
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("UPDATE forward_looks SET verdict='PASS'")
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("DELETE FROM forward_looks")
    text = paper.report(conn)
    assert "INTERIM look through" in text and "FINAL" not in text


def test_final_look_needs_its_own_point_and_records_pass(monkeypatch):
    from isotherm import paper
    conn, dates = _forward_db(monkeypatch, "2025-06-01", 100, sharp=True)   # a clearly better blend
    assert paper.check_looks(conn) == 2
    got = dict(conn.execute("SELECT look, through_date FROM forward_looks").fetchall())
    assert got == {"interim": dates[44], "final": dates[89]}
    assert conn.execute("SELECT verdict FROM forward_looks WHERE look='final'").fetchone()[0].startswith("PASS")


def test_completeness_gates_the_look_and_is_anchored_at_the_start(monkeypatch):
    from isotherm import paper
    # the PC was off for the first 10 days: no decision rows at all for them
    conn, dates = _forward_db(monkeypatch, "2025-06-01", 60, skip_decisions=10)
    share, expected = paper.completeness(conn, dates[44])
    assert expected == 45 * 2 and abs(share - 35 / 45) < 1e-9          # counted from the start
    assert paper.check_looks(conn) == 0                                # 78% < 90%: no look yet
    status = paper.status(conn)
    assert "Settled forward dates: 60" in status and "logscore" not in status


def test_a_date_counts_only_once_every_prediction_on_it_has_settled(monkeypatch):
    from isotherm import paper
    conn, dates = _forward_db(monkeypatch, "2025-06-01", 60)
    # one West Coast-style straggler on the 31st date hasn't settled yet
    late = conn.execute("SELECT ticker FROM markets WHERE target_date=? LIMIT 1", (dates[30],)).fetchone()[0]
    conn.execute("UPDATE markets SET result=NULL WHERE ticker=?", (late,))
    conn.commit()
    assert paper._settled(conn)["target_date"].nunique() == 30         # unbroken run stops there
    assert "Settled forward dates: 30" in paper.status(conn)
    assert paper.check_looks(conn) == 0                                 # 45 not reached
    conn.execute("UPDATE markets SET result='no' WHERE ticker=?", (late,))
    conn.commit()
    assert paper.check_looks(conn) == 1
    assert conn.execute("SELECT through_date FROM forward_looks").fetchone()[0] == dates[44]


def test_void_or_scalar_results_still_settle_a_date(monkeypatch):
    from isotherm import paper
    conn, dates = _forward_db(monkeypatch, "2025-06-01", 50)
    conn.execute("UPDATE markets SET result='scalar' WHERE target_date=?", (dates[3],))
    conn.commit()
    assert paper._settled(conn)["target_date"].nunique() == 49          # date settled, just unscored
    assert paper.check_looks(conn) == 1
