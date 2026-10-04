"""One-off (resumable) history backfills for Phase 2/3.

    python -m isotherm backfill-history --since 2024-02-01

Order: Kalshi markets+results -> forecasts -> NBS -> Kalshi candles (the slow one).
Every step is idempotent; re-running continues where it stopped.
"""
import csv
import io
import logging
from datetime import date, datetime, timedelta, timezone

from . import config, features
from .db import utcnow
from .http import get_json, session
from .kalshi import _f, record_result
from .timeutil import parse_event_date

log = logging.getLogger(__name__)

IEM_MOS_BULK = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"


# ---------------------------------------------------------------- Kalshi markets/results

def _pages(path, params):
    cursor = None
    while True:
        p = dict(params, limit=1000)
        if cursor:
            p["cursor"] = cursor
        j = get_json(config.KALSHI_API + path, params=p, pause=config.KALSHI_PAUSE)
        ms = j.get("markets", [])
        yield from ms
        cursor = j.get("cursor")
        if not cursor or not ms:
            return


def backfill_markets(conn, since: date) -> int:
    n = 0
    for (t,) in conn.execute("SELECT series_ticker FROM series").fetchall():
        for path, extra in (("/historical/markets", {}), ("/markets", {"status": "settled"})):
            for m in _pages(path, dict(series_ticker=t, **extra)):
                try:
                    if parse_event_date(m["event_ticker"]) < since:
                        continue
                except ValueError:
                    continue
                record_result(conn, m, t)
                n += 1
        conn.commit()
        log.info("markets backfill: %s done (%d total)", t, n)
    rebuild_outcomes(conn)
    return n


def rebuild_outcomes(conn):
    conn.execute(
        """INSERT OR IGNORE INTO outcomes(event_ticker, series_ticker, station_id, target_date,
               kind, value, recorded_at)
           SELECT m.event_ticker, m.series_ticker, s.station_id, m.target_date, s.kind,
                  MAX(m.expiration_value), ?
           FROM markets m JOIN series s USING(series_ticker)
           WHERE m.result IS NOT NULL AND m.expiration_value IS NOT NULL
           GROUP BY m.event_ticker""", (utcnow(),))
    conn.commit()


# ---------------------------------------------------------------- forecasts (previous runs)

def store_prev_runs(conn, st, prev, dates, kinds) -> int:
    n, now = 0, utcnow()
    for d in dates:
        for kind in kinds:
            for model in features.HIST_MODELS:
                for hz in features.HORIZONS:
                    v, cov = features.reduce_day(prev, st["tz"], d, kind, model, hz)
                    if v is None:
                        continue
                    conn.execute(
                        """INSERT OR REPLACE INTO hist_forecasts(station_id, target_date, kind, model,
                               horizon, value, hours_covered, fetched_at) VALUES (?,?,?,?,?,?,?,?)""",
                        (st["station_id"], d.isoformat(), kind, model, hz, round(v, 1), cov, now))
                    n += 1
    return n


def backfill_forecasts(conn, since: date, until: date | None = None) -> int:
    until = until or datetime.now(timezone.utc).date() + timedelta(days=2)
    n = 0
    for st in conn.execute("SELECT * FROM stations").fetchall():
        kinds = [r[0] for r in conn.execute(
            "SELECT DISTINCT kind FROM series WHERE station_id=?", (st["station_id"],))]
        chunk_start = since
        while chunk_start <= until:
            chunk_end = min(chunk_start + timedelta(days=364), until)
            # one day of padding either side so every climate-day window is covered
            prev = features.fetch_prev_runs(st["lat"], st["lon"],
                                            chunk_start - timedelta(days=1), chunk_end + timedelta(days=1))
            dates = [chunk_start + timedelta(days=i) for i in range((chunk_end - chunk_start).days + 1)]
            n += store_prev_runs(conn, st, prev, dates, kinds)
            conn.commit()
            chunk_start = chunk_end + timedelta(days=1)
        log.info("forecast backfill: %s done (%d rows total)", st["station_id"], n)
    return n


# ---------------------------------------------------------------- NBS (NBM text) archive

def fetch_nbs(conn, station_id: str, start: datetime, end: datetime) -> int:
    r = session().get(IEM_MOS_BULK, params={
        "station": station_id, "model": "NBS", "format": "csv",
        "sts": start.strftime("%Y-%m-%dT%H:%MZ"), "ets": end.strftime("%Y-%m-%dT%H:%MZ"),
    }, timeout=300)
    r.raise_for_status()
    now, n = utcnow(), 0
    for row in csv.DictReader(io.StringIO(r.text)):
        if not row.get("txn"):
            continue
        rt = datetime.strptime(row["runtime"], "%Y-%m-%d %H:%M:%S")
        if rt.hour not in features.NBS_RUN_HOURS:
            continue
        ft = datetime.strptime(row["ftime"], "%Y-%m-%d %H:%M:%S")
        conn.execute(
            """INSERT OR IGNORE INTO mos_forecasts(station_id, model, runtime_utc, ftime_utc, value,
                   spread, fetched_at) VALUES (?,?,?,?,?,?,?)""",
            (station_id, "NBS", rt.strftime("%Y-%m-%dT%H:%M:%S.000"),
             ft.strftime("%Y-%m-%dT%H:%M:%S.000"), float(row["txn"]),
             float(row["xnd"]) if row.get("xnd") else None, now))
        n += 1
    return n


def backfill_nbs(conn, since: date) -> int:
    n = 0
    today = datetime.now(timezone.utc)
    for (sid,) in conn.execute("SELECT station_id FROM stations").fetchall():
        cur = datetime(since.year, since.month, 1, tzinfo=timezone.utc)
        while cur < today:
            nxt = (cur + timedelta(days=32)).replace(day=1)
            done = conn.execute(
                """SELECT COUNT(*) FROM mos_forecasts WHERE station_id=? AND model='NBS'
                   AND runtime_utc >= ? AND runtime_utc < ?""",
                (sid, cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d"))).fetchall()[0][0]
            if done < 100 or nxt > today:   # skip months already loaded
                try:
                    n += fetch_nbs(conn, sid, cur, min(nxt, today))
                    conn.commit()
                except Exception as e:
                    log.warning("NBS %s %s failed: %s", sid, cur.date(), e)
            cur = nxt
        log.info("NBS backfill: %s done (%d rows total)", sid, n)
    return n


# ---------------------------------------------------------------- Kalshi candles

def _candle_val(c, side):
    x = c.get(side) or {}
    return _f(x.get("close_dollars", x.get("close")))


def _store_candles(conn, ticker, candles) -> int:
    for c in candles:
        price = c.get("price") or {}
        conn.execute(
            """INSERT OR IGNORE INTO market_candles(ticker, end_ts, yes_bid, yes_ask, price, volume,
                   open_interest) VALUES (?,?,?,?,?,?,?)""",
            (ticker, int(c["end_period_ts"]), _candle_val(c, "yes_bid"), _candle_val(c, "yes_ask"),
             _f(price.get("close_dollars", price.get("close"))),
             _f(c.get("volume_fp", c.get("volume"))),
             _f(c.get("open_interest_fp", c.get("open_interest")))))
    conn.execute("INSERT OR REPLACE INTO candles_done(ticker, n, fetched_at) VALUES (?,?,?)",
                 (ticker, len(candles), utcnow()))
    return len(candles)


def backfill_candles(conn, since: date, limit: int | None = None, retry_empty: bool = False) -> int:
    """Fetch hourly candles for settled markets. A market whose fetch returned zero candles is
    recorded with n=0 and skipped on later runs; pass retry_empty=True to fetch those again
    (an empty response can be transient, not proof the market never traded)."""
    cutoff = get_json(config.KALSHI_API + "/historical/cutoff").get("market_settled_ts")
    rows = conn.execute(
        """SELECT m.ticker, m.series_ticker, m.target_date, m.close_time, st.tz
           FROM markets m JOIN series s USING(series_ticker) JOIN stations st USING(station_id)
           WHERE m.target_date >= ? AND m.result IS NOT NULL
             AND m.ticker NOT IN (SELECT ticker FROM candles_done WHERE n > 0 OR NOT ?)
           ORDER BY m.target_date DESC""", (since.isoformat(), retry_empty)).fetchall()
    if limit:
        rows = rows[:limit]
    log.info("candles: %d markets to fetch", len(rows))

    def fetch(r):
        d = date.fromisoformat(r["target_date"])
        start = features.decision_time(r["tz"], d, "open") - timedelta(hours=3)
        end = features.decision_time(r["tz"], d, "eve") + timedelta(hours=1)
        params = {"start_ts": int(start.timestamp()), "end_ts": int(end.timestamp()),
                  "period_interval": 60}
        historical = cutoff and r["close_time"] and r["close_time"] < cutoff
        path = (f"/historical/markets/{r['ticker']}/candlesticks" if historical
                else f"/series/{r['series_ticker']}/markets/{r['ticker']}/candlesticks")
        # Candlestick endpoints are rate-limited harder than plain reads (429s above ~4 req/s),
        # so 4 workers with a 0.6s pause. Failures aren't marked done and retry on the next run.
        return r["ticker"], get_json(config.KALSHI_API + path, params=params, pause=0.6)

    from concurrent.futures import ThreadPoolExecutor
    n = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for c0 in range(0, len(rows), 500):   # chunked so finished responses don't pile up in memory
            for fut in [pool.submit(fetch, dict(r)) for r in rows[c0:c0 + 500]]:
                try:
                    ticker, j = fut.result()
                    n += _store_candles(conn, ticker, j.get("candlesticks", []))
                    conn.commit()   # short write transactions so other jobs never wait on us
                except Exception as e:
                    conn.rollback()
                    log.warning("candles failed: %s", e)
            log.info("candles: %d/%d markets", min(c0 + 500, len(rows)), len(rows))
    return n


def run_all(conn, since: date):
    log.info("backfill: markets"); backfill_markets(conn, since)
    log.info("backfill: forecasts"); backfill_forecasts(conn, since)
    log.info("backfill: NBS"); backfill_nbs(conn, since)
    log.info("backfill: candles"); backfill_candles(conn, since)
    log.info("backfill: complete")
