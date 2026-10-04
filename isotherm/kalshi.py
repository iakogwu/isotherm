"""Kalshi public market data: price snapshots and settlement results. Read-only, no key."""
import logging
from datetime import datetime, timedelta, timezone

from . import config
from .db import utcnow
from .http import get_json
from .timeutil import parse_event_date

log = logging.getLogger(__name__)


def _f(x):
    try:
        return None if x in (None, "") else float(x)
    except (TypeError, ValueError):
        return None


def _markets(params):
    cursor = None
    while True:
        p = dict(params, limit=200)
        if cursor:
            p["cursor"] = cursor
        j = get_json(f"{config.KALSHI_API}/markets", params=p, pause=config.KALSHI_PAUSE)
        yield from j.get("markets", [])
        cursor = j.get("cursor")
        if not cursor:
            return


def _upsert_market(conn, m, series_ticker):
    conn.execute(
        """INSERT INTO markets(ticker, event_ticker, series_ticker, target_date, strike_type,
                               floor_strike, cap_strike, yes_sub_title, rules_primary, close_time,
                               open_time)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(ticker) DO UPDATE SET close_time=excluded.close_time,
             rules_primary=excluded.rules_primary, yes_sub_title=excluded.yes_sub_title,
             open_time=excluded.open_time""",
        (
            m["ticker"], m["event_ticker"], series_ticker,
            parse_event_date(m["event_ticker"]).isoformat(),
            m.get("strike_type"), _f(m.get("floor_strike")), _f(m.get("cap_strike")),
            m.get("yes_sub_title"), m.get("rules_primary"), m.get("close_time"),
            m.get("open_time"),
        ),
    )


def record_result(conn, m, series_ticker):
    """Upsert a settled market including its result. Returns the official value or None."""
    _upsert_market(conn, m, series_ticker)
    val = _f(m.get("expiration_value"))
    if m.get("result"):
        conn.execute(
            "UPDATE markets SET result=?, expiration_value=?, settled_at=? WHERE ticker=?",
            (m["result"], val, m.get("settlement_ts") or utcnow(), m["ticker"]),
        )
    return val


def snapshot(conn) -> int:
    ts = utcnow()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    series = [r[0] for r in conn.execute(
        "SELECT series_ticker FROM series WHERE last_seen >= ?", (cutoff,))]
    n = 0
    for t in series:
        for m in _markets({"series_ticker": t, "status": "open"}):
            _upsert_market(conn, m, t)
            conn.execute(
                """INSERT OR IGNORE INTO market_snapshots(ticker, ts, yes_bid, yes_ask, yes_bid_size,
                       yes_ask_size, last_price, volume, volume_24h, open_interest, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    m["ticker"], ts,
                    _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars")),
                    _f(m.get("yes_bid_size_fp")), _f(m.get("yes_ask_size_fp")),
                    _f(m.get("last_price_dollars")), _f(m.get("volume_fp")),
                    _f(m.get("volume_24h_fp")), _f(m.get("open_interest_fp")),
                    m.get("status"),
                ),
            )
            n += 1
        conn.commit()
    log.info("markets: %d snapshots across %d series", n, len(series))
    return n


def backfill_outcomes(conn, days: int) -> int:
    """Pull already-settled markets from the last `days` days (contract terms + results only;
    historical prices are not captured here)."""
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    series = [r[0] for r in conn.execute("SELECT series_ticker FROM series")]
    for t in series:
        for m in _markets({"series_ticker": t, "status": "settled", "min_close_ts": since}):
            _upsert_market(conn, m, t)
        conn.commit()
    return settle(conn)


def settle(conn) -> int:
    """Fill in results for events whose climate day is over."""
    today = datetime.now(timezone.utc).date().isoformat()
    events = conn.execute(
        """SELECT DISTINCT m.event_ticker, m.series_ticker, m.target_date, s.station_id, s.kind
           FROM markets m JOIN series s ON s.series_ticker = m.series_ticker
           WHERE m.result IS NULL AND m.target_date < ?""",
        (today,),
    ).fetchall()
    n = 0
    now = utcnow()
    for ev in events:
        value = None
        for m in _markets({"event_ticker": ev["event_ticker"]}):
            if not m.get("result"):
                continue
            ev_val = _f(m.get("expiration_value"))
            value = ev_val if ev_val is not None else value
            _upsert_market(conn, m, ev["series_ticker"])
            conn.execute(
                "UPDATE markets SET result=?, expiration_value=?, settled_at=? WHERE ticker=?",
                (m["result"], ev_val, m.get("settlement_ts") or now, m["ticker"]),
            )
        if value is not None:
            conn.execute(
                """INSERT OR REPLACE INTO outcomes(event_ticker, series_ticker, station_id,
                       target_date, kind, value, recorded_at) VALUES (?,?,?,?,?,?,?)""",
                (ev["event_ticker"], ev["series_ticker"], ev["station_id"],
                 ev["target_date"], ev["kind"], value, now),
            )
            n += 1
        conn.commit()
    log.info("settle: %d events settled (%d still pending)", n, len(events) - n)
    return n
