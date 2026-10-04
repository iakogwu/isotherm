"""NWS daily climate reports (CLI) via Iowa Environmental Mesonet.

Two uses: a cross-check on Kalshi's settled value (they should agree almost always),
and years of historical observed highs/lows for calibration work in Phase 2.
"""
import logging
from datetime import datetime, timezone

from . import config
from .db import utcnow
from .http import get_json

log = logging.getLogger(__name__)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None  # "M" (missing) and similar


def fetch(conn, years) -> int:
    fetched_at = utcnow()
    stations = [r[0] for r in conn.execute("SELECT DISTINCT station_id FROM series")]
    n = 0
    for sid in stations:
        for y in years:
            try:
                res = get_json(config.IEM_CLI, params={"station": sid, "year": y},
                               pause=config.OTHER_PAUSE).get("results", [])
            except Exception as e:
                log.warning("CLI %s %s failed: %s", sid, y, e)
                continue
            for r in res:
                conn.execute(
                    """INSERT INTO cli_reports(station_id, date, high, low, product, fetched_at)
                       VALUES (?,?,?,?,?,?)
                       ON CONFLICT(station_id, date) DO UPDATE SET high=excluded.high,
                         low=excluded.low, product=excluded.product, fetched_at=excluded.fetched_at""",
                    (sid, r["valid"], _num(r.get("high")), _num(r.get("low")),
                     r.get("product"), fetched_at),
                )
                n += 1
        conn.commit()
    log.info("cli: %d station-days upserted", n)
    return n


def run(conn) -> int:
    now = datetime.now(timezone.utc)
    years = [now.year - 1, now.year] if now.month == 1 else [now.year]
    return fetch(conn, years)
