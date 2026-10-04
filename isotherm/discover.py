"""Find every open Kalshi daily high/low temperature series and the station it settles on.

Rules text looks like:
  "If the maximum temperature recorded at New York City (CLINYC) for Oct 5, 2026, is ..."
We read the station from the rules instead of hardcoding it, so a change on Kalshi's side
shows up in the data rather than silently breaking it.
"""
import logging
import re

from . import config
from .db import utcnow
from .http import get_json

log = logging.getLogger(__name__)

RULE_RX = re.compile(r"(maximum|minimum) temperature recorded at (.+?) \((CLI\w{3})\) for")


def _station_meta(station_id: str) -> dict:
    j = get_json(f"{config.NWS_API}/stations/{station_id}", pause=config.OTHER_PAUSE)
    lon, lat = j["geometry"]["coordinates"]
    p = j["properties"]
    return {"lat": lat, "lon": lon, "tz": p["timeZone"], "name": p.get("name")}


def run(conn) -> int:
    series = get_json(
        f"{config.KALSHI_API}/series",
        params={"category": config.KALSHI_WEATHER_CATEGORY},
        pause=config.KALSHI_PAUSE,
    ).get("series", [])
    candidates = [
        s for s in series
        if s.get("frequency") == "daily" and ("HIGH" in s["ticker"] or "LOW" in s["ticker"])
    ]
    written = 0
    now = utcnow()
    for s in candidates:
        t = s["ticker"]
        markets = get_json(
            f"{config.KALSHI_API}/markets",
            params={"series_ticker": t, "status": "open", "limit": 1},
            pause=config.KALSHI_PAUSE,
        ).get("markets", [])
        if not markets:
            continue
        m = RULE_RX.search(markets[0].get("rules_primary", ""))
        if not m:
            continue  # international or differently-worded contract; out of scope for now
        kind = "high" if m.group(1) == "maximum" else "low"
        city, cli_code = m.group(2), m.group(3)
        station_id = "K" + cli_code[3:]

        known = conn.execute(
            "SELECT 1 FROM stations WHERE station_id=? AND lat IS NOT NULL", (station_id,)
        ).fetchone()
        if not known:
            try:
                meta = _station_meta(station_id)
            except Exception as e:  # keep going; next discover retries
                log.warning("station lookup failed for %s (%s): %s", station_id, t, e)
                continue
            conn.execute(
                """INSERT INTO stations(station_id, cli_code, city, name, lat, lon, tz, updated_at)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(station_id) DO UPDATE SET cli_code=excluded.cli_code,
                     city=excluded.city, name=excluded.name, lat=excluded.lat,
                     lon=excluded.lon, tz=excluded.tz, updated_at=excluded.updated_at""",
                (station_id, cli_code, city, meta["name"], meta["lat"], meta["lon"], meta["tz"], now),
            )

        detail = get_json(f"{config.KALSHI_API}/series/{t}", pause=config.KALSHI_PAUSE).get("series", {})
        src = ", ".join(x.get("name", "") for x in detail.get("settlement_sources", []))
        conn.execute(
            """INSERT INTO series(series_ticker, kind, station_id, title, fee_type, settlement_src, last_seen)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(series_ticker) DO UPDATE SET kind=excluded.kind,
                 station_id=excluded.station_id, title=excluded.title, fee_type=excluded.fee_type,
                 settlement_src=excluded.settlement_src, last_seen=excluded.last_seen""",
            (t, kind, station_id, s.get("title"), detail.get("fee_type"), src, now),
        )
        written += 1
    conn.commit()
    log.info("discover: %d active temperature series", written)
    return written
