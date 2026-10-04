"""Model forecasts from Open-Meteo, reduced to a max/min per station per climate day.

Every member is stored, not just a mean, so Phase 2 can fit calibration on the raw
ensemble. Rows are keyed by model initialisation time, so re-fetching the same run is
a no-op and the history shows exactly what was knowable at each moment.
"""
import json
import logging
import statistics
from datetime import datetime, timedelta, timezone

from . import config
from .db import utcnow
from .http import get_json
from .timeutil import climate_day_window, climate_today, iso_utc

log = logging.getLogger(__name__)

MIN_HOURS = 20  # require near-complete coverage of the 24h climate day


def _run_inits(base_url: str, meta_names) -> dict:
    out = {}
    for name in meta_names:
        try:
            j = get_json(f"{base_url}/data/{name}/static/meta.json", pause=config.OTHER_PAUSE)
            out[name] = iso_utc(j["last_run_initialisation_time"])
        except Exception as e:
            log.warning("meta.json failed for %s: %s", name, e)
    return out


def _reduce(times, values, start, end, kind):
    vals = [v for t, v in zip(times, values) if start <= t < end and v is not None]
    if len(vals) < MIN_HOURS:
        return None, len(vals)
    return (max(vals) if kind == "high" else min(vals)), len(vals)


def _targets(conn, station):
    kinds = [r[0] for r in conn.execute(
        "SELECT DISTINCT kind FROM series WHERE station_id=?", (station["station_id"],))]
    today = climate_today(station["tz"])
    dates = [today + timedelta(days=i) for i in range(config.FORECAST_DAYS_AHEAD + 1)]
    return kinds, dates


def _parse_times(hourly):
    return [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in hourly["time"]]


def _common_params(st):
    return {
        "latitude": st["lat"], "longitude": st["lon"], "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit", "timezone": "GMT",
        "past_days": 1, "forecast_days": config.FORECAST_DAYS_AHEAD + 2,
    }


def _have(conn, table, station_id, model, run_init) -> bool:
    return conn.execute(
        f"SELECT 1 FROM {table} WHERE station_id=? AND model=? AND run_init=? LIMIT 1",
        (station_id, model, run_init),
    ).fetchone() is not None


def _ensemble(conn, st, kinds, dates, inits, fetched_at) -> int:
    wanted = {
        req: (suffix, inits[meta]) for req, (suffix, meta) in config.ENSEMBLE_MODELS.items()
        if meta in inits and not _have(conn, "ensemble_forecasts", st["station_id"], req, inits[meta])
    }
    if not wanted:
        return 0
    p = _common_params(st)
    p["models"] = ",".join(wanted)
    h = get_json(config.OPEN_METEO_ENSEMBLE, params=p, pause=config.OTHER_PAUSE)["hourly"]
    times = _parse_times(h)
    n = 0
    for req, (suffix, run_init) in wanted.items():
        if len(wanted) == 1 and f"temperature_2m_{suffix}" not in h:
            suffix_tail = ""  # single-model responses drop the model suffix
            keys = ["temperature_2m"] + sorted(
                k for k in h if k.startswith("temperature_2m_member"))
        else:
            suffix_tail = "_" + suffix
            keys = [f"temperature_2m{suffix_tail}"] + sorted(
                k for k in h if k.startswith("temperature_2m_member") and k.endswith(suffix_tail))
        series = [h[k] for k in keys if k in h]
        for d in dates:
            start, end = climate_day_window(st["tz"], d)
            for kind in kinds:
                members, cover = [], 0
                for vals in series:
                    v, c = _reduce(times, vals, start, end, kind)
                    if v is not None:
                        members.append(round(v, 1))
                        cover = max(cover, c)
                if len(members) < 5:
                    continue
                conn.execute(
                    """INSERT OR IGNORE INTO ensemble_forecasts(station_id, target_date, kind, model,
                           run_init, fetched_at, hours_covered, n_members, mean, sd, members_json)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (st["station_id"], d.isoformat(), kind, req, run_init, fetched_at, cover,
                     len(members), round(statistics.fmean(members), 2),
                     round(statistics.pstdev(members), 2), json.dumps(members)),
                )
                n += 1
    return n


def _deterministic(conn, st, kinds, dates, inits, fetched_at) -> int:
    wanted = {
        req: inits[meta] for req, meta in config.DETERMINISTIC_MODELS.items()
        if meta in inits and not _have(conn, "deterministic_forecasts", st["station_id"], req, inits[meta])
    }
    if not wanted:
        return 0
    p = _common_params(st)
    p["models"] = ",".join(wanted)
    h = get_json(config.OPEN_METEO_FORECAST, params=p, pause=config.OTHER_PAUSE)["hourly"]
    times = _parse_times(h)
    n = 0
    for req, run_init in wanted.items():
        vals = h.get(f"temperature_2m_{req}") or (h.get("temperature_2m") if len(wanted) == 1 else None)
        if not vals:
            continue
        for d in dates:
            start, end = climate_day_window(st["tz"], d)
            for kind in kinds:
                v, cover = _reduce(times, vals, start, end, kind)
                if v is None:
                    continue
                conn.execute(
                    """INSERT OR IGNORE INTO deterministic_forecasts(station_id, target_date, kind,
                           model, run_init, fetched_at, hours_covered, value)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (st["station_id"], d.isoformat(), kind, req, run_init, fetched_at, cover, round(v, 1)),
                )
                n += 1
    return n


def run(conn) -> int:
    ens_base = config.OPEN_METEO_ENSEMBLE.split("/v1/")[0]
    det_base = config.OPEN_METEO_FORECAST.split("/v1/")[0]
    ens_inits = _run_inits(ens_base, [m for _, m in config.ENSEMBLE_MODELS.values()])
    det_inits = _run_inits(det_base, config.DETERMINISTIC_MODELS.values())
    fetched_at = utcnow()
    stations = conn.execute(
        "SELECT * FROM stations WHERE station_id IN (SELECT station_id FROM series)").fetchall()
    total, failures = 0, []
    for st in stations:
        kinds, dates = _targets(conn, st)
        try:
            total += _ensemble(conn, st, kinds, dates, ens_inits, fetched_at)
            total += _deterministic(conn, st, kinds, dates, det_inits, fetched_at)
            conn.commit()
        except Exception as e:
            conn.rollback()
            failures.append(st["station_id"])
            log.warning("forecast fetch failed for %s: %s", st["station_id"], e)
    log.info("forecasts: %d rows, %d stations, failures=%s", total, len(stations), failures)
    if failures and len(failures) == len(stations):
        raise RuntimeError("all forecast fetches failed")
    return total
