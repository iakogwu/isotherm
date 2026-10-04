"""NWS statistical guidance (MOS / NBM text) via Iowa Environmental Mesonet.

This is the professional baseline. If a model of ours can't beat NBS, it can't beat the market.
"""
import logging

from . import config
from .db import utcnow
from .http import get_json

log = logging.getLogger(__name__)


def run(conn) -> int:
    fetched_at = utcnow()
    stations = [r[0] for r in conn.execute(
        "SELECT DISTINCT station_id FROM series")]
    n, failures = 0, []
    for sid in stations:
        for model in config.MOS_MODELS:
            try:
                rows = get_json(config.IEM_MOS, params={"station": sid, "model": model},
                                pause=config.OTHER_PAUSE).get("data", [])
            except Exception as e:
                failures.append(f"{sid}/{model}")
                log.warning("MOS %s %s failed: %s", sid, model, e)
                continue
            for r in rows:
                value = r.get("txn") if model == "NBS" else r.get("n_x")
                if value is None:
                    continue
                spread = r.get("xnd") if model == "NBS" else None
                conn.execute(
                    """INSERT OR IGNORE INTO mos_forecasts(station_id, model, runtime_utc, ftime_utc,
                           value, spread, fetched_at) VALUES (?,?,?,?,?,?,?)""",
                    (sid, model, r["runtime_utc"], r["ftime_utc"], float(value),
                     None if spread is None else float(spread), fetched_at),
                )
                n += 1
        conn.commit()
    log.info("mos: %d rows seen, failures=%s", n, failures)
    return n
