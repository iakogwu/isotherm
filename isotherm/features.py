"""Decision times and leak-free model inputs.

Two decision points per climate day D (all times local *standard* time at the station):

  open : D-1 11:00  (Kalshi lists D's markets around 10:00 ET on D-1)
  eve  : D   00:00  (start of the climate day)

Open-Meteo's previous-runs archive gives, for every hour t, the value forecast by a run
initialised at least N days before t ("previous_dayN"). Allowing ~6h for a run to be
computed and published, a previous_dayN value for hour t was public by t - (24N - 6)h.

  open : uses previous_day2 for every hour of D      -> public by D-1 05:00 at the latest
  eve  : previous_day1 for hours up to 18:00 of D    -> public by D 00:00
         previous_day2 for 19:00-23:00               -> public by D-1

So nothing in a backtest row was unknowable at its decision time.
"""
from datetime import datetime, timedelta, timezone

from . import config
from .http import get_json
from .timeutil import climate_day_window, standard_offset

PREV_RUNS_API = "https://previous-runs-api.open-meteo.com/v1/forecast"
HORIZONS = ("open", "eve")
# Models with a usable previous-runs archive. HRRR is dropped: no previous_day2.
HIST_MODELS = ("ncep_nbm_conus", "gfs_global", "ecmwf_ifs025")
# NBM text cycles we rely on. IEM's archive holds 01/07/13/19Z in some periods and
# 00/06/12/18Z in others; the live logger may see any hour, so lookups filter to these too.
NBS_RUN_HOURS = (0, 1, 6, 7, 12, 13, 18, 19)
NBS_LATENCY = timedelta(hours=2)      # assume a run is public 2h after its nominal time
EVE_LEAD1_LAST_HOUR = 18
MIN_HOURS = 20


def decision_time(tz: str, d, horizon: str) -> datetime:
    start, _ = climate_day_window(tz, d)
    if horizon == "eve":
        return start
    return start - timedelta(hours=13)   # D-1 11:00 local standard


def fetch_prev_runs(lat, lon, start_date, end_date) -> dict:
    """Hourly previous_day1/2 temperature (°F) for HIST_MODELS. Keys: (model, lead) -> list."""
    j = get_json(PREV_RUNS_API, params={
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m_previous_day1,temperature_2m_previous_day2",
        "models": ",".join(HIST_MODELS), "temperature_unit": "fahrenheit", "timezone": "GMT",
        "start_date": start_date.isoformat(), "end_date": end_date.isoformat(),
    }, pause=config.OTHER_PAUSE, timeout=180)
    h = j["hourly"]
    times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in h["time"]]
    series = {}
    for m in HIST_MODELS:
        for lead in (1, 2):
            series[(m, lead)] = h.get(f"temperature_2m_previous_day{lead}_{m}") or [None] * len(times)
    return {"times": times, "series": series}


def reduce_day(prev: dict, tz: str, d, kind: str, model: str, horizon: str):
    """Climate-day max/min for one model under the horizon's leak-free lead rule."""
    start, end = climate_day_window(tz, d)
    times, s1, s2 = prev["times"], prev["series"][(model, 1)], prev["series"][(model, 2)]
    vals = []
    # index arithmetic: times are contiguous hourly from times[0]
    i0 = int((start - times[0]).total_seconds() // 3600)
    for k in range(24):
        i = i0 + k
        if i < 0 or i >= len(times):
            continue
        use_lead1 = horizon == "eve" and k <= EVE_LEAD1_LAST_HOUR
        v = s1[i] if use_lead1 else s2[i]
        if v is not None:
            vals.append(v)
    if len(vals) < MIN_HOURS:
        return None, len(vals)
    return (max(vals) if kind == "high" else min(vals)), len(vals)


def nbs_ftime(d, kind: str) -> str:
    """MOS convention: daytime max valid at 00Z of D+1, overnight min at 12Z of D."""
    if kind == "high":
        return (d + timedelta(days=1)).isoformat() + "T00:00"
    return d.isoformat() + "T12:00"


def nbs_runtime_cutoff(decision: datetime) -> str:
    return (decision - NBS_LATENCY).strftime("%Y-%m-%dT%H:%M")


def std_offset_hours(tz: str, d) -> float:
    return standard_offset(tz, d).total_seconds() / 3600
