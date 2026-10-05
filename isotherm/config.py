"""Static configuration. Everything here is read-only public data; no API keys."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
DB_PATH = DATA_DIR / "isotherm.db"

USER_AGENT = "isotherm-research-logger/0.1 (personal research; read-only)"

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_WEATHER_CATEGORY = "Climate and Weather"

OPEN_METEO_ENSEMBLE = "https://ensemble-api.open-meteo.com/v1/ensemble"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
NWS_API = "https://api.weather.gov"
IEM_MOS = "https://mesonet.agron.iastate.edu/api/1/mos.json"
IEM_CLI = "https://mesonet.agron.iastate.edu/json/cli.py"

# Ensemble models: request name -> (response key suffix, metadata name).
# Note native output resolution: GEFS/ECMWF are 3-hourly and AIFS is 6-hourly, so a
# max over interpolated hourly values will run cold for highs and warm for lows.
# That bias is expected and is for Phase 2 calibration to learn, not for us to patch here.
ENSEMBLE_MODELS = {
    "gfs025": ("ncep_gefs025", "ncep_gefs025"),
    "ecmwf_ifs025": ("ecmwf_ifs025_ensemble", "ecmwf_ifs025_ensemble"),
    "ecmwf_aifs025": ("ecmwf_aifs025_ensemble", "ecmwf_aifs025_ensemble"),
    "icon_seamless": ("icon_seamless_eps", "dwd_icon_eps"),
    "gem_global": ("gem_global_ensemble", "cmc_gem_geps"),
}

# Deterministic models: request name -> metadata name (response key suffix == request name).
DETERMINISTIC_MODELS = {
    "ncep_nbm_conus": "ncep_nbm_conus",
    "ncep_hrrr_conus": "ncep_hrrr_conus",
    "gfs_global": "ncep_gfs013",
    "ecmwf_ifs025": "ecmwf_ifs025",
}

# MOS text products archived by Iowa Environmental Mesonet.
# NBS = National Blend of Models station guidance (txn = max/min, xnd = its std dev).
# GFS = GFS MOS (n_x = max/min).
MOS_MODELS = ("NBS", "GFS")

# How often each job is allowed to run when `isotherm run` fires (hourly via Task Scheduler).
INTERVALS_MIN = {
    "discover": 24 * 60,
    "markets": 50,      # every scheduled run
    "forecasts": 50,    # still deduped by model run time, so this is cheap
    "mos": 170,
    "settle": 170,
    "cli": 12 * 60,
    "paper": 50,
    "trades": 6 * 60,
    "books": 50,       # every scheduled run; only markets near a decision are fetched
}

# Polite pacing between HTTP calls (seconds).
KALSHI_PAUSE = 0.15
OTHER_PAUSE = 0.25

FORECAST_DAYS_AHEAD = 2   # log target dates today .. today+2 (station local standard time)
