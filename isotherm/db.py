"""SQLite storage. One file, append-mostly, safe to copy while idle."""
import sqlite3
from datetime import datetime, timezone

from .config import DATA_DIR, DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS stations (
    station_id      TEXT PRIMARY KEY,      -- ICAO, e.g. KNYC
    cli_code        TEXT NOT NULL,         -- NWS climate product id, e.g. CLINYC
    city            TEXT,
    name            TEXT,
    lat             REAL,
    lon             REAL,
    tz              TEXT,                  -- IANA zone
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS series (
    series_ticker   TEXT PRIMARY KEY,      -- e.g. KXHIGHNY
    kind            TEXT NOT NULL,         -- 'high' | 'low'
    station_id      TEXT NOT NULL REFERENCES stations(station_id),
    title           TEXT,
    fee_type        TEXT,
    settlement_src  TEXT,
    last_seen       TEXT
);

-- One row per Kalshi contract; settlement fields filled in later.
CREATE TABLE IF NOT EXISTS markets (
    ticker          TEXT PRIMARY KEY,
    event_ticker    TEXT NOT NULL,
    series_ticker   TEXT NOT NULL,
    target_date     TEXT NOT NULL,         -- climate day (station local standard time)
    strike_type     TEXT,                  -- greater | less | between
    floor_strike    REAL,
    cap_strike      REAL,
    yes_sub_title   TEXT,
    rules_primary   TEXT,
    close_time      TEXT,
    open_time       TEXT,
    result          TEXT,                  -- yes | no (after settlement)
    expiration_value REAL,                 -- official temperature used to settle
    settled_at      TEXT
);
CREATE INDEX IF NOT EXISTS ix_markets_event ON markets(event_ticker);

-- Top-of-book snapshot. Prices in dollars (0..1).
CREATE TABLE IF NOT EXISTS market_snapshots (
    ticker          TEXT NOT NULL,
    ts              TEXT NOT NULL,
    yes_bid         REAL,
    yes_ask         REAL,
    yes_bid_size    REAL,
    yes_ask_size    REAL,
    last_price      REAL,
    volume          REAL,
    volume_24h      REAL,
    open_interest   REAL,
    status          TEXT,
    PRIMARY KEY (ticker, ts)
);

-- Ensemble: all member values for one model run, one station, one climate day.
CREATE TABLE IF NOT EXISTS ensemble_forecasts (
    station_id      TEXT NOT NULL,
    target_date     TEXT NOT NULL,
    kind            TEXT NOT NULL,         -- high | low
    model           TEXT NOT NULL,
    run_init        TEXT NOT NULL,         -- model initialisation time (UTC)
    fetched_at      TEXT NOT NULL,
    hours_covered   INTEGER,               -- hourly samples inside the climate day
    n_members       INTEGER,
    mean            REAL,
    sd              REAL,
    members_json    TEXT,                  -- JSON list, control first
    PRIMARY KEY (station_id, target_date, kind, model, run_init)
);

CREATE TABLE IF NOT EXISTS deterministic_forecasts (
    station_id      TEXT NOT NULL,
    target_date     TEXT NOT NULL,
    kind            TEXT NOT NULL,
    model           TEXT NOT NULL,
    run_init        TEXT NOT NULL,
    fetched_at      TEXT NOT NULL,
    hours_covered   INTEGER,
    value           REAL,
    PRIMARY KEY (station_id, target_date, kind, model, run_init)
);

-- MOS max/min guidance exactly as issued (ftime 00Z = daytime max, 12Z = overnight min).
CREATE TABLE IF NOT EXISTS mos_forecasts (
    station_id      TEXT NOT NULL,
    model           TEXT NOT NULL,         -- NBS | GFS
    runtime_utc     TEXT NOT NULL,
    ftime_utc       TEXT NOT NULL,
    value           REAL,
    spread          REAL,                  -- NBS xnd (std dev); NULL for GFS MOS
    fetched_at      TEXT NOT NULL,
    PRIMARY KEY (station_id, model, runtime_utc, ftime_utc)
);

-- Official outcome per event, from Kalshi settlement.
CREATE TABLE IF NOT EXISTS outcomes (
    event_ticker    TEXT PRIMARY KEY,
    series_ticker   TEXT NOT NULL,
    station_id      TEXT NOT NULL,
    target_date     TEXT NOT NULL,
    kind            TEXT NOT NULL,
    value           REAL NOT NULL,
    recorded_at     TEXT NOT NULL
);

-- NWS daily climate report values (cross-check + historical backfill).
CREATE TABLE IF NOT EXISTS cli_reports (
    station_id      TEXT NOT NULL,
    date            TEXT NOT NULL,
    high            REAL,
    low             REAL,
    product         TEXT,
    fetched_at      TEXT NOT NULL,
    PRIMARY KEY (station_id, date)
);

-- Phase 2/3 history. Deterministic forecasts as they looked N days before each hour
-- (Open-Meteo previous-runs archive), reduced to a climate-day max/min per decision horizon.
CREATE TABLE IF NOT EXISTS hist_forecasts (
    station_id      TEXT NOT NULL,
    target_date     TEXT NOT NULL,
    kind            TEXT NOT NULL,
    model           TEXT NOT NULL,
    horizon         TEXT NOT NULL,         -- 'open' | 'eve'  (see features.py)
    value           REAL,
    hours_covered   INTEGER,
    fetched_at      TEXT NOT NULL,
    PRIMARY KEY (station_id, target_date, kind, model, horizon)
);

-- Hourly Kalshi candles around decision times. Prices in dollars.
CREATE TABLE IF NOT EXISTS market_candles (
    ticker          TEXT NOT NULL,
    end_ts          INTEGER NOT NULL,      -- unix seconds, end of the hour
    yes_bid         REAL,
    yes_ask         REAL,
    price           REAL,
    volume          REAL,
    open_interest   REAL,
    PRIMARY KEY (ticker, end_ts)
);
CREATE TABLE IF NOT EXISTS candles_done (
    ticker          TEXT PRIMARY KEY,
    n               INTEGER,
    fetched_at      TEXT
);

-- Forward paper trading: what the frozen model would have done, recorded at decision time.
CREATE TABLE IF NOT EXISTS paper_predictions (
    ticker          TEXT NOT NULL,
    horizon         TEXT NOT NULL,
    event_ticker    TEXT NOT NULL,
    decision_ts     TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    mu              REAL,
    sigma           REAL,
    p_model         REAL,
    p_nbs           REAL,
    yes_bid         REAL,
    yes_ask         REAL,
    action          TEXT,                  -- buy_yes | buy_no | none
    price           REAL,                  -- price paid per contract (excl. fee)
    fee             REAL,
    exp_edge        REAL,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (ticker, horizon)
);

CREATE TABLE IF NOT EXISTS job_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job             TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    ok              INTEGER,
    rows_written    INTEGER,
    detail          TEXT
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(markets)")}
    if "open_time" not in cols:
        conn.execute("ALTER TABLE markets ADD COLUMN open_time TEXT")
        conn.commit()
    pcols = {r[1] for r in conn.execute("PRAGMA table_info(paper_predictions)")}
    if "p_trade" not in pcols:
        conn.execute("ALTER TABLE paper_predictions ADD COLUMN p_trade REAL")   # v2 blend probability
        conn.commit()


def last_success(conn, job: str):
    row = conn.execute(
        "SELECT MAX(finished_at) FROM job_runs WHERE job=? AND ok=1", (job,)
    ).fetchone()
    return row[0]
