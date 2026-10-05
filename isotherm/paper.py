"""Forward paper trading: the only test the model can't have been tuned on.

Runs inside the hourly job. When a decision time (features.py) has just passed for a
station/day, it rebuilds the inputs exactly as the backtest does, fits the frozen model,
prices every bracket, and records what it *would* trade at the live quote. Nothing is sent
to Kalshi. `python -m isotherm paper-report` scores it once markets settle.
"""
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import evaluate, features, history, model
from .db import utcnow
from .timeutil import climate_today

log = logging.getLogger(__name__)

# Frozen settings come from data/reports/blend.json (written once by `evaluate`, fit on DEV only).
WINDOW_AFTER_DECISION = timedelta(hours=3)
# A quote counts as "the price at decision time" only if its snapshot is this close to the
# decision. A late or missed hourly run records a missed decision instead of trading at a
# later price under the original timestamp.
QUOTE_MAX_LAG = timedelta(minutes=15)
ENABLED = True    # on since 2026-10-04, with blend.json frozen from the validation run


def _load_blend():
    """The frozen blend, or (None, reason) if it is missing or was fit under different code."""
    import json
    try:
        blend = json.loads(evaluate.BLEND_FILE.read_text())
    except (OSError, ValueError) as e:
        return None, f"no blend.json ({e.__class__.__name__})"
    if blend.get("model_version") != model.VERSION:
        return None, f"blend.json is for {blend.get('model_version')}, code is {model.VERSION}"
    if blend.get("config_hash") != evaluate.config_hash():
        return None, f"blend.json config {blend.get('config_hash')} != code config {evaluate.config_hash()}"
    return blend, "ok"


def _due(conn, now):
    out = []
    for st in conn.execute(
            "SELECT * FROM stations WHERE station_id IN (SELECT station_id FROM series)").fetchall():
        today = climate_today(st["tz"], now)
        for d in (today, today + timedelta(days=1)):
            for hz in features.HORIZONS:
                dec = features.decision_time(st["tz"], d, hz)
                if dec <= now < dec + WINDOW_AFTER_DECISION:
                    out.append((st, d, hz, dec))
    return out


def _pending_kinds(conn, station_id, d, hz) -> list:
    """Kinds (high/low) at this station whose decision hasn't been logged yet."""
    return [r[0] for r in conn.execute(
        """SELECT DISTINCT kind FROM series WHERE station_id=? AND kind NOT IN
               (SELECT kind FROM paper_decisions WHERE station_id=? AND target_date=? AND horizon=?)""",
        (station_id, station_id, d.isoformat(), hz))]


def _mark(conn, st, d, hz, kind, dec, status, detail=None, contracts=0, blend_hash=None):
    conn.execute(
        """INSERT OR REPLACE INTO paper_decisions(station_id, target_date, horizon, kind, decision_ts,
               status, detail, contracts, blend_hash, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (st["station_id"], d.isoformat(), hz, kind, dec.isoformat(), status, detail, contracts,
         blend_hash, utcnow()))


def _quotes_at(conn, station_id, d, dec) -> pd.DataFrame:
    """Per market, the snapshot nearest the decision time within QUOTE_MAX_LAG (else no row)."""
    q = pd.read_sql_query(
        """SELECT x.ticker, x.ts AS quote_ts, x.yes_bid, x.yes_ask, x.yes_bid_size, x.yes_ask_size
           FROM market_snapshots x JOIN markets m USING(ticker) JOIN series s USING(series_ticker)
           WHERE s.station_id=? AND m.target_date=? AND x.ts BETWEEN ? AND ?""",
        conn, params=(station_id, d.isoformat(), (dec - QUOTE_MAX_LAG).isoformat(),
                      (dec + QUOTE_MAX_LAG).isoformat()))
    if q.empty:
        return q
    q["lag"] = (pd.to_datetime(q["quote_ts"], utc=True) - dec).abs()
    return q.sort_values("lag").drop_duplicates("ticker").drop(columns="lag")


def _num(x):
    return None if pd.isna(x) else float(x)


def _record(conn, st, d, hz, dec, kinds, blend) -> int:
    # refresh inputs exactly the way the backfill built them
    prev = features.fetch_prev_runs(st["lat"], st["lon"], d - timedelta(days=2), d + timedelta(days=1))
    history.store_prev_runs(conn, st, prev, [d], kinds)
    now = datetime.now(timezone.utc)
    history.fetch_nbs(conn, st["station_id"], now - timedelta(days=2), now)
    conn.commit()

    blend_hash = (blend or {}).get("config_hash")
    ds = model.build_dataset(conn, hz, st["station_id"])
    pr = model.fit_predict(ds, only_dates=[d.isoformat()])
    mk = pd.read_sql_query(
        """SELECT m.ticker, m.event_ticker, m.strike_type, m.floor_strike, m.cap_strike, s.kind
           FROM markets m JOIN series s USING(series_ticker)
           WHERE s.station_id=? AND m.target_date=?""",
        conn, params=(st["station_id"], d.isoformat()))
    quotes = _quotes_at(conn, st["station_id"], d, dec)
    mk = mk.merge(quotes, on="ticker", how="left") if not quotes.empty else mk.assign(
        quote_ts=None, yes_bid=np.nan, yes_ask=np.nan, yes_bid_size=np.nan, yes_ask_size=np.nan)
    threshold = (blend or {}).get("thresholds", {}).get(hz)
    n = 0
    for kind in kinds:
        k = mk[mk["kind"] == kind]
        if not pr.empty:
            k = k.merge(pr[["kind", "mu", "sigma", "nbs", "nbs_sd"]], on="kind")
        if pr.empty or k.empty:
            _mark(conn, st, d, hz, kind, dec, "no_prediction", "not enough history or no markets",
                  blend_hash=blend_hash)
            continue
        if k["quote_ts"].isna().all():
            _mark(conn, st, d, hz, kind, dec, "missed_quote",
                  f"no snapshot within {QUOTE_MAX_LAG} of decision", blend_hash=blend_hash)
            continue
        b = [model.bracket_bounds(t, f, c) for t, f, c in zip(k["strike_type"], k["floor_strike"], k["cap_strike"])]
        k = k[[x is not None for x in b]].copy()
        b = [x for x in b if x is not None]
        k["lo"], k["hi"] = [x[0] for x in b], [x[1] for x in b]
        k["p_model"] = model.prob_in(k["mu"], k["sigma"], k["lo"], k["hi"])
        k["p_nbs"] = np.where(k["nbs"].notna(), model.prob_in(
            k["nbs"].fillna(0), np.maximum(k["nbs_sd"].fillna(2.0), 0.7), k["lo"], k["hi"]), np.nan)
        k["y"] = 0  # unknown yet; trades() only needs it for pnl, which we don't store here
        # v2: trade on the frozen DEV-fit market+model blend, never on the model alone
        k["p_mkt"] = (k["yes_bid"] + k["yes_ask"]) / 2
        k["p_trade"] = np.nan
        if blend and hz in blend["coefs"]:
            cf = blend["coefs"][hz]
            ok = (k["yes_ask"] > k["yes_bid"]) & (k["yes_bid"] > 0) & (k["yes_ask"] < 1)
            k.loc[ok, "p_trade"] = evaluate.apply_blend(
                k[ok], [cf["b0"], cf["b_market"], cf["b_model"]])
        t = (evaluate.trades(k, threshold, "p_trade") if threshold is not None
             else k.iloc[0:0].assign(side=None, price=None, fee=None, exp_edge=None))
        t = t.set_index("ticker")
        for _, r in k.iterrows():
            tr = t.loc[r["ticker"]] if r["ticker"] in t.index else None
            conn.execute(
                """INSERT OR IGNORE INTO paper_predictions(ticker, horizon, event_ticker, decision_ts,
                       model_version, mu, sigma, p_model, p_nbs, yes_bid, yes_ask, action, price, fee,
                       exp_edge, created_at, p_trade, quote_ts, yes_bid_size, yes_ask_size, blend_hash)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (r["ticker"], hz, r["event_ticker"], dec.isoformat(),
                 f"{model.VERSION}+blend/th{threshold}", r["mu"], r["sigma"], r["p_model"],
                 _num(r["p_nbs"]), _num(r["yes_bid"]), _num(r["yes_ask"]),
                 "none" if tr is None else f"buy_{tr['side']}",
                 None if tr is None else tr["price"], None if tr is None else tr["fee"],
                 None if tr is None else tr["exp_edge"], utcnow(), _num(r["p_trade"]),
                 None if pd.isna(r["quote_ts"]) else r["quote_ts"], _num(r["yes_bid_size"]),
                 _num(r["yes_ask_size"]), blend_hash))
            n += 1
        _mark(conn, st, d, hz, kind, dec, "recorded", contracts=len(k), blend_hash=blend_hash)
        log.info("paper: %s %s %s %s recorded %d contracts, %d paper trades",
                 st["station_id"], d, hz, kind, len(k), len(t))
    conn.commit()
    return n


def run(conn) -> int:
    if not ENABLED:
        return 0
    blend, why = _load_blend()
    if blend is None:
        log.warning("paper: not trading (%s); recording model probabilities and quotes only", why)
    n = 0
    for st, d, hz, dec in _due(conn, datetime.now(timezone.utc)):
        kinds = _pending_kinds(conn, st["station_id"], d, hz)
        if not kinds:
            continue
        try:
            n += _record(conn, st, d, hz, dec, kinds, blend)
        except Exception as e:   # one station's failure must not cost the others their decision
            conn.rollback()
            log.exception("paper: %s %s %s failed", st["station_id"], d, hz)
            for kind in kinds:
                _mark(conn, st, d, hz, kind, dec, "error", f"{e.__class__.__name__}: {e}"[:300],
                      blend_hash=(blend or {}).get("config_hash"))
            conn.commit()
    return n


def report(conn) -> str:
    df = pd.read_sql_query(
        """SELECT p.*, m.result, m.target_date FROM paper_predictions p JOIN markets m USING(ticker)""", conn)
    if df.empty:
        return "No paper predictions yet. They start at the next decision time after the hourly job runs."
    s = df[df["result"].isin(["yes", "no"])].copy()
    lines = [f"Paper predictions: {len(df)} contracts recorded, {len(s)} settled.",
             "Decisions: " + ", ".join(f"{st}={c}" for st, c in conn.execute(
                 "SELECT status, COUNT(*) FROM paper_decisions GROUP BY status ORDER BY status")), ""]
    if s.empty:
        return "\n".join(lines)
    s["y"] = (s["result"] == "yes").astype(int)
    q = s.dropna(subset=["yes_bid", "yes_ask"])
    q = q[(q["yes_ask"] > q["yes_bid"]) & (q["yes_bid"] > 0) & (q["yes_ask"] < 1)]
    if not q.empty:
        pm = (q["yes_bid"] + q["yes_ask"]) / 2
        bl = q["p_trade"].notna()
        lines.append(f"Brier  model={np.mean((q['p_model'] - q['y']) ** 2):.4f}  "
                     f"market={np.mean((pm - q['y']) ** 2):.4f}  (n={len(q)})")
        if bl.any():
            lines.append(f"Brier  blend={np.mean((q.loc[bl, 'p_trade'] - q.loc[bl, 'y']) ** 2):.4f}  "
                         f"market={np.mean((pm[bl] - q.loc[bl, 'y']) ** 2):.4f}  (n={int(bl.sum())})  "
                         "<- blend must beat market here")
    t = s[s["action"].isin(["buy_yes", "buy_no"])].copy()
    if not t.empty:
        yes = t["action"] == "buy_yes"
        t["won"] = np.where(yes, t["y"] == 1, t["y"] == 0).astype(int)
        t["pnl"] = t["won"] - t["price"] - t["fee"]
        t["station_id"] = t["event_ticker"]
        lo, hi = evaluate._boot_ci(t)
        lines += [f"Paper trades settled: {len(t)}  win rate {t['won'].mean():.1%}  "
                  f"avg price {t['price'].mean():.2f}",
                  f"Expected edge {t['exp_edge'].mean():+.3f}/contract  realized {t['pnl'].mean():+.3f}/contract  "
                  f"90% CI [{lo:+.3f}, {hi:+.3f}]  total ${t['pnl'].sum():+.2f} per 1-contract stake"]
        for hz, g in t.groupby("horizon"):
            lines.append(f"  {hz:<5} trades={len(g)}  realized {g['pnl'].mean():+.3f}/contract")
    return "\n".join(lines)
