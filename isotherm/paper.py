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

# Frozen settings come from data/reports/blend.json (written by `evaluate`, fit on DEV only).
WINDOW_AFTER_DECISION = timedelta(hours=3)
ENABLED = False   # flipped on once history backfill + first evaluation are done


def _load_blend():
    import json
    try:
        return json.loads(evaluate.BLEND_FILE.read_text())
    except (OSError, ValueError):
        return None


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


def _already(conn, station_id, d, hz) -> bool:
    return conn.execute(
        """SELECT 1 FROM paper_predictions p JOIN markets m USING(ticker)
           JOIN series s USING(series_ticker)
           WHERE s.station_id=? AND m.target_date=? AND p.horizon=? LIMIT 1""",
        (station_id, d.isoformat(), hz)).fetchone() is not None


def run(conn) -> int:
    if not ENABLED:
        return 0
    now = datetime.now(timezone.utc)
    n = 0
    for st, d, hz, dec in _due(conn, now):
        if _already(conn, st["station_id"], d, hz):
            continue
        kinds = [r[0] for r in conn.execute(
            "SELECT DISTINCT kind FROM series WHERE station_id=?", (st["station_id"],))]
        # refresh inputs exactly the way the backfill built them
        prev = features.fetch_prev_runs(st["lat"], st["lon"], d - timedelta(days=2), d + timedelta(days=1))
        history.store_prev_runs(conn, st, prev, [d], kinds)
        history.fetch_nbs(conn, st["station_id"], now - timedelta(days=2), now)
        conn.commit()

        ds = model.build_dataset(conn, hz, st["station_id"])
        pr = model.fit_predict(ds, only_dates=[d.isoformat()])
        if pr.empty:
            log.info("paper: no prediction for %s %s %s (not enough history)", st["station_id"], d, hz)
            continue
        mk = pd.read_sql_query(
            """SELECT m.ticker, m.event_ticker, m.strike_type, m.floor_strike, m.cap_strike, s.kind,
                      (SELECT yes_bid FROM market_snapshots x WHERE x.ticker=m.ticker ORDER BY ts DESC LIMIT 1) yes_bid,
                      (SELECT yes_ask FROM market_snapshots x WHERE x.ticker=m.ticker ORDER BY ts DESC LIMIT 1) yes_ask
               FROM markets m JOIN series s USING(series_ticker)
               WHERE s.station_id=? AND m.target_date=?""",
            conn, params=(st["station_id"], d.isoformat()))
        mk = mk.merge(pr[["kind", "mu", "sigma", "nbs", "nbs_sd"]], on="kind")
        if mk.empty:
            continue
        b = [model.bracket_bounds(t, f, c) for t, f, c in zip(mk["strike_type"], mk["floor_strike"], mk["cap_strike"])]
        mk = mk[[x is not None for x in b]].copy()
        b = [x for x in b if x is not None]
        mk["lo"], mk["hi"] = [x[0] for x in b], [x[1] for x in b]
        mk["p_model"] = model.prob_in(mk["mu"], mk["sigma"], mk["lo"], mk["hi"])
        mk["p_nbs"] = np.where(mk["nbs"].notna(), model.prob_in(
            mk["nbs"].fillna(0), np.maximum(mk["nbs_sd"].fillna(2.0), 0.7), mk["lo"], mk["hi"]), np.nan)
        mk["y"] = 0  # unknown yet; trades() only needs it for pnl, which we don't store here
        # v2: trade on the frozen DEV-fit market+model blend, never on the model alone
        blend = _load_blend()
        mk["p_mkt"] = (mk["yes_bid"] + mk["yes_ask"]) / 2
        mk["p_trade"] = np.nan
        if blend and hz in blend["coefs"]:
            cf = blend["coefs"][hz]
            ok = (mk["yes_ask"] > mk["yes_bid"]) & (mk["yes_bid"] > 0) & (mk["yes_ask"] < 1)
            mk.loc[ok, "p_trade"] = evaluate.apply_blend(
                mk[ok], [cf["b0"], cf["b_market"], cf["b_model"]])
        threshold = (blend or {}).get("thresholds", {}).get(hz)
        t = (evaluate.trades(mk, threshold, "p_trade") if threshold is not None
             else mk.iloc[0:0].assign(side=None, price=None, fee=None, exp_edge=None))
        t = t.set_index("ticker")
        for _, r in mk.iterrows():
            tr = t.loc[r["ticker"]] if r["ticker"] in t.index else None
            conn.execute(
                """INSERT OR IGNORE INTO paper_predictions(ticker, horizon, event_ticker, decision_ts,
                       model_version, mu, sigma, p_model, p_nbs, yes_bid, yes_ask, action, price, fee,
                       exp_edge, created_at, p_trade) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (r["ticker"], hz, r["event_ticker"], dec.isoformat(),
                 f"{model.VERSION}+blend/th{threshold}", r["mu"], r["sigma"], r["p_model"],
                 None if pd.isna(r["p_nbs"]) else r["p_nbs"], r["yes_bid"], r["yes_ask"],
                 "none" if tr is None else f"buy_{tr['side']}",
                 None if tr is None else tr["price"], None if tr is None else tr["fee"],
                 None if tr is None else tr["exp_edge"], utcnow(),
                 None if pd.isna(r["p_trade"]) else r["p_trade"]))
            n += 1
        conn.commit()
        log.info("paper: %s %s %s recorded %d contracts, %d paper trades",
                 st["station_id"], d, hz, len(mk), len(t))
    return n


def report(conn) -> str:
    df = pd.read_sql_query(
        """SELECT p.*, m.result, m.target_date FROM paper_predictions p JOIN markets m USING(ticker)""", conn)
    if df.empty:
        return "No paper predictions yet. They start at the next decision time after the hourly job runs."
    s = df[df["result"].isin(["yes", "no"])].copy()
    lines = [f"Paper predictions: {len(df)} contracts recorded, {len(s)} settled.", ""]
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
