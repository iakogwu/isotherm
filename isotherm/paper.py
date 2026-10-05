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


# Forward-test stopping rule, registered before any forward result existed:
# docs/forward-test-plan.md. Each look is computed once, at a point fixed by the data, stored in
# forward_looks and never recomputed. Between looks only collection health is shown.
FORWARD_START = "2026-10-05"
INTERIM_DATES = 45   # futility look only: stop if the blend isn't ahead of the market
FINAL_DATES = 90     # the one success look
MIN_COMPLETE = 0.90  # share of expected decisions actually recorded, counted from FORWARD_START


def _settled(conn) -> pd.DataFrame:
    """Forward predictions on fully settled dates only. A date counts once every prediction
    recorded for it has a final result (any result: yes, no, or a void/scalar that scoring then
    drops), and only as part of an unbroken run from FORWARD_START, so a look can't freeze
    while some city's events, on that date or an earlier one, are still unsettled."""
    f = pd.read_sql_query(
        """SELECT p.*, m.result, m.target_date FROM paper_predictions p JOIN markets m USING(ticker)
           WHERE m.target_date >= ?""", conn, params=(FORWARD_START,))
    done = f.groupby("target_date")["result"].apply(lambda r: r.notna().all()).sort_index()
    prefix = done.index[:int(done.cummin().sum())] if len(done) else []
    s = f[f["target_date"].isin(prefix) & f["result"].isin(["yes", "no"])].copy()
    s["y"] = (s["result"] == "yes").astype(int)
    return s


def completeness(conn, through: str) -> tuple:
    """(recorded share, expected count) of decisions for target dates FORWARD_START..through.
    Expected = every series x horizon x day, so a PC that was off (no row at all) counts as missed."""
    n_series = conn.execute("SELECT COUNT(*) FROM series").fetchone()[0]
    days = (pd.Timestamp(through) - pd.Timestamp(FORWARD_START)).days + 1
    expected = max(days, 0) * n_series * len(features.HORIZONS)
    recorded = conn.execute(
        """SELECT COUNT(*) FROM paper_decisions WHERE status='recorded'
           AND target_date BETWEEN ? AND ?""", (FORWARD_START, through)).fetchone()[0]
    return (recorded / expected if expected else 0.0), expected


def _look_point(conn, s: pd.DataFrame, need: int, after: str | None):
    """The first settled date at which `need` settled dates exist and completeness through it is
    at least MIN_COMPLETE. Fixed by the data, not by when this runs."""
    dates = sorted(s["target_date"].unique())
    for i, d in enumerate(dates):
        if i + 1 >= need and (after is None or d > after) and completeness(conn, d)[0] >= MIN_COMPLETE:
            return d
    return None


def _evaluate_look(conn, s: pd.DataFrame, through: str) -> dict:
    s = s[s["target_date"] <= through]
    ev = _forward_events(s)
    if ev.empty:
        return {"estimate": None, "ci_lo": None, "ci_hi": None, "table": "(no fully quoted events)"}
    ev["part"] = "forward"
    tab = evaluate._event_table(pd.concat([ev, ev.assign(horizon="both")]), "part")
    d = tab[tab["horizon"] == "both"].iloc[0]
    text = evaluate._fmt(tab)
    t = s[s["action"].isin(["buy_yes", "buy_no"])].copy()
    if not t.empty:
        yes = t["action"] == "buy_yes"
        t["won"] = np.where(yes, t["y"] == 1, t["y"] == 0).astype(int)
        t["pnl"] = t["won"] - t["price"] - t["fee"]
        lo, hi = evaluate._boot_ci(t)
        text += (f"\nPaper trades: {len(t)} on {t['target_date'].nunique()} dates, expected edge "
                 f"{t['exp_edge'].mean():+.3f}/contract, realized {t['pnl'].mean():+.3f}/contract, "
                 f"90% CI [{lo:+.3f}, {hi:+.3f}] (dates resampled)")
    return {"estimate": float(d["blend_minus_market"]), "ci_lo": float(d["ci90_lo"]),
            "ci_hi": float(d["ci90_hi"]), "table": text}


def check_looks(conn) -> int:
    """Record the interim and final looks once each, when their registered conditions are met.
    Runs inside the hourly job; never replaces a recorded look."""
    done = {r[0]: r[1] for r in conn.execute("SELECT look, verdict FROM forward_looks")}
    s = _settled(conn)
    if s.empty:
        return 0
    n = 0
    for look, need in (("interim", INTERIM_DATES), ("final", FINAL_DATES)):
        if look in done:
            continue
        if look == "final" and ("interim" not in done or done["interim"].startswith("STOP")):
            break
        after = (conn.execute("SELECT through_date FROM forward_looks WHERE look='interim'").fetchone()
                 or [None])[0] if look == "final" else None
        through = _look_point(conn, s, need, after)
        if through is None:
            break
        r = _evaluate_look(conn, s, through)
        est, hi = r["estimate"], r["ci_hi"]
        if look == "interim":
            verdict = ("STOP (futility): blend not ahead of the market" if est is None or est >= 0
                       else "CONTINUE: blend ahead at the interim look")
        else:
            verdict = ("PASS: blend beats the market (90% CI below zero)" if hi is not None and hi < 0
                       else "FAIL: no clear improvement over the market")
        share, expected = completeness(conn, through)
        conn.execute(
            """INSERT INTO forward_looks(look, through_date, settled_dates, completeness, expected,
                   estimate, ci_lo, ci_hi, verdict, detail, blend_hash, recorded_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (look, through, int(s.loc[s["target_date"] <= through, "target_date"].nunique()),
             share, expected, est, r["ci_lo"], hi, verdict, r["table"],
             (_load_blend()[0] or {}).get("config_hash"), utcnow()))
        conn.commit()
        log.info("forward look %s recorded through %s: %s", look, through, verdict)
        done[look] = verdict
        n += 1
    return n


def _forward_events(s: pd.DataFrame) -> pd.DataFrame:
    """Per-event scores for settled forward predictions, as in evaluate.event_scores."""
    c = s.assign(p_mkt=(s["yes_bid"] + s["yes_ask"]) / 2, p_blend=s["p_trade"],
                 split="forward", dev_part=None)
    return evaluate.event_scores(c, ["p_mkt", "p_model", "p_blend"])


def status(conn) -> str:
    """Collection health only. No performance numbers, so routine checks can't peek."""
    dec = dict(conn.execute(
        "SELECT status, COUNT(*) FROM paper_decisions WHERE target_date >= ? GROUP BY status",
        (FORWARD_START,)).fetchall())
    s = _settled(conn)
    dates = s["target_date"].nunique()
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    lines = [f"Forward test from {FORWARD_START}. Decisions: "
             + (", ".join(f"{k}={v}" for k, v in sorted(dec.items())) or "none yet")]
    if yesterday >= FORWARD_START:
        share, expected = completeness(conn, yesterday)
        lines.append(f"Recorded {share:.1%} of {expected} expected decisions through {yesterday} "
                     f"(looks need >= {MIN_COMPLETE:.0%})")
    lines.append(f"Settled forward dates: {dates}  (interim look at {INTERIM_DATES}, final at {FINAL_DATES})")
    looks = conn.execute("SELECT look, through_date FROM forward_looks ORDER BY recorded_at").fetchall()
    lines.append("Looks recorded: " + (", ".join(f"{a} through {b}" for a, b in looks) or "none")
                 + ". Results: python -m isotherm paper-report")
    return "\n".join(lines)


def report(conn) -> str:
    """The recorded looks, exactly as stored. Nothing is computed here."""
    rows = conn.execute("SELECT * FROM forward_looks ORDER BY recorded_at").fetchall()
    if not rows:
        return ("No forward look recorded yet. Results stay hidden until the registered looks "
                f"({INTERIM_DATES} and {FINAL_DATES} settled dates, >= {MIN_COMPLETE:.0%} recorded); "
                "see `paper-status` for collection health.")
    out = []
    for r in rows:
        out += [f"== {r['look'].upper()} look through {r['through_date']} (recorded {r['recorded_at']}, "
                f"blend {r['blend_hash']})",
                f"Settled dates {r['settled_dates']}, completeness {r['completeness']:.1%} of {r['expected']}",
                r["detail"], f"Verdict: {r['verdict']}", ""]
    return "\n".join(out)
