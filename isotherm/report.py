"""Read-only views over the database: collection health and a single-event sanity check."""
import json
from datetime import date, timedelta

from .config import DB_PATH

TABLES = ("stations", "series", "markets", "market_snapshots", "ensemble_forecasts",
          "deterministic_forecasts", "mos_forecasts", "outcomes", "cli_reports")


def status(conn):
    print(f"Database: {DB_PATH}  ({DB_PATH.stat().st_size / 1e6:.1f} MB)\n")
    for t in TABLES:
        print(f"  {t:<26}{conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>12,}")
    print("\nLast run per job:")
    rows = conn.execute(
        """SELECT job, MAX(started_at) AS last, SUM(ok=1) AS ok, SUM(ok=0) AS failed
           FROM job_runs GROUP BY job ORDER BY job""").fetchall()
    for r in rows:
        last = conn.execute(
            "SELECT ok, detail, rows_written FROM job_runs WHERE job=? ORDER BY id DESC LIMIT 1",
            (r["job"],)).fetchone()
        state = "OK  " if last["ok"] == 1 else ("FAIL" if last["ok"] == 0 else "....")
        print(f"  {r['job']:<10} {state} last={r['last']}  runs ok={r['ok']} failed={r['failed']}"
              f"  [{last['detail'] or ''}]")
    print("\nSettled events by kind:")
    for r in conn.execute("SELECT kind, COUNT(*) n, MIN(target_date) a, MAX(target_date) b "
                          "FROM outcomes GROUP BY kind"):
        print(f"  {r['kind']:<5} {r['n']:>6}  {r['a']} .. {r['b']}")
    mism = conn.execute(
        """SELECT o.event_ticker, o.value, CASE o.kind WHEN 'high' THEN c.high ELSE c.low END nws
           FROM outcomes o JOIN cli_reports c ON c.station_id=o.station_id AND c.date=o.target_date
           WHERE nws IS NOT NULL AND nws != o.value""").fetchall()
    print(f"\nKalshi vs NWS climate report disagreements: {len(mism)}")
    for r in mism[:10]:
        print(f"  {r['event_ticker']}: kalshi={r['value']} nws={r['nws']}")


def _contains(strike_type, floor, cap, v):
    if strike_type == "greater":
        return v > floor
    if strike_type == "less":
        return v < cap
    return floor <= v <= cap


def peek(conn, event_ticker):
    ms = conn.execute(
        """SELECT m.*, s.station_id, s.kind FROM markets m JOIN series s USING(series_ticker)
           WHERE event_ticker=? ORDER BY COALESCE(floor_strike, cap_strike - 100)""",
        (event_ticker,)).fetchall()
    if not ms:
        print("unknown event (has `markets` run since it opened?)")
        return
    sid, kind, d = ms[0]["station_id"], ms[0]["kind"], ms[0]["target_date"]
    print(f"{event_ticker}  station={sid}  {kind}  climate day {d}\n")

    pooled, by_model = [], []
    for r in conn.execute(
        """SELECT model, run_init, members_json, mean, sd FROM ensemble_forecasts e
           WHERE station_id=? AND target_date=? AND kind=? AND run_init =
             (SELECT MAX(run_init) FROM ensemble_forecasts WHERE station_id=e.station_id
              AND target_date=e.target_date AND kind=e.kind AND model=e.model)""",
            (sid, d, kind)):
        mem = [round(x) for x in json.loads(r["members_json"])]
        pooled += mem
        by_model.append(f"  {r['model']:<15} run {r['run_init']}  n={len(mem):>3}  "
                        f"mean={r['mean']:.1f}  sd={r['sd']:.1f}")
    print("Latest ensemble runs:")
    print("\n".join(by_model) or "  (none)")

    target = date.fromisoformat(d)
    ftime = (target + timedelta(days=1)).isoformat() + "T00:00" if kind == "high" \
        else d + "T12:00"
    nbs = conn.execute(
        """SELECT runtime_utc, value, spread FROM mos_forecasts WHERE station_id=? AND model='NBS'
           AND ftime_utc LIKE ? ORDER BY runtime_utc DESC LIMIT 1""", (sid, ftime + "%")).fetchone()
    if nbs:
        print(f"  NBS (NBM text)  run {nbs['runtime_utc']}  value={nbs['value']:.0f}  "
              f"sd={nbs['spread'] or float('nan'):.1f}   <- the baseline to beat")

    print("\nRAW pooled-ensemble share vs market. Uncalibrated - NOT a trading signal.\n")
    print(f"  {'bracket':<16}{'bid':>6}{'ask':>6}{'raw ens':>9}{'result':>8}")
    for m in ms:
        snap = conn.execute(
            "SELECT yes_bid, yes_ask FROM market_snapshots WHERE ticker=? ORDER BY ts DESC LIMIT 1",
            (m["ticker"],)).fetchone()
        share = (sum(_contains(m["strike_type"], m["floor_strike"], m["cap_strike"], v)
                     for v in pooled) / len(pooled)) if pooled else None
        bid = f"{snap['yes_bid']:.2f}" if snap and snap["yes_bid"] is not None else "-"
        ask = f"{snap['yes_ask']:.2f}" if snap and snap["yes_ask"] is not None else "-"
        print(f"  {m['yes_sub_title'] or m['ticker']:<16}{bid:>6}{ask:>6}"
              f"{'-' if share is None else f'{share:.0%}':>9}{m['result'] or '':>8}")
    out = conn.execute("SELECT value FROM outcomes WHERE event_ticker=?", (event_ticker,)).fetchone()
    if out:
        print(f"\nOfficial value: {out['value']:.0f}")
