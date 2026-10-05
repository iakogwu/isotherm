"""Isotherm Phase 1 logger.

    python -m isotherm run            # what the scheduler calls; each job runs only when due
    python -m isotherm run --force    # run every job now
    python -m isotherm <job>          # discover | markets | forecasts | mos | settle | cli
    python -m isotherm backfill-cli 2015 2026
    python -m isotherm status
    python -m isotherm peek KXHIGHNY-26OCT05
"""
import argparse
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler

from . import climate, config, discover, forecasts, kalshi, mos, paper, report
from .db import connect, last_success, utcnow

JOBS = {
    "discover": discover.run,
    "markets": kalshi.snapshot,
    "forecasts": forecasts.run,
    "mos": mos.run,
    "settle": kalshi.settle,
    "cli": climate.run,
    "paper": paper.run,
    "trades": kalshi.trades,
    "books": kalshi.books,
    "looks": paper.check_looks,
}
LOCK = config.DATA_DIR / "run.lock"

log = logging.getLogger("isotherm")


def _setup_logging():
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = RotatingFileHandler(config.LOG_DIR / "isotherm.log", maxBytes=5_000_000, backupCount=5,
                             encoding="utf-8")
    fh.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    if sys.stdout is not None and sys.stdout.isatty():
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _due(conn, job) -> bool:
    last = last_success(conn, job)
    if not last:
        return True
    age = datetime.now(timezone.utc) - datetime.fromisoformat(last)
    return age >= timedelta(minutes=config.INTERVALS_MIN[job])


def run_job(conn, job) -> bool:
    cur = conn.execute("INSERT INTO job_runs(job, started_at) VALUES (?,?)", (job, utcnow()))
    conn.commit()
    run_id = cur.lastrowid
    t0 = time.time()
    try:
        rows = JOBS[job](conn)
        conn.execute("UPDATE job_runs SET finished_at=?, ok=1, rows_written=?, detail=? WHERE id=?",
                     (utcnow(), rows, f"{time.time() - t0:.1f}s", run_id))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        log.error("job %s failed: %s\n%s", job, e, traceback.format_exc())
        conn.execute("UPDATE job_runs SET finished_at=?, ok=0, detail=? WHERE id=?",
                     (utcnow(), f"{type(e).__name__}: {e}"[:500], run_id))
        conn.commit()
        return False


def _acquire_lock() -> bool:
    try:
        if LOCK.exists() and time.time() - LOCK.stat().st_mtime > 2 * 3600:
            LOCK.unlink()  # stale lock from a crashed run
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def cmd_run(conn, force: bool) -> int:
    if not _acquire_lock():
        log.info("another run is in progress; exiting")
        return 0
    try:
        ok = True
        # discover first so a fresh install has series/stations to work with
        for job in ("discover", "markets", "books", "forecasts", "mos", "settle", "cli", "paper",
                    "trades", "looks"):
            if force or _due(conn, job):
                ok &= run_job(conn, job)
        return 0 if ok else 1
    finally:
        LOCK.unlink(missing_ok=True)


def main(argv=None) -> int:
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _setup_logging()
    ap = argparse.ArgumentParser(prog="isotherm", description="Isotherm Phase 1 data logger")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run all due jobs (used by the scheduler)")
    r.add_argument("--force", action="store_true", help="ignore intervals; run every job")
    for job in JOBS:
        sub.add_parser(job, help=f"run the {job} job now")
    b = sub.add_parser("backfill-cli", help="load historical NWS climate reports")
    b.add_argument("start_year", type=int)
    b.add_argument("end_year", type=int)
    h = sub.add_parser("backfill-history", help="Phase 2/3 history (resumable; slow)")
    h.add_argument("--since", default="2024-02-01")
    h.add_argument("--step", choices=("all", "markets", "forecasts", "nbs", "candles", "trades"),
                   default="all")
    h.add_argument("--retry-empty", action="store_true",
                   help="candles: re-fetch markets whose earlier fetch returned no candles")
    o = sub.add_parser("backfill-outcomes", help="load recently settled Kalshi results")
    o.add_argument("--days", type=int, default=30)
    ev = sub.add_parser("evaluate", help="Phase 2 skill + Phase 3 backtest report")
    ev.add_argument("--reuse", action="store_true", help="reuse the latest saved predictions (fast)")
    sub.add_parser("paper-report", help="the recorded forward-test looks (hidden until registered)")
    sub.add_parser("paper-status", help="forward-test collection health (no performance numbers)")
    sub.add_parser("status", help="collection health summary")
    p = sub.add_parser("peek", help="sanity-check one event (raw, uncalibrated)")
    p.add_argument("event_ticker")
    args = ap.parse_args(argv)

    conn = connect()
    if args.cmd == "run":
        return cmd_run(conn, args.force)
    if args.cmd in JOBS:
        return 0 if run_job(conn, args.cmd) else 1
    if args.cmd == "backfill-cli":
        climate.fetch(conn, range(args.start_year, args.end_year + 1))
        return 0
    if args.cmd == "backfill-history":
        from datetime import date
        from . import history
        since = date.fromisoformat(args.since)
        step = {"all": history.run_all, "markets": history.backfill_markets,
                "forecasts": history.backfill_forecasts, "nbs": history.backfill_nbs,
                "candles": history.backfill_candles}[args.step]
        if args.step == "candles":
            history.backfill_candles(conn, since, retry_empty=args.retry_empty)
        elif args.step == "trades":   # not part of "all": optional, slow
            kalshi.trades(conn, since=since.isoformat())
        else:
            step(conn, since)
        return 0
    if args.cmd == "paper-report":
        print(paper.report(conn))
        return 0
    if args.cmd == "paper-status":
        print(paper.status(conn))
        return 0
    if args.cmd == "evaluate":
        from . import evaluate
        print(evaluate.run(conn, reuse=args.reuse))
        return 0
    if args.cmd == "backfill-outcomes":
        print(f"settled events loaded: {kalshi.backfill_outcomes(conn, args.days)}")
        return 0
    if args.cmd == "status":
        report.status(conn)
        return 0
    if args.cmd == "peek":
        report.peek(conn, args.event_ticker)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
