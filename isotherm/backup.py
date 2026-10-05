"""Daily copy of the database. It holds the only record of the forward test."""
import logging
import os
import sqlite3
import time
from datetime import date

from .config import DATA_DIR

log = logging.getLogger(__name__)

BACKUP_DIR = DATA_DIR / "backups"
KEEP = 7


def run(conn) -> int:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for stale in BACKUP_DIR.glob("isotherm-*.db.part"):
        stale.unlink(missing_ok=True)
    final = BACKUP_DIR / f"isotherm-{date.today().isoformat()}.db"
    part = final.with_name(final.name + ".part")
    t0 = time.time()
    # Online backup in small steps so the hourly writer is never blocked for long.
    dst = sqlite3.connect(part)
    try:
        conn.backup(dst, pages=1024, sleep=0.05)
        dst.execute("PRAGMA journal_mode=DELETE")   # self-contained file, no -wal sidecar
        check = dst.execute("PRAGMA quick_check").fetchone()[0]
    finally:
        dst.close()
    if check != "ok":
        part.unlink(missing_ok=True)
        raise RuntimeError(f"backup failed quick_check: {check}")
    os.replace(part, final)
    _prune()
    log.info("backup %s: %.1f MB in %.1fs", final.name, final.stat().st_size / 1e6, time.time() - t0)
    return 1


def _prune():
    # ISO dates in the name sort chronologically.
    for old in sorted(BACKUP_DIR.glob("isotherm-*.db"), reverse=True)[KEEP:]:
        old.unlink()
