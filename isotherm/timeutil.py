"""Climate-day handling.

NWS daily climate reports (which Kalshi's settlement data is built on) cover midnight to
midnight *local standard time*, all year. During daylight saving time that is 1 AM to
1 AM on the wall clock. Getting this window wrong quietly corrupts every max/min.
"""
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo


def standard_offset(tz_name: str, d: date) -> timedelta:
    probe = datetime(d.year, d.month, d.day, 12, tzinfo=ZoneInfo(tz_name))
    return probe.utcoffset() - probe.dst()


def climate_day_window(tz_name: str, d: date):
    """Return [start, end) of climate day `d` as aware UTC datetimes."""
    off = standard_offset(tz_name, d)
    start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc) - off
    return start, start + timedelta(days=1)


def climate_today(tz_name: str, now_utc: datetime | None = None) -> date:
    now_utc = now_utc or datetime.now(timezone.utc)
    off = standard_offset(tz_name, now_utc.date())
    return (now_utc + off).date()


def parse_event_date(event_ticker: str) -> date:
    """KXHIGHNY-26OCT05 -> 2026-10-05."""
    return datetime.strptime(event_ticker.rsplit("-", 1)[1], "%y%b%d").date()


def iso_utc(ts: float | int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).replace(microsecond=0).isoformat()
