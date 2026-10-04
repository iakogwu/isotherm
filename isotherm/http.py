"""Shared HTTP session: GET only, retries with backoff, polite pacing."""
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import USER_AGENT

_session = None


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(
            total=4,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            respect_retry_after_header=True,
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.headers["User-Agent"] = USER_AGENT
        s.headers["Accept"] = "application/json"
        _session = s
    return _session


def get_json(url: str, params=None, pause: float = 0.0, timeout: int = 60):
    r = session().get(url, params=params, timeout=timeout)
    r.raise_for_status()
    if pause:
        time.sleep(pause)
    return r.json()
