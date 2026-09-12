"""
One HTTP path for every provider: timeouts, exponential backoff, rate-limit awareness, a
per-host circuit breaker, and a sync-log row for every call.

curl is used rather than a Python HTTP library so the job has no dependency the workflow
does not already carry, and so a hung connection is bounded by --max-time rather than hope.

Secrets: a key travels only in a header or query string built here from an environment
variable. URLs written to the sync log have query strings stripped, so a key can never land
in the repository through a log line.
"""
import json
import os
import subprocess
import time
from urllib.parse import urlsplit, urlunsplit

from . import store

DEFAULT_TIMEOUT = 20
MAX_RETRIES = 3
BACKOFF = 1.5          # seconds, doubled each retry
_BREAKER = {}          # host -> {"fails": n, "open_until": ts}
BREAK_AFTER = 4        # consecutive failures before the host is skipped
BREAK_FOR = 15 * 60    # seconds

# NWS asks every client to identify itself. Set NWS_USER_AGENT in the workflow secrets or
# env; the default names the project, never a person.
USER_AGENT = os.environ.get("NWS_USER_AGENT") or "snap-judgment (github.com/isaaczheng2-droid/snap-judgment)"


def _redact(url):
    u = urlsplit(url)
    return urlunsplit((u.scheme, u.netloc, u.path, "", ""))


def _host(url):
    return urlsplit(url).netloc


def breaker_open(url):
    b = _BREAKER.get(_host(url))
    return bool(b and b.get("open_until", 0) > time.time())


def _note_failure(url):
    b = _BREAKER.setdefault(_host(url), {"fails": 0, "open_until": 0})
    b["fails"] += 1
    if b["fails"] >= BREAK_AFTER:
        b["open_until"] = time.time() + BREAK_FOR


def _note_success(url):
    _BREAKER[_host(url)] = {"fails": 0, "open_until": 0}


def get_json(url, provider, endpoint, headers=None, timeout=DEFAULT_TIMEOUT, fixture=None, log=print):
    """
    -> (data or None, info). info carries http_status, latency_ms, error, retries.
    A fixture path short-circuits the network (used for tests and for the two dev machines
    that cannot reach the providers); the sync log records that it was a fixture.
    """
    started = store.now_iso()
    t0 = time.time()
    info = {"provider": provider, "endpoint": endpoint, "url": _redact(url), "started_at": started,
            "http_status": None, "error": None, "retry_count": 0, "records_fetched": 0, "fixture": bool(fixture)}
    if fixture:
        try:
            data = json.load(open(fixture))
            info.update(status="ok", http_status=200)
            return data, _finish(info, t0)
        except Exception as e:
            info.update(status="error", error=f"fixture: {e}")
            return None, _finish(info, t0)
    if breaker_open(url):
        info.update(status="skipped", error="circuit open for host")
        return None, _finish(info, t0)

    hdr = ["-H", "Accept: application/json, application/geo+json", "-H", f"User-Agent: {USER_AGENT}"]
    for k, v in (headers or {}).items():
        hdr += ["-H", f"{k}: {v}"]
    delay = BACKOFF
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = subprocess.run(["curl", "-sS", "-L", "--max-time", str(timeout), "-w", "\n%{http_code}", *hdr, url],
                               capture_output=True, timeout=timeout + 10)
            body, _, code = r.stdout.rpartition(b"\n")
            status = int(code or 0)
            info["http_status"] = status
            if status == 200 and body:
                data = json.loads(body)
                info["status"] = "ok"
                _note_success(url)
                return data, _finish(info, t0)
            if status in (429, 500, 502, 503, 504) or status == 0:
                info["error"] = f"HTTP {status}"
                raise RuntimeError(info["error"])
            # 4xx other than 429: not retryable
            info.update(status="error", error=f"HTTP {status}: {body[:120].decode(errors='ignore')}")
            _note_failure(url)
            return None, _finish(info, t0)
        except Exception as e:
            info["error"] = str(e)[:200]
            if attempt < MAX_RETRIES:
                info["retry_count"] += 1
                time.sleep(delay)
                delay *= 2
                continue
            info["status"] = "error"
            _note_failure(url)
            return None, _finish(info, t0)
    return None, _finish(info, t0)


def _finish(info, t0):
    info["latency_ms"] = int((time.time() - t0) * 1000)
    info["completed_at"] = store.now_iso()
    store.append("api_sync_log", dict(info))
    return info
