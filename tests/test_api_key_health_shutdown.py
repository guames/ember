"""API key auth, /health, and graceful SIGTERM shutdown (issue #27).

Three angles, all model-free:
1. EMBER_API_KEY off (default): /v1/* works unauthenticated, as today.
2. EMBER_API_KEY set: /v1/* rejects missing/wrong bearer tokens with 401, accepts the
   right one, and /health stays open regardless (supervisors shouldn't need the key).
3. Shutdown: do_POST rejects new work with 503 once `_shutting_down` is set, and
   `_wait_for_drain` blocks while the worker is busy and returns once it clears.
"""

import http.client
import threading

import pytest

from ember import server


@pytest.fixture
def live_server():
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        t.join()


@pytest.fixture
def api_key():
    server.API_KEY = "secret123"
    try:
        yield server.API_KEY
    finally:
        server.API_KEY = None


def _get(conn, path, headers=None):
    conn.request("GET", path, headers=headers or {})
    r = conn.getresponse()
    body = r.read()
    return r.status, body


def test_health_is_unauthenticated_and_ok(live_server):
    host, port = live_server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        status, body = _get(conn, "/health")
        assert status == 200
        assert b'"ok"' in body
    finally:
        conn.close()


def test_v1_unauthenticated_by_default(live_server):
    assert server.API_KEY is None
    host, port = live_server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        status, _ = _get(conn, "/v1/models")
        assert status == 200
    finally:
        conn.close()


def test_v1_requires_bearer_token_when_api_key_set(live_server, api_key):
    host, port = live_server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        status, body = _get(conn, "/v1/models")
        assert status == 401
        assert b"invalid_api_key" in body

        status, body = _get(conn, "/v1/models", {"Authorization": "Bearer wrong"})
        assert status == 401

        status, _ = _get(conn, "/v1/models", {"Authorization": f"Bearer {api_key}"})
        assert status == 200
    finally:
        conn.close()


def test_health_stays_open_even_with_api_key_set(live_server, api_key):
    host, port = live_server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        status, _ = _get(conn, "/health")
        assert status == 200
    finally:
        conn.close()


def test_info_routes_require_bearer_token_when_api_key_set(live_server, api_key):
    """/status, /memory, /metrics leak model names/memory/traffic -- issue #52 widens
    auth to cover them, not just /v1/*."""
    host, port = live_server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        for path in ("/status", "/memory", "/metrics"):
            status, body = _get(conn, path)
            assert status == 401, path
            assert b"invalid_api_key" in body

            status, _ = _get(conn, path, {"Authorization": f"Bearer {api_key}"})
            assert status == 200, path
    finally:
        conn.close()


def test_info_routes_unauthenticated_by_default(live_server):
    assert server.API_KEY is None
    host, port = live_server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        for path in ("/status", "/memory", "/metrics"):
            status, _ = _get(conn, path)
            assert status == 200, path
    finally:
        conn.close()


def test_unload_and_clear_require_bearer_token_when_api_key_set(live_server, api_key):
    """POST /unload and /clear are state-mutating (evict every model / drop KV caches)
    and must not be reachable without the key once one is set (issue #52). Needs the
    queue worker running -- unlike the auth-only checks above, the authorized branch
    goes all the way through job dispatch."""
    threading.Thread(target=server._worker, daemon=True).start()
    host, port = live_server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        for path in ("/unload", "/clear"):
            conn.request("POST", path, body=b"{}")
            r = conn.getresponse()
            body = r.read()
            assert r.status == 401, path
            assert b"invalid_api_key" in body

            conn.request(
                "POST",
                path,
                body=b"{}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            r = conn.getresponse()
            r.read()
            assert r.status == 200, path
    finally:
        conn.close()


def test_shutting_down_rejects_new_posts(live_server):
    server._shutting_down.set()
    try:
        host, port = live_server.server_address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        try:
            conn.request("POST", "/v1/chat/completions", body=b"{}")
            r = conn.getresponse()
            body = r.read()
            assert r.status == 503
            assert b"shutting_down" in body
        finally:
            conn.close()
    finally:
        server._shutting_down.clear()


def test_wait_for_drain_blocks_until_worker_idle():
    server._worker_busy.set()
    try:
        assert server._wait_for_drain(timeout=0.2) is False
    finally:
        server._worker_busy.clear()
    assert server._wait_for_drain(timeout=1) is True


def test_wait_for_drain_true_when_queue_and_worker_idle():
    assert server._q.empty()
    assert not server._worker_busy.is_set()
    assert server._wait_for_drain(timeout=0.1) is True


# ---------------------------------------------------------------- drain/dequeue race (issue #82)
class _AlwaysEmptyQueue:
    def empty(self):
        return True


def _fake_clock(step=0.02):
    """Deterministic stand-in for time.monotonic: advances by `step` on every call, so a
    timeout-bounded loop runs a fixed, fast number of iterations instead of spinning on
    the real wall clock."""
    state = {"t": 0.0}

    def tick():
        state["t"] += step
        return state["t"]

    return tick


def test_wait_for_drain_rechecks_race_between_dequeue_and_busy_set(monkeypatch):
    """_worker does `_q.get()` *then* `_worker_busy.set()` -- a drain check landing in
    that gap sees "queue empty + worker idle" even though a job is about to run. Simulates
    exactly that gap: the queue always looks empty, and _worker_busy.is_set() reads False
    on the very first check (the race) but True on every read after (the job has started).
    The old single-check implementation would have returned True here; the fix's re-check
    must catch it and keep waiting instead."""
    is_set_calls = {"n": 0}

    class _BusyFromSecondCheck:
        def is_set(self):
            is_set_calls["n"] += 1
            return is_set_calls["n"] > 1

    monkeypatch.setattr(server, "_q", _AlwaysEmptyQueue())
    monkeypatch.setattr(server, "_worker_busy", _BusyFromSecondCheck())
    monkeypatch.setattr(server.time, "sleep", lambda s: None)  # don't actually wait in the test
    monkeypatch.setattr(server.time, "monotonic", _fake_clock())

    assert server._wait_for_drain(timeout=0.3) is False
    assert is_set_calls["n"] > 1  # the re-check ran at least once


def test_wait_for_drain_recheck_confirms_true_drain(monkeypatch):
    """A genuine drain (idle on the first check AND the re-check) still reports True --
    the re-check doesn't turn into a second, redundant wait for the timeout."""

    class _NeverBusy:
        def is_set(self):
            return False

    monkeypatch.setattr(server, "_q", _AlwaysEmptyQueue())
    monkeypatch.setattr(server, "_worker_busy", _NeverBusy())
    monkeypatch.setattr(server.time, "sleep", lambda s: None)
    monkeypatch.setattr(server.time, "monotonic", _fake_clock())

    assert server._wait_for_drain(timeout=5) is True
