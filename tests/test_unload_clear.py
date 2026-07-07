"""Issue #71: _run_unload/_run_clear/_run_evict must never leave job.out empty.

Before this fix, an exception inside these worker-side jobs was caught only by the outer
_worker loop (which just logs it) -- job.out never got a message, so a client blocked in
Handler._unload/_clear (job.out.get() with no timeout) would hang forever. Mirrors the
style of tests/test_ac_em_keep_alive.py: a minimal _Job stand-in, module globals stubbed.
"""

import queue
import threading

import pytest

from ember import server


class _Job:
    def __init__(self, payload):
        self.payload = payload
        self.out = queue.Queue()
        self.cancel = threading.Event()


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.setattr(server, "_chat", {}, raising=True)
    monkeypatch.setattr(
        server,
        "_ac",
        {"model": None, "tok": None, "pc": None, "pctoks": None, "last": 0.0, "ka": -1},
    )
    monkeypatch.setattr(server, "_em", {"model": None, "proc": None, "last": 0.0, "ka": -1})


def test_run_unload_puts_error_on_exception(clean, monkeypatch):
    monkeypatch.setattr(server, "_chat", {"m": {}})
    monkeypatch.setattr(server, "_evict", lambda n: (_ for _ in ()).throw(RuntimeError("boom")))
    job = _Job({"target": "chat"})
    server._run_unload(job)
    kind, data = job.out.get_nowait()
    assert kind == "error"
    assert "boom" in data


def test_run_clear_puts_error_on_exception(clean, monkeypatch):
    monkeypatch.setattr(server, "_chat", {"m": {"slots": [{"pc": object(), "pctoks": [1]}]}})
    monkeypatch.setattr(
        server.mx, "clear_cache", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    job = _Job({"target": "cache"})
    server._run_clear(job)
    kind, data = job.out.get_nowait()
    assert kind == "error"
    assert "boom" in data


def test_run_evict_puts_error_on_exception(clean, monkeypatch):
    monkeypatch.setattr(server, "_chat", {"m": {"ka": 1.0, "last": 0.0}})
    monkeypatch.setattr(server.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(server, "_evict", lambda n: (_ for _ in ()).throw(RuntimeError("boom")))
    job = _Job({"names": ["m"], "ac": False, "em": False})
    server._run_evict(job)
    kind, data = job.out.get_nowait()
    assert kind == "error"
    assert "boom" in data


def test_run_unload_still_puts_result_on_success(clean, monkeypatch):
    monkeypatch.setattr(server, "_chat", {"m": {}})
    monkeypatch.setattr(server, "_evict", lambda n: server._chat.pop(n, None))
    job = _Job({"target": "chat"})
    server._run_unload(job)
    kind, data = job.out.get_nowait()
    assert (kind, data) == ("result", ["m"])


# ---------------------------------------------------------------- Handler wait-based handlers
class _FakeJobWithOut:
    """A pre-populated job.out, standing in for a job the worker already finished."""

    def __init__(self, kind, data):
        self.out = queue.Queue()
        self.out.put((kind, data))
        self.cancel = threading.Event()


def test_unload_handler_returns_500_on_worker_error(clean, monkeypatch):
    h = server.Handler.__new__(server.Handler)
    errors = []
    h._error = lambda code, msg, **kw: errors.append((code, msg))
    monkeypatch.setattr(server, "_mem", lambda: {})
    monkeypatch.setattr(
        server, "_submit", lambda prio, kind, payload: _FakeJobWithOut("error", "boom")
    )
    h._unload({"target": "chat"})
    assert errors == [(500, "boom")]


def test_clear_handler_returns_500_on_worker_error(clean, monkeypatch):
    h = server.Handler.__new__(server.Handler)
    errors = []
    h._error = lambda code, msg, **kw: errors.append((code, msg))
    monkeypatch.setattr(server, "_mem", lambda: {})
    monkeypatch.setattr(
        server, "_submit", lambda prio, kind, payload: _FakeJobWithOut("error", "boom")
    )
    h._clear({"target": "all"})
    assert errors == [(500, "boom")]
