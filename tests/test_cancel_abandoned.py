"""Cancel abandoned non-streaming/queued requests (issue #31).

Before this, `job.cancel` was only ever set from `_stream_out`'s `except BrokenPipeError`
(a *write*-triggered signal) -- so a non-streaming request whose client disconnected, or any
job (streaming or not) still sitting in the queue when its client gave up, ran to completion
regardless. `Handler._wait_out` replaces the plain blocking `job.out.get()` used by every
job-wait site with a short-timeout poll that also checks `Handler._client_gone` (a read-side
liveness probe via MSG_PEEK) and sets `job.cancel` once the connection is confirmed dead --
covering a job that hasn't been dispatched yet as well as one that's running. `_run_chat`/
`_run_fim`/`_run_embed` also bail out immediately if `job.cancel` is already set, so a
cancelled-while-queued job skips model load / generation entirely instead of only stopping at
the next per-token check.
"""

import io
import queue
import socket
import threading

import pytest

from ember import server


class _FakeJob:
    """Minimal stand-in for server.Job: payload + out + cancel, no queue/thread plumbing."""

    def __init__(self, payload):
        self.payload = payload
        self.out = queue.Queue()
        self.cancel = threading.Event()


# ---------------------------------------------------------------- Handler._client_gone
@pytest.fixture
def sockpair():
    a, b = socket.socketpair()
    a.setblocking(False)
    try:
        yield a, b
    finally:
        a.close()
        b.close()


def test_client_gone_false_while_connected_and_idle(sockpair):
    a, _b = sockpair
    h = server.Handler.__new__(server.Handler)
    h.connection = a
    assert h._client_gone() is False


def test_client_gone_true_after_peer_closes(sockpair):
    a, b = sockpair
    h = server.Handler.__new__(server.Handler)
    h.connection = a
    b.close()
    assert h._client_gone() is True


def test_client_gone_false_on_pipelined_data_and_leaves_it_for_later_read(sockpair):
    """Bytes waiting to be read (e.g. a pipelined next request under keep-alive) must read as
    'still connected', and MSG_PEEK must not consume them."""
    a, b = sockpair
    h = server.Handler.__new__(server.Handler)
    h.connection = a
    b.sendall(b"hello")
    assert h._client_gone() is False
    assert a.recv(5) == b"hello"  # still there -- peek didn't eat it


# ---------------------------------------------------------------- Handler._wait_out
def test_wait_out_returns_immediately_without_checking_liveness():
    h = server.Handler.__new__(server.Handler)

    def boom():
        raise AssertionError("_client_gone should not be consulted when a message is ready")

    h._client_gone = boom
    job = _FakeJob({})
    job.out.put(("delta", "hi"))
    assert h._wait_out(job) == ("delta", "hi")


def test_wait_out_cancels_and_returns_none_when_client_gone(monkeypatch):
    monkeypatch.setattr(server, "JOB_WAIT_POLL_S", 0.01)
    h = server.Handler.__new__(server.Handler)
    h._client_gone = lambda: True
    job = _FakeJob({})  # job.out never gets anything -- as if still queued
    assert h._wait_out(job) == (None, None)
    assert job.cancel.is_set()


def test_wait_out_keeps_polling_while_alive_then_returns_message(monkeypatch):
    monkeypatch.setattr(server, "JOB_WAIT_POLL_S", 0.01)
    h = server.Handler.__new__(server.Handler)
    h._client_gone = lambda: False
    job = _FakeJob({})

    def deliver_late():
        job.out.put(("done", {"prompt_tokens": 1}))

    threading.Timer(0.05, deliver_late).start()
    assert h._wait_out(job) == ("done", {"prompt_tokens": 1})
    assert not job.cancel.is_set()


# ---------------------------------------------------------------- job-wait call sites
def test_collect_out_returns_when_client_gone():
    h = server.Handler.__new__(server.Handler)
    h._wait_out = lambda job: (None, None)

    def boom(*a, **kw):
        raise AssertionError("no response should be sent once the client is gone")

    h._json = boom
    assert h._collect_out(_FakeJob({}), "cid", 0, "model") is None


def test_completions_returns_when_client_gone(monkeypatch):
    h = server.Handler.__new__(server.Handler)
    monkeypatch.setattr(server, "_submit", lambda prio, kind, payload: _FakeJob(payload))
    h._wait_out = lambda job: (None, None)

    def boom(*a, **kw):
        raise AssertionError("no response should be sent once the client is gone")

    h._json = boom
    assert h._completions({}) is None


def test_embeddings_returns_when_client_gone(monkeypatch):
    h = server.Handler.__new__(server.Handler)
    monkeypatch.setattr(server, "_submit", lambda prio, kind, payload: _FakeJob(payload))
    h._wait_out = lambda job: (None, None)

    def boom(*a, **kw):
        raise AssertionError("no response should be sent once the client is gone")

    h._json = boom
    assert h._embeddings({"input": "x"}) is None


def test_stream_out_returns_when_client_gone_before_first_token():
    h = server.Handler.__new__(server.Handler)
    h._wait_out = lambda job: (None, None)
    assert h._stream_out(_FakeJob({}), "cid", 0, "model", include_usage=False) is None


def test_stream_out_stops_mid_stream_when_client_gone():
    h = server.Handler.__new__(server.Handler)
    h.wfile = io.BytesIO()
    h.request_version = "HTTP/1.1"
    h.requestline = "POST /v1/chat/completions HTTP/1.1"

    # first event is consumed by _stream_out's pre-headers wait; the loop below sees the rest
    events = iter([("meta", None), ("delta", "hi"), (None, None)])
    h._wait_out = lambda job: next(events)

    result = h._stream_out(_FakeJob({}), "cid", 0, "model", include_usage=False)
    assert result is None
    raw = h.wfile.getvalue().decode()
    assert '"content": "hi"' in raw
    assert "[DONE]" not in raw  # gave up before the terminator -- nobody's listening


# ---------------------------------------------------------------- worker-side early bail
def test_run_chat_skips_already_cancelled_job(monkeypatch):
    def boom(name):
        raise AssertionError("chat_model() should not run for an already-cancelled job")

    monkeypatch.setattr(server, "chat_model", boom)
    job = _FakeJob({"name": "whatever", "body": {"messages": []}})
    job.cancel.set()
    server._run_chat(job)
    assert job.out.empty()


def test_run_fim_skips_already_cancelled_job(monkeypatch):
    def boom(body):
        raise AssertionError("gen_fim() should not run for an already-cancelled job")

    monkeypatch.setattr(server, "gen_fim", boom)
    job = _FakeJob({"body": {}})
    job.cancel.set()
    server._run_fim(job)
    assert job.out.empty()


def test_run_embed_skips_already_cancelled_job(monkeypatch):
    def boom(chunk):
        raise AssertionError("embeddings() should not run for an already-cancelled job")

    monkeypatch.setattr(server, "embeddings", boom)
    job = _FakeJob({"texts": ["a", "b"]})
    job.cancel.set()
    server._run_embed(job)
    assert job.out.empty()
