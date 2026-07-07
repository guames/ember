"""Structured metrics tests (issue #28): JSONL request log + GET /metrics.

Model-free: _record_metrics/_metrics_text are pure (plus disk I/O for the log), and the
_run_fim/_run_embed/_run_chat hook points are exercised with monkeypatched model calls,
mirroring tests/test_fim_cache.py's and tests/test_chat_cache.py's style.
"""

import builtins
import http.client
import json
import threading

import pytest

from ember import server


@pytest.fixture(autouse=True)
def clean_metrics(monkeypatch, tmp_path):
    """Every test gets an empty in-memory counter set, its own JSONL path, and a fresh
    (closed) persistent log handle -- the handle is a module global tied to whatever path
    was open when it was created, so it must not leak across tests that swap the path."""
    monkeypatch.setattr(server, "_metrics", {})
    monkeypatch.setattr(server, "METRICS_LOG_PATH", str(tmp_path / "metrics.jsonl"))
    monkeypatch.setattr(server, "METRICS_LOG_MAX_MB", 64.0)
    monkeypatch.setattr(server, "METRICS_LOG_MAX_BYTES", 64 * 1024 * 1024)
    monkeypatch.setattr(server, "_metrics_log_fh", None)
    return tmp_path


def _lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------- _record_metrics
def test_record_metrics_writes_jsonl_and_updates_counters(clean_metrics):
    server._record_metrics(
        "chat", "qwen3-8b", 0.25, prompt_tokens=10, completion_tokens=5, cached_tokens=3
    )
    entries = _lines(clean_metrics / "metrics.jsonl")
    assert len(entries) == 1
    e = entries[0]
    assert e["endpoint"] == "chat"
    assert e["model"] == "qwen3-8b"
    assert e["status"] == "ok"
    assert e["latency_ms"] == 250.0
    assert e["prompt_tokens"] == 10
    assert e["completion_tokens"] == 5
    assert e["cached_tokens"] == 3
    assert "error" not in e

    m = server._metrics[("chat", "qwen3-8b", "ok")]
    assert m["count"] == 1
    assert m["prompt_tokens"] == 10
    assert m["completion_tokens"] == 5
    assert m["cached_tokens"] == 3
    assert m["latency_sum"] == pytest.approx(0.25)


def test_record_metrics_error_status_separate_key(clean_metrics):
    server._record_metrics("chat", "qwen3-8b", 0.1, error="boom")
    entries = _lines(clean_metrics / "metrics.jsonl")
    assert entries[0]["status"] == "error"
    assert entries[0]["error"] == "boom"
    assert ("chat", "qwen3-8b", "error") in server._metrics
    assert ("chat", "qwen3-8b", "ok") not in server._metrics


def test_record_metrics_accumulates_across_calls(clean_metrics):
    server._record_metrics("embed", "embed", 0.1, prompt_tokens=4)
    server._record_metrics("embed", "embed", 0.2, prompt_tokens=6)
    m = server._metrics[("embed", "embed", "ok")]
    assert m["count"] == 2
    assert m["prompt_tokens"] == 10
    assert m["latency_sum"] == pytest.approx(0.3)


def test_record_metrics_disabled_log_path_still_updates_counters(clean_metrics, monkeypatch):
    monkeypatch.setattr(server, "METRICS_LOG_PATH", None)
    server._record_metrics("fim", server.AC_NAME, 0.05, prompt_tokens=1, completion_tokens=1)
    assert not (clean_metrics / "metrics.jsonl").exists()
    assert server._metrics[("fim", server.AC_NAME, "ok")]["count"] == 1


def test_record_metrics_bad_log_path_does_not_raise(clean_metrics, monkeypatch, capsys):
    monkeypatch.setattr(server, "METRICS_LOG_PATH", "/nonexistent-root-dir/x/metrics.jsonl")
    server._record_metrics("chat", "m", 0.1)  # should not raise
    assert server._metrics[("chat", "m", "ok")]["count"] == 1


# ---------------------------------------------------------------- log rotation (issue #57)
def test_metrics_log_reuses_one_handle_across_writes(clean_metrics, monkeypatch):
    """_record_metrics used to os.makedirs + open + close on every call; now it keeps one
    handle open behind _metrics_lock and only opens once."""
    opened = []
    real_open = builtins.open

    def spy_open(path, mode):
        opened.append(path)
        return real_open(path, mode)

    monkeypatch.setattr(server, "open", spy_open, raising=False)
    server._record_metrics("chat", "m", 0.1)
    server._record_metrics("chat", "m", 0.1)
    server._record_metrics("chat", "m", 0.1)
    assert len(opened) == 1
    assert len(_lines(clean_metrics / "metrics.jsonl")) == 3


def test_metrics_log_does_not_rotate_under_cap(clean_metrics):
    for i in range(5):
        server._record_metrics("chat", "m", 0.1, prompt_tokens=i)
    assert len(_lines(clean_metrics / "metrics.jsonl")) == 5
    assert not (clean_metrics / "metrics.jsonl.1").exists()


def test_metrics_log_rotates_past_cap_and_keeps_one_generation(clean_metrics, monkeypatch):
    """Once a write pushes the file past METRICS_LOG_MAX_BYTES, the whole file (including
    the line that just crossed the cap) is renamed to `.1` and the next write starts a
    fresh file -- so disk usage stays bounded at ~2 generations, not O(requests)."""
    monkeypatch.setattr(server, "METRICS_LOG_MAX_BYTES", 1)  # any single line already exceeds it
    current = clean_metrics / "metrics.jsonl"
    rotated = clean_metrics / "metrics.jsonl.1"

    for i in range(3):
        server._record_metrics("chat", "m", 0.1, prompt_tokens=i)
        # every write immediately crosses the 1-byte cap, so it's rotated out right away
        assert not current.exists()
        assert len(_lines(rotated)) == 1
        assert _lines(rotated)[0]["prompt_tokens"] == i

    # only the most recent write survives in .1 -- older generations aren't kept around
    assert _lines(rotated)[0]["prompt_tokens"] == 2


def test_metrics_log_reopens_after_handle_error(clean_metrics, monkeypatch):
    """If a write on the cached handle fails, the handle is dropped and the next call
    reopens (and succeeds) instead of silently staying broken forever."""
    real_open = builtins.open
    state = {"n": 0}

    class _BoomOnce:
        def write(self, data):
            state["n"] += 1
            if state["n"] == 1:
                raise OSError("disk full")

        def flush(self):
            pass

        def tell(self):
            return 0

        def close(self):
            pass

    def spy_open(path, mode):
        if state["n"] == 0:
            return _BoomOnce()
        return real_open(path, mode)

    monkeypatch.setattr(server, "open", spy_open, raising=False)
    server._record_metrics("chat", "m", 0.1)  # write fails, handle dropped
    server._record_metrics("chat", "m", 0.1)  # reopens for real, succeeds
    assert len(_lines(clean_metrics / "metrics.jsonl")) == 1
    assert server._metrics[("chat", "m", "ok")]["count"] == 2  # counters unaffected either way


# ---------------------------------------------------------------- _metrics_text
def test_metrics_text_prometheus_shape_and_bucket_counts(clean_metrics):
    server._record_metrics("chat", "m", 0.05, prompt_tokens=1, completion_tokens=1)
    server._record_metrics("chat", "m", 5.0, prompt_tokens=2, completion_tokens=2)
    text = server._metrics_text()
    assert "# TYPE ember_requests_total counter" in text
    assert "# TYPE ember_request_latency_seconds histogram" in text
    assert 'ember_requests_total{endpoint="chat",model="m",status="ok"} 2' in text
    assert 'ember_request_latency_seconds_count{endpoint="chat",model="m",status="ok"} 2' in text
    assert 'ember_prompt_tokens_total{endpoint="chat",model="m",status="ok"} 3' in text
    assert 'ember_completion_tokens_total{endpoint="chat",model="m",status="ok"} 3' in text

    # Cumulative histogram: the 0.05s sample lands in every bucket up to +Inf, the 5.0s
    # sample only from le=5 upward. Values must be non-decreasing as le grows, and +Inf
    # must equal the total count -- this locks the cumulative semantics (regression: an
    # earlier version double-accumulated and produced values *larger* than the count).
    expected = {0.1: 1, 0.25: 1, 0.5: 1, 1: 1, 2.5: 1, 5: 2, 10: 2, 30: 2, 60: 2, 120: 2}
    prev = 0
    for le, want in expected.items():
        line = f'ember_request_latency_seconds_bucket{{endpoint="chat",model="m",status="ok",le="{le}"}} {want}'
        assert line in text
        assert want >= prev
        assert want <= 2
        prev = want
    assert (
        'ember_request_latency_seconds_bucket{endpoint="chat",model="m",status="ok",le="+Inf"} 2'
        in text
    )


def test_metrics_text_empty_when_no_requests(clean_metrics):
    text = server._metrics_text()
    assert "ember_requests_total" not in text.split("\n")[-2]  # only HELP/TYPE header lines
    assert "# TYPE ember_requests_total counter" in text


# ---------------------------------------------------------------- GET /metrics (live)
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


def test_metrics_endpoint_returns_prometheus_text(live_server, clean_metrics):
    server._record_metrics("chat", "m", 0.2, prompt_tokens=1, completion_tokens=1)
    host, port = live_server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        conn.request("GET", "/metrics")
        r = conn.getresponse()
        body = r.read().decode()
        assert r.status == 200
        assert "text/plain" in r.getheader("Content-Type")
        assert 'ember_requests_total{endpoint="chat",model="m",status="ok"} 1' in body
    finally:
        conn.close()


def test_metrics_endpoint_requires_bearer_token_when_api_key_set(
    live_server, clean_metrics, monkeypatch
):
    """/metrics leaks model names/traffic counters, so it's guarded like /v1/* once an
    API key is set (issue #52)."""
    monkeypatch.setattr(server, "API_KEY", "secret123")
    host, port = live_server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        conn.request("GET", "/metrics")
        r = conn.getresponse()
        r.read()
        assert r.status == 401

        conn.request("GET", "/metrics", headers={"Authorization": "Bearer secret123"})
        r = conn.getresponse()
        r.read()
        assert r.status == 200
    finally:
        conn.close()


# ---------------------------------------------------------------- hook points
class _FakeJob:
    def __init__(self, payload):
        self.payload = payload
        self.out = []
        self.cancel = threading.Event()

    def put(self, item):
        self.out.append(item)


def test_run_fim_records_metrics_on_success(clean_metrics, monkeypatch):
    usage = {"prompt_tokens": 7, "completion_tokens": 3, "cached_tokens": 2}
    monkeypatch.setattr(server, "gen_fim", lambda body: ("hello", usage))
    job = _FakeJob({"body": {"model": server.AC_NAME, "prompt": "x"}})
    job.out = server.queue.Queue()
    server._run_fim(job)
    kind, data = job.out.get_nowait()
    assert kind == "result"
    m = server._metrics[("fim", server.AC_NAME, "ok")]
    assert m["count"] == 1
    assert m["prompt_tokens"] == 7
    assert m["completion_tokens"] == 3
    assert m["cached_tokens"] == 2


def test_run_fim_records_metrics_on_error(clean_metrics, monkeypatch):
    def boom(body):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(server, "gen_fim", boom)
    job = _FakeJob({"body": {}})
    job.out = server.queue.Queue()
    server._run_fim(job)
    kind, data = job.out.get_nowait()
    assert kind == "error"
    m = server._metrics[("fim", server.AC_NAME, "error")]
    assert m["count"] == 1


def test_run_embed_records_metrics_once_after_all_chunks(clean_metrics, monkeypatch):
    monkeypatch.setattr(server, "EMBED_CHUNK", 2)
    calls = []

    def fake_embeddings(chunk):
        calls.append(list(chunk))
        return [[0.0] for _ in chunk], len(chunk)

    monkeypatch.setattr(server, "embeddings", fake_embeddings)
    job = _FakeJob({"texts": ["a", "b", "c", "d", "e"]})
    job.out = server.queue.Queue()
    # _run_embed re-queues itself via _q when a batch spans multiple chunks; drive it directly
    # by looping until it produces a result, like the real worker loop would.
    while job.out.empty():
        server._run_embed(job)
        if not job.out.empty():
            break
        # pull the job back off _q (it re-queued itself for the next chunk)
        _, _, requeued = server._q.get_nowait()
        job = requeued
    kind, data = job.out.get_nowait()
    assert kind == "result"
    vecs, tokens = data
    assert len(vecs) == 5
    assert tokens == 5
    m = server._metrics[("embed", server.EM_NAME, "ok")]
    assert m["count"] == 1  # one record despite 3 chunks (2+2+1)
    assert m["prompt_tokens"] == 5


def test_run_chat_records_error_metric_for_non_vision_image(clean_metrics):
    name = next(iter(server.CFG))  # any configured (non-vision) chat model
    assert not server.CFG[name].get("vision")
    job = _FakeJob(
        {
            "name": name,
            "body": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "data:,x"}}],
                    }
                ]
            },
        }
    )
    job.out = server.queue.Queue()
    server._run_chat(job)
    kind, data = job.out.get_nowait()
    assert kind == "error"
    m = server._metrics[("chat", name, "error")]
    assert m["count"] == 1


class _FakeTok:
    """No chat_template -> _fmt_chat falls back to a plain join; encode returns 3 fixed ids."""

    def encode(self, s, add_special_tokens=False):
        return [1, 2, 3]


def test_run_chat_records_error_metric_for_num_ctx_rejection(clean_metrics, monkeypatch):
    """The num_ctx-overflow rejection in _run_chat returned without ever calling
    _record_metrics (issue #57), so context-overflow errors were invisible in /metrics and
    the JSONL log."""
    name = next(iter(server.CFG))
    monkeypatch.setattr(
        server,
        "_chat",
        {
            name: {
                "model": object(),
                "tok": _FakeTok(),
                "size_gb": 1.0,
                "last": 0.0,
                "ka": -1,
                "vlm": False,
                "slots": [{"pc": None, "pctoks": None, "last": 0.0}],
            }
        },
    )
    monkeypatch.setitem(server.CFG[name], "params", {"num_ctx": 2})  # _FakeTok.encode -> 3 ids
    job = _FakeJob({"name": name, "body": {"messages": [{"role": "user", "content": "hi"}]}})
    job.out = server.queue.Queue()
    server._run_chat(job)
    kinds = []
    while not job.out.empty():
        kind, data = job.out.get_nowait()
        kinds.append((kind, data))
    assert kinds[-1][0] == "error"
    assert "num_ctx" in kinds[-1][1]
    entries = _lines(clean_metrics / "metrics.jsonl")
    assert len(entries) == 1
    assert entries[0]["endpoint"] == "chat"
    assert entries[0]["model"] == name
    assert entries[0]["status"] == "error"
    m = server._metrics[("chat", name, "error")]
    assert m["count"] == 1
