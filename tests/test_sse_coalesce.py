"""Tests for SSE delta coalescing in Handler._stream_out (issue #79, model-free).

At 80-150+ tok/s a write()+flush() (plus a json.dumps) per generated token is measurable
overhead and floods the socket with tiny TCP segments. _stream_out now buffers "delta" text
and flushes it to the wire every ~SSE_COALESCE_S or once SSE_COALESCE_CHARS has piled up,
whichever comes first, while tool-call/done/error events still flush immediately (never
delayed behind buffered text) and job.out itself still gets a put per token upstream (so
cancellation granularity is unchanged). These tests drive Handler._stream_out directly against
a fake job (same pattern as test_http_keepalive.py's test_stream_out_forces_connection_close)
so no real model or socket is involved.
"""

import io
import json
import queue
import threading

import pytest

from ember import server


class _FakeJob:
    """_stream_out consumes one leading job.out event before it commits to sending HTTP
    headers (used to detect cancellation/error before anything is generated) -- in production
    that's always the ("meta", name) event _run_chat puts first. So every fake job here gets
    that same leading event automatically; `events` is what arrives *after* it."""

    def __init__(self, events):
        self.out = queue.Queue()
        self.out.put(("meta", "qwen2.5-coder-1.5b"))
        for e in events:
            self.out.put(e)
        self.cancel = threading.Event()


def _run_stream(job, include_usage=False):
    h = server.Handler.__new__(server.Handler)
    h.wfile = io.BytesIO()
    h.request_version = "HTTP/1.1"
    h.requestline = "POST /v1/chat/completions HTTP/1.1"
    h._stream_out(job, "chatcmpl-test", 0, "qwen2.5-coder-1.5b", include_usage=include_usage)
    return h.wfile.getvalue().decode()


def _frames(raw):
    """SSE body -> list of parsed JSON payloads, in order, dropping the terminal [DONE]."""
    body = raw.split("\r\n\r\n", 1)[1]
    out = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if not chunk or chunk == "data: [DONE]":
            continue
        assert chunk.startswith("data: ")
        out.append(json.loads(chunk[len("data: ") :]))
    return out


def _contents(frames):
    """Pull delta.content out of each chunk-completion frame that carries one."""
    out = []
    for f in frames:
        for c in f.get("choices", []):
            content = c.get("delta", {}).get("content")
            if content:
                out.append(content)
    return out


def test_fast_deltas_coalesce_into_one_frame(monkeypatch):
    """Back-to-back tokens arriving well within the coalesce window must land in a single SSE
    write, not one write per token."""
    monkeypatch.setattr(server, "SSE_COALESCE_S", 999)  # never trip the time-based flush
    monkeypatch.setattr(server, "SSE_COALESCE_CHARS", 10_000)  # never trip the size-based flush

    job = _FakeJob(
        [("delta", "Hel"), ("delta", "lo"), ("delta", ", "), ("delta", "world"), ("done", {})]
    )
    raw = _run_stream(job)
    frames = _frames(raw)

    # frame 0 is the opening role delta; frame 1 must be the single coalesced content chunk.
    assert _contents(frames) == ["Hello, world"]


def test_char_threshold_flushes_before_time_would(monkeypatch):
    """Once the buffered text crosses SSE_COALESCE_CHARS, it flushes right away even though the
    (disabled) time-based threshold hasn't fired."""
    monkeypatch.setattr(server, "SSE_COALESCE_S", 999)
    monkeypatch.setattr(server, "SSE_COALESCE_CHARS", 5)

    job = _FakeJob(
        [("delta", "ab"), ("delta", "cd"), ("delta", "ef"), ("delta", "gh"), ("done", {})]
    )
    raw = _run_stream(job)
    frames = _frames(raw)

    # "ab"+"cd"+"ef" crosses the 5-char threshold on "ef" -> flush "abcdef"; "gh" flushes at done.
    assert _contents(frames) == ["abcdef", "gh"]


def test_zero_time_threshold_flushes_every_delta(monkeypatch):
    """SSE_COALESCE_S=0 means "elapsed since last flush" is always past the threshold, so each
    delta is sent on its own -- the timing branch degrades to the old per-token behavior."""
    monkeypatch.setattr(server, "SSE_COALESCE_S", 0)
    monkeypatch.setattr(server, "SSE_COALESCE_CHARS", 10_000)

    job = _FakeJob([("delta", "a"), ("delta", "b"), ("delta", "c"), ("done", {})])
    raw = _run_stream(job)
    frames = _frames(raw)

    assert _contents(frames) == ["a", "b", "c"]


def test_toolcalls_flush_buffered_text_immediately(monkeypatch):
    """A buffered delta must not sit behind a tool_calls event: it flushes first, in order,
    even though the coalescing thresholds haven't been crossed."""
    monkeypatch.setattr(server, "SSE_COALESCE_S", 999)
    monkeypatch.setattr(server, "SSE_COALESCE_CHARS", 10_000)

    calls = [{"name": "f", "arguments": "{}"}]  # _openai_tool_calls' input shape
    job = _FakeJob(
        [
            ("delta", "he"),
            ("toolcalls", (calls, "llo")),
            ("done", {}),
        ]
    )
    raw = _run_stream(job)
    frames = _frames(raw)

    contents = _contents(frames)
    assert contents == ["he", "llo"]  # buffered "he" flushed before the toolcalls' own content
    tool_call_frames = [f for f in frames if "tool_calls" in f["choices"][0]["delta"]]
    assert len(tool_call_frames) == 1


def test_error_flushes_buffered_text_immediately(monkeypatch):
    """A buffered delta must not sit behind an error event either."""
    monkeypatch.setattr(server, "SSE_COALESCE_S", 999)
    monkeypatch.setattr(server, "SSE_COALESCE_CHARS", 10_000)

    job = _FakeJob([("delta", "partial"), ("error", "boom")])
    raw = _run_stream(job)
    frames = _frames(raw)

    assert _contents(frames) == ["partial"]
    error_frames = [f for f in frames if "error" in f]
    assert len(error_frames) == 1
    assert error_frames[0]["error"]["message"] == "boom"


def test_empty_delta_strings_are_not_flushed(monkeypatch):
    """Falsy delta payloads (e.g. "") must be ignored, same as before coalescing existed."""
    monkeypatch.setattr(server, "SSE_COALESCE_S", 999)
    monkeypatch.setattr(server, "SSE_COALESCE_CHARS", 10_000)

    job = _FakeJob([("delta", ""), ("delta", "hi"), ("delta", ""), ("done", {})])
    raw = _run_stream(job)
    frames = _frames(raw)

    assert _contents(frames) == ["hi"]


@pytest.mark.parametrize("include_usage", [False, True])
def test_done_still_reports_finish_reason_and_usage(monkeypatch, include_usage):
    """Coalescing must not disturb the existing done/[DONE] contract."""
    monkeypatch.setattr(server, "SSE_COALESCE_S", 999)
    monkeypatch.setattr(server, "SSE_COALESCE_CHARS", 10_000)

    job = _FakeJob(
        [
            ("delta", "hi"),
            ("done", {"prompt_tokens": 3, "completion_tokens": 1, "cached_tokens": 0}),
        ]
    )
    raw = _run_stream(job, include_usage=include_usage)
    frames = _frames(raw)

    finish_frames = [f for f in frames if f["choices"] and f["choices"][0].get("finish_reason")]
    assert len(finish_frames) == 1
    assert finish_frames[0]["choices"][0]["finish_reason"] == "stop"
    assert raw.rstrip().endswith("data: [DONE]")

    usage_frames = [f for f in frames if "usage" in f]
    assert len(usage_frames) == (1 if include_usage else 0)
