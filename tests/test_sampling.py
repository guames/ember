"""Tests for stop sequences, keep_alive, cache prefix and multimodal payload."""

from ember.server import _error_obj, _extract_images, _normalize_messages, _parse_ka, _StopBuf


def test_stop_simple():
    b = _StopBuf(["END"])
    e1, h1 = b.push("hello ")
    e2, h2 = b.push("world END trailing")
    assert not h1 and h2
    assert e1 + e2 == "hello world "


def test_stop_split_across_tokens():
    """A stop that arrives across several tokens must not leak (hold-back)."""
    b = _StopBuf(["<|done|>"])
    out, hit = "", False
    for tok in ["hi", "<|", "do", "ne", "|>", "rest"]:
        e, h = b.push(tok)
        out += e
        if h:
            hit = True
            break
    assert hit
    assert out == "hi"
    assert "<|" not in out


def test_stop_absent_flush_returns_tail():
    b = _StopBuf(["ZZZ"])
    e, _ = b.push("normal text")
    assert e + b.flush() == "normal text"


def test_stop_split_across_many_short_tokens():
    """A match whose start falls inside the windowed region (not just the very
    last token) must still be caught — regression for the windowed scan
    (issue #54): the window covers `hold` chars back from `checked`, not just
    the newest push."""
    b = _StopBuf(["STOP"])
    out, hit = "", False
    for tok in ["a", "b", "c", "S", "T", "O", "P", "d"]:
        e, h = b.push(tok)
        out += e
        if h:
            hit = True
            break
    assert hit
    assert out == "abc"


def test_stop_scan_is_not_quadratic(monkeypatch):
    """Per-token scan must start near the tail of the accumulator (bounded by
    the stop length), not rescan from position 0 every time — otherwise a long
    generation pays O(n^2) (issue #54)."""
    b = _StopBuf(["never-matches-xyz"])  # len 17, so hold == 16
    starts = []
    orig_earliest_stop = _StopBuf.earliest_stop

    def recording_earliest_stop(text, stops, start=0):
        starts.append(start)
        return orig_earliest_stop(text, stops, start)

    monkeypatch.setattr(_StopBuf, "earliest_stop", staticmethod(recording_earliest_stop))
    for _ in range(2000):
        b.push("x")
    # after warmup, each scan's start position must sit within `hold` chars of
    # the accumulator's current length, not stay pinned at (or near) 0.
    for i, start in enumerate(starts[20:], start=20):
        assert start >= i - b.hold


def test_earliest_stop_used_by_tools_path():
    """The tools branch of _run_chat truncates its raw text buffer at the earliest
    stop sequence via this static helper (issue #26: stop was previously ignored
    whenever tools were present)."""
    assert _StopBuf.earliest_stop("hello END world", ["END"]) == 6
    assert _StopBuf.earliest_stop("no stop here", ["END"]) == -1
    # earliest of multiple candidates wins
    assert _StopBuf.earliest_stop("aaa STOP2 bbb STOP1", ["STOP1", "STOP2"]) == 4


def test_parse_ka():
    assert _parse_ka(None) is None
    assert _parse_ka(30) == 30.0
    assert _parse_ka("30s") == 30.0
    assert _parse_ka("5m") == 300.0
    assert _parse_ka("1h") == 3600.0


def test_extract_images():
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            ],
        }
    ]
    assert _extract_images(msgs) == ["data:image/png;base64,AAA"]
    assert _extract_images([{"role": "user", "content": "text only"}]) == []


def test_error_obj_is_openai_shaped():
    """issue #26: error responses must be {message, type, code}, not a bare string."""
    e = _error_obj("boom")
    assert e == {"message": "boom", "type": "internal_error", "code": None}
    e2 = _error_obj("bad request", err_type="invalid_request_error", err_code="not_found")
    assert e2 == {"message": "bad request", "type": "invalid_request_error", "code": "not_found"}


def test_normalize_messages_parses_tool_call_args():
    msgs = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "f", "arguments": '{"x":1}'}}],
        }
    ]
    out = _normalize_messages(msgs)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"x": 1}
    assert out[0]["content"] == ""
