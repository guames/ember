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
