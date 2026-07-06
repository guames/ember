"""Tests for stop sequences, keep_alive, cache prefix and multimodal payload."""

from ember.server import _extract_images, _normalize_messages, _parse_ka, _StopBuf


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
