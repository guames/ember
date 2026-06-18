"""Tests for tool-call parsing and tool_choice (imports ember.server)."""

from ember.server import (
    _balanced_json,
    _calls_from_obj,
    _openai_tool_calls,
    _parse_tool_calls,
    _tool_prefill,
)


def test_hermes_block():
    calls, content = _parse_tool_calls(
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Lisbon"}}\n</tool_call>'
    )
    assert calls == [{"name": "get_weather", "arguments": {"city": "Lisbon"}}]
    assert content == ""


def test_text_before_call_is_kept():
    calls, content = _parse_tool_calls(
        'Let me check.<tool_call>{"name":"f","arguments":{"x":1}}</tool_call>'
    )
    assert calls == [{"name": "f", "arguments": {"x": 1}}]
    assert content == "Let me check."


def test_two_calls():
    calls, _ = _parse_tool_calls(
        '<tool_call>{"name":"a","arguments":{}}</tool_call>'
        '<tool_call>{"name":"b","arguments":{"y":2}}</tool_call>'
    )
    assert [c["name"] for c in calls] == ["a", "b"]


def test_fenced_and_raw_json():
    assert _parse_tool_calls('```json\n{"name":"calc","arguments":{"e":"2+2"}}\n```')[0]
    assert _parse_tool_calls('{"name":"ping","arguments":{}}')[0]


def test_dangling_tool_call_without_close():
    calls, _ = _parse_tool_calls('<tool_call>\n{"name":"f","arguments":{"x":1}}')
    assert calls == [{"name": "f", "arguments": {"x": 1}}]


def test_nested_function_and_string_args():
    assert _calls_from_obj(
        {"type": "function", "function": {"name": "g", "arguments": {"z": 3}}}
    ) == [{"name": "g", "arguments": {"z": 3}}]
    calls = _calls_from_obj({"name": "f", "arguments": '{"k":1}'})
    assert calls == [{"name": "f", "arguments": {"k": 1}}]


def test_no_tool_call_returns_text():
    calls, content = _parse_tool_calls("Hi, no tools here.")
    assert calls == []
    assert content == "Hi, no tools here."


def test_openai_format_arguments_is_string():
    oa = _openai_tool_calls([{"name": "f", "arguments": {"a": 1}}])
    assert oa[0]["type"] == "function"
    assert oa[0]["id"].startswith("call_")
    assert oa[0]["function"]["arguments"] == '{"a": 1}'  # JSON string


def test_balanced_json():
    assert _balanced_json('{"a":1}') == '{"a":1}'
    assert _balanced_json('junk {"a":{"b":2}} rest') == '{"a":{"b":2}}'
    assert _balanced_json('{"s":"}"}') == '{"s":"}"}'  # } inside a string
    assert _balanced_json("no json") is None


def test_tool_prefill():
    p = "...<tool_call>..."  # Hermes template
    assert _tool_prefill("auto", p) == ""
    assert _tool_prefill("none", p) == ""
    assert _tool_prefill("required", p).startswith("<tool_call>")
    named = _tool_prefill({"type": "function", "function": {"name": "go"}}, p)
    assert '"name": "go"' in named
    assert _tool_prefill("required", "no tag") == ""  # non-Hermes template
