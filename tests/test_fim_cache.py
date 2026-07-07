"""FIM prompt-cache tests: KV reuse for the fixed autocomplete slot (_ac).

Mirrors tests/test_memory.py's style — monkeypatch module globals, fake cache
objects (plain lists), no real weights or mlx_lm calls.
"""

import pytest

from ember import server


class FakeCache(list):
    """Stands in for the list mlx_lm.models.cache returns; trim just pops."""


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.setattr(
        server, "_ac", {"model": object(), "tok": object(), "pc": None, "pctoks": None}
    )
    monkeypatch.setattr(server, "PROMPT_CACHE", True)

    trims = []

    def fake_trim(cache, n):
        trims.append(n)
        del cache[max(0, len(cache) - n) :]
        return n

    monkeypatch.setattr(server, "trim_prompt_cache", fake_trim)
    monkeypatch.setattr(server, "can_trim_prompt_cache", lambda cache: True)
    monkeypatch.setattr(server, "make_prompt_cache", lambda model: FakeCache(["fresh"]))
    return trims


def test_reuse_ac_cache_miss_when_empty(clean):
    cache, suffix, reused = server._reuse_ac_cache(model=object(), ptoks=[1, 2, 3])
    assert reused == 0
    assert suffix == [1, 2, 3]
    assert list(cache) == ["fresh"]


def test_reuse_ac_cache_partial_hit_trims_divergent_suffix(clean):
    trims = clean
    server._ac["pc"] = FakeCache(["a", "b", "c"])  # 3 KV entries, matching pctoks below
    server._ac["pctoks"] = [1, 2, 3]
    cache, suffix, reused = server._reuse_ac_cache(model=object(), ptoks=[1, 2, 9, 9])
    assert reused == 2
    assert suffix == [9, 9]
    assert trims == [1]  # trimmed the 1 divergent token ("c")
    assert list(cache) == ["a", "b"]


def test_reuse_ac_cache_prompt_equals_cached_prefix(clean):
    """prompt is an exact prefix of the cache -> trim(1) + regenerate last token."""
    trims = clean
    server._ac["pc"] = FakeCache(["a", "b", "c"])
    server._ac["pctoks"] = [1, 2, 3]
    cache, suffix, reused = server._reuse_ac_cache(model=object(), ptoks=[1, 2, 3])
    assert reused == 3
    assert suffix == [3]
    assert trims == [1]


def test_reuse_ac_cache_disabled_always_fresh(clean, monkeypatch):
    monkeypatch.setattr(server, "PROMPT_CACHE", False)
    server._ac["pc"] = FakeCache(["a", "b", "c"])
    server._ac["pctoks"] = [1, 2, 3]
    cache, suffix, reused = server._reuse_ac_cache(model=object(), ptoks=[1, 2, 3])
    assert reused == 0
    assert suffix == [1, 2, 3]
    assert list(cache) == ["fresh"]


def test_store_ac_cache_writes_slot(clean):
    fake = FakeCache(["x"])
    server._store_ac_cache([1, 2, 3], fake)
    assert server._ac["pc"] is fake
    assert server._ac["pctoks"] == [1, 2, 3]


def test_gen_fim_stops_on_marker_split_across_tokens(clean, monkeypatch):
    """The windowed scanner (issue #54) must still catch a marker that arrives
    split across multiple stream_generate tokens, and must stop consuming
    further tokens instead of rescanning/rebuilding the full text each time."""
    fake_tok = type(
        "FakeTok", (), {"encode": lambda self, s, add_special_tokens=False: [1, 2, 3]}
    )()
    monkeypatch.setattr(server, "ac_model", lambda: (object(), fake_tok))

    class R:
        def __init__(self, text, token):
            self.text = text
            self.token = token

    pieces = ["hel", "lo ", "<", "|", "endoftext", "|>", "SHOULD NOT APPEAR"]

    def fake_stream_generate(model, tok, arr, **kw):
        for i, p in enumerate(pieces):
            yield R(p, i)

    monkeypatch.setattr(server, "stream_generate", fake_stream_generate)
    text, usage = server.gen_fim({"prompt": "x", "suffix": "y"})
    assert "SHOULD NOT APPEAR" not in text
    assert usage["completion_tokens"] < len(pieces)
