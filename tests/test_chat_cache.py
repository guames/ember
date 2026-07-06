"""Chat prompt-cache tests: multi-slot KV reuse per runner (issue #21).

Mirrors tests/test_fim_cache.py's style — monkeypatch module globals, fake cache
objects (plain lists), no real weights or mlx_lm calls.
"""

import pytest

from ember import server


class FakeCache(list):
    """Stands in for the list mlx_lm.models.cache returns; trim just pops."""


def _runner(n_slots=2):
    return {
        "model": object(),
        "tok": object(),
        "size_gb": 1.0,
        "last": 0.0,
        "ka": -1,
        "vlm": False,
        "slots": [{"pc": None, "pctoks": None, "last": 0.0} for _ in range(n_slots)],
    }


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.setattr(server, "_chat", {"m": _runner(2)}, raising=True)
    monkeypatch.setattr(server, "PROMPT_CACHE", True)

    fresh_counter = iter(range(1000))

    def fake_make(model):
        return FakeCache([f"fresh{next(fresh_counter)}"])

    trims = []

    def fake_trim(cache, n):
        trims.append(n)
        del cache[max(0, len(cache) - n):]
        return n

    monkeypatch.setattr(server, "trim_prompt_cache", fake_trim)
    monkeypatch.setattr(server, "can_trim_prompt_cache", lambda cache: True)
    monkeypatch.setattr(server, "make_prompt_cache", fake_make)
    return trims


def test_reuse_cache_miss_when_pool_empty(clean):
    cache, suffix, reused, slot_idx = server._reuse_cache("m", model=object(), ptoks=[1, 2, 3])
    assert reused == 0
    assert suffix == [1, 2, 3]
    assert slot_idx == 0  # first empty slot


def test_reuse_cache_partial_hit_trims_divergent_suffix(clean):
    trims = clean
    server._chat["m"]["slots"][0] = {"pc": FakeCache(["a", "b", "c"]), "pctoks": [1, 2, 3], "last": 1.0}
    cache, suffix, reused, slot_idx = server._reuse_cache("m", model=object(), ptoks=[1, 2, 9, 9])
    assert (reused, suffix, slot_idx) == (2, [9, 9], 0)
    assert trims == [1]


def test_two_interleaved_conversations_each_keep_their_own_slot(clean):
    """The bug this issue fixes: two conversations on the same runner must not evict
    each other's cache every turn -- each gets its own slot in the pool."""
    convo_a = [1, 2, 3]
    convo_b = [7, 8, 9]

    cache_a, suffix_a, reused_a, slot_a = server._reuse_cache("m", object(), convo_a)
    server._store_cache("m", convo_a, cache_a, slot_a)
    cache_b, suffix_b, reused_b, slot_b = server._reuse_cache("m", object(), convo_b)
    server._store_cache("m", convo_b, cache_b, slot_b)

    assert reused_a == 0 and reused_b == 0
    assert slot_a != slot_b  # landed in different pool slots, not thrashing one

    # next turn on convo_a should hit its own slot, unaffected by convo_b's turn in between
    cache_a2, suffix_a2, reused_a2, slot_a2 = server._reuse_cache("m", object(), convo_a + [4])
    assert (reused_a2, suffix_a2, slot_a2) == (3, [4], slot_a)


def test_reuse_cache_pool_full_no_match_evicts_lru_slot(clean):
    server._chat["m"]["slots"][0] = {"pc": FakeCache(["a"]), "pctoks": [1, 1, 1], "last": 5.0}
    server._chat["m"]["slots"][1] = {"pc": FakeCache(["b"]), "pctoks": [2, 2, 2], "last": 1.0}
    cache, suffix, reused, slot_idx = server._reuse_cache("m", model=object(), ptoks=[9, 9, 9])
    assert reused == 0
    assert slot_idx == 1  # the LRU slot (last=1.0), not slot 0


def test_reuse_cache_disabled_always_fresh(clean, monkeypatch):
    monkeypatch.setattr(server, "PROMPT_CACHE", False)
    server._chat["m"]["slots"][0] = {"pc": FakeCache(["a", "b", "c"]), "pctoks": [1, 2, 3], "last": 1.0}
    cache, suffix, reused, slot_idx = server._reuse_cache("m", model=object(), ptoks=[1, 2, 3])
    assert reused == 0
    assert suffix == [1, 2, 3]


def test_store_cache_writes_chosen_slot(clean):
    fake = FakeCache(["x"])
    server._store_cache("m", [1, 2, 3], fake, 1)
    assert server._chat["m"]["slots"][1]["pc"] is fake
    assert server._chat["m"]["slots"][1]["pctoks"] == [1, 2, 3]
    assert server._chat["m"]["slots"][0]["pc"] is None  # untouched
