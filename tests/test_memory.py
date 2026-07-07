"""Memory admission tests: proactive eviction BEFORE loading a model.

These exercise the budget logic (_make_room / _enforce_memory / _estimate_size_gb)
without loading real weights — _evict and _free_gb are stubbed.
"""

import pytest

from ember import server


@pytest.fixture
def clean(monkeypatch):
    """Empty registry; _evict just pops (no MLX); restore module state after."""
    monkeypatch.setattr(server, "_chat", {}, raising=True)
    monkeypatch.setattr(server, "_sizes", {}, raising=True)
    evicted = []

    def fake_evict(name):
        server._chat.pop(name, None)
        evicted.append(name)

    monkeypatch.setattr(server, "_evict", fake_evict)
    return evicted


def _put(name, size_gb, last):
    server._chat[name] = {
        "model": object(),
        "tok": object(),
        "size_gb": size_gb,
        "last": last,
        "ka": -1,
        "vlm": False,
        "slots": [{"pc": None, "pctoks": None, "last": 0.0}],
    }


def test_make_room_evicts_lru_before_a_big_model(clean, monkeypatch):
    """The reported bug: two ~12G models on 24G. Loading the 2nd must evict the
    1st BEFORE the load, not after."""
    monkeypatch.setattr(server, "MIN_FREE_GB", 2.0)
    monkeypatch.setattr(server, "MAX_RUNNERS", 4)
    _put("old", 12.0, last=1.0)  # resident, LRU
    monkeypatch.setattr(server, "_free_gb", lambda: 11.0)  # only 11G free now
    # incoming model ~12G -> 11 - 12 = -1 < MIN_FREE_GB -> must evict "old"
    server._make_room("new", est=12.0)
    assert clean == ["old"]
    assert "old" not in server._chat


def test_make_room_keeps_models_that_fit(clean, monkeypatch):
    monkeypatch.setattr(server, "MIN_FREE_GB", 2.0)
    monkeypatch.setattr(server, "MAX_RUNNERS", 4)
    _put("small", 2.0, last=1.0)
    monkeypatch.setattr(server, "_free_gb", lambda: 15.0)
    server._make_room("new", est=4.0)  # 15 - 4 = 11 >= 2 -> nothing evicted
    assert clean == []
    assert "small" in server._chat


def test_make_room_enforces_max_runners(clean, monkeypatch):
    """Even with plenty of RAM, the runner ceiling forces an eviction."""
    monkeypatch.setattr(server, "MIN_FREE_GB", 2.0)
    monkeypatch.setattr(server, "MAX_RUNNERS", 2)
    _put("a", 1.0, last=1.0)
    _put("b", 1.0, last=2.0)
    monkeypatch.setattr(server, "_free_gb", lambda: 100.0)  # RAM is fine
    server._make_room("c", est=1.0)  # 2 resident + 1 new = 3 > 2 -> evict LRU "a"
    assert clean == ["a"]


def test_make_room_evicts_lru_order(clean, monkeypatch):
    monkeypatch.setattr(server, "MIN_FREE_GB", 2.0)
    monkeypatch.setattr(server, "MAX_RUNNERS", 4)
    _put("oldest", 8.0, last=1.0)
    _put("newer", 8.0, last=5.0)
    monkeypatch.setattr(server, "_free_gb", lambda: 3.0)
    # need 12 + 2 = 14 free; have 3. evict oldest(+8=11), still short -> evict newer(+8=19)
    server._make_room("new", est=12.0)
    assert clean == ["oldest", "newer"]


def test_make_room_never_evicts_target(clean, monkeypatch):
    """If `name` is already resident (re-entry), it is never the victim."""
    monkeypatch.setattr(server, "MIN_FREE_GB", 2.0)
    monkeypatch.setattr(server, "MAX_RUNNERS", 4)
    _put("me", 12.0, last=1.0)
    monkeypatch.setattr(server, "_free_gb", lambda: 0.5)  # critically low
    server._make_room("me", est=12.0)  # only resident is the target -> nothing to evict
    assert clean == []
    assert "me" in server._chat


# ---------------------------------------------------------------- unconditional clear (#56)
def test_enforce_memory_clears_pool_even_with_nothing_to_evict(clean, monkeypatch):
    """A retry after a failed allocation must get a clean pool even when the eviction
    plan is empty (e.g. single hot model, nothing else to drop) — otherwise the retry
    can fail again for the same reason the first attempt did."""
    monkeypatch.setattr(server, "MIN_FREE_GB", 2.0)
    monkeypatch.setattr(server, "MAX_RUNNERS", 4)
    _put("me", 12.0, last=1.0)
    monkeypatch.setattr(server, "_free_gb", lambda: 0.5)  # critically low, no cache to relieve
    calls = []
    monkeypatch.setattr(server.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(server.mx, "clear_cache", lambda: calls.append("mx"))
    server._enforce_memory(keep="me")  # only resident is the target -> nothing to evict
    assert clean == []  # confirms the eviction plan was indeed empty
    assert calls == ["gc", "mx"]


def test_estimate_prefers_measured_size(clean):
    server._sizes["m"] = 7.5
    assert server._estimate_size_gb("m") == 7.5


def test_estimate_falls_back_to_largest_hot(clean, monkeypatch):
    """Unknown model, no disk weights -> use the largest hot model as the guess."""
    monkeypatch.setattr(server, "CFG", {}, raising=False)
    _put("big", 13.0, last=1.0)
    _put("small", 3.0, last=2.0)
    assert server._estimate_size_gb("unknown") == 13.0


def test_estimate_default_when_nothing_known(clean, monkeypatch):
    monkeypatch.setattr(server, "CFG", {}, raising=False)
    monkeypatch.setattr(server, "DEFAULT_EST_GB", 8.0)
    assert server._estimate_size_gb("unknown") == 8.0


# ---------------------------------------------------------------- sizes cache persistence (#32)
def test_save_sizes_then_load_round_trips(clean, monkeypatch, tmp_path):
    path = tmp_path / "sizes.json"
    monkeypatch.setattr(server, "SIZES_CACHE_PATH", str(path))
    server._sizes["big"] = 13.0
    server._sizes["small"] = 3.0
    server._save_sizes()
    assert server._load_sizes() == {"big": 13.0, "small": 3.0}


def test_load_sizes_missing_file_returns_empty(clean, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "SIZES_CACHE_PATH", str(tmp_path / "does-not-exist.json"))
    assert server._load_sizes() == {}


def test_load_sizes_disabled_returns_empty(clean, monkeypatch, tmp_path):
    path = tmp_path / "sizes.json"
    path.write_text('{"m": 1.0}')
    monkeypatch.setattr(server, "SIZES_CACHE_PATH", None)
    assert server._load_sizes() == {}


def test_load_sizes_corrupt_file_is_ignored(clean, monkeypatch, tmp_path):
    path = tmp_path / "sizes.json"
    path.write_text("not json")
    monkeypatch.setattr(server, "SIZES_CACHE_PATH", str(path))
    assert server._load_sizes() == {}


def test_save_sizes_disabled_is_a_noop(clean, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "SIZES_CACHE_PATH", None)
    server._sizes["m"] = 1.0
    server._save_sizes()  # must not raise, and must not create anything
    assert list(tmp_path.iterdir()) == []
