"""Pure unit tests for the shared memory-policy planners.

No server, no MLX, no monkeypatch — these call `memory_policy` directly with plain dicts,
which is the whole point of extracting the decision logic from the effectful server.
"""

from ember import memory_policy as mp


def _models(*rows):
    """rows of (name, size_gb, last) -> snapshot dict the planners expect."""
    return {name: {"last": last, "size_gb": size} for name, size, last in rows}


# ---------------------------------------------------------------- estimate_size_gb
def test_estimate_prefers_measured():
    assert mp.estimate_size_gb(7.5, 9.0, [13.0], 8.0) == 7.5


def test_estimate_falls_back_to_disk():
    assert mp.estimate_size_gb(None, 9.0, [13.0], 8.0) == 9.0


def test_estimate_falls_back_to_largest_hot():
    assert mp.estimate_size_gb(None, None, [13.0, 3.0], 8.0) == 13.0


def test_estimate_default_when_nothing_known():
    assert mp.estimate_size_gb(None, None, [], 8.0) == 8.0


# ---------------------------------------------------------------- plan_make_room
def test_make_room_evicts_lru_before_a_big_model():
    """Two ~12G models on 24G: loading the 2nd must evict the 1st before the load."""
    models = _models(("old", 12.0, 1.0))
    assert mp.plan_make_room("new", 12.0, 11.0, models, 2.0, 4) == ["old"]


def test_make_room_keeps_models_that_fit():
    models = _models(("small", 2.0, 1.0))
    assert mp.plan_make_room("new", 4.0, 15.0, models, 2.0, 4) == []


def test_make_room_enforces_max_runners():
    """Even with plenty of RAM, the runner ceiling forces an eviction."""
    models = _models(("a", 1.0, 1.0), ("b", 1.0, 2.0))
    assert mp.plan_make_room("c", 1.0, 100.0, models, 2.0, 2) == ["a"]


def test_make_room_evicts_in_lru_order_until_it_fits():
    models = _models(("oldest", 8.0, 1.0), ("newer", 8.0, 5.0))
    # need 12 + 2 = 14 free; have 3. evict oldest(+8=11) still short -> newer(+8=19) ok
    assert mp.plan_make_room("new", 12.0, 3.0, models, 2.0, 4) == ["oldest", "newer"]


def test_make_room_never_evicts_target():
    """If the target is already resident (re-entry) it is never the victim."""
    models = _models(("me", 12.0, 1.0))
    assert mp.plan_make_room("me", 12.0, 0.5, models, 2.0, 4) == []


def test_make_room_free_none_drives_only_by_runner_count():
    """When free RAM can't be measured, only the runner ceiling can force evictions."""
    models = _models(("a", 5.0, 1.0), ("b", 5.0, 2.0))
    assert mp.plan_make_room("c", 5.0, None, models, 2.0, 4) == []  # 3 <= 4, no RAM signal
    assert mp.plan_make_room("c", 5.0, None, models, 2.0, 2) == ["a"]  # 3 > 2 -> LRU


# ---------------------------------------------------------------- plan_enforce
def test_enforce_keeps_at_least_one():
    models = _models(("only", 12.0, 1.0))
    assert mp.plan_enforce("only", 0.1, models, 2.0, 4) == []


def test_enforce_over_by_runner_count():
    models = _models(("a", 1.0, 1.0), ("b", 1.0, 2.0), ("keep", 1.0, 3.0))
    assert mp.plan_enforce("keep", 100.0, models, 2.0, 2) == ["a"]  # 3 > 2 -> drop LRU


def test_enforce_over_by_ram_evicts_until_ok():
    models = _models(("a", 3.0, 1.0), ("b", 3.0, 2.0), ("keep", 3.0, 3.0))
    # free 1.0 < 2.0: evict a(+3=4) -> ok, b spared
    assert mp.plan_enforce("keep", 1.0, models, 2.0, 4) == ["a"]


def test_enforce_never_evicts_keep():
    models = _models(("keep", 3.0, 1.0), ("other", 3.0, 2.0))
    victims = mp.plan_enforce("keep", 0.0, models, 2.0, 4)
    assert "keep" not in victims and victims == ["other"]


# ---------------------------------------------------------------- order_cache_relief
def _cache_models(*rows):
    return {name: {"last": last, "has_cache": hc} for name, hc, last in rows}


def test_cache_relief_lru_first_keep_last():
    models = _cache_models(("a", True, 1.0), ("b", True, 5.0), ("keep", True, 3.0))
    assert mp.order_cache_relief("keep", models) == ["a", "b", "keep"]


def test_cache_relief_skips_models_without_cache():
    models = _cache_models(("a", False, 1.0), ("b", True, 2.0), ("keep", False, 3.0))
    assert mp.order_cache_relief("keep", models) == ["b"]


# ---------------------------------------------------------------- is_oom_error
def test_is_oom_error_matches_metal_buffer_size_message():
    msg = "[metal::malloc] Attempting to allocate 160000000000 bytes which is greater than..."
    assert mp.is_oom_error(msg)


def test_is_oom_error_matches_metal_resource_limit_message():
    assert mp.is_oom_error("[metal::malloc] Resource limit (100.0 GB) exceeded.")


def test_is_oom_error_matches_cpu_malloc_message():
    assert mp.is_oom_error("[malloc] Unable to allocate 4294967296 bytes.")


def test_is_oom_error_rejects_unrelated_errors():
    assert not mp.is_oom_error("connection reset by peer")
    assert not mp.is_oom_error("KeyError: 'qwen3-8b'")


def test_is_oom_error_handles_empty_message():
    assert not mp.is_oom_error("")
    assert not mp.is_oom_error(None)


# ---------------------------------------------------------------- scale_defaults
def test_scale_defaults_small_ram_is_conservative():
    d = mp.scale_defaults(8.0)
    assert d["max_runners"] == 1
    assert d["min_free_gb"] < 2.0
    assert d["wired_headroom_gb"] < 5.0


def test_scale_defaults_matches_historical_24gb_values():
    """The dev box's 24GB defaults must not change: min_free=2.0, est=8.0, runners=4,
    wired_headroom=5.0 (i.e. wired_limit = total-5)."""
    d = mp.scale_defaults(24.0)
    assert d == {
        "min_free_gb": 2.0,
        "default_est_gb": 8.0,
        "max_runners": 4,
        "wired_headroom_gb": 5.0,
    }


def test_scale_defaults_large_ram_uses_more_headroom_and_runners():
    d = mp.scale_defaults(64.0)
    assert d["max_runners"] > 4
    assert d["min_free_gb"] > 2.0


def test_scale_defaults_xlarge_ram_scales_further():
    small, large = mp.scale_defaults(64.0), mp.scale_defaults(128.0)
    assert large["max_runners"] > small["max_runners"]
    assert large["min_free_gb"] > small["min_free_gb"]
    assert large["wired_headroom_gb"] > small["wired_headroom_gb"]


def test_scale_defaults_boundaries_are_inclusive_on_lower_tier():
    assert mp.scale_defaults(10.0)["max_runners"] == 1
    assert mp.scale_defaults(40.0)["max_runners"] == 4
    assert mp.scale_defaults(80.0)["max_runners"] == 6


# ---------------------------------------------------------------- common_prefix
def test_common_prefix():
    assert mp.common_prefix([1, 2, 3], [1, 2, 9]) == 2
    assert mp.common_prefix([1, 2], [1, 2, 3]) == 2
    assert mp.common_prefix([], [1]) == 0


# ---------------------------------------------------------------- select_prompt_cache_slot
def _slots(*rows):
    """rows of (tokens_or_None, last) -> the slot-pool snapshot the planner expects."""
    return [{"tokens": toks, "last": last} for toks, last in rows]


def test_select_slot_empty_pool_writes_first_empty():
    slots = _slots((None, 0.0), (None, 0.0))
    assert mp.select_prompt_cache_slot(slots, [1, 2, 3]) == (None, 0, 0)


def test_select_slot_picks_longest_common_prefix():
    slots = _slots(([1, 2, 9, 9], 1.0), ([1, 2, 3, 4], 2.0))
    # slot 1 shares a longer prefix ([1,2,3]) with the incoming prompt than slot 0 ([1,2])
    match_idx, common_len, write_idx = mp.select_prompt_cache_slot(slots, [1, 2, 3, 9])
    assert (match_idx, common_len, write_idx) == (1, 3, 1)


def test_select_slot_ties_break_to_most_recently_used():
    slots = _slots(([1, 2, 3], 1.0), ([1, 2, 3], 5.0))
    match_idx, common_len, write_idx = mp.select_prompt_cache_slot(slots, [1, 2, 3, 9])
    assert (match_idx, common_len, write_idx) == (1, 3, 1)


def test_select_slot_no_overlap_uses_first_empty_when_pool_has_room():
    slots = _slots(([9, 9, 9], 1.0), (None, 0.0))
    assert mp.select_prompt_cache_slot(slots, [1, 2, 3]) == (None, 0, 1)


def test_select_slot_no_overlap_and_pool_full_evicts_lru():
    slots = _slots(([9, 9, 9], 5.0), ([8, 8, 8], 1.0))
    match_idx, common_len, write_idx = mp.select_prompt_cache_slot(slots, [1, 2, 3])
    assert (match_idx, common_len, write_idx) == (None, 0, 1)  # slot 1 is the LRU


def test_select_slot_single_slot_pool_matches_old_single_slot_behavior():
    slots = _slots(([1, 2, 3], 1.0))
    assert mp.select_prompt_cache_slot(slots, [1, 2, 9]) == (0, 2, 0)
    assert mp.select_prompt_cache_slot(slots, [9, 9, 9]) == (None, 0, 0)
