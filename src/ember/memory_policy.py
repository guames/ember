"""Pure memory-admission policy — the *decision* half of the router's RAM management.

These functions contain no side effects and import nothing heavy (no mlx, no psutil):
given a snapshot of the hot models and the current free RAM, they return *which* models
to evict (or *how big* an incoming model is). The effectful half — reading free RAM,
actually evicting, printing — stays in the server, which snapshots its state, calls a
planner here, and acts on the result.

Splitting it out lets the package server (`server.py`) and the PROD router
(`bench/mlx_router.py`) share one tested implementation instead of two hand-synced copies.

`models` is a mapping `name -> {"last": float, "size_gb": float}` (a snapshot of the hot
chat models); `free` is free RAM in GB, or None when it can't be measured.
"""

DISK_ESTIMATE_MARGIN = 1.15  # on-disk weight size undercounts activations/working-set overhead


def estimate_size_gb(measured, disk, hot_sizes, default_est):
    """Best estimate (GB) of a model's resident size BEFORE loading it: a real
    measurement from a prior load (this session, or persisted from an earlier one), else
    its on-disk weight size inflated by `DISK_ESTIMATE_MARGIN` (weights-on-disk alone
    undercounts true resident size), else the largest hot model (or `default_est` when
    nothing is hot)."""
    if measured is not None:
        return measured
    if disk is not None:
        return disk * DISK_ESTIMATE_MARGIN
    return max(hot_sizes) if hot_sizes else default_est


def plan_make_room(target, est, free, models, min_free_gb, max_runners):
    """Plan the LRU evictions to run BEFORE loading `target` (~`est` GB) so it fits the
    budget: free-after-load >= `min_free_gb` and runners <= `max_runners`. This is the
    proactive admission gate — running it before the (memory-spiking) load is what keeps
    a second big model from overflowing RAM during load and being evicted too late.

    Returns the victim names in eviction order (never `target`). Mirrors the original
    loop: it simulates the expected RAM recovery (`free += size`) and the shrinking
    runner count as it goes, stopping as soon as the budget is met."""
    victims = []
    others = sorted((x for x in models if x != target), key=lambda x: models[x]["last"])
    target_present = target in models
    cur_free = free
    n_chat = len(models)
    for victim in others:
        n_after = n_chat + (0 if target_present else 1)
        short = cur_free is not None and (cur_free - est) < min_free_gb
        if not short and n_after <= max_runners:
            break
        victims.append(victim)
        if cur_free is not None:
            cur_free += models[victim]["size_gb"]
        n_chat -= 1
    return victims


def plan_enforce(keep, free, models, min_free_gb, max_runners):
    """Post-load safety net: plan LRU evictions while the budget is exceeded
    (runners > `max_runners`, or free < `min_free_gb`), never evicting `keep` and always
    leaving at least one model hot (Ollama-style). Returns victim names in eviction order."""
    victims = []
    cur_free = free
    remaining = dict(models)
    while True:
        n = len(remaining)
        if n <= 1:
            break
        over = n > max_runners or (cur_free is not None and cur_free < min_free_gb)
        if not over:
            break
        cand = sorted((x for x in remaining if x != keep), key=lambda x: remaining[x]["last"])
        if not cand:
            break
        victim = cand[0]
        victims.append(victim)
        if cur_free is not None:
            cur_free += remaining[victim]["size_gb"]
        del remaining[victim]
    return victims


def plan_emergency_evict(
    free_gb,
    pageout_mb_s,
    chat_models,
    ac_size_gb,
    em_size_gb,
    trigger_free_gb,
    recover_free_gb,
    trigger_pageout_mb_s,
):
    """Reactive safety net for when the *whole machine* is under memory pressure, not just
    this process's own admission budget (issue #93) -- proactive admission only ever runs
    at load time, so a model that's already resident can still get squeezed into swap by
    something else running on the box. Emergency is declared when EITHER signal fires:
    `free_gb < trigger_free_gb` (RAM is nearly gone) OR `pageout_mb_s > trigger_pageout_mb_s`
    (the kernel is already paging, before free RAM necessarily looks low). Returns `[]`
    when neither is true.

    Once triggered, evicts at least one victim -- chat models by LRU (`chat_models` is
    `name -> {"last": float, "size_gb": float}`), then `"autocomplete"`, then `"embed"`
    (only when their resident size is known, i.e. currently loaded) -- simulating
    recovered RAM per victim, continuing until the simulated free RAM reaches
    `recover_free_gb`. That target is set above `trigger_free_gb` (hysteresis) so eviction
    doesn't stop right at the line and immediately re-trigger next tick. A pageout-only
    trigger (free RAM already looks fine) still forces one eviction -- the whole point of
    that signal is that free bytes alone can lag behind real pressure."""
    if not (free_gb < trigger_free_gb or pageout_mb_s > trigger_pageout_mb_s):
        return []
    candidates = sorted(chat_models, key=lambda n: chat_models[n]["last"])
    sizes = {n: chat_models[n]["size_gb"] for n in candidates}
    if ac_size_gb is not None:
        candidates.append("autocomplete")
        sizes["autocomplete"] = ac_size_gb
    if em_size_gb is not None:
        candidates.append("embed")
        sizes["embed"] = em_size_gb
    victims = []
    cur_free = free_gb
    for name in candidates:
        victims.append(name)
        cur_free += sizes[name]
        if cur_free >= recover_free_gb:
            break
    return victims


def scale_defaults(total_gb):
    """Derive memory-policy defaults from total system RAM (GB).

    The historical hardcoded defaults (`min_free_gb=2.0`, `default_est_gb=8.0`,
    `max_runners=4`, `wired_headroom_gb=5.0`) were tuned for the ~24GB Apple Silicon dev
    machine. On an 8GB machine they're too aggressive (real OOM risk); on a 64-128GB
    machine they leave most of the RAM unused. Bucket into small/medium/large/xlarge RAM
    profiles instead. Envs, when set, always override these (see server.py) — this only
    picks the out-of-the-box behavior for whoever hasn't set them.

    `prefill_step` (issue #81) got the same treatment: it used to be a flat 512
    regardless of machine size. Chunked prefill trades peak RAM for cold-prompt prefill
    speed (a bigger step means fewer chunks, so faster, but a higher transient peak) — a
    512-token step sized for an 8GB machine's OOM safety leaves prefill throughput on the
    table on a 64-128GB machine. With the prompt cache hot, prefill is usually just the
    new suffix, so this mostly matters for cold first turns and cache-relief
    reprocessing. The bucket values below are taken as-proposed from issue #81 (not
    independently re-benched — see the PR for why).
    """
    if total_gb <= 10:
        return {
            "min_free_gb": 1.0,
            "default_est_gb": 3.0,
            "max_runners": 1,
            "wired_headroom_gb": 2.0,
            "prefill_step": 512,
        }
    if total_gb <= 40:
        return {
            "min_free_gb": 2.0,
            "default_est_gb": 8.0,
            "max_runners": 4,
            "wired_headroom_gb": 5.0,
            "prefill_step": 1024,
        }
    if total_gb <= 80:
        return {
            "min_free_gb": 4.0,
            "default_est_gb": 8.0,
            "max_runners": 6,
            "wired_headroom_gb": 8.0,
            "prefill_step": 2048,
        }
    return {
        "min_free_gb": 8.0,
        "default_est_gb": 8.0,
        "max_runners": 8,
        "wired_headroom_gb": 16.0,
        "prefill_step": 4096,
    }


def order_cache_relief(keep, models):
    """Order in which to drop runners' KV caches under RAM pressure: oldest (LRU) first,
    with `keep` last (the current request's cache is the last resort). Only models whose
    snapshot has `has_cache` truthy are candidates. `models` here is
    `name -> {"last": float, "has_cache": bool}`."""
    order = sorted(
        (n for n in models if n != keep and models[n].get("has_cache")),
        key=lambda n: models[n]["last"],
    )
    if keep in models and models[keep].get("has_cache"):
        order.append(keep)
    return order


def is_oom_error(msg):
    """True when an exception message matches MLX's OOM-shaped allocator failures:
    `[malloc] Unable to allocate ...` (CPU) or `[metal::malloc] Attempting to allocate ...`
    / `[metal::malloc] Resource limit (...` (GPU). MLX raises these as a plain builtin
    `RuntimeError` with no dedicated exception type, so message-sniffing is the only
    signal available to tell an OOM apart from any other runtime failure."""
    return bool(msg) and ("[malloc]" in msg or "[metal::malloc]" in msg)


def common_prefix(a, b):
    """Length of the longest shared prefix of two token-id sequences."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def select_prompt_cache_slot(slots, ptoks):
    """Chooses which KV-cache slot (of a runner's small pool) to reuse for an incoming
    prompt, and which slot the post-generation cache should be written back into.

    `slots` is a snapshot list of `{"tokens": [...] | None, "last": float}` — one entry
    per pool slot, in pool order. `ptoks` is the incoming prompt's token ids.

    Returns `(match_idx, common_len, write_idx)`:
      - `match_idx` / `common_len`: the slot with the longest common prefix against
        `ptoks` (ties broken by most-recently-used), and that prefix's length. `match_idx`
        is `None` when no slot has any token overlap (empty pool or zero-length match).
      - `write_idx`: where to store the cache after generation. Equal to `match_idx` on a
        hit (overwrite in place); otherwise the first empty slot, or — when the pool is
        full — the least-recently-used slot (evicted to make room)."""
    best_idx, best_len = None, 0
    for i, s in enumerate(slots):
        toks = s.get("tokens")
        if not toks:
            continue
        n = common_prefix(toks, ptoks)
        if n > best_len or (n == best_len and n > 0 and s["last"] > slots[best_idx]["last"]):
            best_idx, best_len = i, n
    if best_idx is not None and best_len > 0:
        return best_idx, best_len, best_idx
    empty = next((i for i, s in enumerate(slots) if not s.get("tokens")), None)
    if empty is not None:
        return None, 0, empty
    lru = min(range(len(slots)), key=lambda i: slots[i]["last"])
    return None, 0, lru
