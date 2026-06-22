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


def estimate_size_gb(measured, disk, hot_sizes, default_est):
    """Best estimate (GB) of a model's resident size BEFORE loading it: a real
    measurement from a prior load this session, else its on-disk weight size, else
    the largest hot model (or `default_est` when nothing is hot)."""
    if measured is not None:
        return measured
    if disk is not None:
        return disk
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
