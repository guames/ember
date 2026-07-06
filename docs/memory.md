# Adaptive memory

The point of Ember is to keep several models **warm** on one Mac without ever pushing it
into swap. One policy governs that: keep models hot while there's RAM, evict the least-
recently-used when there isn't, and *decide before loading* so a big model never overflows
RAM mid-load. This guide explains the policy and every knob that tunes it.

The eviction/sizing **decisions** live in a small pure module,
[`src/ember/memory_policy.py`](../src/ember/memory_policy.py) (no MLX, no psutil — just the
math), which the server calls after snapshotting its state. The same module backs the
private PROD router, so both share one tested implementation. The unit tests in
[`tests/test_memory_policy.py`](../tests/test_memory_policy.py) pin the behavior described
below.

## The model slots

- **Chat / code / vision** — a multi-runner pool, loaded on demand, up to `MLX_MAX_RUNNERS`
  hot at once, the rest LRU-evicted. This is what the policy manages.
- **Autocomplete (FIM)** and **embeddings** — pinned in RAM, always hot, so they can
  preempt chat instantly. They don't participate in eviction.

`GET /status` (or `ember status`) shows what's hot, the memory breakdown, the queue, and
the live policy values.

## Proactive admission control (evict *before* loading)

The original failure mode: load a second ~12 GB model on top of an existing one and the
machine overflows RAM *during* the load, before any reactive eviction can fire. Ember
prevents this by making room **before** the memory-spiking load:

1. **Estimate** the incoming model's resident size (`memory_policy.estimate_size_gb`):
   a real measurement from a prior load this session → its on-disk `.safetensors` size →
   the largest hot model → `MLX_DEFAULT_EST_GB` (8 GB).
2. **Plan evictions** (`memory_policy.plan_make_room`): evict LRU chat models — never the
   target — until the budget holds: `free − estimate ≥ MLX_MIN_FREE_GB` **and**
   `runners ≤ MLX_MAX_RUNNERS`. The plan simulates the RAM freed by each eviction, so it
   stops as soon as the model will fit.
3. **Load** and measure the real size (remembered for the next estimate).

Each eviction logs, e.g.:

```
[router] admission: evicting LRU qwen3-30b-a3b (~12.4GB) to fit qwen2.5-coder-32b (~13.4GB, free 4.1GB)
```

A post-load safety net (`plan_enforce`) runs the same LRU rule after a load, always
keeping at least one model hot (Ollama-style).

## KV-cache relief (cheaper than eviction)

Under *critical* pressure — free RAM below `MLX_MIN_FREE_CACHE_GB` (default 1 GB) — Ember
first drops runners' **KV/prompt caches**, oldest-first, before evicting any whole model.
This is much cheaper: the weights stay hot, and the only cost is reprocessing that
conversation's prompt next turn (see [prompt-cache.md](prompt-cache.md)). The current
request's own cache is the last to go.

## keep_alive & idle unload

Each chat model has a keep-alive. After `MLX_IDLE_TIMEOUT` seconds (default 300) idle, an
idle model is unloaded automatically; the next request reloads it. Override per request
with the `keep_alive` field — a number of seconds or a string like `"30s"`, `"5m"`,
`"1h"` (`0`/negative = never expire). `ember ps` shows each model's idle time and
keep-alive.

## Boot-time tuning

- **Wired-memory pinning** keeps the weights resident so the OS doesn't compress/page them
  near the RAM limit (which would make speed erratic). Auto by default — `total − 5 GB`,
  leaving headroom for the OS — or set `MLX_WIRED_LIMIT_GB` explicitly.
- **Chunked prefill** (`MLX_PREFILL_STEP`, default 512) processes a cold prompt in chunks
  to lower peak RAM. With the prompt cache, normal prefill is already just the new suffix,
  so this mostly matters for the first long prompt.
- **Cache limit** (`MLX_CACHE_LIMIT_GB`) optionally caps MLX's buffer pool, returning RAM
  to the OS. Off by default.

## Every memory knob

| Env | Default | Meaning |
|---|---|---|
| `MLX_MAX_RUNNERS` | auto by RAM (`4` on 24GB) | max chat models hot at once |
| `MLX_MIN_FREE_GB` | auto by RAM (`2.0` on 24GB) | evict a model when free RAM would fall below this |
| `MLX_MIN_FREE_CACHE_GB` | `1.0` | drop KV caches when free RAM falls below this |
| `MLX_DEFAULT_EST_GB` | auto by RAM (`8.0` on 24GB) | size guess for an unknown incoming model |
| `MLX_IDLE_TIMEOUT` | `300` | idle seconds before unloading a chat model (`0` = never) |
| `MLX_MAX_QUEUE` | `32` | queue depth before returning `503` |
| `MLX_KV_BITS` | off | `8`/`4` to quantize the KV cache (~2× smaller at 8-bit) |
| `MLX_KV_GROUP_SIZE` | `64` | KV quantization group size |
| `MLX_KV_QUANT_START` | `0` | quantize the KV cache from token N onward |
| `MLX_PREFILL_STEP` | `512` | prefill chunk size (lower peak RAM) |
| `MLX_WIRED_LIMIT_GB` | auto by RAM | wired-memory ceiling (`total − headroom`; headroom scales with RAM, `5GB` on 24GB) |
| `MLX_CACHE_LIMIT_GB` | off | cap the MLX buffer pool |
| `MLX_PROMPT_CACHE` | `1` | prefix KV-cache reuse (see [prompt-cache.md](prompt-cache.md)) |

## Inspecting & nudging it

```bash
ember status            # models + memory + queue + policy
ember memory            # MLX + system memory (in use / free)
ember ps                # hot models: size, idle, keep-alive, cached tokens
ember unload <model>    # evict one model now ( | chat | all )
ember clear context     # drop KV caches, keep models hot
```
