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
   a real measurement from a prior load (this session, or persisted from an earlier run —
   see below) → its on-disk `.safetensors` size inflated by `DISK_ESTIMATE_MARGIN` (1.15×,
   since weights-on-disk alone undercounts activations/working-set overhead) → the largest
   hot model → `MLX_DEFAULT_EST_GB` (8 GB).
2. **Plan evictions** (`memory_policy.plan_make_room`): evict LRU chat models — never the
   target — until the budget holds: `free − estimate ≥ MLX_MIN_FREE_GB` **and**
   `runners ≤ MLX_MAX_RUNNERS`. The plan simulates the RAM freed by each eviction, so it
   stops as soon as the model will fit.
3. **Load** and measure the real size (remembered for the next estimate, and persisted to
   `EMBER_SIZES_CACHE` so the *first* load after a restart is already as accurate as every
   load after it — not just a disk-size guess).

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

## Emergency memory watchdog

Everything above only runs around a *load* — proactive admission decides before loading,
the post-load safety net runs right after. Neither reacts once a model is already sitting
resident and something else on the machine (another app, a second Ember instance, the OS
itself) starts eating RAM. That gap is real: it caused a SIGABRT + jetsam kill in
production. A background thread closes it by polling **system-wide** memory pressure —
independent of Ember's own admission budget — every `MLX_MEMWATCH_INTERVAL_S` seconds
(default 2.5s):

- **Trigger** — an emergency is declared when **either** signal fires: free RAM drops
  below `MLX_EMERGENCY_FREE_GB` (default 1.5 GB), **or** the kernel's swap pageout rate
  exceeds `MLX_EMERGENCY_PAGEOUT_RATE` MB/s (default 50). The pageout signal catches
  pressure that free-RAM-alone can miss or lag behind (macOS's memory compressor can keep
  "free" looking fine while the kernel is already paging).
- **Eviction** (`memory_policy.plan_emergency_evict`) — LRU chat models first, then the
  pinned autocomplete slot, then the pinned embed slot (only once the chat runners are
  exhausted), stopping once simulated free RAM recovers past `MLX_EMERGENCY_FREE_GB + 0.5`
  (fixed hysteresis margin, not configurable) — so eviction doesn't stop right at the
  trigger line and immediately fire again next tick.
- Every emergency eviction prints loudly and, when `EMBER_METRICS_LOG` is enabled, appends
  a JSONL line (`event: "emergency_evict"`) alongside the normal request metrics.
- Set `MLX_MEMWATCH=0` to disable the watchdog entirely (e.g. if you're running your own
  external memory monitor). It only starts when `psutil` is installed, same as the rest of
  the system-memory reporting.

## keep_alive & idle unload

Each chat model has a keep-alive. After `MLX_IDLE_TIMEOUT` seconds (default 300) idle, an
idle model is unloaded automatically; the next request reloads it. Override per request
with the `keep_alive` field — a number of seconds or a string like `"30s"`, `"5m"`,
`"1h"` (`0`/negative = never expire). `ember ps` shows each model's idle time and
keep-alive.

### The `warm` model alias

`model: "warm"` on `/v1/chat/completions` resolves to the **most recently used loaded
chat model** — whatever you warmed last. Useful for clients that want "the current
model" without hard-coding a name (e.g. pointing a fixed model slot of another tool at
Ember). The response echoes the resolved model name. With nothing loaded, the request
fails with a clear 404 rather than loading a model on its own; set `MLX_WARM_DEFAULT`
to a known model name to opt in to a cold-start fallback.

## RAM profiles (auto defaults)

Ember picks its out-of-the-box memory defaults from your Mac's **total RAM**
(`memory_policy.scale_defaults`), so the same install behaves sensibly on an 8 GB Air and
a 128 GB Studio. Setting the corresponding env var always overrides the profile.

| Total RAM | `MLX_MAX_RUNNERS` | `MLX_MIN_FREE_GB` | `MLX_DEFAULT_EST_GB` | wired headroom | `MLX_PREFILL_STEP` |
|---|--:|--:|--:|--:|--:|
| ≤ 10 GB | 1 | 1.0 | 3.0 | 2 GB | 512 |
| 10–40 GB | 4 | 2.0 | 8.0 | 5 GB | 1024 |
| 40–80 GB | 6 | 4.0 | 8.0 | 8 GB | 2048 |
| > 80 GB | 8 | 8.0 | 8.0 | 16 GB | 4096 |

("Wired headroom" is what `MLX_WIRED_LIMIT_GB` leaves for the OS: the auto ceiling is
`total − headroom`.)

These profiles were dogfooded on a 24 GB machine (the 10–40 GB row); the other rows are
principled extrapolations. If a bucket misbehaves on your hardware, override the envs and
please report it in [#84](https://github.com/guames/ember/issues/84).

## Boot-time tuning

- **Wired-memory pinning** keeps the weights resident so the OS doesn't compress/page them
  near the RAM limit (which would make speed erratic). Auto by default — `total −
  headroom`, where the headroom comes from your [RAM profile](#ram-profiles-auto-defaults)
  — or set `MLX_WIRED_LIMIT_GB` explicitly.
- **Chunked prefill** (`MLX_PREFILL_STEP`, auto by [RAM profile](#ram-profiles-auto-defaults),
  512 on the smallest tier) processes a cold prompt in chunks to lower peak RAM — a bigger
  step means fewer chunks (faster prefill) at the cost of a higher transient peak, so the
  default scales up with headroom the same way the other RAM-profile knobs do (issue #81).
  With the prompt cache, normal prefill is already just the new suffix, so this mostly
  matters for the first long prompt (or cache-relief reprocessing).
- **Cache limit** (`MLX_CACHE_LIMIT_GB`) optionally caps MLX's buffer pool, returning RAM
  to the OS. Off by default.

## Every memory knob

| Env | Default | Meaning |
|---|---|---|
| `MLX_MAX_RUNNERS` | auto by [RAM profile](#ram-profiles-auto-defaults) | max chat models hot at once |
| `MLX_MIN_FREE_GB` | auto by [RAM profile](#ram-profiles-auto-defaults) | evict a model when free RAM would fall below this |
| `MLX_MIN_FREE_CACHE_GB` | `1.0` | drop KV caches when free RAM falls below this |
| `MLX_DEFAULT_EST_GB` | auto by [RAM profile](#ram-profiles-auto-defaults) | size guess for an unknown incoming model |
| `MLX_IDLE_TIMEOUT` | `300` | idle seconds before unloading a chat model (`0` = never) |
| `MLX_MAX_QUEUE` | `32` | queue depth before returning `503` |
| `MLX_KV_BITS` | `8` | KV cache quantization bits; `4` for more aggressive, `0` for fp16 |
| `MLX_KV_GROUP_SIZE` | `64` | KV quantization group size |
| `MLX_KV_QUANT_START` | `0` | quantize the KV cache from token N onward |
| `MLX_PREFILL_STEP` | auto by [RAM profile](#ram-profiles-auto-defaults) | prefill chunk size (lower peak RAM) |
| `MLX_WIRED_LIMIT_GB` | auto by [RAM profile](#ram-profiles-auto-defaults) | wired-memory ceiling (`total − headroom`) |
| `MLX_CACHE_LIMIT_GB` | off | cap the MLX buffer pool |
| `MLX_PROMPT_CACHE` | `1` | prefix KV-cache reuse (see [prompt-cache.md](prompt-cache.md)) |
| `EMBER_SIZES_CACHE` | `~/.cache/ember/sizes.json` | persisted measured model sizes across restarts (`0` disables) |
| `MLX_MEMWATCH` | `1` | [emergency watchdog](#emergency-memory-watchdog) for whole-machine memory pressure (`0` disables) |
| `MLX_MEMWATCH_INTERVAL_S` | `2.5` | how often the emergency watchdog samples free RAM/pageout |
| `MLX_EMERGENCY_FREE_GB` | `1.5` | emergency trigger: evict when free RAM drops below this |
| `MLX_EMERGENCY_PAGEOUT_RATE` | `50` | emergency trigger: evict when swap pageout exceeds this (MB/s) |

## Inspecting & nudging it

```bash
ember status            # models + memory + queue + policy
ember memory            # MLX + system memory (in use / free)
ember ps                # hot models: size, idle, keep-alive, cached tokens
ember unload <model>    # evict one model now ( | chat | all )
ember clear context     # drop KV caches, keep models hot
```
