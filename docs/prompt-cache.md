# Prompt cache (KV reuse)

Ember reuses the KV cache across requests the way llama.cpp and Ollama do: each runner
keeps a small **pool of cache slots** (2 by default), and on every request it picks
whichever slot has the **longest common prefix** of tokens with the incoming prompt,
processing only the new suffix. In a conversation or an edit loop — where each turn
repeats the whole history plus a little more — this cuts time-to-first-token
dramatically (see [benchmarks.md](benchmarks.md): TTFT ~5× lower on a 1.3k-token
prompt). There is **zero deepcopy**; the chosen slot is trimmed in place.

It's on by default. Turn matching off with `MLX_PROMPT_CACHE=0`. Size the pool with
`MLX_PROMPT_CACHE_SLOTS` (default 2; each extra slot costs one more runner's worth of KV
cache RAM while it holds a conversation).

## How it works

For an incoming prompt's tokens `ptoks`, across the runner's slot pool:

1. Score each slot by `n = common_prefix(slot_tokens, ptoks)`; pick the slot with the
   largest `n` (ties go to the most recently used slot).
2. On a hit (`n > 0`), trim that slot's KV cache back to the prefix (drop the
   `len(slot_tokens) - n` divergent tokens), then generate over only `ptoks[n:]`.
3. If the new prompt *is* exactly the cached prefix, keep all but one token so there's
   always something to generate.
4. No slot has any overlap (or cache off) → a fresh cache over the whole prompt, written
   into the first empty slot, or — if the pool is full — the least-recently-used slot.

After generation the chosen slot is updated to `prompt + generated` tokens, so the next
turn's prefix match includes the model's own last answer. When a prefix is reused the
server logs `[router] cache <model>: reused <n>/<total> prompt tokens`.

This is what lets two interleaved conversations (or agents) on the same model each keep
their own slot instead of evicting each other's cache every turn.

`ember ps` shows each hot model's `CACHE` column (total cached tokens across its pool);
`cached_tokens` also appears in `GET /status`.

## What resets or limits it

- **Divergent prefix.** Change something early in the prompt (e.g. the system message)
  and the common prefix shrinks — only the unchanged head is reused.
- **Small, fixed pool.** A model holds `MLX_PROMPT_CACHE_SLOTS` conversations' caches at
  once; a `(N+1)`th interleaved conversation evicts the pool's least-recently-used slot.
- **Memory pressure.** Under low RAM, Ember drops a model's whole pool of KV caches
  (oldest models first) *before* evicting a model — cheap, because the weights stay hot
  and only the prompt is reprocessed next turn. See [memory.md](memory.md).
- **Eviction / unload.** Unloading a model (idle timeout, `ember unload`, admission
  eviction) discards its pool.
- **Vision models** don't use the prompt cache yet (see the
  [Roadmap](../README.md#roadmap)).

## Quantized KV cache

To fit more context in the same RAM, quantize the KV cache with `MLX_KV_BITS=8` (or `4`).
8-bit is ~2× smaller than fp16 and practically lossless, and it stays compatible with the
prompt cache (the quantized cache is trimmable). Off by default. Related knobs:
`MLX_KV_GROUP_SIZE` (default 64) and `MLX_KV_QUANT_START` (quantize from token N onward).

## Manually clearing it

```bash
ember clear context     # drop the KV/prompt caches, keep models hot
ember clear cache       # release the MLX buffer pool (mx.clear_cache + reset peak)
ember clear all         # both
```

`clear context` is the lightweight reset: it frees the conversation caches without
unloading any weights, so the next request reloads nothing but reprocesses its prompt.
