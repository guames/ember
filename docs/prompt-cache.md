# Prompt cache (KV reuse)

Ember reuses the KV cache across requests the way llama.cpp and Ollama do: each runner
keeps **one** cache slot, and on every request it reuses the **longest common prefix**
of tokens with what's already cached, processing only the new suffix. In a conversation
or an edit loop — where each turn repeats the whole history plus a little more — this
cuts time-to-first-token dramatically (see [benchmarks.md](benchmarks.md): TTFT ~5×
lower on a 1.3k-token prompt). There is **zero deepcopy**; the slot is trimmed in place.

It's on by default. Turn it off with `MLX_PROMPT_CACHE=0`.

## How it works

For an incoming prompt's tokens `ptoks`:

1. If the runner's slot has cached tokens `slot_t`, compute `n = common_prefix(slot_t, ptoks)`.
2. Trim the slot's KV cache back to that prefix (drop the `len(slot_t) - n` divergent
   tokens), then generate over only `ptoks[n:]`.
3. If the new prompt *is* exactly the cached prefix, keep all but one token so there's
   always something to generate.
4. No overlap (or cache off) → a fresh cache over the whole prompt.

After generation the slot is updated to `prompt + generated` tokens, so the next turn's
prefix match includes the model's own last answer. When a prefix is reused the server
logs `[router] cache <model>: reused <n>/<total> prompt tokens`.

`ember ps` shows each hot model's `CACHE` column (cached token count); `cached_tokens`
also appears in `GET /status`.

## What resets or limits it

- **Divergent prefix.** Change something early in the prompt (e.g. the system message)
  and the common prefix shrinks — only the unchanged head is reused.
- **One slot per runner.** A model holds a single conversation's cache at a time;
  interleaving two very different conversations on the same model trades reuse back and
  forth.
- **Memory pressure.** Under low RAM, Ember drops KV caches (oldest first) *before*
  evicting a model — cheap, because the weights stay hot and only the prompt is
  reprocessed next turn. See [memory.md](memory.md).
- **Eviction / unload.** Unloading a model (idle timeout, `ember unload`, admission
  eviction) discards its slot.
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
