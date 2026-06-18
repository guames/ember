# Benchmarks

Real measurements taken while building Ember. **Hardware: Apple M5, 24 GB unified
memory, 10 cores** (macOS, `mlx-lm` 0.31). Numbers are a guide, not a guarantee — your
mileage varies with quant, context and thermals.

- **tok/s** = generation throughput (reasoning/"thinking" off where applicable).
- **RAM** = resident memory with the model loaded (weights + working set), at the
  context length shown.
- **Ollama** column (llama.cpp / GGUF, same machine) is for reference only.

## Generation speed & memory (MLX)

| Model | Quant | tok/s · MLX | RAM · MLX | tok/s · Ollama | RAM · Ollama |
|---|---|--:|--:|--:|--:|
| **DeepSeek-Coder-V2-16B** (MoE) | 4-bit | **77** | 9 GB | 68 | 11 GB |
| DeepSeek-Coder-V2-16B | 6-bit | 53 | 13 GB | 52 | 16 GB |
| DeepSeek-Coder-V2-16B | 8-bit | 43 | 17 GB | 38 | 17 GB |
| **Qwen3-30B-A3B** (MoE) | 3-bit | **68** | 13 GB | 56 | 15 GB |
| Qwen3-30B-A3B | 4-bit | 53 | 19 GB | 50 | 18 GB |
| Qwen3-8B | 3-bit | 36 | 4 GB | 30 | 7 GB |
| Qwen3-8B | 4-bit | 28 | 5 GB | 24 | 8 GB |
| Qwen3-8B | 8-bit | 15 | 9 GB | 15 | 10 GB |
| Qwen3-8B | 16-bit | 9 | 16 GB | 9 | 15 GB |
| Phi-4-14B | 3-bit | 19 | 6 GB | 16 | 10 GB |
| Phi-4-14B | 4-bit | 15 | 8 GB | 14 | 11 GB |
| Phi-4-14B | 8-bit | 8 | 16 GB | 8 | 16 GB |
| Codestral-22B | 3-bit | 12 | 9 GB | 11 | 12 GB |
| Codestral-22B | 4-bit | 10 | 13 GB | 9 | 14 GB |
| Gemma-3-27B | 3-bit | 9 | 13 GB | 8 | 14 GB |
| GLM-4-32B | 3-bit | 9 | 14 GB | 8 | 16 GB |
| GLM-4-32B | 4-bit | 7 | 18 GB | 6 | 19 GB |
| Qwen2.5-Coder-32B | 3-bit | 8 | 15 GB | 8 | 16 GB |

> MLX is consistently lighter on RAM than Ollama for the same model/quant (often 1–3 GB
> less), and matches or beats its throughput on Apple Silicon.

## KV-cache memory (the "context" cost)

The prompt/KV cache grows with conversation length. Cost per token depends on layers ×
KV-heads × head-dim (GQA keeps it small). At fp16; **8-bit KV (`MLX_KV_BITS=8`) halves it.**

| Model | per 1k tokens | at full context |
|---|--:|--:|
| Qwen3-8B | 0.14 MB | 4.5 GB @ 32k |
| Gemma-3-27B | 0.48 MB | 3.9 GB @ 8k |
| DeepSeek-Coder-V2-16B | 0.21 MB | 3.4 GB @ 16k |
| Qwen2.5-Coder-32B | 0.25 MB | 2.0 GB @ 8k |
| Phi-4-14B | 0.20 MB | 1.6 GB @ 8k |
| Qwen3-30B-A3B | 0.09 MB | 0.8 GB @ 8k |
| Qwen2.5-VL-3B | 0.04 MB | 0.6 GB @ 16k |
| GLM-4-32B | 0.06 MB | 0.5 GB @ 8k |

In normal use the cache is tens to a few hundred MB (conversations rarely fill the
window). Ember drops it oldest-first under memory pressure (`MLX_MIN_FREE_CACHE_GB`).

## Optimization wins (measured)

| Feature | Effect |
|---|---|
| **Prompt cache** (prefix reuse) | TTFT **396 ms → 80 ms** (~5×) on a 1343-token prompt repeated |
| **Prefill chunking** (`MLX_PREFILL_STEP`) | prefill peak RAM **1.78 → 1.45 GB** (−19%) on a 6.4k-token prompt |
| **8-bit KV cache** | KV cache **~2× smaller**, near-lossless; still prefix-reusable |

## Takeaways

- **MoE wins big.** DeepSeek-Coder-V2-16B and Qwen3-30B-A3B hit 50–77 tok/s; dense 22–32B
  models sit at 7–12 tok/s with similar RAM, because **generation is memory-bandwidth
  bound** — not compute or free-RAM bound.
- **3-bit is ~20–25 % faster than 4-bit** on the same model (fewer bytes per token).
- **Speed is a hardware ceiling.** Ember can't make a given model generate faster; its
  wins are TTFT (prompt cache), footprint (KV quant), peak RAM (chunked prefill) and
  consistency (wired memory). MLX matches or slightly beats Ollama here (~1.0–1.3×).
