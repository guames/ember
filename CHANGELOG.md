# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/), versions follow [SemVer](https://semver.org/).

## [0.1.0] — 2026-06-17

First public release. Extracted and hardened from a private MLX benchmarking project.

### Added
- OpenAI-compatible server: `/v1/chat/completions` (stream & non-stream),
  `/v1/completions` (FIM autocomplete), `/v1/embeddings`, `/v1/models`.
- Ops endpoints: `/status`, `/memory`, `/unload`.
- Single GPU worker + priority queue with **cooperative preemption** (autocomplete/embed
  run between chat tokens).
- **Adaptive memory**: multi-runner with RAM budget, LRU eviction, `keep_alive`/idle-unload,
  and KV-cache relief under memory pressure.
- **Prompt cache**: longest-common-prefix KV reuse (zero-copy).
- **Tools / function-calling**: `tools` + `tool_choice` (`auto`/`none`/`required`/named).
- **Vision** (`[vision]` extra): mlx-vlm models with image inputs.
- **Structured output**: `response_format` JSON object / JSON schema via llguidance.
- Sampling: `temperature`, `top_p`, `top_k`, `min_p`, `stop`, `seed`,
  `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `logit_bias`.
- Memory tuning: 8-bit KV cache, chunked prefill, wired-memory pinning.
- File-based model registry (`ember.yaml` / `EMBER_CONFIG`).

[0.1.0]: https://github.com/gustavoames/ember/releases/tag/v0.1.0
