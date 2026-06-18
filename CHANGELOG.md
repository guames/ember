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
- Management **CLI** with `--help` for every command: `serve`, `ps`, `list`, `status`,
  `memory`, `run`, `warm`, `unload`, `clear`, `config`, `version`.
- `POST /clear` (and `ember clear`): drop prompt cache / MLX buffer pool without
  unloading models.
- Benchmarks: [docs/benchmarks.md](docs/benchmarks.md) (Apple M5 24 GB — tok/s, MLX vs
  Ollama RAM).
- Beginner-friendly **Getting started** in the README and
  [INSTALL_WITH_AI.md](INSTALL_WITH_AI.md) — a guide you can hand to an AI assistant to
  install and configure Ember interactively.

[0.1.0]: https://github.com/guames/ember/releases/tag/v0.1.0
