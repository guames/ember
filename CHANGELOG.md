# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/), versions follow [SemVer](https://semver.org/).

## [Unreleased]

Not yet published to PyPI — see [Publishing](README.md#publishing-maintainer).

### Fixed
- **Proactive admission control** — estimate an incoming model's size and evict LRU
  models *before* loading, so a second large model can't overflow RAM mid-load (#1, #2).
- **Embeddings**: embed one text at a time to avoid all-NaN vectors (and invalid JSON)
  from mixed-length batches (#5, #6).

### Changed
- Extracted the memory-admission *decision* logic into a pure, dependency-free
  `ember.memory_policy` module (`estimate_size_gb`, `plan_make_room`, `plan_enforce`,
  `order_cache_relief`), with unit tests; the server keeps thin effectful wrappers (#7, #8).

### Added
- CI / PyPI / Python / MIT badges and a Brazilian-Portuguese README
  ([README.pt-br.md](README.pt-br.md)) with a language switcher (#9, #10).
- Release workflow for PyPI **Trusted Publishing** (OIDC, no API tokens) on `v*` tags
  ([`.github/workflows/release.yml`](.github/workflows/release.yml)) (#11, #12).
- Feature guides under [`docs/`](docs/README.md): tools, vision, structured output,
  prompt cache, and adaptive memory (#15).
- OpenAI-shaped `usage` (`prompt_tokens`/`completion_tokens`/`total_tokens` +
  `prompt_tokens_details.cached_tokens`) on chat, FIM, and embeddings responses; chat
  streaming exposes it via `stream_options: {"include_usage": true}` (#18).
- **Multi-slot KV prompt cache per chat runner** (`MLX_PROMPT_CACHE_SLOTS`, default 2):
  interleaved conversations on the same model now get their own cache slot instead of
  evicting each other every turn. Slot selection (longest-common-prefix match, LRU
  eviction when the pool is full) is the pure `memory_policy.select_prompt_cache_slot`
  (#21).

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

[Unreleased]: https://github.com/guames/ember/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/guames/ember/releases/tag/v0.1.0
