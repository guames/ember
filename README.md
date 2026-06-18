# 🔥 Ember

**A warm, memory-smart MLX inference server for Apple Silicon.**

One process serves **chat, tool-calling, vision, embeddings and code autocomplete** —
all on [MLX](https://github.com/ml-explore/mlx), all OpenAI-compatible, with a single
adaptive memory policy that keeps your models *warm* and never blows past your RAM.

Built for local coding assistants (e.g. [Continue](https://continue.dev)) on a single Mac.

> ⚠️ Status: **beta**. Runs in daily use, but the API may still shift before 1.0.

---

## Why Ember?

There are already OpenAI-compatible MLX servers (`mlx_lm.server`, FastMLX, LM Studio's
backend…). Ember's niche is being the **unified, memory-adaptive** one for a single Mac:

- 🧩 **One server, every role.** Chat/code, FIM autocomplete, embeddings and vision in a
  single process — instead of juggling three servers and three memory budgets.
- ⏱️ **Cooperative preemption.** Autocomplete and embedding requests *jump the queue and
  run between the chat's tokens*, so typing never stalls a long generation.
- 🧠 **Adaptive memory.** Multiple models stay hot while RAM allows (LRU eviction,
  idle-unload, `keep_alive`). Under pressure it **drops KV caches oldest-first** before
  evicting a whole model.
- ⚡ **Prompt cache (prefix reuse).** Llama.cpp/Ollama-style longest-common-prefix KV reuse
  → much lower TTFT when continuing a conversation. Zero-copy.
- 🎯 **Real constrained decoding.** `response_format` with JSON schema is *guaranteed* via
  [llguidance](https://github.com/guidance-ai/llguidance) (token-level masking), not prompt
  nudging.
- 🛠️ **Full OpenAI surface.** Tools/function-calling (`tool_choice` incl. forced),
  streaming, `stop`, `seed`, repetition/presence/frequency penalties, `logit_bias`.
- 💾 **Tuned for 24 GB.** 8-bit KV cache, chunked prefill (lower peak RAM), wired-memory
  pinning for consistent speed near the limit.

---

## Benchmarks

Measured on an **Apple M5, 24 GB** (MLX). Generation is memory-bandwidth bound, so MoE
models fly while dense 30B-class models trade speed for quality:

| Model | Quant | tok/s | RAM |
|---|---|--:|--:|
| DeepSeek-Coder-V2-16B (MoE) | 4-bit | **77** | 9 GB |
| Qwen3-30B-A3B (MoE) | 3-bit | **68** | 13 GB |
| Qwen3-8B | 3-bit | 36 | 4 GB |
| Phi-4-14B | 3-bit | 19 | 6 GB |
| Qwen2.5-Coder-32B | 3-bit | 8 | 15 GB |

Optimizations (measured): prompt cache cuts **TTFT ~5×** (396 → 80 ms on a 1.3k-token
prompt); chunked prefill drops peak RAM ~19 %; 8-bit KV cache is ~2× smaller.

➡️ Full tables (all 18 configs, KV-cache memory per model, Ollama comparison) in
[docs/benchmarks.md](docs/benchmarks.md).

## Requirements

- macOS on **Apple Silicon** (M-series).
- Python **3.10+**.
- Enough RAM for the models you load (16 GB works; 24 GB+ recommended).

## Install

```bash
pip install ember-mlx                 # core (chat, autocomplete, embeddings)
pip install "ember-mlx[vision]"       # + vision (mlx-vlm) and response_format/JSON schema
```

Or from source:

```bash
git clone https://github.com/gustavoames/ember && cd ember
pip install -e ".[vision,dev]"
```

## Quickstart

1. Create an `ember.yaml` (see [`examples/models.yaml`](examples/models.yaml)):

```yaml
models:
  - name: qwen3-8b
    mlx: mlx-community/Qwen3-8B-4bit
    params: { temperature: 0.0, top_p: 0.95, num_ctx: 32768 }
  - name: qwen2.5-vl
    mlx: mlx-community/Qwen2.5-VL-3B-Instruct-4bit
    vision: true
```

2. Run it (models download on first use):

```bash
ember                      # http://127.0.0.1:8000/v1
```

3. Call it like the OpenAI API:

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3-8b",
  "messages": [{"role": "user", "content": "Write a haiku about Metal shaders."}]
}'
```

## Manage it from the terminal

Ember ships a small CLI. Run `ember --help` or `ember <command> --help` for details.

| Command | What it does |
|---|---|
| `ember serve` | start the server (`--host` `--port` `--config`) |
| `ember ps` | list **hot** models in RAM (size, idle, keep-alive, cached tokens) |
| `ember list` | list **configured** models and which are hot |
| `ember status` | full status: models + memory + queue + policy |
| `ember memory` | memory breakdown (MLX + system) |
| `ember run <model> [prompt]` | one-off streamed chat (prompt via arg or stdin) |
| `ember warm <model>` | preload a model into RAM (no generation) |
| `ember unload [target]` | unload `chat` (default) / `all` / `<model>` |
| `ember config` | show the resolved config file and validate models |
| `ember version` | print the version |

```console
$ ember ps
MODELO                            TAM  VISÃO  OCIOSO   KEEP   CACHE
qwen3-8b                         3.3G      -      0s   5.0m      50

$ echo "refactor this loop" | ember run qwen3-8b
```

Management commands talk to a running server (`--url`, default `http://127.0.0.1:8000`).

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | chat/code — stream & non-stream; `tools`, `response_format`, images |
| `POST /v1/completions` | FIM autocomplete (kept hot) |
| `POST /v1/embeddings` | embeddings (kept hot) |
| `GET /v1/models` | list configured models |
| `GET /status` | hot models, memory, queue, policy |
| `GET /memory` | MLX + system memory |
| `POST /unload` | unload `chat` / `all` / `<model>` |

## Configuration (env)

| Var | Default | Meaning |
|---|---|---|
| `MLX_ROUTER_PORT` / `MLX_ROUTER_HOST` | `8000` / `127.0.0.1` | bind address |
| `MLX_MAX_RUNNERS` | `4` | max models hot at once |
| `MLX_MIN_FREE_GB` | `2.0` | evict a model below this free RAM |
| `MLX_MIN_FREE_CACHE_GB` | `1.0` | drop KV caches below this free RAM |
| `MLX_IDLE_TIMEOUT` | `300` | idle seconds before unloading a chat model |
| `MLX_MAX_QUEUE` | `32` | queue depth before returning 503 |
| `MLX_PROMPT_CACHE` | `1` | prefix KV-cache reuse |
| `MLX_KV_BITS` | off | `8`/`4` to quantize the KV cache (~2× smaller at 8-bit) |
| `MLX_PREFILL_STEP` | `512` | prefill chunk size (lower peak RAM) |
| `MLX_WIRED_LIMIT_GB` | auto | wired-memory ceiling (RAM−5 GB) |
| `EMBER_CONFIG` | — | explicit path to the models config file |

See [`docs/`](docs/) for tools, vision, `response_format`, prompt cache and memory details.

## Use with Continue

Point Continue at Ember as an OpenAI provider — see
[`examples/continue.config.yaml`](examples/continue.config.yaml). Vision models get
`capabilities: [image_input]`.

## Roadmap

- [ ] Prompt cache for vision models
- [ ] Context shifting (generate past `num_ctx`)
- [ ] Native `/api/*` compatibility layer
- [ ] Optional batching for concurrent requests to the same model

## License

[MIT](LICENSE) © Gustavo Ames. Not affiliated with Apple or the MLX project.
