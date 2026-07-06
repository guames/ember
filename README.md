# 🔥 Ember

[![CI](https://github.com/guames/ember/actions/workflows/ci.yml/badge.svg)](https://github.com/guames/ember/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ember-mlx.svg)](https://pypi.org/project/ember-mlx/)
[![Python](https://img.shields.io/pypi/pyversions/ember-mlx.svg)](https://pypi.org/project/ember-mlx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

🇬🇧 **English** · [🇧🇷 Português](README.pt-br.md)

**A warm, memory-smart MLX inference server for Apple Silicon.**

One process serves **chat, tool-calling, vision, embeddings and code autocomplete** —
all on [MLX](https://github.com/ml-explore/mlx), all OpenAI-compatible, with a single
adaptive memory policy that keeps your models *warm* and never blows past your RAM.

Built for local coding assistants (e.g. [Continue](https://continue.dev)) on a single Mac.

> ⚠️ Status: **beta**. Runs in daily use, but the API may still shift before 1.0.

---

## What is Ember?

**Ember is a local AI server for your Mac.** You point your tools (a coding assistant, a
script, `curl`) at it as if it were the OpenAI API, and it runs the models **on your own
machine** — private, offline, no subscription. It handles chat and code, code
autocomplete, text embeddings, and image understanding, all from one address.

**What it does.** A real coding setup needs several models at once — a big one to chat
about code, a tiny fast one for inline autocomplete, an embedder for codebase search,
maybe a vision model for screenshots. Running those as separate servers means three
processes fighting over the same RAM. Ember runs them in **one** process with a single
brain that decides what to keep loaded ("warm"), what to unload, and how to share memory —
so things stay fast without ever overflowing your RAM.

**Why it exists.** [Ollama](https://ollama.com) made local models genuinely pleasant:
load by name, keep models warm, reuse the conversation cache, manage it from a simple CLI.
But Ollama runs on llama.cpp/GGUF. On **Apple Silicon (the M-series chips)**, Apple's own
[MLX](https://github.com/ml-explore/mlx) framework is typically **faster and lighter** for
the same model. Ember takes the parts of Ollama that make it nice to use — the warm
models, the prefix cache, the no-fuss CLI — and rebuilds them **natively on MLX, tuned for
M-series Macs**: it consistently uses 1–3 GB less RAM per model and matches or beats
Ollama's speed (see [Benchmarks](#benchmarks)). It is **not** a fork of Ollama — it shares
none of its code; it borrows its *ergonomics* and reimplements them for Apple's stack,
adding things a coding assistant wants out of the box (tools, vision, JSON-schema output,
cooperative autocomplete).

In short: **Ollama-style ease, MLX speed, built for the M-series.**

## What makes it different

Other OpenAI-compatible MLX servers exist (`mlx_lm.server`, FastMLX, LM Studio's
backend…). Ember's niche is being the **unified, memory-adaptive** one for a single Mac:

- 🧩 **One server, every role.** Chat/code, FIM autocomplete, embeddings and vision in a
  single process — instead of juggling three servers and three memory budgets.
- ⏱️ **Cooperative preemption.** Autocomplete and embedding requests *jump the queue and
  run between the chat's tokens*, so typing never stalls a long generation.
- 🧠 **Adaptive memory.** Multiple models stay hot while RAM allows (LRU eviction,
  idle-unload, `keep_alive`). Under pressure it **drops KV caches oldest-first** before
  evicting a whole model.
- ⚡ **Prompt cache (prefix reuse).** Llama.cpp/Ollama-style longest-common-prefix KV reuse,
  multi-slot per runner so interleaved conversations don't evict each other → much lower
  TTFT when continuing a conversation. Zero-copy.
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

| Model | Quant | tok/s · MLX | tok/s · Ollama | MLX faster | RAM · MLX | RAM · Ollama | MLX lighter |
|---|---|--:|--:|--:|--:|--:|--:|
| DeepSeek-Coder-V2-16B (MoE) | 4-bit | **77** | 68 | **+13%** | **9 GB** | 11 GB | **−18%** |
| Qwen3-30B-A3B (MoE) | 3-bit | **68** | 56 | **+21%** | **13 GB** | 15 GB | **−13%** |
| Qwen3-8B | 3-bit | **36** | 30 | **+20%** | **4 GB** | 7 GB | **−43%** |
| Phi-4-14B | 3-bit | **19** | 16 | **+19%** | **6 GB** | 10 GB | **−40%** |
| Qwen2.5-Coder-32B | 3-bit | 8 | 8 | ±0% | **15 GB** | 16 GB | **−6%** |

Optimizations (measured): prompt cache cuts **TTFT ~5×** (396 → 80 ms on a 1.3k-token
prompt); chunked prefill drops peak RAM ~19 %; 8-bit KV cache is ~2× smaller.

➡️ Full tables (all 18 configs, KV-cache memory per model, Ollama comparison) in
[docs/benchmarks.md](docs/benchmarks.md).

## Getting started

> 🤖 **Want an AI assistant to set this up for you?** Hand it
> [INSTALL_WITH_AI.md](INSTALL_WITH_AI.md) — it walks the assistant through installing
> and configuring Ember while *asking you* which models and options you want.

### 0. Requirements

- A Mac with **Apple Silicon** (M1 or newer). Ember does not run on Intel Macs.
- **Python 3.10+** — check with `python3 --version`. (Get it from [python.org](https://www.python.org/downloads/macos/) or `brew install python`.)
- Free disk + RAM for the models you pick (8 GB works for small models; 24 GB+ for the
  big ones — see [Benchmarks](#benchmarks)).

### 1. Install

> ℹ️ **Not yet on PyPI.** `pip install ember-mlx` won't work until the package is published —
> for now, install from a clone (`pip install .`). Releasing is wired and ready; see
> [Publishing](#publishing-maintainer).

```bash
# recommended: an isolated environment
python3 -m venv ~/.ember-venv
source ~/.ember-venv/bin/activate

pip install ember-mlx                # core: chat, autocomplete, embeddings (once published)
# or, to also get vision + JSON-schema output:
pip install "ember-mlx[vision]"
```

Check it worked:

```bash
ember --help
```

### 2. Configure your models

Create a file named **`ember.yaml`** in the folder where you'll run Ember (start from
[`examples/models.yaml`](examples/models.yaml)). Each entry has a `name` (what you'll call
it in requests) and an `mlx` Hugging Face repo:

```yaml
models:
  - name: qwen3-8b                       # small & fast — good first pick
    mlx: mlx-community/Qwen3-8B-4bit
    params: { temperature: 0.0, num_ctx: 32768 }

  - name: qwen2.5-vl                     # optional: vision (needs the [vision] extra)
    mlx: mlx-community/Qwen2.5-VL-3B-Instruct-4bit
    vision: true
```

Not sure which models? See the [Benchmarks](#benchmarks) for speed/RAM, then validate your
file with `ember config`. Models download automatically the first time they're used.

### 3. Start the server

```bash
ember serve                            # serves http://127.0.0.1:8000/v1
```

Leave it running in this terminal (or set it to start at login — see
[`examples/com.ember.server.plist`](examples/com.ember.server.plist)).

### 4. Use it

From another terminal — the friendly way:

```bash
ember run qwen3-8b "Write a haiku about Metal shaders."
ember ps          # what's loaded right now
```

…or as a normal OpenAI API:

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3-8b",
  "messages": [{"role": "user", "content": "Write a haiku about Metal shaders."}]
}'
```

That's it. To wire it into your editor, see [Use with Continue](#use-with-continue).

## Manage it from the terminal

Ember ships a small CLI. Run `ember --help` or `ember <command> --help` for details.

| Command | What it does |
|---|---|
| `ember serve` | start the server (`--host` `--port` `--config`) |
| `ember ps` | list **hot** models in RAM (size, idle, keep-alive, cached tokens) |
| `ember list` | list **configured** models and which are hot |
| `ember status` | full status: models + memory + queue + policy |
| `ember memory` | memory breakdown (MLX + system) |
| `ember metrics` | request counters + latency histogram (Prometheus text) |
| `ember run <model> [prompt]` | one-off streamed chat (prompt via arg or stdin) |
| `ember warm <model>` | preload a model into RAM (no generation) |
| `ember unload [target]` | unload `chat` (default) / `all` / `<model>` |
| `ember config` | show the resolved config file and validate models |
| `ember version` | print the version |

```console
$ ember ps
MODEL                            SIZE  VISION    IDLE   KEEP   CACHE
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
| `GET /health` | trivial 200 for process supervisors (unauthenticated, no `EMBER_API_KEY` needed) |
| `GET /status` | hot models, memory, queue, policy |
| `GET /memory` | MLX + system memory |
| `GET /metrics` | request counters + latency histogram, Prometheus text format |
| `POST /unload` | unload `chat` / `all` / `<model>` |

`/v1/*` routes require `Authorization: Bearer <key>` when `EMBER_API_KEY` is set (off by
default). `SIGTERM` stops accepting new requests, waits for the in-flight job to finish
(up to `EMBER_SHUTDOWN_TIMEOUT`), then exits.

Every chat/FIM/embed request also appends a JSON line (endpoint, model, latency,
prompt/completion/cached tokens, status) to `EMBER_METRICS_LOG` — additive to the existing
`print(...)` logging, so downstream tools (dashboards, cost tracking) can tail it instead of
polling `/status`. `GET /metrics` serves the same data pre-aggregated as Prometheus counters
and a latency histogram, reset on restart.

## Configuration (env)

| Var | Default | Meaning |
|---|---|---|
| `MLX_ROUTER_PORT` / `MLX_ROUTER_HOST` | `8000` / `127.0.0.1` | bind address |
| `EMBER_API_KEY` | off | require `Authorization: Bearer <key>` on `/v1/*` |
| `EMBER_SHUTDOWN_TIMEOUT` | `30` | seconds to drain the in-flight job on `SIGTERM` |
| `EMBER_METRICS_LOG` | `~/.cache/ember/metrics.jsonl` | JSONL request log path (`0` disables it) |
| `MLX_MAX_RUNNERS` | auto by RAM (`4` on 24GB) | max models hot at once |
| `MLX_MIN_FREE_GB` | auto by RAM (`2.0` on 24GB) | evict a model below this free RAM |
| `MLX_MIN_FREE_CACHE_GB` | `1.0` | drop KV caches below this free RAM |
| `MLX_IDLE_TIMEOUT` | `300` | idle seconds before unloading a chat model |
| `MLX_MAX_QUEUE` | `32` | queue depth before returning 503 |
| `MLX_PROMPT_CACHE` | `1` | prefix KV-cache reuse |
| `MLX_PROMPT_CACHE_SLOTS` | `2` | KV-cache slots per runner (interleaved conversations) |
| `MLX_KV_BITS` | off | `8`/`4` to quantize the KV cache (~2× smaller at 8-bit) |
| `MLX_PREFILL_STEP` | `512` | prefill chunk size (lower peak RAM) |
| `MLX_WIRED_LIMIT_GB` | auto by RAM | wired-memory ceiling (RAM − headroom, headroom scales with RAM) |
| `EMBER_CONFIG` | — | explicit path to the models config file |

**Feature guides** ([`docs/`](docs/README.md)): [Tools & function-calling](docs/tools.md) ·
[Vision](docs/vision.md) · [Structured output](docs/response-format.md) ·
[Prompt cache](docs/prompt-cache.md) · [Adaptive memory](docs/memory.md) ·
[Benchmarks](docs/benchmarks.md).

## Use with Continue

Point Continue at Ember as an OpenAI provider — see
[`examples/continue.config.yaml`](examples/continue.config.yaml). Vision models get
`capabilities: [image_input]`.

## Roadmap

Planned work, roughly in priority order — none of these is implemented yet.
Contributions welcome.

- [ ] **Prompt cache for vision models.** Text chat already reuses the KV prefix
  ([prompt cache](docs/prompt-cache.md)); vision models reprocess their prompt each turn.
- [ ] **Context shifting.** Generate past a model's `num_ctx` by dropping the oldest
  tokens instead of stopping at the limit.
- [ ] **Native `/api/*` compatibility layer.** Ember speaks the OpenAI surface (`/v1/*`)
  today; an Ollama-style `/api/*` would let more tools point at it unchanged.
- [ ] **Optional batching.** The GPU worker is serial today (with cooperative
  preemption — see [adaptive memory](docs/memory.md)); batching concurrent requests to
  the same model would raise throughput.

## Publishing (maintainer)

> 📦 **Status: not yet on PyPI — pending, by choice.** The release path is fully wired
> ([`.github/workflows/release.yml`](.github/workflows/release.yml), build + publish via GitHub
> Actions **Trusted Publishing** / OIDC — no API tokens). Only the steps below remain; do them
> whenever you actually want `ember-mlx` on PyPI.

1. **One-time, on [pypi.org](https://pypi.org/manage/account/publishing/)** → *Add a pending publisher*:
   - PyPI Project Name: `ember-mlx`
   - Owner: `guames` · Repository: `ember`
   - Workflow name: `release.yml` · Environment: `pypi`
2. **Tag a release:** `git tag v0.1.0 && git push origin v0.1.0`
3. The Release workflow builds the sdist + wheel and publishes via OIDC. Verify with
   `pip install ember-mlx` in a clean venv. (The PyPI badges above go live at this point.)

## License

[MIT](LICENSE) © Gustavo Ames. Not affiliated with Apple or the MLX project.
