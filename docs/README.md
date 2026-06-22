# Ember documentation

Feature guides for the Ember MLX server. Each one is grounded in the actual
behavior of [`src/ember/server.py`](../src/ember/server.py); request/response shapes
are OpenAI-compatible unless noted.

| Guide | What it covers |
|---|---|
| [Tools & function-calling](tools.md) | `tools` / `tool_choice`, forced calls, how tool-calls are parsed and returned |
| [Vision (multimodal)](vision.md) | image inputs, `vision: true` models, the mlx-vlm path |
| [Structured output](response-format.md) | `response_format` — guaranteed JSON object / JSON-schema via constrained decoding |
| [Prompt cache](prompt-cache.md) | longest-common-prefix KV reuse and what resets it |
| [Adaptive memory](memory.md) | multi-runner budget, admission control, KV-cache relief, every env knob |
| [Benchmarks](benchmarks.md) | Apple M5 24 GB — tok/s and RAM, MLX vs Ollama |

New to Ember? Start with the [README](../README.md) (install + getting started),
then come back here for the per-feature details.

## The shared model surface

All chat features below run through one endpoint, `POST /v1/chat/completions`, and
share the same machinery:

- **One GPU worker + priority queue.** Requests are serialized on a single worker;
  autocomplete/embedding jobs preempt chat *between tokens* (see [memory.md](memory.md)).
  A full queue returns `503 {"error": "queue full (maxQueue)"}`.
- **Streaming or not.** Set `"stream": true` for Server-Sent Events
  (`chat.completion.chunk` deltas, terminated by `data: [DONE]`); otherwise you get a
  single `chat.completion` JSON. Both report `finish_reason` (`stop`, `tool_calls`, or
  `error`).
- **Per-model defaults.** Every model in your `ember.yaml` may carry a `params:` block;
  request fields override those per call.
