# Tools & function-calling

Ember speaks the OpenAI tools API on `POST /v1/chat/completions`: you pass `tools`
(and optionally `tool_choice`), the model decides whether to call one, and Ember
returns OpenAI-shaped `tool_calls`.

Tool-calling rides on the model's own chat template, so it works on models whose
template understands tools — the Hermes/Qwen/GLM family that emits
`<tool_call>…</tool_call>` blocks is the best-supported. There is no separate
"tools model"; any chat model you've configured can be called with `tools`.

## Request

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3-8b",
  "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Current weather for a city",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }],
  "tool_choice": "auto"
}'
```

The `tools` array is injected into the prompt through the tokenizer's
`apply_chat_template(..., tools=…)`. If a model's template ignores tools, the call
still runs as a plain chat completion.

### `tool_choice`

| Value | Behavior |
|---|---|
| `"auto"` (default) | the model decides whether to call a tool |
| `"none"` | tools are **dropped entirely** — a normal completion |
| `"required"` | force *some* call (prefill opens `<tool_call>`) |
| `{"type":"function","function":{"name":"get_weather"}}` | force *this* tool by name |

Forcing (`required` / named) works by **prefilling** the start of a `<tool_call>`
block onto the prompt, so it only applies to Hermes-style templates that contain the
`<tool_call>` tag. On templates without it, Ember can't force a call and leaves the
decision to the model.

## Response

When the model calls a tool, Ember parses the call out of the generated text and
returns it in OpenAI format — `finish_reason` becomes `tool_calls`:

```json
{
  "choices": [{
    "index": 0,
    "finish_reason": "tool_calls",
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_3f9a…",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}
      }]
    }
  }]
}
```

`arguments` is always a JSON **string** (OpenAI convention), even though the model
emits it as an object. When the model answers in prose instead of calling a tool,
you get a normal `content` message with `finish_reason: stop`.

In streaming mode the tool-call arrives as a single chunk whose `delta.tool_calls`
holds the complete call (Ember buffers the generation and parses it at the end, so
tool arguments are not streamed token-by-token).

## What Ember accepts when parsing

The parser is deliberately lenient, because different model families format calls
differently. It recognizes, in order:

1. `<tool_call>{…}</tool_call>` blocks (one or many) — Qwen / Hermes / GLM.
2. An unclosed `<tool_call>{…` (truncated or prefill-forced) — the first balanced
   JSON object is recovered.
3. A fallback ```` ```json ```` fence, or raw text that is itself a JSON object/array.

Inside, any of these object shapes work: `{name, arguments}`, `{name, parameters}`,
`{function: {name, arguments}}`, and `{tool_calls: [...]}` (and lists of them).

## Multi-turn

Feed the result back like the OpenAI API: append the assistant message (with its
`tool_calls`) and one `{"role": "tool", "content": "<result>"}` message per call,
then call again. The conversation prefix is reused from the [prompt cache](prompt-cache.md).
