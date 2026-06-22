# Structured output (`response_format`)

Ember can **guarantee** that a chat response is valid JSON — optionally conforming to a
JSON schema — using constrained decoding, not prompt nudging. At each generation step a
logits processor masks every token that would break the grammar, so the output is valid
by construction.

This is powered by [llguidance](https://github.com/guidance-ai/llguidance) (via
`mlx_vlm.structured`), which ships in the optional `[vision]` extra:

```bash
pip install "ember-mlx[vision]"
```

## `json_object` — any JSON object

```json
{
  "model": "qwen3-8b",
  "messages": [{"role": "user", "content": "Give me a sample user as JSON."}],
  "response_format": {"type": "json_object"}
}
```

The model is constrained to emit a single well-formed JSON object (`{"type":"object"}`).

## `json_schema` — conform to a schema

```json
{
  "model": "qwen3-8b",
  "messages": [{"role": "user", "content": "Extract the city and temperature."}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "weather",
      "schema": {
        "type": "object",
        "properties": {
          "city": {"type": "string"},
          "temp_c": {"type": "number"}
        },
        "required": ["city", "temp_c"]
      }
    }
  }
}
```

Ember reads the schema from `json_schema.schema` (it also tolerates a nested
`json_schema.json_schema`). Every sampled token is masked against the grammar, so the
result parses *and* matches the schema.

`{"type": "text"}` (or an unknown type) applies no constraint — a normal completion.

## Behavior & limits

- **Composes with everything.** `response_format` stacks on top of your sampler and
  repetition penalties; it's just another logits processor in the chain. It also works
  alongside [tools](tools.md), though you'll usually use one or the other.
- **Fail-open.** If the constraint can't be built (e.g. the `[vision]` extra isn't
  installed, or the schema is rejected), Ember logs
  `[router] response_format ignored (<model>): <reason>` and falls back to an
  unconstrained completion rather than erroring the request. If you require strict JSON,
  check the server log when output isn't constrained.
- **Set `temperature: 0`** for the most reliable extraction.
