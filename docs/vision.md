# Vision (multimodal)

Ember serves vision-language models through the same
`POST /v1/chat/completions` endpoint, using OpenAI's multimodal message format. Image
models run on [mlx-vlm](https://github.com/Blaizzy/mlx-vlm), which ships in the
optional `[vision]` extra.

## Setup

1. Install the extra:

   ```bash
   pip install "ember-mlx[vision]"
   ```

2. Mark the model `vision: true` in your `ember.yaml`:

   ```yaml
   models:
     - name: qwen2.5-vl
       mlx: mlx-community/Qwen2.5-VL-3B-Instruct-4bit
       vision: true
   ```

The `vision: true` flag is what routes the request down the mlx-vlm path
(`model, processor` loaded via `mlx_vlm.load`) instead of the text path.

## Request

Content becomes an array of parts mixing text and images:

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen2.5-vl",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What is in this image?"},
      {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}
    ]
  }]
}'
```

### Accepted image parts

Ember collects image sources from each message and is flexible about the shape:

- `type` may be `image_url`, `input_image`, or `image`.
- the URL may be `image_url.url` (object form), or a bare string under
  `image_url` / `image` / `url`.
- the source itself can be a remote **URL** or a **data URI**
  (`data:image/png;base64,…`); mlx-vlm loads either.

Multiple images per message are supported — they're passed to the model together.

## Guardrail

If a request carries images but the target model is **not** configured `vision: true`,
Ember rejects it *before loading anything*:

```json
{"error": "model 'qwen3-8b' is not a vision model (config vision:true); got 1 image(s)"}
```

This keeps a stray image from silently loading a text model that would just ignore it.

## Notes & limits

- **Sampling:** the VLM path honors `max_tokens`, `temperature`, and `top_p` (from the
  request, falling back to the model's `params`). The richer sampler/penalty options of
  the text path don't all apply here.
- **Streaming** works (SSE deltas), the same as text chat.
- **No prompt cache yet** for vision models — see the [Roadmap](../README.md#roadmap).
  Each turn reprocesses its prompt.
- Combine with [structured output](response-format.md) to get schema-valid JSON
  *about* an image.
