"""Ember model registry — loads model definitions from a config file.

Search order (the first one that exists wins):
  1. $EMBER_CONFIG               (explicit path to a .yaml/.yml/.json)
  2. ./ember.yaml  ./ember.yml  ./ember.json   (current directory)
  3. ~/.config/ember/config.yaml  (or .yml/.json)

Format (YAML; JSON with the same keys also works):

    models:
      - name: qwen3-8b                       # name used in the API ("model" field)
        mlx: mlx-community/Qwen3-8B-4bit      # HF repo or local path
        params: {temperature: 0.0, top_p: 0.95, num_ctx: 32768}
      - name: qwen2.5-vl
        mlx: mlx-community/Qwen2.5-VL-3B-Instruct-4bit
        vision: true                         # loads via mlx-vlm; accepts images

    autocomplete:                            # optional (has a default)
      name: autocomplete
      mlx: mlx-community/Qwen2.5-Coder-1.5B-4bit
    embed:                                   # optional (has a default)
      name: embed
      mlx: mlx-community/nomicai-modernbert-embed-base-4bit

If no file is found, it boots with a minimal set of defaults (1 small chat model +
autocomplete + embed) just so it doesn't fail — edit the config for real use.
"""

import json
import os

DEFAULT_AUTOCOMPLETE = {"name": "autocomplete", "mlx": "mlx-community/Qwen2.5-Coder-1.5B-4bit"}
DEFAULT_EMBED = {"name": "embed", "mlx": "mlx-community/nomicai-modernbert-embed-base-4bit"}
DEFAULT_MODELS = [
    {
        "name": "qwen2.5-coder-1.5b",
        "mlx": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
        "params": {"temperature": 0.0},
    },
]

_SEARCH = [
    "ember.yaml",
    "ember.yml",
    "ember.json",
    os.path.expanduser("~/.config/ember/config.yaml"),
    os.path.expanduser("~/.config/ember/config.yml"),
    os.path.expanduser("~/.config/ember/config.json"),
]


def _find_config():
    env = os.environ.get("EMBER_CONFIG")
    if env:
        if not os.path.exists(env):
            raise FileNotFoundError(f"EMBER_CONFIG points to a nonexistent file: {env}")
        return env
    for path in _SEARCH:
        if os.path.exists(path):
            return path
    return None


def _parse(path):
    with open(path) as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "YAML config needs pyyaml (pip install pyyaml) — or use .json."
            ) from exc
        return yaml.safe_load(text) or {}
    return json.loads(text or "{}")


def _norm_model(spec):
    """Normalizes a model entry into the server's internal format."""
    if not isinstance(spec, dict) or "name" not in spec or "mlx" not in spec:
        raise ValueError(f"invalid model entry (needs name+mlx): {spec!r}")
    return {
        "name": spec["name"],
        "mlx": spec["mlx"],
        "params": spec.get("params", {}) or {},
        "vision": bool(spec.get("vision", False)),
    }


def load_registry(path=None):
    """Returns (CFG, autocomplete, embed).
    CFG = {name: {name, mlx, params, vision}} of the chat/code/vision models.
    autocomplete/embed = {name, mlx} of the fixed models."""
    path = path or _find_config()
    if path is None:
        models, ac, em = DEFAULT_MODELS, DEFAULT_AUTOCOMPLETE, DEFAULT_EMBED
    else:
        data = _parse(path)
        models = data.get("models") or DEFAULT_MODELS
        ac = {**DEFAULT_AUTOCOMPLETE, **(data.get("autocomplete") or {})}
        em = {**DEFAULT_EMBED, **(data.get("embed") or {})}
    cfg = {}
    for spec in models:
        m = _norm_model(spec)
        cfg[m["name"]] = m
    if not cfg:
        raise ValueError("no chat model defined in the config")
    return cfg, ac, em
