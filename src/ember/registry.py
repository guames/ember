"""Registro de modelos do Ember — carrega as definições de um arquivo de config.

Ordem de busca (o primeiro que existir vence):
  1. $EMBER_CONFIG               (caminho explícito p/ um .yaml/.yml/.json)
  2. ./ember.yaml  ./ember.yml  ./ember.json   (diretório atual)
  3. ~/.config/ember/config.yaml  (ou .yml/.json)

Formato (YAML; JSON com as mesmas chaves também serve):

    models:
      - name: qwen3-8b                       # nome usado na API (campo "model")
        mlx: mlx-community/Qwen3-8B-4bit      # repo HF ou caminho local
        params: {temperature: 0.0, top_p: 0.95, num_ctx: 32768}
      - name: qwen2.5-vl
        mlx: mlx-community/Qwen2.5-VL-3B-Instruct-4bit
        vision: true                         # carrega via mlx-vlm; aceita imagens

    autocomplete:                            # opcional (tem default)
      name: autocomplete
      mlx: mlx-community/Qwen2.5-Coder-1.5B-4bit
    embed:                                   # opcional (tem default)
      name: embed
      mlx: mlx-community/nomicai-modernbert-embed-base-4bit

Se nenhum arquivo for encontrado, sobe com um conjunto mínimo de defaults (1 modelo de
chat pequeno + autocomplete + embed) só p/ não falhar — edite o config p/ uso real.
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
            raise FileNotFoundError(f"EMBER_CONFIG aponta p/ arquivo inexistente: {env}")
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
                "Config YAML precisa do pyyaml (pip install pyyaml) — ou use .json."
            ) from exc
        return yaml.safe_load(text) or {}
    return json.loads(text or "{}")


def _norm_model(spec):
    """Normaliza uma entrada de modelo p/ o formato interno do servidor."""
    if not isinstance(spec, dict) or "name" not in spec or "mlx" not in spec:
        raise ValueError(f"entrada de modelo inválida (precisa de name+mlx): {spec!r}")
    return {
        "name": spec["name"],
        "mlx": spec["mlx"],
        "params": spec.get("params", {}) or {},
        "vision": bool(spec.get("vision", False)),
    }


def load_registry(path=None):
    """Retorna (CFG, autocomplete, embed).
    CFG = {nome: {name, mlx, params, vision}} dos modelos de chat/código/visão.
    autocomplete/embed = {name, mlx} dos modelos fixos."""
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
        raise ValueError("nenhum modelo de chat definido no config")
    return cfg, ac, em
