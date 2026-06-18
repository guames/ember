"""Testes do registro de modelos (puro — não importa MLX, roda em qualquer SO)."""

import json

import pytest

from ember.registry import load_registry


def _write(tmp_path, data, name="ember.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return str(p)


def test_defaults_when_no_file():
    cfg, ac, em = load_registry(path=None)  # sem arquivo -> defaults
    assert cfg  # ao menos um modelo de chat
    assert "mlx" in ac and "mlx" in em


def test_loads_models_and_flags(tmp_path):
    path = _write(
        tmp_path,
        {
            "models": [
                {"name": "chat", "mlx": "org/Chat-4bit", "params": {"temperature": 0.1}},
                {"name": "vlm", "mlx": "org/VL-4bit", "vision": True},
            ],
            "autocomplete": {"name": "ac", "mlx": "org/AC"},
            "embed": {"name": "em", "mlx": "org/EM"},
        },
    )
    cfg, ac, em = load_registry(path=path)
    assert set(cfg) == {"chat", "vlm"}
    assert cfg["chat"]["params"]["temperature"] == 0.1
    assert cfg["chat"]["vision"] is False
    assert cfg["vlm"]["vision"] is True
    assert ac["name"] == "ac" and em["mlx"] == "org/EM"


def test_autocomplete_embed_fall_back_to_defaults(tmp_path):
    path = _write(tmp_path, {"models": [{"name": "c", "mlx": "org/C"}]})
    _, ac, em = load_registry(path=path)
    assert "mlx" in ac and "mlx" in em  # preenchidos pelos defaults


def test_invalid_model_raises(tmp_path):
    path = _write(tmp_path, {"models": [{"name": "x"}]})  # falta "mlx"
    with pytest.raises(ValueError):
        load_registry(path=path)


def test_missing_env_config_raises(monkeypatch):
    monkeypatch.setenv("EMBER_CONFIG", "/nao/existe/ember.yaml")
    from ember.registry import _find_config

    with pytest.raises(FileNotFoundError):
        _find_config()
