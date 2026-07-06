"""Default KV cache quantization (issue #23): MLX_KV_BITS defaults to 8-bit,
still overridable to fp16 (0) or 4-bit via env."""

import importlib

from ember import server


def _reload_with_env(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("MLX_KV_BITS", raising=False)
    else:
        monkeypatch.setenv("MLX_KV_BITS", value)
    return importlib.reload(server)


def test_defaults_to_8bit(monkeypatch):
    mod = _reload_with_env(monkeypatch, None)
    try:
        assert mod.KV_BITS == 8
        assert mod._kv_kwargs()["kv_bits"] == 8
    finally:
        _reload_with_env(monkeypatch, None)


def test_zero_falls_back_to_fp16(monkeypatch):
    mod = _reload_with_env(monkeypatch, "0")
    try:
        assert mod.KV_BITS is None
        assert mod._kv_kwargs() == {}
    finally:
        _reload_with_env(monkeypatch, None)


def test_explicit_4bit_still_honored(monkeypatch):
    mod = _reload_with_env(monkeypatch, "4")
    try:
        assert mod.KV_BITS == 4
    finally:
        _reload_with_env(monkeypatch, None)
