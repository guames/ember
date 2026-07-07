"""Default KV cache quantization (issue #23): MLX_KV_BITS defaults to 8-bit,
still overridable to fp16 (0) or 4-bit via env.

Also covers issue #82: MLX_KV_BITS now understands the same boolean-ish "off" spellings
as MLX_EMBED_CACHE/MLX_PROMPT_CACHE (e.g. "false") instead of crashing at import with
`int("false")`.
"""

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


def test_false_falls_back_to_fp16_instead_of_crashing(monkeypatch):
    """Previously `int("false")` blew up at import time -- MLX_KV_BITS=false must be
    understood the same way MLX_EMBED_CACHE=false already is (issue #82)."""
    mod = _reload_with_env(monkeypatch, "false")
    try:
        assert mod.KV_BITS is None
        assert mod._kv_kwargs() == {}
    finally:
        _reload_with_env(monkeypatch, None)


def test_false_case_and_whitespace_insensitive(monkeypatch):
    mod = _reload_with_env(monkeypatch, "  FALSE  ")
    try:
        assert mod.KV_BITS is None
    finally:
        _reload_with_env(monkeypatch, None)


def test_empty_string_still_falls_back_to_fp16(monkeypatch):
    mod = _reload_with_env(monkeypatch, "")
    try:
        assert mod.KV_BITS is None
    finally:
        _reload_with_env(monkeypatch, None)


# ---------------------------------------------------------------- shared env helpers (issue #82)
def test_env_bool_consistent_falsy_spellings(monkeypatch):
    for value in ("0", "false", "False", "  false  ", ""):
        monkeypatch.setenv("MLX_TEST_FLAG", value)
        assert server._env_bool("MLX_TEST_FLAG", True) is False, value
    monkeypatch.delenv("MLX_TEST_FLAG", raising=False)


def test_env_bool_truthy_and_default(monkeypatch):
    monkeypatch.setenv("MLX_TEST_FLAG", "1")
    assert server._env_bool("MLX_TEST_FLAG", False) is True
    monkeypatch.delenv("MLX_TEST_FLAG", raising=False)
    assert server._env_bool("MLX_TEST_FLAG", True) is True
    assert server._env_bool("MLX_TEST_FLAG", False) is False


def test_env_int_or_none_matches_env_bool_falsy_set(monkeypatch):
    """MLX_KV_BITS's "N, or off" parsing should treat exactly the same strings as falsy
    that _env_bool does, so MLX_EMBED_CACHE=false and MLX_KV_BITS=false behave the same
    way (issue #82)."""
    for value in ("0", "false", "False", ""):
        monkeypatch.setenv("MLX_TEST_INT", value)
        assert server._env_int_or_none("MLX_TEST_INT", "8") is None, value
    monkeypatch.setenv("MLX_TEST_INT", "4")
    assert server._env_int_or_none("MLX_TEST_INT", "8") == 4
    monkeypatch.delenv("MLX_TEST_INT", raising=False)
    assert server._env_int_or_none("MLX_TEST_INT", "8") == 8
    monkeypatch.delenv("MLX_TEST_INT", raising=False)
