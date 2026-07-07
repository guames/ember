"""Tests for keep_alive/idle-timeout on the fixed autocomplete (_ac) / embed (_em)
slots (issue #34). Mirrors tests/test_memory.py's style -- monkeypatch module
globals, stub eviction, no real weights or mlx calls.
"""

import queue
import threading

import pytest

from ember import server


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.setattr(
        server,
        "_ac",
        {"model": object(), "tok": object(), "pc": None, "pctoks": None, "last": 0.0, "ka": -1},
    )
    monkeypatch.setattr(server, "_em", {"model": object(), "proc": object(), "last": 0.0, "ka": -1})
    monkeypatch.setattr(server, "_chat", {}, raising=True)
    evicted = []
    monkeypatch.setattr(server, "_evict_ac", lambda: evicted.append("ac"))
    monkeypatch.setattr(server, "_evict_em", lambda: evicted.append("em"))
    return evicted


class _Job:
    def __init__(self, payload):
        self.payload = payload
        self.out = queue.Queue()
        self.cancel = threading.Event()


# ---------------------------------------------------------------- _run_evict
def test_run_evict_never_evicts_ac_em_by_default(clean, monkeypatch):
    """ka == -1 (today's behavior) -> never due, even idle forever."""
    monkeypatch.setattr(server.time, "monotonic", lambda: 10_000.0)
    server._run_evict(_Job({"names": [], "ac": True, "em": True}))
    assert clean == []


def test_run_evict_evicts_ac_once_idle_exceeds_ka(clean, monkeypatch):
    monkeypatch.setattr(server.time, "monotonic", lambda: 100.0)
    server._ac["ka"] = 5.0
    server._ac["last"] = 90.0  # 10s idle > 5s ka
    server._run_evict(_Job({"names": [], "ac": True, "em": False}))
    assert clean == ["ac"]


def test_run_evict_evicts_em_once_idle_exceeds_ka(clean, monkeypatch):
    monkeypatch.setattr(server.time, "monotonic", lambda: 100.0)
    server._em["ka"] = 5.0
    server._em["last"] = 90.0
    server._run_evict(_Job({"names": [], "ac": False, "em": True}))
    assert clean == ["em"]


def test_run_evict_ignores_ac_em_when_not_flagged(clean, monkeypatch):
    """The watchdog only sets ac/em=True when it already thinks they're expired;
    _run_evict must still gate on ka/idle itself, but a False flag short-circuits."""
    monkeypatch.setattr(server.time, "monotonic", lambda: 100.0)
    server._ac["ka"] = 5.0
    server._ac["last"] = 0.0
    server._run_evict(_Job({"names": [], "ac": False, "em": False}))
    assert clean == []


def test_run_evict_keeps_ac_when_still_within_ka(clean, monkeypatch):
    monkeypatch.setattr(server.time, "monotonic", lambda: 100.0)
    server._ac["ka"] = 30.0
    server._ac["last"] = 90.0  # 10s idle < 30s ka
    server._run_evict(_Job({"names": [], "ac": True, "em": False}))
    assert clean == []


# ---------------------------------------------------------------- ac_model/em_model touch "last"
def test_ac_model_refreshes_last_on_each_call(clean, monkeypatch):
    monkeypatch.setattr(server, "load", lambda repo: (object(), object()))
    server._ac["last"] = 0.0
    server.ac_model()
    assert server._ac["last"] > 0.0


def test_em_model_refreshes_last_on_each_call(clean, monkeypatch):
    import types

    fake_mlx_embeddings = types.ModuleType("mlx_embeddings")
    fake_mlx_embeddings.load = lambda repo: (object(), object())
    monkeypatch.setitem(__import__("sys").modules, "mlx_embeddings", fake_mlx_embeddings)
    server._em["model"] = None
    server._em["last"] = 0.0
    server.em_model()
    assert server._em["last"] > 0.0


# ---------------------------------------------------------------- opt-in via request keep_alive
def test_gen_fim_sets_ac_ka_from_keep_alive(clean, monkeypatch):
    fake_tok = type(
        "FakeTok", (), {"encode": lambda self, s, add_special_tokens=False: [1, 2, 3]}
    )()
    monkeypatch.setattr(server, "ac_model", lambda: (object(), fake_tok))
    monkeypatch.setattr(server, "make_prompt_cache", lambda model: [])

    class R:
        def __init__(self, text, token):
            self.text = text
            self.token = token

    def fake_stream_generate(model, tok, arr, **kw):
        yield R("ok", 1)

    monkeypatch.setattr(server, "stream_generate", fake_stream_generate)
    server.gen_fim({"prompt": "x", "suffix": "y", "keep_alive": "30s"})
    assert server._ac["ka"] == 30.0


def test_gen_fim_without_keep_alive_leaves_ac_ka_untouched(clean, monkeypatch):
    fake_tok = type(
        "FakeTok", (), {"encode": lambda self, s, add_special_tokens=False: [1, 2, 3]}
    )()
    monkeypatch.setattr(server, "ac_model", lambda: (object(), fake_tok))
    monkeypatch.setattr(server, "make_prompt_cache", lambda model: [])

    class R:
        def __init__(self, text, token):
            self.text = text
            self.token = token

    def fake_stream_generate(model, tok, arr, **kw):
        yield R("ok", 1)

    monkeypatch.setattr(server, "stream_generate", fake_stream_generate)
    server._ac["ka"] = -1
    server.gen_fim({"prompt": "x", "suffix": "y"})
    assert server._ac["ka"] == -1


def test_run_embed_sets_em_ka_from_keep_alive(clean, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "METRICS_LOG_PATH", str(tmp_path / "metrics.jsonl"))
    monkeypatch.setattr(server, "_metrics", {})
    monkeypatch.setattr(server, "_metrics_log_fh", None)
    monkeypatch.setattr(server, "embeddings", lambda texts: ([[0.1]] * len(texts), 1))
    job = _Job({"texts": ["hi"], "keep_alive": "1m"})
    server._run_embed(job)
    assert server._em["ka"] == 60.0
