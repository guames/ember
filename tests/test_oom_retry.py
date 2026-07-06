"""OOM-shaped-error retry tests (issue #30): on a caught MLX allocator failure during
generation, _run_chat/_run_fim/_run_embed relieve RAM (_enforce_memory) and retry the
request once before surfacing a 500 -- but only where it's safe to do so (nothing already
streamed to the caller). Mirrors tests/test_metrics.py's _FakeJob/monkeypatch style.
"""

import threading

import pytest

from ember import server


class _FakeJob:
    def __init__(self, payload):
        self.payload = payload
        self.out = server.queue.Queue()
        self.cancel = threading.Event()


_OOM = "[metal::malloc] Attempting to allocate 999999999999 bytes which is greater than..."


@pytest.fixture(autouse=True)
def spy_enforce_memory(monkeypatch):
    calls = []
    monkeypatch.setattr(server, "_enforce_memory", lambda keep=None: calls.append(keep))
    return calls


# ---------------------------------------------------------------- _run_fim
def test_run_fim_retries_once_on_oom_then_succeeds(spy_enforce_memory, monkeypatch):
    calls = []

    def flaky(body):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError(_OOM)
        return "hello", {"prompt_tokens": 1, "completion_tokens": 1, "cached_tokens": 0}

    monkeypatch.setattr(server, "gen_fim", flaky)
    job = _FakeJob({"body": {}})
    server._run_fim(job)
    kind, data = job.out.get_nowait()
    assert kind == "result"
    assert len(calls) == 2  # one retry
    assert spy_enforce_memory == [None]  # relief ran exactly once


def test_run_fim_gives_up_after_one_retry(spy_enforce_memory, monkeypatch):
    def always_oom(body):
        raise RuntimeError(_OOM)

    monkeypatch.setattr(server, "gen_fim", always_oom)
    job = _FakeJob({"body": {}})
    server._run_fim(job)
    kind, data = job.out.get_nowait()
    assert kind == "error"
    assert spy_enforce_memory == [None]  # only ever retries once


def test_run_fim_does_not_retry_non_oom_errors(spy_enforce_memory, monkeypatch):
    calls = []

    def boom(body):
        calls.append(1)
        raise RuntimeError("KeyError: 'qwen'")

    monkeypatch.setattr(server, "gen_fim", boom)
    job = _FakeJob({"body": {}})
    server._run_fim(job)
    kind, data = job.out.get_nowait()
    assert kind == "error"
    assert len(calls) == 1  # no retry for a non-OOM error
    assert spy_enforce_memory == []


# ---------------------------------------------------------------- _run_embed
def test_run_embed_retries_once_on_oom_mid_batch(spy_enforce_memory, monkeypatch):
    monkeypatch.setattr(server, "EMBED_CHUNK", 2)
    attempts = {"n": 0}

    def flaky_embeddings(chunk):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError(_OOM)
        return [[0.0] for _ in chunk], len(chunk)

    monkeypatch.setattr(server, "embeddings", flaky_embeddings)
    job = _FakeJob({"texts": ["a", "b", "c"]})
    while job.out.empty():
        server._run_embed(job)
        if not job.out.empty():
            break
        _, _, requeued = server._q.get_nowait()
        job = requeued
    kind, data = job.out.get_nowait()
    assert kind == "result"
    vecs, tokens = data
    assert len(vecs) == 3
    assert spy_enforce_memory == [None]  # relief ran exactly once for the whole request


def test_run_embed_gives_up_after_one_retry(spy_enforce_memory, monkeypatch):
    monkeypatch.setattr(server, "EMBED_CHUNK", 2)

    def always_oom(chunk):
        raise RuntimeError(_OOM)

    monkeypatch.setattr(server, "embeddings", always_oom)
    job = _FakeJob({"texts": ["a", "b"]})
    server._run_embed(job)
    kind, data = job.out.get_nowait()
    assert kind == "error"
    assert spy_enforce_memory == [None]


# ---------------------------------------------------------------- _run_chat
class _FakeTok:
    """No chat_template -> _fmt_chat falls back to a plain join; encode returns fixed ids."""

    def encode(self, s, add_special_tokens=False):
        return [1, 2, 3]


class _R:
    def __init__(self, token, text):
        self.token = token
        self.text = text
        self.generation_tokens = 1


def _runner(tok):
    return {
        "model": object(),
        "tok": tok,
        "size_gb": 1.0,
        "last": 0.0,
        "ka": -1,
        "vlm": False,
        "slots": [{"pc": None, "pctoks": None, "last": 0.0}],
    }


@pytest.fixture
def chat_env(monkeypatch):
    """Pre-load a fake chat runner so chat_model() short-circuits the real mlx_lm load,
    and stub the cache-pool primitives (mirrors tests/test_chat_cache.py's `clean`)."""
    name = next(iter(server.CFG))  # any configured model name (chat_model requires name in CFG)
    tok = _FakeTok()
    monkeypatch.setattr(server, "_chat", {name: _runner(tok)})
    monkeypatch.setattr(server, "PROMPT_CACHE", True)
    monkeypatch.setattr(server, "make_prompt_cache", lambda model: [])
    monkeypatch.setattr(server, "can_trim_prompt_cache", lambda cache: True)
    monkeypatch.setattr(server, "trim_prompt_cache", lambda cache, n: None)
    return name


def _body():
    return {"messages": [{"role": "user", "content": "hi"}]}


def test_run_chat_retries_once_on_oom_before_first_token(chat_env, spy_enforce_memory, monkeypatch):
    name = chat_env
    calls = {"n": 0}

    def flaky_stream_generate(model, tok, arr, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError(_OOM)
            yield  # pragma: no cover - makes this a generator function
        yield _R(1, "hi")

    monkeypatch.setattr(server, "stream_generate", flaky_stream_generate)
    job = _FakeJob({"name": name, "body": _body()})
    server._run_chat(job)

    kinds = []
    while not job.out.empty():
        kinds.append(job.out.get_nowait()[0])
    assert "meta" in kinds
    assert kinds.count("meta") == 1  # not re-sent on retry
    assert "done" in kinds
    assert "error" not in kinds
    assert calls["n"] == 2  # one retry
    assert spy_enforce_memory == [name]


def test_run_chat_does_not_retry_after_a_token_was_already_streamed(
    chat_env, spy_enforce_memory, monkeypatch
):
    """Once a delta has reached job.out, retrying would double-send content to the
    client -- so an OOM after that point must surface as an error, not a silent retry."""
    name = chat_env
    calls = {"n": 0}

    def flaky_stream_generate(model, tok, arr, **kw):
        calls["n"] += 1
        yield _R(1, "partial")
        raise RuntimeError(_OOM)

    monkeypatch.setattr(server, "stream_generate", flaky_stream_generate)
    job = _FakeJob({"name": name, "body": _body()})
    server._run_chat(job)

    kinds = []
    while not job.out.empty():
        kinds.append(job.out.get_nowait()[0])
    assert kinds.count("delta") == 1
    assert "error" in kinds
    assert "done" not in kinds
    assert calls["n"] == 1  # no retry once content has already been emitted
    assert spy_enforce_memory == []


def test_run_chat_load_oom_retries_then_succeeds(chat_env, spy_enforce_memory, monkeypatch):
    """OOM raised by chat_model() itself (e.g. loading a big model) also gets one retry."""
    name = chat_env
    real_chat = dict(server._chat)
    monkeypatch.setattr(server, "_chat", {})  # force chat_model() down the "load" path
    attempts = {"n": 0}

    def flaky_chat_model(n):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError(_OOM)
        server._chat[n] = real_chat[name]
        return real_chat[name]["model"], real_chat[name]["tok"], False

    monkeypatch.setattr(server, "chat_model", flaky_chat_model)
    monkeypatch.setattr(server, "stream_generate", lambda *a, **kw: iter([_R(1, "hi")]))
    job = _FakeJob({"name": name, "body": _body()})
    server._run_chat(job)

    kinds = []
    while not job.out.empty():
        kinds.append(job.out.get_nowait()[0])
    assert "done" in kinds
    assert "error" not in kinds
    assert attempts["n"] == 2
    assert spy_enforce_memory == [name]
