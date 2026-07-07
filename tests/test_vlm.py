"""Tests for the VLM generation path: stop/seed parity with text chat (issue #34).

Mirrors tests/test_sampling.py's _StopBuf coverage and tests/test_chat_cache.py's
monkeypatch style — stubs mlx_vlm's template + stream_generate so _gen_vlm runs
without real vision weights.
"""

import queue
import threading

import pytest

from ember import server


class _Chunk:
    def __init__(self, text):
        self.text = text


class _Job:
    def __init__(self):
        self.out = queue.Queue()
        self.cancel = threading.Event()


def _deltas(job):
    out = []
    while not job.out.empty():
        out.append(job.out.get_nowait())
    return "".join(d for k, d in out if k == "delta")


def _fake_model():
    return type("FakeModel", (), {"config": object()})()


@pytest.fixture
def fake_vlm(monkeypatch):
    import mlx_vlm
    from mlx_vlm import prompt_utils

    monkeypatch.setattr(prompt_utils, "apply_chat_template", lambda *a, **kw: "PROMPT")
    chunks = []

    def fake_stream_generate(model, proc, prompt, image=None, **kw):
        yield from chunks

    monkeypatch.setattr(mlx_vlm, "stream_generate", fake_stream_generate)
    monkeypatch.setattr(server, "CFG", {}, raising=False)
    monkeypatch.setattr(server, "_chat", {}, raising=True)
    return chunks


def test_gen_vlm_stop_sequence_truncates_output(fake_vlm):
    fake_vlm.extend([_Chunk("hello "), _Chunk("world STOP trailing")])
    job = _Job()
    usage = server._gen_vlm(job, "vlm", _fake_model(), object(), {"stop": "STOP"}, [], [])
    assert _deltas(job) == "hello world "
    assert usage is not None


def test_gen_vlm_stop_split_across_chunks_does_not_leak(fake_vlm):
    """A stop sequence that arrives split across several stream chunks must not
    leak into the output -- same hold-back guarantee as the text path."""
    fake_vlm.extend(
        [_Chunk("hi"), _Chunk("<|"), _Chunk("do"), _Chunk("ne"), _Chunk("|>"), _Chunk("rest")]
    )
    job = _Job()
    server._gen_vlm(job, "vlm", _fake_model(), object(), {"stop": "<|done|>"}, [], [])
    assert _deltas(job) == "hi"


def test_gen_vlm_no_stop_flushes_everything(fake_vlm):
    fake_vlm.extend([_Chunk("a"), _Chunk("b"), _Chunk("c")])
    job = _Job()
    server._gen_vlm(job, "vlm", _fake_model(), object(), {}, [], [])
    assert _deltas(job) == "abc"


def test_gen_vlm_stop_from_model_params_when_absent_in_body(fake_vlm):
    fake_vlm.extend([_Chunk("keep IT stop")])
    job = _Job()
    monkeypatch_cfg = {"vlm": {"params": {"stop": "stop"}}}
    server.CFG.update(monkeypatch_cfg)
    server._gen_vlm(job, "vlm", _fake_model(), object(), {}, [], [])
    assert _deltas(job) == "keep IT "


def test_gen_vlm_seed_seeds_mx_random(fake_vlm, monkeypatch):
    fake_vlm.append(_Chunk("ok"))
    seeded = []
    monkeypatch.setattr(server.mx.random, "seed", lambda s: seeded.append(s))
    job = _Job()
    server._gen_vlm(job, "vlm", _fake_model(), object(), {"seed": 42}, [], [])
    assert seeded == [42]


def test_gen_vlm_no_seed_does_not_seed(fake_vlm, monkeypatch):
    fake_vlm.append(_Chunk("ok"))
    seeded = []
    monkeypatch.setattr(server.mx.random, "seed", lambda s: seeded.append(s))
    job = _Job()
    server._gen_vlm(job, "vlm", _fake_model(), object(), {}, [], [])
    assert seeded == []
