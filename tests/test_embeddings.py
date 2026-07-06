"""Tests for the embeddings endpoint helper (model-free, via monkeypatch).

Regression for issue #5: mlx_embeddings.generate pads to the longest text in a batch, so
pooling over the padded positions returns all-NaN embeddings for the shorter texts. The fix
embeds one text at a time; these tests lock that invariant without loading any model.
"""

import types

import mlx_embeddings

from ember import server


def _fake_out(vec):
    """Mimic mlx_embeddings' return: `.text_embeds.tolist()` -> [[...]] (one row per call)."""
    return types.SimpleNamespace(text_embeds=types.SimpleNamespace(tolist=lambda: [vec]))


class _FakeProc:
    """Fake tokenizer: one token per character, so expected counts are easy to assert."""

    def encode(self, text, truncation=True, max_length=512):
        return list(text)[:max_length]


def test_embeddings_one_text_per_call(monkeypatch):
    """Each text must be embedded on its own — never batched (that is what NaNs the short ones)."""
    proc = _FakeProc()
    monkeypatch.setattr(server, "em_model", lambda: ("M", proc))
    calls = []

    def fake_generate(model, p, texts):
        assert (model, p) == ("M", proc)
        assert len(texts) == 1  # the invariant: never a mixed-length batch
        calls.append(texts[0])
        return _fake_out([float(len(texts[0]))] * 3)

    monkeypatch.setattr(mlx_embeddings, "generate", fake_generate)

    vecs, prompt_tokens = server.embeddings(["aa", "bbbb", "c"])

    assert calls == ["aa", "bbbb", "c"]  # order preserved
    assert vecs == [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [1.0, 1.0, 1.0]]
    assert prompt_tokens == 2 + 4 + 1  # sum of per-text token counts


def test_embeddings_empty_input(monkeypatch):
    monkeypatch.setattr(server, "em_model", lambda: ("M", _FakeProc()))
    monkeypatch.setattr(
        mlx_embeddings, "generate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called"))
    )
    assert server.embeddings([]) == ([], 0)


class _FakeJob:
    """Minimal stand-in for server.Job: just payload + out, no thread/queue plumbing."""

    def __init__(self, texts):
        self.kind = "embed"
        self.payload = {"texts": texts}
        self.out = server.queue.Queue()


def test_run_embed_chunks_and_yields(monkeypatch):
    """Regression for issue #25: a batch bigger than EMBED_CHUNK must not be embedded in one
    shot — _run_embed should slice it, re-queue the remainder, and return in between so a
    chat step gets a chance to run before the next slice."""
    monkeypatch.setattr(server, "EMBED_CHUNK", 2)
    calls = []

    def fake_embeddings(texts):
        calls.append(list(texts))
        return [[float(len(t))] for t in texts], sum(len(t) for t in texts)

    monkeypatch.setattr(server, "embeddings", fake_embeddings)

    texts = ["a", "bb", "ccc", "dddd", "e"]  # 5 texts, chunk=2 -> 3 slices
    job = _FakeJob(texts)

    server._run_embed(job)
    assert calls == [["a", "bb"]]  # only the first slice ran
    assert job.out.empty()  # not done yet: nothing posted to the caller
    assert server._q.qsize() == 1  # continuation re-queued for the next _drain_short call

    # drain the rest as _drain_short would, one slice per call
    server._drain_short()
    assert calls == [["a", "bb"], ["ccc", "dddd"]]
    assert job.out.empty()
    assert server._q.qsize() == 1

    server._drain_short()
    assert calls == [["a", "bb"], ["ccc", "dddd"], ["e"]]
    kind, (vecs, prompt_tokens) = job.out.get_nowait()
    assert kind == "result"
    assert vecs == [[1.0], [2.0], [3.0], [4.0], [1.0]]
    assert prompt_tokens == sum(len(t) for t in texts)


def test_run_embed_single_slice_batch(monkeypatch):
    """A batch that fits in one slice should still finish in a single _run_embed call."""
    monkeypatch.setattr(server, "EMBED_CHUNK", 8)
    monkeypatch.setattr(server, "embeddings", lambda texts: ([[1.0]] * len(texts), len(texts)))

    job = _FakeJob(["x", "y"])
    server._run_embed(job)

    assert server._q.empty()
    kind, (vecs, prompt_tokens) = job.out.get_nowait()
    assert kind == "result"
    assert vecs == [[1.0], [1.0]]
    assert prompt_tokens == 2


def test_drain_short_yields_after_one_job(monkeypatch):
    """_drain_short must process exactly one queued short job per call — not drain the whole
    queue — so a chat generation loop calling it between tokens actually gets control back."""
    ran = []
    monkeypatch.setattr(server, "_dispatch", lambda job: ran.append(job))

    server._q.put_nowait((server.P_SHORT, next(server._seq), "job-a"))
    server._q.put_nowait((server.P_SHORT, next(server._seq), "job-b"))

    server._drain_short()
    assert ran == ["job-a"]
    assert server._q.qsize() == 1

    server._drain_short()
    assert ran == ["job-a", "job-b"]
    assert server._q.empty()
