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
