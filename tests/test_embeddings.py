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


def test_embeddings_one_text_per_call(monkeypatch):
    """Each text must be embedded on its own — never batched (that is what NaNs the short ones)."""
    monkeypatch.setattr(server, "em_model", lambda: ("M", "P"))
    calls = []

    def fake_generate(model, proc, texts):
        assert (model, proc) == ("M", "P")
        assert len(texts) == 1  # the invariant: never a mixed-length batch
        calls.append(texts[0])
        return _fake_out([float(len(texts[0]))] * 3)

    monkeypatch.setattr(mlx_embeddings, "generate", fake_generate)

    out = server.embeddings(["aa", "bbbb", "c"])

    assert calls == ["aa", "bbbb", "c"]  # order preserved
    assert out == [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [1.0, 1.0, 1.0]]


def test_embeddings_empty_input(monkeypatch):
    monkeypatch.setattr(server, "em_model", lambda: ("M", "P"))
    monkeypatch.setattr(
        mlx_embeddings, "generate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called"))
    )
    assert server.embeddings([]) == []
