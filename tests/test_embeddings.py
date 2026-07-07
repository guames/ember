"""Tests for the embeddings endpoint helper (model-free, via monkeypatch).

Regression for issue #5: mlx_embeddings.generate pads to the longest text in a batch, so
pooling over the padded positions returns all-NaN embeddings for the shorter texts. The fix
(issue #24) buckets by exact tokenized length before batching — same-length texts share no
padding, so they can go through generate() as a real batch, while different-length texts must
never land in the same call. These tests lock both the length-bucketing invariant and the
content-hash cache (issue #24) without loading any model.
"""

import queue
import types

import mlx_embeddings
import pytest

from ember import server


def _fake_out(vecs):
    """Mimic mlx_embeddings' return: `.text_embeds.tolist()` -> one row per input text."""
    return types.SimpleNamespace(text_embeds=types.SimpleNamespace(tolist=lambda: vecs))


class _FakeProc:
    """Fake tokenizer: one token per character, so expected counts/lengths are easy to assert."""

    def encode(self, text, truncation=True, max_length=512):
        return list(text)[:max_length]


class _FakeCache:
    """In-memory stand-in for _EmbedCache: same get/put shape, no disk."""

    def __init__(self):
        self.store = {}
        self.puts = []

    def get(self, h):
        return self.store.get(h)

    def put(self, h, vec):
        self.puts.append(h)
        self.store[h] = vec


def test_embeddings_never_mixes_lengths_in_one_batch(monkeypatch):
    """Different-length texts must never share a generate() call (that is what NaNs the short
    ones); same-length texts are batched together."""
    monkeypatch.setattr(server, "em_model", lambda: ("M", _FakeProc()))
    monkeypatch.setattr(server, "EMBED_CACHE", False)
    calls = []

    def fake_generate(model, p, texts):
        assert model == "M"
        lens = {len(t) for t in texts}
        assert len(lens) == 1  # the invariant: never a mixed-length batch
        calls.append(list(texts))
        return _fake_out([[float(len(t))] * 3 for t in texts])

    monkeypatch.setattr(mlx_embeddings, "generate", fake_generate)

    # "aa" and "cd" share length 2 and must batch together; "efg" (length 3) is separate.
    vecs, prompt_tokens = server.embeddings(["aa", "cd", "efg"])

    assert calls == [["aa", "cd"], ["efg"]]
    assert vecs == [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    assert prompt_tokens == 2 + 2 + 3


def test_embeddings_dedups_repeated_text_within_one_call(monkeypatch):
    """The same text appearing twice in one request should only cost one forward pass."""
    monkeypatch.setattr(server, "em_model", lambda: ("M", _FakeProc()))
    monkeypatch.setattr(server, "EMBED_CACHE", False)
    calls = []

    def fake_generate(model, p, texts):
        calls.append(list(texts))
        return _fake_out([[float(len(t))] for t in texts])

    monkeypatch.setattr(mlx_embeddings, "generate", fake_generate)

    vecs, prompt_tokens = server.embeddings(["same", "same", "other"])

    assert calls == [["same"], ["other"]]  # "same" embedded once despite appearing twice
    assert vecs[0] == vecs[1] == [4.0]
    assert vecs[2] == [5.0]
    assert prompt_tokens == 4 + 4 + 5  # usage still reflects what was actually requested


def test_embeddings_empty_input(monkeypatch):
    monkeypatch.setattr(server, "em_model", lambda: ("M", _FakeProc()))
    monkeypatch.setattr(
        mlx_embeddings, "generate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called"))
    )
    assert server.embeddings([]) == ([], 0)


def test_embeddings_cache_hit_skips_generate(monkeypatch):
    """A text already in the content-hash cache must not trigger a forward pass at all."""
    monkeypatch.setattr(server, "em_model", lambda: ("M", _FakeProc()))
    monkeypatch.setattr(server, "EMBED_CACHE", True)
    cache = _FakeCache()
    monkeypatch.setattr(server, "_embed_cache", lambda: cache)
    cache.store[server._embed_hash("cached")] = [9.0, 9.0]

    calls = []

    def fake_generate(model, p, texts):
        calls.append(list(texts))
        return _fake_out([[float(len(t))] for t in texts])

    monkeypatch.setattr(mlx_embeddings, "generate", fake_generate)

    vecs, prompt_tokens = server.embeddings(["cached", "new"])

    assert calls == [["new"]]  # only the uncached text hit the model
    assert vecs == [[9.0, 9.0], [3.0]]
    assert prompt_tokens == 3  # only the uncached text is tokenized/counted, not "cached" too
    assert cache.puts == [server._embed_hash("new")]  # newly embedded text gets cached


def test_embed_cache_put_get_roundtrip(tmp_path):
    """Vectors survive the float32 BLOB pack/unpack round-trip (issue #55)."""
    cache = server._EmbedCache(str(tmp_path / "cache.sqlite3"))
    vec = [1.5, -2.25, 0.0, 3.0]
    cache.put("h1", vec)
    assert cache.get("h1") == vec


def test_embed_cache_miss_returns_none(tmp_path):
    cache = server._EmbedCache(str(tmp_path / "cache.sqlite3"))
    assert cache.get("nope") is None


def test_embed_cache_evicts_oldest_past_size_cap(tmp_path):
    """Once the on-disk cache exceeds its cap, the oldest (by last_used) rows are evicted first,
    and the cache stays bounded rather than growing forever (issue #55)."""
    cache = server._EmbedCache(str(tmp_path / "cache.sqlite3"))
    for i in range(10):
        cache.put(f"h{i}", [float(i)] * 64)
    # Cap it at the size it happens to be at after 10 rows, then keep writing past that.
    page_count, page_size = (
        cache._conn.execute("PRAGMA page_count").fetchone()[0],
        cache._conn.execute("PRAGMA page_size").fetchone()[0],
    )
    cache._max_bytes = page_count * page_size

    for i in range(10, 30):
        cache.put(f"h{i}", [float(i)] * 64)

    n = cache._conn.execute("SELECT COUNT(*) FROM embeddings_v2").fetchone()[0]
    assert 0 < n < 30  # eviction ran, but the cache isn't emptied outright
    assert cache.get("h0") is None  # oldest entries are the ones gone
    assert cache.get("h29") == [29.0] * 64  # most recently put entry survives


def test_embed_cache_enables_wal_mode(tmp_path):
    """journal_mode=WAL (+ synchronous=NORMAL) so recency updates don't each pay a synchronous
    fsync (issue #76)."""
    cache = server._EmbedCache(str(tmp_path / "cache.sqlite3"))
    mode = cache._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    sync = cache._conn.execute("PRAGMA synchronous").fetchone()[0]
    assert sync == 1  # NORMAL


def test_embed_cache_get_batches_last_used_touches(tmp_path, monkeypatch):
    """Repeated get() hits accumulate last_used in memory and don't commit (i.e. don't fsync)
    until the batch threshold is reached -- a fully-cached re-index run shouldn't pay one
    write-transaction per hit (issue #76)."""
    monkeypatch.setattr(server, "_EMBED_CACHE_TOUCH_BATCH", 3)
    path = str(tmp_path / "cache.sqlite3")
    cache = server._EmbedCache(path)

    clock = iter([100.0, 200.0, 201.0, 202.0])
    monkeypatch.setattr(server.time, "time", lambda: next(clock))

    cache.put("h1", [1.0, 2.0])  # last_used = 100.0 (consumes first clock value)

    def on_disk_last_used():
        # A second connection only sees committed data -- this proves whether get() flushed.
        other = server.sqlite3.connect(path)
        try:
            return other.execute(
                "SELECT last_used FROM embeddings_v2 WHERE hash = ?", ("h1",)
            ).fetchone()[0]
        finally:
            other.close()

    assert cache.get("h1") == [1.0, 2.0]  # hit #1 -> pending last_used = 200.0
    assert cache.get("h1") == [1.0, 2.0]  # hit #2 -> pending last_used = 201.0
    assert cache._pending_touches == {"h1": 201.0}
    assert on_disk_last_used() == 100.0  # neither hit has been committed yet

    assert cache.get("h1") == [1.0, 2.0]  # hit #3 crosses the threshold -> flush + commit
    assert cache._pending_touches == {}
    assert on_disk_last_used() == 202.0


def test_embed_cache_put_flushes_pending_touches(tmp_path, monkeypatch):
    """put() flushes any pending last_used touches (folded into its own commit) instead of
    leaving them stranded in memory (issue #76)."""
    monkeypatch.setattr(server, "_EMBED_CACHE_TOUCH_BATCH", 1000)  # never hit via count alone
    cache = server._EmbedCache(str(tmp_path / "cache.sqlite3"))
    cache.put("h1", [1.0])
    cache.get("h1")
    assert cache._pending_touches  # one touch pending, below the batch threshold

    cache.put("h2", [2.0])
    assert cache._pending_touches == {}  # put() flushed it


def test_embed_cache_close_flushes_pending_touches(tmp_path, monkeypatch):
    """close() flushes pending touches so a clean shutdown doesn't drop recency data that only
    existed in memory (issue #76)."""
    monkeypatch.setattr(server, "_EMBED_CACHE_TOUCH_BATCH", 1000)
    path = str(tmp_path / "cache.sqlite3")
    cache = server._EmbedCache(path)
    cache.put("h1", [1.0])
    cache.get("h1")
    assert cache._pending_touches
    cache.close()

    other = server.sqlite3.connect(path)
    try:
        last_used = other.execute(
            "SELECT last_used FROM embeddings_v2 WHERE hash = ?", ("h1",)
        ).fetchone()[0]
    finally:
        other.close()
    assert last_used > 0


def test_embeddings_cache_disabled_never_touches_cache(monkeypatch):
    monkeypatch.setattr(server, "em_model", lambda: ("M", _FakeProc()))
    monkeypatch.setattr(server, "EMBED_CACHE", False)

    def boom():
        raise AssertionError("_embed_cache() should not be called when EMBED_CACHE is False")

    monkeypatch.setattr(server, "_embed_cache", boom)
    monkeypatch.setattr(mlx_embeddings, "generate", lambda m, p, t: _fake_out([[1.0] for _ in t]))

    vecs, _ = server.embeddings(["x"])
    assert vecs == [[1.0]]


class _FakeJob:
    """Minimal stand-in for server.Job: just payload + out + cancel, no thread/queue plumbing."""

    def __init__(self, texts):
        self.kind = "embed"
        self.payload = {"texts": texts}
        self.out = server.queue.Queue()
        self.cancel = server.threading.Event()


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


def test_drain_short_fails_chat_job_when_put_back_races_full(monkeypatch):
    """Issue #58: if the put-back of a chat job loses a race to a producer filling the queue,
    _drain_short must fail that job with the same queue_full error _submit's callers get at
    admission time -- not dispatch it inline, which would nest a full chat generation loop
    inside the caller's own token loop (_drain_short is called from _run_chat's token loop)."""

    class _AlwaysFullQueue(queue.PriorityQueue):
        def put_nowait(self, item):
            raise queue.Full

    fake_q = _AlwaysFullQueue()
    job = server.Job("chat", {})
    fake_q.put((server.P_CHAT, next(server._seq), job))  # seed via blocking put, not put_nowait
    monkeypatch.setattr(server, "_q", fake_q)

    ran = []
    monkeypatch.setattr(server, "_dispatch", lambda j: ran.append(j))

    server._drain_short()

    assert ran == []  # never dispatched inline
    kind, data = job.out.get_nowait()
    assert kind == "error"
    assert data == "queue full (maxQueue)"


def test_drain_short_empty_queue_is_a_noop(monkeypatch):
    """Issue #79: the overwhelmingly common call to _drain_short (called once per generated
    token) finds an empty queue. It must return without ever calling _q.get_nowait() -- that's
    the queue.Empty exception this guard exists to avoid raising on every token."""
    assert server._q.empty()

    def _boom():
        raise AssertionError("get_nowait() should not be called when the queue is empty")

    monkeypatch.setattr(server._q, "get_nowait", _boom)
    monkeypatch.setattr(server, "_dispatch", lambda job: pytest.fail("must not dispatch"))

    server._drain_short()  # must not raise, must not touch get_nowait/_dispatch


def test_drain_short_still_works_when_queue_non_empty(monkeypatch):
    """The empty-queue fast path in _drain_short must not break the normal (non-empty) path."""
    ran = []
    monkeypatch.setattr(server, "_dispatch", lambda job: ran.append(job))
    server._q.put_nowait((server.P_SHORT, next(server._seq), "job-x"))

    server._drain_short()

    assert ran == ["job-x"]
    assert server._q.empty()
