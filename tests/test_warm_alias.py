"""The 'warm' model alias (issue #89).

`model: "warm"` resolves to the most recently used loaded chat model — whatever
was warmed last (e.g. from the menu bar). Nothing loaded -> MLX_WARM_DEFAULT
when it names a known model, else a clear 404: the router never picks (and
silently loads) a model on its own.

Mirrors tests/test_ac_em_keep_alive.py's style — monkeypatch module globals,
no real weights or mlx calls; the 404 path runs against a live model-free server
like tests/test_body_validation.py.
"""

import http.client
import json
import threading

import pytest

from ember import server


def _slot(last):
    return {"last": last}


# ---------------------------------------------------------------- _warm_model
def test_resolves_to_most_recently_used_chat(monkeypatch):
    monkeypatch.setattr(
        server, "_chat", {"a": _slot(1.0), "b": _slot(5.0), "c": _slot(3.0)}
    )
    assert server._warm_model() == "b"


def test_nothing_loaded_and_no_default_is_none(monkeypatch):
    monkeypatch.setattr(server, "_chat", {})
    monkeypatch.delenv("MLX_WARM_DEFAULT", raising=False)
    assert server._warm_model() is None


def test_default_env_used_when_cold(monkeypatch):
    monkeypatch.setattr(server, "_chat", {})
    monkeypatch.setattr(server, "CFG", {"m1": {"name": "m1", "mlx": "org/M1"}})
    monkeypatch.setenv("MLX_WARM_DEFAULT", "m1")
    assert server._warm_model() == "m1"


def test_default_env_must_name_a_known_model(monkeypatch):
    monkeypatch.setattr(server, "_chat", {})
    monkeypatch.setattr(server, "CFG", {"m1": {"name": "m1", "mlx": "org/M1"}})
    monkeypatch.setenv("MLX_WARM_DEFAULT", "nope")
    assert server._warm_model() is None


def test_loaded_model_wins_over_default(monkeypatch):
    monkeypatch.setattr(server, "_chat", {"hot": _slot(2.0)})
    monkeypatch.setenv("MLX_WARM_DEFAULT", "m1")
    assert server._warm_model() == "hot"


# ---------------------------------------------------------------- handler 404
@pytest.fixture
def live_server():
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        t.join()


def test_warm_cold_router_answers_clear_404(live_server, monkeypatch):
    monkeypatch.setattr(server, "_chat", {})
    monkeypatch.delenv("MLX_WARM_DEFAULT", raising=False)
    host, port = live_server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps({"model": "warm", "messages": []}).encode(),
        )
        r = conn.getresponse()
        body = r.read()
        assert r.status == 404
        assert b"model_not_found" in body
        assert b"warm" in body and b"MLX_WARM_DEFAULT" in body
    finally:
        conn.close()
