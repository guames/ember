"""Malformed JSON body and request body size cap (issue #53); invalid `seed`/`logit_bias`
values (issue #82).

Angles, all model-free, against a live server unless noted:
1. Invalid JSON in a POST body -> 400 with the OpenAI-shaped error envelope, connection
   stays usable (no dropped connection).
2. Content-Length above EMBER_MAX_BODY_MB -> 413 before the body is read off the socket.
3. A non-numeric `seed` or `logit_bias` key/value -> 400 invalid_request_error from the
   handler, before a job ever reaches the GPU worker (previously an opaque 500 from
   `int("abc")` deep inside the worker/_logits_processors).
"""

import http.client
import json
import threading

import pytest

from ember import server


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


def test_malformed_json_returns_400_envelope(live_server):
    host, port = live_server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        conn.request("POST", "/v1/chat/completions", body=b"{not json")
        r = conn.getresponse()
        body = r.read()
        assert r.status == 400
        assert b"invalid_json" in body
        assert b"invalid_request_error" in body

        # connection must still be usable afterwards -- no dropped socket
        conn.request("GET", "/health")
        r = conn.getresponse()
        assert r.status == 200
        r.read()
    finally:
        conn.close()


def test_empty_body_defaults_to_empty_object(live_server):
    """Existing behavior: no body at all is treated as {} rather than a parse error.

    Posts to an unrouted path (404) so the assertion isolates body-parsing from the
    worker queue -- the worker thread isn't running in this test harness.
    """
    host, port = live_server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        conn.request("POST", "/nope", body=b"")
        r = conn.getresponse()
        r.read()
        assert r.status == 404
    finally:
        conn.close()


@pytest.fixture
def tiny_body_cap():
    server.MAX_BODY_BYTES = 16
    try:
        yield
    finally:
        server.MAX_BODY_BYTES = int(server.MAX_BODY_MB * 1024 * 1024)


def test_oversized_body_returns_413_without_reading(live_server, tiny_body_cap):
    host, port = live_server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        oversized = b'{"messages": ' + b"[1]" * 20 + b"}"
        assert len(oversized) > server.MAX_BODY_BYTES
        conn.request("POST", "/v1/chat/completions", body=oversized)
        r = conn.getresponse()
        body = r.read()
        assert r.status == 413
        assert b"request_too_large" in body
    finally:
        conn.close()


def test_body_at_or_under_cap_is_accepted(live_server, tiny_body_cap):
    """Posts to an unrouted path (404) so the assertion isolates the cap check from the
    worker queue -- the worker thread isn't running in this test harness."""
    host, port = live_server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        conn.request("POST", "/nope", body=b"{}")
        r = conn.getresponse()
        r.read()
        assert r.status == 404
    finally:
        conn.close()


# ---------------------------------------------------------------- seed/logit_bias (issue #82)
def test_non_numeric_seed_returns_400(live_server):
    """seed: "abc" used to reach `int(seed)` inside the worker and surface as a 500; the
    handler now rejects it before the job is ever queued (no worker thread needed here)."""
    host, port = live_server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        payload = json.dumps({"model": "whatever", "seed": "abc"}).encode()
        conn.request("POST", "/v1/chat/completions", body=payload)
        r = conn.getresponse()
        body = r.read()
        assert r.status == 400
        assert b"invalid_request_error" in body
        assert b"seed" in body

        # connection must still be usable afterwards -- no dropped socket
        conn.request("GET", "/health")
        r = conn.getresponse()
        assert r.status == 200
        r.read()
    finally:
        conn.close()


def test_non_numeric_logit_bias_key_returns_400(live_server):
    """logit_bias with a non-token-id key used to reach `int(k)` inside
    _logits_processors and surface as a 500; the handler now rejects it up front."""
    host, port = live_server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        payload = json.dumps({"model": "whatever", "logit_bias": {"not-a-token": 1}}).encode()
        conn.request("POST", "/v1/chat/completions", body=payload)
        r = conn.getresponse()
        body = r.read()
        assert r.status == 400
        assert b"invalid_request_error" in body
        assert b"logit_bias" in body
    finally:
        conn.close()


def test_non_dict_logit_bias_returns_400(live_server):
    host, port = live_server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        payload = json.dumps({"model": "whatever", "logit_bias": "nope"}).encode()
        conn.request("POST", "/v1/chat/completions", body=payload)
        r = conn.getresponse()
        body = r.read()
        assert r.status == 400
        assert b"invalid_request_error" in body
    finally:
        conn.close()


@pytest.mark.parametrize(
    "body",
    [
        {},  # no seed/logit_bias at all
        {"seed": None},
        {"seed": 42},
        {"seed": "42"},  # numeric string is fine, mirrors int(seed) in the worker
        {"logit_bias": None},
        {"logit_bias": {}},
        {"logit_bias": {"123": 1.5, "456": -2}},
        {"logit_bias": {123: 1.5}},  # int keys (valid JSON only allows string keys, but
        # body validation itself should tolerate them since json.loads never produces them)
    ],
)
def test_validate_generation_params_accepts_valid_values(body):
    assert server._validate_generation_params(body) is None


@pytest.mark.parametrize(
    "body",
    [
        {"seed": "abc"},
        {"seed": []},
        {"seed": {}},
        {"logit_bias": "nope"},
        {"logit_bias": [1, 2]},
        {"logit_bias": {"not-a-token": 1}},
        {"logit_bias": {"123": "not-a-number"}},
    ],
)
def test_validate_generation_params_rejects_bad_values(body):
    assert server._validate_generation_params(body) is not None
