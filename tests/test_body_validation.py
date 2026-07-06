"""Malformed JSON body and request body size cap (issue #53).

Two angles, both model-free, against a live server:
1. Invalid JSON in a POST body -> 400 with the OpenAI-shaped error envelope, connection
   stays usable (no dropped connection).
2. Content-Length above EMBER_MAX_BODY_MB -> 413 before the body is read off the socket.
"""

import http.client
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
