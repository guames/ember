"""HTTP/1.1 keep-alive tests (issue #19).

Two angles, both model-free:
1. A real ThreadingHTTPServer + Handler round trip on a model-free GET
   endpoint (/v1/models), proving the server now negotiates HTTP/1.1
   keep-alive and reuses the TCP connection across requests.
2. A direct unit test of Handler._stream_out, proving the SSE streaming
   path explicitly closes its connection (it has no Content-Length/chunked
   framing, so keep-alive there would leave the client unable to tell where
   the response ends).
"""

import http.client
import io
import queue
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


def test_handler_negotiates_http11(live_server):
    assert server.Handler.protocol_version == "HTTP/1.1"


def test_json_endpoint_keeps_connection_alive(live_server):
    host, port = live_server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        conn.request("GET", "/v1/models")
        r1 = conn.getresponse()
        body1 = r1.read()
        assert r1.status == 200
        assert body1
        assert (r1.getheader("Connection") or "").lower() != "close"

        # Same connection, second request -> proves the socket wasn't closed.
        conn.request("GET", "/status")
        r2 = conn.getresponse()
        body2 = r2.read()
        assert r2.status == 200
        assert body2
    finally:
        conn.close()


class _FakeJob:
    def __init__(self, events):
        self.out = queue.Queue()
        for e in events:
            self.out.put(e)
        self.cancel = threading.Event()


def test_stream_out_forces_connection_close():
    h = server.Handler.__new__(server.Handler)
    h.wfile = io.BytesIO()
    h.request_version = "HTTP/1.1"
    h.requestline = "POST /v1/chat/completions HTTP/1.1"

    job = _FakeJob([("delta", "hi"), ("done", {"prompt": 1, "completion": 1})])
    h._stream_out(job, "chatcmpl-test", 0, "qwen2.5-coder-1.5b", include_usage=False)

    raw = h.wfile.getvalue().decode()
    header_block = raw.split("\r\n\r\n", 1)[0]
    assert "Connection: close" in header_block
    assert h.close_connection is True
