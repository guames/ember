"""CLI auth support (issue #75): `--api-key` / `EMBER_API_KEY` must be sent as a bearer
token on every management request, so the CLI can talk to a server with auth enabled.

Uses a tiny stand-in HTTP server (not `ember.server`) so these tests stay model-free and
don't pull in MLX.
"""

import http.server
import json
import threading

import pytest

from ember import cli


class _AuthHandler(http.server.BaseHTTPRequestHandler):
    """Mimics the bits of the real server's auth gate that the CLI talks to: /status
    requires `Authorization: Bearer <REQUIRED_KEY>` and returns a shape cmd_ps/cmd_status
    can parse; anything else is a 401 in the same envelope as ember.server."""

    REQUIRED_KEY = "secret123"

    def _authorized(self):
        return self.headers.get("Authorization") == f"Bearer {self.REQUIRED_KEY}"

    def do_GET(self):
        if self.path != "/status":
            self.send_response(404)
            self.end_headers()
            return
        if not self._authorized():
            body = json.dumps(
                {
                    "error": {
                        "message": "invalid or missing API key",
                        "type": "authentication_error",
                        "code": "invalid_api_key",
                    }
                }
            ).encode()
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = json.dumps(
            {
                "loaded": {"chat": [], "autocomplete": None, "embed": None},
                "memory": {},
                "queue": {"depth": 0, "max": 1},
                "policy": {
                    "max_runners": 1,
                    "min_free_gb": 1,
                    "min_free_cache_gb": 1,
                    "idle_timeout_s": 60,
                    "prompt_cache": True,
                    "kv_bits": None,
                    "prefill_step": 1,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # silence test output
        pass


@pytest.fixture
def auth_server():
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _AuthHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        host, port = httpd.server_address
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        t.join()


# ------------------------------------------------------------ argparse wiring
def test_api_key_flag_defaults_from_env(monkeypatch):
    monkeypatch.setenv("EMBER_API_KEY", "from-env")
    args = cli.build_parser().parse_args(["status"])
    assert args.api_key == "from-env"


def test_api_key_flag_overrides_env(monkeypatch):
    monkeypatch.setenv("EMBER_API_KEY", "from-env")
    args = cli.build_parser().parse_args(["status", "--api-key", "from-flag"])
    assert args.api_key == "from-flag"


def test_api_key_defaults_to_none_without_env(monkeypatch):
    monkeypatch.delenv("EMBER_API_KEY", raising=False)
    args = cli.build_parser().parse_args(["status"])
    assert args.api_key is None


def test_local_only_commands_have_no_api_key_flag():
    for argv in (["serve"], ["config"], ["version"]):
        args = cli.build_parser().parse_args(argv)
        assert not hasattr(args, "api_key")


# ------------------------------------------------------------ end-to-end against a stub server
def test_cli_succeeds_with_env_var(auth_server, monkeypatch, capsys):
    monkeypatch.setenv("EMBER_API_KEY", _AuthHandler.REQUIRED_KEY)
    args = cli.build_parser().parse_args(["ps", "--url", auth_server])
    cli.cmd_ps(args)
    assert "No chat model is hot." in capsys.readouterr().out


def test_cli_succeeds_with_flag(monkeypatch, auth_server, capsys):
    monkeypatch.delenv("EMBER_API_KEY", raising=False)
    args = cli.build_parser().parse_args(
        ["ps", "--url", auth_server, "--api-key", _AuthHandler.REQUIRED_KEY]
    )
    cli.cmd_ps(args)
    assert "No chat model is hot." in capsys.readouterr().out


def test_cli_gets_clean_401_without_key(monkeypatch, auth_server):
    monkeypatch.delenv("EMBER_API_KEY", raising=False)
    args = cli.build_parser().parse_args(["ps", "--url", auth_server])
    with pytest.raises(SystemExit) as e:
        cli.cmd_ps(args)
    assert "401" in str(e.value)
    assert "invalid_api_key" in str(e.value) or "invalid or missing API key" in str(e.value)


def test_cli_gets_clean_401_with_wrong_key(auth_server):
    args = cli.build_parser().parse_args(["ps", "--url", auth_server, "--api-key", "wrong"])
    with pytest.raises(SystemExit) as e:
        cli.cmd_ps(args)
    assert "401" in str(e.value)
