"""Ember management CLI.

`ember <command> --help` shows what each one does. The management commands (status, ps,
memory, metrics, list, run, warm, unload) talk HTTP to an already-running server; `serve`
starts the server; `config`/`version` are local. Only `serve` imports MLX — the rest is
lightweight.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

from . import __version__


# ----------------------------------------------------------------- HTTP client
def _default_url():
    host = os.environ.get("MLX_ROUTER_HOST", "127.0.0.1")
    port = os.environ.get("MLX_ROUTER_PORT", "8000")
    return f"http://{host}:{port}"


def _request(url, path, method="GET", body=None, timeout=600, api_key=None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=data,
        method=method,
        headers=headers,
    )
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310


def _call(url, path, method="GET", body=None, api_key=None):
    """Makes the request and returns the JSON; exits with a friendly message if it fails."""
    try:
        with _request(url, path, method, body, api_key=api_key) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        try:
            err = json.load(e).get("error", str(e))
        except Exception:  # noqa: BLE001
            err = str(e)
        sys.exit(f"error {e.code}: {err}")
    except (urllib.error.URLError, ConnectionError, OSError):
        sys.exit(f"Ember did not respond at {url}. Start it with `ember serve` (or set --url).")


def _call_text(url, path, api_key=None):
    """Like _call, but returns the raw response body (for the Prometheus text endpoint)."""
    try:
        with _request(url, path, api_key=api_key) as resp:
            return resp.read().decode()
    except urllib.error.HTTPError as e:
        sys.exit(f"error {e.code}: {e.read().decode(errors='replace')}")
    except (urllib.error.URLError, ConnectionError, OSError):
        sys.exit(f"Ember did not respond at {url}. Start it with `ember serve` (or set --url).")


def _dur(s):
    if s is None or s < 0:
        return "∞"
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    return f"{s // 3600}h"


# ----------------------------------------------------------------- commands
def cmd_serve(args):
    if args.config:
        os.environ["EMBER_CONFIG"] = args.config  # before importing (registry runs at import)
    from . import server  # import MLX only here

    server.serve(host=args.host, port=args.port)


def cmd_ps(args):
    chat = _call(args.url, "/status", api_key=args.api_key)["loaded"]
    rows = chat["chat"]
    if not rows:
        print("No chat model is hot.")
    else:
        print(f"{'MODEL':<30}{'SIZE':>7}{'VISION':>7}{'IDLE':>8}{'KEEP':>7}{'CACHE':>8}")
        for c in rows:
            vis = "yes" if c["vision"] else "-"
            ka = _dur(c["keep_alive_s"])
            print(
                f"{c['name']:<30}{c['size_gb']:>6.1f}G{vis:>7}"
                f"{_dur(c['idle_s']):>8}{ka:>7}{c['cached_tokens']:>8}"
            )
    fixed = [n for n in ("autocomplete", "embed") if chat.get(n)]
    if fixed:
        print("hot fixed:", ", ".join(fixed))


def cmd_status(args):
    st = _call(args.url, "/status", api_key=args.api_key)
    cmd_ps(args)
    m = st["memory"]
    sysm = m.get("system", {})
    if sysm:
        print(
            f"\nmemory: system {sysm['used_gb']:.1f}/{sysm['total_gb']:.0f}G "
            f"(free {sysm['free_gb']:.1f}G), MLX active {m['mlx']['active_gb']:.1f}G"
        )
    q, p = st["queue"], st["policy"]
    print(f"queue: {q['depth']}/{q['max']}")
    print(
        "policy: "
        f"runners≤{p['max_runners']} · min_free {p['min_free_gb']}G · "
        f"cache_relief<{p['min_free_cache_gb']}G · idle {_dur(int(p['idle_timeout_s']))} · "
        f"prompt_cache {'on' if p['prompt_cache'] else 'off'} · "
        f"kv_bits {p['kv_bits'] or 'fp16'} · prefill_step {p['prefill_step']}"
    )


def cmd_memory(args):
    m = _call(args.url, "/memory", api_key=args.api_key)
    mlx = m["mlx"]
    print(
        f"MLX     active {mlx['active_gb']:.2f}G  cache {mlx['cache_gb']:.2f}G  peak {mlx['peak_gb']:.2f}G"
    )
    sysm = m.get("system")
    if sysm:
        print(
            f"system  used {sysm['used_gb']:.1f}G / {sysm['total_gb']:.0f}G  "
            f"free {sysm['free_gb']:.1f}G ({sysm['used_pct']}%)"
        )
    if m.get("router_rss_gb") is not None:
        print(f"ember   RSS {m['router_rss_gb']:.2f}G")


def cmd_metrics(args):
    print(_call_text(args.url, "/metrics", api_key=args.api_key), end="")


def cmd_list(args):
    try:
        with _request(args.url, "/status", api_key=args.api_key) as r:
            hot = {c["name"] for c in json.load(r)["loaded"]["chat"]}
        models = _call(args.url, "/v1/models", api_key=args.api_key)["data"]
        print(f"{'MODEL':<34}STATE")
        for mdl in models:
            mark = "● hot" if mdl["id"] in hot else "○ cold"
            print(f"{mdl['id']:<34}{mark}")
    except (urllib.error.URLError, ConnectionError, OSError):
        from .registry import _find_config, load_registry

        cfg, ac, em = load_registry()
        print(f"(server offline — reading local config: {_find_config() or 'defaults'})\n")
        for name in cfg:
            print(f"  {name}{'  [vision]' if cfg[name]['vision'] else ''}")
        print(f"  {ac['name']} (autocomplete), {em['name']} (embed)")


def cmd_run(args):
    prompt = args.prompt
    if not prompt:
        prompt = sys.stdin.read().strip()
    if not prompt:
        sys.exit("nothing to send: pass a prompt or send it via stdin.")
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    try:
        with _request(args.url, "/v1/chat/completions", "POST", body, api_key=args.api_key) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content"):
                    sys.stdout.write(delta["content"])
                    sys.stdout.flush()
                if chunk["choices"][0].get("finish_reason"):
                    break
        print()
    except (urllib.error.URLError, ConnectionError, OSError):
        sys.exit(f"Ember did not respond at {args.url}. Start it with `ember serve`.")


def cmd_warm(args):
    body = {"model": args.model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
    _call(args.url, "/v1/chat/completions", "POST", body, api_key=args.api_key)
    print(f"'{args.model}' loaded and hot.")
    cmd_ps(args)


def cmd_unload(args):
    res = _call(args.url, "/unload", "POST", {"target": args.target}, api_key=args.api_key)
    freed = res.get("unloaded") or []
    print(f"unloaded: {', '.join(freed) if freed else 'nothing'}")


def cmd_clear(args):
    res = _call(args.url, "/clear", "POST", {"target": args.target}, api_key=args.api_key)
    cleared = res.get("cleared") or []
    print(f"cleared ({args.target}): {', '.join(cleared) if cleared else 'nothing'}")


def cmd_config(args):
    from .registry import _find_config, load_registry

    path = _find_config()
    print(f"config: {path or '(no file — using defaults; see `ember config --help`)'}")
    try:
        cfg, ac, em = load_registry()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"invalid config: {e}")
    print(f"\n{len(cfg)} chat model(s):")
    for name, m in cfg.items():
        print(f"  {name:<28}{m['mlx']}{'  [vision]' if m['vision'] else ''}")
    print(f"autocomplete: {ac['name']} -> {ac['mlx']}")
    print(f"embed:        {em['name']} -> {em['mlx']}")


def cmd_version(args):
    print(f"ember {__version__}")


# ----------------------------------------------------------------- parser
def build_parser():
    p = argparse.ArgumentParser(
        prog="ember",
        description="Ember — OpenAI-compatible MLX inference server for Apple Silicon.",
        epilog="Use `ember <command> --help` for details on each command.",
    )
    p.add_argument("--version", action="version", version=f"ember {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="<command>")

    def add(name, func, help, url=True):
        sp = sub.add_parser(name, help=help, description=help)
        sp.set_defaults(func=func)
        if url:
            sp.add_argument(
                "--url", default=_default_url(), help="server base (default %(default)s)"
            )
            sp.add_argument(
                "--api-key",
                default=os.environ.get("EMBER_API_KEY") or None,
                help="bearer token for an authenticated server (default $EMBER_API_KEY)",
            )
        return sp

    s = add(
        "serve", cmd_serve, "Start the server (chat, autocomplete, embeddings, vision).", url=False
    )
    s.add_argument("--host", help="bind address (default 127.0.0.1 / MLX_ROUTER_HOST)")
    s.add_argument("--port", type=int, help="port (default 8000 / MLX_ROUTER_PORT)")
    s.add_argument("--config", help="path to the models file (overrides EMBER_CONFIG)")

    add("ps", cmd_ps, "List the hot models in RAM (size, idle, keep_alive, cache).")
    add("status", cmd_status, "Full status: hot models + memory + queue + policy.")
    add("memory", cmd_memory, "Memory usage (MLX + system).")
    add("metrics", cmd_metrics, "Request counters + latency histogram (Prometheus text).")
    add("list", cmd_list, "List the configured models and which ones are hot.")

    r = add("run", cmd_run, "Quick chat in the terminal (stream). Prompt via arg or stdin.")
    r.add_argument("model", help="model name (as in the config)")
    r.add_argument("prompt", nargs="?", help="message; if omitted, reads from stdin")

    w = add("warm", cmd_warm, "Preload a model into RAM (without generating a response).")
    w.add_argument("model", help="name of the model to load")

    u = add("unload", cmd_unload, "Unload models: chat (default) | all | <name>.")
    u.add_argument("target", nargs="?", default="chat", help="chat | all | <model name>")

    c = add(
        "clear",
        cmd_clear,
        "Clear context (prompt cache) and/or the MLX cache — keeps the models hot.",
    )
    c.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=["context", "cache", "all"],
        help="context (conversation KV) | cache (MLX buffers) | all (default)",
    )

    add("config", cmd_config, "Show the resolved config file and validate the models.", url=False)
    add("version", cmd_version, "Show the Ember version.", url=False)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
