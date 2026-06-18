"""CLI de gestão do Ember.

`ember <comando> --help` mostra o que cada um faz. Os comandos de gestão (status, ps,
memory, list, run, warm, unload) falam HTTP com um servidor já rodando; `serve` sobe o
servidor; `config`/`version` são locais. Só `serve` importa o MLX — o resto é leve.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

from . import __version__


# ----------------------------------------------------------------- cliente HTTP
def _default_url():
    host = os.environ.get("MLX_ROUTER_HOST", "127.0.0.1")
    port = os.environ.get("MLX_ROUTER_PORT", "8000")
    return f"http://{host}:{port}"


def _request(url, path, method="GET", body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310


def _call(url, path, method="GET", body=None):
    """Faz a requisição e devolve o JSON; encerra com mensagem amigável se cair."""
    try:
        with _request(url, path, method, body) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        try:
            err = json.load(e).get("error", str(e))
        except Exception:  # noqa: BLE001
            err = str(e)
        sys.exit(f"erro {e.code}: {err}")
    except (urllib.error.URLError, ConnectionError, OSError):
        sys.exit(f"Ember não respondeu em {url}. Suba com `ember serve` (ou ajuste --url).")


def _dur(s):
    if s is None or s < 0:
        return "∞"
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    return f"{s // 3600}h"


# ----------------------------------------------------------------- comandos
def cmd_serve(args):
    if args.config:
        os.environ["EMBER_CONFIG"] = args.config  # antes de importar (registry no import)
    from . import server  # importa MLX só aqui

    server.serve(host=args.host, port=args.port)


def cmd_ps(args):
    chat = _call(args.url, "/status")["loaded"]
    rows = chat["chat"]
    if not rows:
        print("Nenhum modelo de chat quente.")
    else:
        print(f"{'MODELO':<30}{'TAM':>7}{'VISÃO':>7}{'OCIOSO':>8}{'KEEP':>7}{'CACHE':>8}")
        for c in rows:
            vis = "sim" if c["vision"] else "-"
            ka = _dur(c["keep_alive_s"])
            print(
                f"{c['name']:<30}{c['size_gb']:>6.1f}G{vis:>7}"
                f"{_dur(c['idle_s']):>8}{ka:>7}{c['cached_tokens']:>8}"
            )
    fixed = [n for n in ("autocomplete", "embed") if chat.get(n)]
    if fixed:
        print("fixos quentes:", ", ".join(fixed))


def cmd_status(args):
    st = _call(args.url, "/status")
    cmd_ps(args)
    m = st["memory"]
    sysm = m.get("system", {})
    if sysm:
        print(
            f"\nmemória: sistema {sysm['used_gb']:.1f}/{sysm['total_gb']:.0f}G "
            f"(livre {sysm['free_gb']:.1f}G), MLX active {m['mlx']['active_gb']:.1f}G"
        )
    q, p = st["queue"], st["policy"]
    print(f"fila: {q['depth']}/{q['max']}")
    print(
        "política: "
        f"runners≤{p['max_runners']} · min_free {p['min_free_gb']}G · "
        f"cache_relief<{p['min_free_cache_gb']}G · idle {_dur(int(p['idle_timeout_s']))} · "
        f"prompt_cache {'on' if p['prompt_cache'] else 'off'} · "
        f"kv_bits {p['kv_bits'] or 'fp16'} · prefill_step {p['prefill_step']}"
    )


def cmd_memory(args):
    m = _call(args.url, "/memory")
    mlx = m["mlx"]
    print(
        f"MLX     active {mlx['active_gb']:.2f}G  cache {mlx['cache_gb']:.2f}G  peak {mlx['peak_gb']:.2f}G"
    )
    sysm = m.get("system")
    if sysm:
        print(
            f"sistema usado {sysm['used_gb']:.1f}G / {sysm['total_gb']:.0f}G  "
            f"livre {sysm['free_gb']:.1f}G ({sysm['used_pct']}%)"
        )
    if m.get("router_rss_gb") is not None:
        print(f"ember   RSS {m['router_rss_gb']:.2f}G")


def cmd_list(args):
    try:
        with _request(args.url, "/status") as r:
            hot = {c["name"] for c in json.load(r)["loaded"]["chat"]}
        models = _call(args.url, "/v1/models")["data"]
        print(f"{'MODELO':<34}ESTADO")
        for mdl in models:
            mark = "● quente" if mdl["id"] in hot else "○ frio"
            print(f"{mdl['id']:<34}{mark}")
    except (urllib.error.URLError, ConnectionError, OSError):
        from .registry import _find_config, load_registry

        cfg, ac, em = load_registry()
        print(f"(servidor offline — lendo config local: {_find_config() or 'defaults'})\n")
        for name in cfg:
            print(f"  {name}{'  [visão]' if cfg[name]['vision'] else ''}")
        print(f"  {ac['name']} (autocomplete), {em['name']} (embed)")


def cmd_run(args):
    prompt = args.prompt
    if not prompt:
        prompt = sys.stdin.read().strip()
    if not prompt:
        sys.exit("nada p/ enviar: passe um prompt ou mande via stdin.")
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    try:
        with _request(args.url, "/v1/chat/completions", "POST", body) as resp:
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
        sys.exit(f"Ember não respondeu em {args.url}. Suba com `ember serve`.")


def cmd_warm(args):
    body = {"model": args.model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
    _call(args.url, "/v1/chat/completions", "POST", body)
    print(f"'{args.model}' carregado e quente.")
    cmd_ps(args)


def cmd_unload(args):
    res = _call(args.url, "/unload", "POST", {"target": args.target})
    freed = res.get("unloaded") or []
    print(f"descarregado: {', '.join(freed) if freed else 'nada'}")


def cmd_clear(args):
    res = _call(args.url, "/clear", "POST", {"target": args.target})
    cleared = res.get("cleared") or []
    print(f"limpo ({args.target}): {', '.join(cleared) if cleared else 'nada'}")


def cmd_config(args):
    from .registry import _find_config, load_registry

    path = _find_config()
    print(f"config: {path or '(nenhum arquivo — usando defaults; ver `ember config --help`)'}")
    try:
        cfg, ac, em = load_registry()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"config inválido: {e}")
    print(f"\n{len(cfg)} modelo(s) de chat:")
    for name, m in cfg.items():
        print(f"  {name:<28}{m['mlx']}{'  [visão]' if m['vision'] else ''}")
    print(f"autocomplete: {ac['name']} -> {ac['mlx']}")
    print(f"embed:        {em['name']} -> {em['mlx']}")


def cmd_version(args):
    print(f"ember {__version__}")


# ----------------------------------------------------------------- parser
def build_parser():
    p = argparse.ArgumentParser(
        prog="ember",
        description="Ember — servidor de inferência MLX OpenAI-compatible p/ Apple Silicon.",
        epilog="Use `ember <comando> --help` p/ detalhes de cada comando.",
    )
    p.add_argument("--version", action="version", version=f"ember {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="<comando>")

    def add(name, func, help, url=True):
        sp = sub.add_parser(name, help=help, description=help)
        sp.set_defaults(func=func)
        if url:
            sp.add_argument(
                "--url", default=_default_url(), help="base do servidor (default %(default)s)"
            )
        return sp

    s = add(
        "serve", cmd_serve, "Sobe o servidor (chat, autocomplete, embeddings, visão).", url=False
    )
    s.add_argument("--host", help="endereço de bind (default 127.0.0.1 / MLX_ROUTER_HOST)")
    s.add_argument("--port", type=int, help="porta (default 8000 / MLX_ROUTER_PORT)")
    s.add_argument("--config", help="caminho do arquivo de modelos (sobrepõe EMBER_CONFIG)")

    add("ps", cmd_ps, "Lista os modelos quentes na RAM (tamanho, ocioso, keep_alive, cache).")
    add("status", cmd_status, "Status completo: modelos quentes + memória + fila + política.")
    add("memory", cmd_memory, "Uso de memória (MLX + sistema).")
    add("list", cmd_list, "Lista os modelos configurados e quais estão quentes.")

    r = add("run", cmd_run, "Conversa rápida no terminal (stream). Prompt via arg ou stdin.")
    r.add_argument("model", help="nome do modelo (como no config)")
    r.add_argument("prompt", nargs="?", help="mensagem; se omitido, lê do stdin")

    w = add("warm", cmd_warm, "Pré-carrega um modelo na RAM (sem gerar resposta).")
    w.add_argument("model", help="nome do modelo a carregar")

    u = add("unload", cmd_unload, "Descarrega modelos: chat (default) | all | <nome>.")
    u.add_argument("target", nargs="?", default="chat", help="chat | all | <nome do modelo>")

    c = add(
        "clear",
        cmd_clear,
        "Limpa contexto (prompt cache) e/ou cache do MLX — mantém os modelos quentes.",
    )
    c.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=["context", "cache", "all"],
        help="context (KV da conversa) | cache (buffers MLX) | all (default)",
    )

    add(
        "config", cmd_config, "Mostra o arquivo de config resolvido e valida os modelos.", url=False
    )
    add("version", cmd_version, "Mostra a versão do Ember.", url=False)
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
