"""Ember — servidor de inferência MLX OpenAI-compatible p/ Apple Silicon.

Um processo, três capacidades, uma política de memória (multi-runner, keep_alive,
preempção, prompt cache). Pensado p/ assistentes de código locais (ex.: Continue).

Serve 3 capacidades, todas MLX:
  • /v1/chat/completions  -> modelo de chat/código (multi-runner com budget de RAM;
                             aceita `tools`/`tool_choice` OpenAI e devolve tool_calls)
  • /v1/completions       -> autocomplete FIM (Qwen2.5-Coder-1.5B base, fixo na RAM)
  • /v1/embeddings        -> embeddings (nomic-modernbert, fixo na RAM)
Operacao/observabilidade:
  • GET  /status          -> modelos quentes + memória + fila + política
  • GET  /memory          -> memória MLX e do sistema (em uso / livre)
  • POST /unload {target}  -> descarrega ('chat' | 'all' | '<nome-do-modelo>')

Robustez estilo Ollama (1 worker de GPU + fila com prioridade):
  - autocomplete/embed têm prioridade e furam a fila do chat (rodam ENTRE os
    tokens do chat -> digitar não trava durante a geração);
  - maxQueue rejeita com 503 quando sobrecarregado; cancela se o cliente cai;
  - multi-runner: mantém >1 modelo de chat quente enquanto sobrar >= MLX_MIN_FREE_GB
    de RAM livre (e <= MLX_MAX_RUNNERS, teto de segurança); LRU evicta o resto;
  - keep_alive: idle-unload por modelo (env MLX_IDLE_TIMEOUT, ou campo keep_alive
    por requisição). A próxima chamada recarrega sozinho.

Prompt cache (KV reuse, estilo Ollama/llama.cpp): cada runner mantém 1 slot de KV
cache; a cada request reusa o maior prefixo comum de tokens (system+histórico) e só
processa o sufixo novo -> corta TTFT em conversas/edições. Zero deepcopy. MLX_PROMPT_CACHE=0 desliga.

Sob pressão de RAM (free < MLX_MIN_FREE_CACHE_GB, default 1GB) o router larga os KV
caches (LRU, mais antigo primeiro) ANTES de evictar um modelo inteiro.

KV cache quantizado (opcional, mais contexto na mesma RAM): MLX_KV_BITS=8 (ou 4) liga;
8-bit é ~2x menor que fp16 e praticamente lossless. Compatível com o prompt cache (o
QuantizedKVCache é trimável). Desligado por padrão.

Tuning de memória no boot: wired_limit (pesos residentes, sem compressão do SO perto do
limite) + prefill chunkado (MLX_PREFILL_STEP, pico de RAM ↓ no prompt longo a frio; com
o prompt cache o prefill normal já é só o sufixo, então quase não custa).

Envs: MLX_ROUTER_PORT(8000) MLX_ROUTER_HOST(127.0.0.1) MLX_IDLE_TIMEOUT(300)
      MLX_MAX_RUNNERS(4) MLX_MIN_FREE_GB(2.0) MLX_MIN_FREE_CACHE_GB(1.0)
      MLX_MAX_QUEUE(32) MLX_PROMPT_CACHE(1) MLX_KV_BITS(off) MLX_KV_GROUP_SIZE(64)
      MLX_KV_QUANT_START(0) MLX_PREFILL_STEP(512) MLX_WIRED_LIMIT_GB(auto)
      MLX_CACHE_LIMIT_GB(off)

    ember                          # CLI (porta 8000 ou env MLX_ROUTER_PORT)
    python -m ember
"""

import gc
import itertools
import json
import os
import queue
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from .registry import load_registry

try:
    import psutil
except Exception:  # noqa: BLE001
    psutil = None

# Registro de modelos (arquivo de config; ver registry.py). CFG = chat/código/visão.
CFG, _AC, _EM = load_registry()
AC_NAME, AC_REPO = _AC["name"], _AC["mlx"]  # autocomplete FIM (fixo na RAM)
EM_NAME, EM_REPO = _EM["name"], _EM["mlx"]  # embeddings (fixo na RAM)

# ---- política (envs) ----
IDLE_TIMEOUT = float(os.environ.get("MLX_IDLE_TIMEOUT", "300"))  # s; 0/neg = nunca
MAX_RUNNERS = int(os.environ.get("MLX_MAX_RUNNERS", "4"))  # teto de segurança
MIN_FREE_GB = float(os.environ.get("MLX_MIN_FREE_GB", "2.0"))  # headroom p/ evictar modelo
MIN_FREE_CACHE_GB = float(os.environ.get("MLX_MIN_FREE_CACHE_GB", "1.0"))  # piso p/ largar KV cache
MAX_QUEUE = int(os.environ.get("MLX_MAX_QUEUE", "32"))
PROMPT_CACHE = os.environ.get("MLX_PROMPT_CACHE", "1") not in ("0", "false", "")  # KV reuse
_KVB = os.environ.get("MLX_KV_BITS")  # 8/4 = quantiza KV cache
KV_BITS = int(_KVB) if _KVB not in (None, "", "0") else None
KV_GROUP_SIZE = int(os.environ.get("MLX_KV_GROUP_SIZE", "64"))
KV_QUANT_START = int(os.environ.get("MLX_KV_QUANT_START", "0"))  # quantiza a partir do token N
PREFILL_STEP = int(os.environ.get("MLX_PREFILL_STEP", "512"))  # chunk do prefill (pico de RAM ↓)
WIRED_LIMIT_GB = float(os.environ.get("MLX_WIRED_LIMIT_GB", "0"))  # 0 = auto (total-5GB)
CACHE_LIMIT_GB = float(os.environ.get("MLX_CACHE_LIMIT_GB", "0"))  # 0 = default do MLX (sem cap)
_DEFAULT_KA = IDLE_TIMEOUT if IDLE_TIMEOUT > 0 else -1  # -1 = nunca expira


def _kv_kwargs():
    """kwargs de KV-cache quantizado p/ o generate_step (vazio = KV em fp16, default)."""
    if KV_BITS is None:
        return {}
    return {
        "kv_bits": KV_BITS,
        "kv_group_size": KV_GROUP_SIZE,
        "quantized_kv_start": KV_QUANT_START,
    }


def _tune_memory():
    """Tuning de memória no boot: wired_limit mantém os pesos residentes (sem o SO
    comprimir/paginar perto do limite de RAM -> velocidade consistente); cache_limit
    (opcional) limita o pool de buffers do MLX, devolvendo RAM ao SO."""
    try:
        total = (psutil.virtual_memory().total / 1024**3) if psutil else 24.0
        wl = WIRED_LIMIT_GB or max(4.0, total - 5.0)  # auto: deixa ~5GB p/ o SO
        mx.set_wired_limit(int(wl * 1024**3))
        extra = ""
        if CACHE_LIMIT_GB > 0:
            mx.set_cache_limit(int(CACHE_LIMIT_GB * 1024**3))
            extra = f" cache_limit={CACHE_LIMIT_GB:.0f}GB"
        print(
            f"[router] mem: wired_limit={wl:.0f}GB prefill_step={PREFILL_STEP}{extra}", flush=True
        )
    except Exception as e:  # noqa: BLE001
        print(f"[router] mem tuning falhou (segue sem): {e}", flush=True)


# ---- estado dos modelos (mutado SÓ pelo worker; _reg_lock guarda a estrutura) ----
_reg_lock = threading.Lock()
_chat = {}  # name -> {model, tok, size_gb, last, ka}
_ac = {"model": None, "tok": None}  # autocomplete (fixo)
_em = {"model": None, "proc": None}  # embed (fixo)

# ---- fila de GPU (1 worker) ----
P_SHORT, P_CHAT = 0, 1  # menor = maior prioridade
_q = queue.PriorityQueue(maxsize=MAX_QUEUE)
_seq = itertools.count()


class Job:
    __slots__ = ("kind", "payload", "out", "cancel")

    def __init__(self, kind, payload):
        self.kind = kind
        self.payload = payload
        self.out = queue.Queue()
        self.cancel = threading.Event()


def _submit(prio, kind, payload):
    """Enfileira um job de GPU. Retorna o Job, ou None se a fila estiver cheia."""
    job = Job(kind, payload)
    try:
        _q.put_nowait((prio, next(_seq), job))
    except queue.Full:
        return None
    return job


def _gb(b):
    return round(b / 1024**3, 2)


# ---------------------------------------------------------------- modelos (worker)
def _evict(name):
    """Descarrega um modelo de chat (chamado só no worker)."""
    with _reg_lock:
        m = _chat.pop(name, None)
    if m is None:
        return
    m["model"] = m["tok"] = m["pc"] = m["pctoks"] = None  # libera tb o KV cache do slot
    gc.collect()
    mx.clear_cache()
    print(f"[router] evict {name}", flush=True)


def _free_gb():
    return psutil.virtual_memory().available / 1024**3 if psutil else None


def _cache_bytes(pc):
    try:
        return sum(getattr(c, "nbytes", 0) for c in pc)
    except Exception:  # noqa: BLE001
        return 0


def _relieve_cache(keep):
    """Sob pressão de RAM (free < MIN_FREE_CACHE_GB) descarta os KV caches dos runners —
    do mais antigo (LRU) ao mais novo, `keep` por último. Bem mais barato que evictar o
    modelo (o peso fica quente; só reprocessa o prompt no próximo turno). Roda no worker.
    Usa o tamanho do cache (.nbytes) como recuperação prevista (o SO demora a refletir)."""
    free = _free_gb()
    if free is None or free >= MIN_FREE_CACHE_GB:
        return free
    with _reg_lock:
        order = sorted(
            (n for n in _chat if n != keep and _chat[n].get("pc") is not None),
            key=lambda n: _chat[n]["last"],
        )
        if keep in _chat and _chat[keep].get("pc") is not None:
            order.append(keep)  # o do request atual só em último caso
    for n in order:
        with _reg_lock:
            e = _chat.get(n)
            if not e or e.get("pc") is None:
                continue
            freed = _cache_bytes(e["pc"]) / 1024**3
            e["pc"] = e["pctoks"] = None
        gc.collect()
        mx.clear_cache()
        print(
            f"[router] RAM baixa (<{MIN_FREE_CACHE_GB:.1f}GB): descartou KV cache de "
            f"{n} (~{freed:.2f}GB)",
            flush=True,
        )
        free = free + freed if free is not None else None
        if free is None or free >= MIN_FREE_CACHE_GB:
            break
    return free


def _enforce_memory(keep):
    """Política de RAM: 1) sob pressão crítica (<MIN_FREE_CACHE_GB) larga KV caches (LRU,
    barato); 2) se ainda faltar (<MIN_FREE_GB ou >MAX_RUNNERS) evicta modelo LRU (nunca
    `keep`). Usa tamanho medido como recuperação prevista (o SO demora a refletir o free)."""
    _relieve_cache(keep)
    free = _free_gb()
    while True:
        with _reg_lock:
            n = len(_chat)
            if n <= 1:
                return  # 1o modelo sempre fica (estilo Ollama)
            over = n > MAX_RUNNERS or (free is not None and free < MIN_FREE_GB)
            if not over:
                return
            victims = sorted((x for x in _chat if x != keep), key=lambda x: _chat[x]["last"])
            if not victims:
                return
            victim = victims[0]
            vsize = _chat[victim]["size_gb"]
        _evict(victim)
        if free is not None:
            free += vsize  # recuperação prevista


def chat_model(name):
    """Garante o modelo de chat residente; carrega+mede tamanho+aplica budget.
    Modelos com `vision: true` no config carregam via mlx_vlm (model, processor).
    Retorna (model, tok_ou_processor, is_vlm)."""
    if name not in CFG:
        raise KeyError(name)
    with _reg_lock:
        m = _chat.get(name)
        if m is not None:
            m["last"] = time.monotonic()
            return m["model"], m["tok"], m["vlm"]
    vlm = bool(CFG[name].get("vision"))
    before = mx.get_active_memory()
    print(f"[router] {'vlm' if vlm else 'chat'}: carregando {name} ...", flush=True)
    if vlm:
        import mlx_vlm

        model, tok = mlx_vlm.load(CFG[name]["mlx"])
    else:
        model, tok = load(CFG[name]["mlx"])
        # alguns tokenizers só listam <|endoftext|> em eos_token_ids; o terminador do
        # chat template (ex.: <|im_end|>) fica de fora e vazaria como texto no fim.
        eid = getattr(tok, "eos_token_id", None)
        if eid is not None:
            try:
                tok.eos_token_ids.add(eid)
            except (AttributeError, TypeError):
                pass
    mx.eval([v for _, v in tree_flatten(model.parameters())])  # materializa p/ medir
    size = _gb(mx.get_active_memory() - before)
    with _reg_lock:
        _chat[name] = {
            "model": model,
            "tok": tok,
            "size_gb": size,
            "last": time.monotonic(),
            "ka": _DEFAULT_KA,
            "vlm": vlm,
            "pc": None,
            "pctoks": None,
        }  # slot de prompt cache (KV reuse)
    _enforce_memory(keep=name)
    return model, tok, vlm


def ac_model():
    if _ac["model"] is None:
        print("[router] autocomplete: carregando 1.5B FIM ...", flush=True)
        _ac["model"], _ac["tok"] = load(AC_REPO)
    return _ac["model"], _ac["tok"]


def em_model():
    if _em["model"] is None:
        import mlx_embeddings

        print("[router] embed: carregando modernbert ...", flush=True)
        _em["model"], _em["proc"] = mlx_embeddings.load(EM_REPO)
    return _em["model"], _em["proc"]


def _normalize_messages(messages):
    """Normaliza mensagens p/ o chat template. Em assistant com tool_calls, converte
    function.arguments de string JSON -> objeto (a maioria dos templates espera dict)
    e garante content presente; deixa role:'tool' (resultado) intacto."""
    out = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            m = dict(m)
            tcs = []
            for tc in m["tool_calls"]:
                fn = dict(tc.get("function", {}))
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except (ValueError, TypeError):
                        pass
                tcs.append({**tc, "function": fn})
            m["tool_calls"] = tcs
            if m.get("content") is None:
                m["content"] = ""
        out.append(m)
    return out


def _fmt_chat(tok, messages, tools=None):
    messages = _normalize_messages(messages)
    if getattr(tok, "chat_template", None):
        kw = {"add_generation_prompt": True, "tokenize": False}
        if tools:
            kw["tools"] = tools
        try:
            return tok.apply_chat_template(messages, enable_thinking=False, **kw)
        except TypeError:
            return tok.apply_chat_template(messages, **kw)
    return "\n".join(m.get("content") or "" for m in messages)


# ---------------------------------------------------------------- tools (Fase 2)
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json|tool_call)?\s*(.*?)```", re.DOTALL)


def _calls_from_obj(obj):
    """obj (dict|list) -> lista de {name, arguments}. Vazio se não for tool-call.
    Aceita formatos: {name, arguments}, {name, parameters},
    {function:{name, arguments}}, {tool_calls:[...]}, e listas desses."""
    out = []
    if isinstance(obj, list):
        for o in obj:
            out += _calls_from_obj(o)
        return out
    if not isinstance(obj, dict):
        return out
    if isinstance(obj.get("tool_calls"), list):
        return _calls_from_obj(obj["tool_calls"])
    if isinstance(obj.get("function"), dict):
        obj = obj["function"]
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return out
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, TypeError):
            pass
    out.append({"name": name, "arguments": args})
    return out


def _parse_tool_calls(text):
    """Extrai tool-calls do texto gerado. Retorna (calls, content_restante).
    1) blocos <tool_call>...</tool_call> (Qwen/Hermes/GLM);
    2) fallback: bloco ```json``` ou texto cru que seja objeto/array JSON de call."""
    blocks = _TOOLCALL_RE.findall(text)
    if blocks:
        calls = []
        for b in blocks:
            try:
                calls += _calls_from_obj(json.loads(b))
            except (ValueError, TypeError):
                pass
        if calls:
            return calls, _TOOLCALL_RE.sub("", text).strip()
    if "<tool_call>" in text:  # abertura sem fechamento (prefill/truncado)
        seg = text.split("<tool_call>", 1)[1]
        obj = _balanced_json(seg)
        if obj:
            try:
                calls = _calls_from_obj(json.loads(obj))
                if calls:
                    return calls, text.split("<tool_call>")[0].strip()
            except (ValueError, TypeError):
                pass
    candidate = text.strip()
    fence = _FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()
    if candidate[:1] in "{[":
        try:
            calls = _calls_from_obj(json.loads(candidate))
            if calls:
                return calls, ""
        except (ValueError, TypeError):
            pass
    return [], text


def _openai_tool_calls(calls):
    """Converte [{name, arguments}] -> formato OpenAI (arguments como string JSON)."""
    return [
        {
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {
                "name": c["name"],
                "arguments": c["arguments"]
                if isinstance(c["arguments"], str)
                else json.dumps(c["arguments"], ensure_ascii=False),
            },
        }
        for c in calls
    ]


def _balanced_json(s):
    """Extrai o 1o objeto JSON {...} balanceado de s (respeita strings/escapes)."""
    start = s.find("{")
    if start < 0:
        return None
    depth, instr, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            instr = not instr
        elif not instr:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    return None


def _tool_prefill(tool_choice, prompt):
    """tool_choice forçado -> prefill (abertura de <tool_call>) p/ anexar ao prompt.
    Só age em modelos Hermes-style (tag <tool_call> presente nas instruções do template):
    'required' abre uma chamada qualquer; {function:{name}} fixa o nome. 'auto'/'none'/
    vazio = sem prefill (o modelo decide)."""
    if not tool_choice or tool_choice in ("auto", "none"):
        return ""
    if "<tool_call>" not in prompt:  # template não-Hermes: não dá p/ forçar via prefill
        return ""
    if tool_choice == "required":
        return '<tool_call>\n{"name": "'  # abre o objeto p/ evitar lixo antes do JSON
    name = None
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function", tool_choice)
        name = fn.get("name") if isinstance(fn, dict) else None
    if name:
        return '<tool_call>\n{"name": "' + name + '", "arguments":'
    return "<tool_call>\n"


def _sampler(name, body):
    p = CFG.get(name, {}).get("params", {})
    return make_sampler(
        temp=body.get("temperature", p.get("temperature", 0.0)),
        top_p=body.get("top_p", p.get("top_p", 0.0)) or 0.0,
        top_k=p.get("top_k", -1),
        min_p=p.get("min_p", 0.0),
    )


def _logits_processors(name, body):
    """Penalidades de repetição (OpenAI + aliases Ollama). Request sobrepõe os params
    do modelo. Retorna lista de processors p/ o generate_step, ou None se nada setado."""
    p = CFG.get(name, {}).get("params", {})

    def g(*keys, default=None):
        for src in (body, p):
            for k in keys:
                if src.get(k) is not None:
                    return src[k]
        return default

    rep = g("repetition_penalty", "repeat_penalty")  # multiplicativo (Ollama)
    pres = g("presence_penalty")  # aditivo (OpenAI)
    freq = g("frequency_penalty")  # aditivo proporcional (OpenAI)
    bias = g("logit_bias")
    ctx = int(g("repetition_context_size", "repeat_last_n", default=20))
    kw = {}
    if rep:
        kw["repetition_penalty"] = float(rep)
        kw["repetition_context_size"] = ctx
    if pres:
        kw["presence_penalty"] = float(pres)
    if freq:
        kw["frequency_penalty"] = float(freq)
    if isinstance(bias, dict) and bias:
        kw["logit_bias"] = {int(k): float(v) for k, v in bias.items()}
    return make_logits_processors(**kw) if kw else None


class _StopBuf:
    """Detecção de stop sequences segura p/ streaming. Segura uma cauda (até maxlen-1
    chars) antes de emitir, p/ não vazar um stop que cruza a fronteira entre tokens."""

    def __init__(self, stops):
        self.stops = [s for s in stops if s]
        self.hold = max((len(s) for s in self.stops), default=0) - 1
        self.acc = ""
        self.sent = 0

    def push(self, text):
        """Adiciona texto novo. Retorna (texto_a_emitir, parou)."""
        self.acc += text
        cut = -1
        for s in self.stops:
            j = self.acc.find(s)
            if j != -1 and (cut == -1 or j < cut):
                cut = j
        if cut != -1:  # stop encontrado: corta nele
            emit = self.acc[self.sent : cut] if cut > self.sent else ""
            self.sent = len(self.acc)
            return emit, True
        safe = len(self.acc) - self.hold  # segura a cauda
        emit = self.acc[self.sent : safe] if safe > self.sent else ""
        self.sent += len(emit)
        return emit, False

    def flush(self):
        emit = self.acc[self.sent :]
        self.sent = len(self.acc)
        return emit


def _response_format_processor(name, tok, body):
    """response_format OpenAI -> logits processor de decodificação RESTRITA (llguidance,
    via mlx_vlm.structured). `json_object` = qualquer objeto JSON; `json_schema` = conforme
    o schema dado. Garante saída válida (mascara tokens inválidos a cada passo)."""
    rf = body.get("response_format")
    if not isinstance(rf, dict):
        return None
    t = rf.get("type")
    if t == "json_object":
        schema = {"type": "object"}
    elif t == "json_schema":
        js = rf.get("json_schema") or {}
        schema = js.get("schema") or js.get("json_schema") or {"type": "object"}
    else:
        return None  # "text" ou desconhecido = sem restrição
    try:
        from mlx_vlm.structured import build_json_schema_logits_processor

        hf_tok = getattr(tok, "_tokenizer", tok)
        return build_json_schema_logits_processor(hf_tok, schema)
    except Exception as e:  # noqa: BLE001
        print(f"[router] response_format ignorado ({name}): {e}", flush=True)
        return None


def gen_fim(body):
    """Autocomplete FIM (Qwen2.5-Coder): <|fim_prefix|>pre<|fim_suffix|>suf<|fim_middle|>."""
    model, tok = ac_model()
    pre = body.get("prompt", "")
    suf = body.get("suffix", "") or ""
    prompt = f"<|fim_prefix|>{pre}<|fim_suffix|>{suf}<|fim_middle|>"
    stops = body.get("stop") or []
    if isinstance(stops, str):
        stops = [stops]
    sampler = make_sampler(temp=body.get("temperature", 0.1))
    out = []
    for r in stream_generate(
        model, tok, prompt, max_tokens=body.get("max_tokens") or 256, sampler=sampler
    ):
        out.append(r.text)
        text = "".join(out)
        if "<|" in text or any(s and s in text for s in stops):
            break
    text = "".join(out)
    for marker in ("<|endoftext|>", "<|fim_pad|>", "<|file_sep|>", "<|repo_name|>"):
        text = text.split(marker)[0]
    for s in stops:
        if s:
            text = text.split(s)[0]
    return text


def embeddings(texts):
    import mlx_embeddings

    model, proc = em_model()
    out = mlx_embeddings.generate(model, proc, texts)
    return out.text_embeds.tolist()  # (n, 768)


# ---------------------------------------------------------------- keep_alive / mem
def _parse_ka(v):
    """keep_alive -> segundos. Aceita número ou string '30s'/'5m'/'1h'. None = default."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        s = str(v).strip().lower()
        mult = {"s": 1, "m": 60, "h": 3600}.get(s[-1:])
        return float(s[:-1]) * mult if mult else float(s)
    except (ValueError, TypeError):
        return None


def _loaded():
    now = time.monotonic()
    with _reg_lock:
        chat = [
            {
                "name": n,
                "size_gb": m["size_gb"],
                "vision": m.get("vlm", False),
                "cached_tokens": len(m["pctoks"]) if m.get("pctoks") else 0,
                "idle_s": round(now - m["last"]),
                "keep_alive_s": m["ka"],
            }
            for n, m in _chat.items()
        ]
    return {
        "chat": chat,
        "autocomplete": AC_NAME if _ac["model"] is not None else None,
        "embed": EM_NAME if _em["model"] is not None else None,
    }


def _mem():
    out = {}
    try:
        out["mlx"] = {
            "active_gb": _gb(mx.get_active_memory()),
            "cache_gb": _gb(mx.get_cache_memory()),
            "peak_gb": _gb(mx.get_peak_memory()),
        }
    except Exception as e:  # noqa: BLE001
        out["mlx"] = {"error": str(e)}
    if psutil is not None:
        vm = psutil.virtual_memory()
        out["system"] = {
            "total_gb": _gb(vm.total),
            "used_gb": _gb(vm.total - vm.available),
            "free_gb": _gb(vm.available),
            "used_pct": vm.percent,
        }
        out["router_rss_gb"] = _gb(psutil.Process().memory_info().rss)
    return out


# ---------------------------------------------------------------- multimodal (Fase 3)
def _extract_images(messages):
    """Coleta as fontes de imagem das mensagens OpenAI multimodais (data URI ou URL);
    o mlx-vlm carrega cada uma via load_image. Aceita type image_url/input_image/image."""
    imgs = []
    for m in messages:
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for part in c:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("image_url", "input_image", "image"):
                u = part.get("image_url") or part.get("image") or part.get("url")
                if isinstance(u, dict):
                    u = u.get("url")
                if u:
                    imgs.append(u)
    return imgs


def _gen_vlm(job, name, model, proc, body, messages, images):
    """Geração multimodal via mlx-vlm. Streama deltas como o caminho de texto
    (sem tools). skip_special_tokens evita vazar tokens do template na saída."""
    import mlx_vlm
    from mlx_vlm.prompt_utils import apply_chat_template as vlm_template

    p = CFG.get(name, {}).get("params", {})
    last = None
    try:
        prompt = vlm_template(proc, model.config, messages, num_images=len(images))
        kw = {
            "max_tokens": body.get("max_tokens") or 1024,
            "temperature": body.get("temperature", p.get("temperature", 0.0)),
            "skip_special_tokens": True,
            "prefill_step_size": PREFILL_STEP,
        }
        top_p = body.get("top_p", p.get("top_p", 0.0))
        if top_p:
            kw["top_p"] = top_p
        for r in mlx_vlm.stream_generate(model, proc, prompt, image=images or None, **kw):
            if job.cancel.is_set():
                break
            job.out.put(("delta", r.text))
            last = r
            _drain_short()
        job.out.put(("done", getattr(last, "generation_tokens", 0)))
    except Exception as e:  # noqa: BLE001
        job.out.put(("error", str(e)))
    finally:
        with _reg_lock:
            if name in _chat:
                _chat[name]["last"] = time.monotonic()


# ---------------------------------------------------------------- prompt cache (KV reuse)
def _common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _reuse_cache(name, model, ptoks):
    """Reuso de KV cache por maior-prefixo-comum (slot único por runner, estilo
    Ollama/llama.cpp; zero deepcopy). Trima o sufixo divergente do cache do slot e
    devolve (cache, tokens_a_processar). Sem match -> cache novo + prompt inteiro."""
    if PROMPT_CACHE:
        with _reg_lock:
            e = _chat.get(name)
            slot_c = e["pc"] if e else None
            slot_t = e["pctoks"] if e else None
        if slot_c is not None and slot_t and can_trim_prompt_cache(slot_c):
            n = _common_prefix(slot_t, ptoks)
            if n > 0:
                extra = len(slot_t) - n  # tokens do slot além do prefixo
                if extra > 0:
                    trim_prompt_cache(slot_c, extra)
                suffix = ptoks[n:]
                if not suffix:  # prompt == prefixo do cache
                    trim_prompt_cache(slot_c, 1)  # garante >=1 token p/ gerar
                    suffix = ptoks[-1:]
                return slot_c, suffix, n
    return make_prompt_cache(model), ptoks, 0


def _store_cache(name, all_toks, cache):
    with _reg_lock:
        if name in _chat:
            _chat[name]["pc"] = cache
            _chat[name]["pctoks"] = all_toks


# ---------------------------------------------------------------- worker de GPU
def _run_chat(job):
    name, body = job.payload["name"], job.payload["body"]
    messages = body.get("messages", [])
    images = _extract_images(messages)
    if images and not CFG.get(name, {}).get("vision"):  # rejeita antes de carregar
        job.out.put(
            (
                "error",
                f"modelo '{name}' nao e de visao (config vision:true); "
                f"recebeu {len(images)} imagem(ns)",
            )
        )
        return
    try:
        model, tok, vlm = chat_model(name)
    except Exception as e:  # noqa: BLE001
        job.out.put(("error", str(e)))
        return
    ka = _parse_ka(body.get("keep_alive"))
    if ka is not None:
        with _reg_lock:
            if name in _chat:
                _chat[name]["ka"] = ka
    job.out.put(("meta", name))
    if vlm:  # Fase 3: caminho multimodal (mlx-vlm)
        _gen_vlm(job, name, model, tok, body, messages, images)
        return
    tc = body.get("tool_choice")
    tools = body.get("tools") if tc != "none" else None
    prompt = _fmt_chat(tok, messages, tools)
    prefill = _tool_prefill(tc, prompt) if tools else ""  # tool_choice forçado
    if prefill:
        prompt += prefill
    ptoks = tok.encode(prompt, add_special_tokens=False)  # template já tem os specials
    cache, suffix, reused = _reuse_cache(name, model, ptoks)
    p = CFG.get(name, {}).get("params", {})
    seed = body.get("seed", p.get("seed"))
    if seed is not None:  # reprodutibilidade (temp>0)
        mx.random.seed(int(seed))
    stops = body.get("stop", p.get("stop"))  # stop sequences (str ou lista)
    if isinstance(stops, str):
        stops = [stops]
    stopbuf = _StopBuf(stops) if stops else None
    lps = _logits_processors(name, body) or []  # penalidades de repetição
    rf = _response_format_processor(name, tok, body)  # decod. restrita (JSON/schema)
    if rf is not None:
        lps = lps + [rf]
    last = None
    buf = []  # com tools, bufferiza p/ parsear no fim
    gen_ids = []
    stopped = False
    try:
        for r in stream_generate(
            model,
            tok,
            mx.array(suffix),
            prompt_cache=cache,
            max_tokens=body.get("max_tokens") or 1024,
            sampler=_sampler(name, body),
            logits_processors=lps or None,
            prefill_step_size=PREFILL_STEP,
            **_kv_kwargs(),
        ):
            if job.cancel.is_set():
                break
            last = r
            gen_ids.append(int(r.token))
            if tools:
                buf.append(r.text)
            elif stopbuf is not None:
                emit, hit = stopbuf.push(r.text)
                if emit:
                    job.out.put(("delta", emit))
                if hit:
                    stopped = True
                    break
            else:
                job.out.put(("delta", r.text))
            _drain_short()  # deixa autocomplete/embed passar na frente
        if stopbuf is not None and not stopped:
            tail = stopbuf.flush()  # esvazia a cauda segurada (fim natural)
            if tail:
                job.out.put(("delta", tail))
        _store_cache(name, ptoks + gen_ids, cache)  # slot reflete prompt+geração
        if reused:
            print(
                f"[router] cache {name}: reusou {reused}/{len(ptoks)} tokens do prompt", flush=True
            )
        if tools:
            calls, content = _parse_tool_calls(prefill + "".join(buf))
            if calls:
                job.out.put(("toolcalls", (calls, content)))
            elif content:
                job.out.put(("delta", content))
        job.out.put(("done", getattr(last, "generation_tokens", 0)))
        _relieve_cache(name)  # resposta já enviada; alivia RAM se preciso
    except Exception as e:  # noqa: BLE001
        job.out.put(("error", str(e)))
    finally:
        with _reg_lock:
            if name in _chat:
                _chat[name]["last"] = time.monotonic()


def _run_fim(job):
    try:
        job.out.put(("result", gen_fim(job.payload["body"])))
    except Exception as e:  # noqa: BLE001
        job.out.put(("error", str(e)))


def _run_embed(job):
    try:
        job.out.put(("result", embeddings(job.payload["texts"])))
    except Exception as e:  # noqa: BLE001
        job.out.put(("error", str(e)))


def _run_unload(job):
    target = job.payload["target"]
    freed = []
    if target in ("chat", "all"):
        for n in list(_chat):
            _evict(n)
            freed.append(n)
    if target == "all":
        if _ac["model"] is not None:
            _ac["model"] = _ac["tok"] = None
            freed.append(AC_NAME)
        if _em["model"] is not None:
            _em["model"] = _em["proc"] = None
            freed.append(EM_NAME)
    if target not in ("chat", "all") and target in _chat:
        _evict(target)
        freed.append(target)
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    print(f"[router] unload({target}) -> liberou {freed or 'nada'}", flush=True)
    job.out.put(("result", freed))


def _run_evict(job):
    now = time.monotonic()
    with _reg_lock:
        due = [
            n
            for n in job.payload["names"]
            if n in _chat and _chat[n]["ka"] >= 0 and now - _chat[n]["last"] > _chat[n]["ka"]
        ]
    for n in due:
        print(f"[router] idle: {n} excedeu keep_alive", flush=True)
        _evict(n)
    if due:
        mx.reset_peak_memory()


def _dispatch(job):
    {
        "chat": _run_chat,
        "fim": _run_fim,
        "embed": _run_embed,
        "unload": _run_unload,
        "evict": _run_evict,
    }[job.kind](job)


def _drain_short():
    """Executa jobs de prioridade alta (short) que chegaram durante a geração do chat."""
    while True:
        try:
            item = _q.get_nowait()
        except queue.Empty:
            return
        prio, _seqn, job = item
        if prio <= P_SHORT:
            _dispatch(job)
            _q.task_done()
        else:  # é chat: devolve e para de drenar
            try:
                _q.put_nowait(item)
            except queue.Full:
                _dispatch(job)
                _q.task_done()
            return


def _worker():
    while True:
        item = _q.get()
        try:
            _dispatch(item[2])
        except Exception as e:  # noqa: BLE001
            print(f"[router] worker erro: {e}", flush=True)
        finally:
            _q.task_done()


def _watchdog():
    """Enfileira eviction dos modelos ociosos (a remoção em si roda no worker)."""
    while True:
        time.sleep(10)
        now = time.monotonic()
        with _reg_lock:
            expired = [n for n, m in _chat.items() if m["ka"] >= 0 and now - m["last"] > m["ka"]]
        if expired:
            _submit(P_SHORT, "evict", {"names": expired})


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.rstrip("/")
        if path.endswith("/v1/models"):
            ids = list(CFG) + [AC_NAME, EM_NAME]
            self._json(200, {"object": "list", "data": [{"id": n, "object": "model"} for n in ids]})
        elif path.endswith("/status"):
            self._json(
                200,
                {
                    "loaded": _loaded(),
                    "memory": _mem(),
                    "queue": {"depth": _q.qsize(), "max": MAX_QUEUE},
                    "policy": {
                        "max_runners": MAX_RUNNERS,
                        "min_free_gb": MIN_FREE_GB,
                        "min_free_cache_gb": MIN_FREE_CACHE_GB,
                        "idle_timeout_s": IDLE_TIMEOUT,
                        "prompt_cache": PROMPT_CACHE,
                        "kv_bits": KV_BITS,
                        "prefill_step": PREFILL_STEP,
                    },
                },
            )
        elif path.endswith("/memory"):
            self._json(200, _mem())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.rstrip("/")
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        try:
            if path.endswith("/chat/completions"):
                self._chat(body)
            elif path.endswith("/completions"):
                self._completions(body)
            elif path.endswith("/embeddings"):
                self._embeddings(body)
            elif path.endswith("/unload"):
                self._unload(body)
            else:
                self._json(404, {"error": "not found"})
        except BrokenPipeError:
            pass
        except Exception as e:  # noqa: BLE001
            try:
                self._json(500, {"error": str(e)})
            except Exception:
                pass

    # ---- chat (multi-runner) ----
    def _chat(self, body):
        name = body.get("model", "")
        if name not in CFG:
            return self._json(404, {"error": f"modelo '{name}' desconhecido"})
        job = _submit(P_CHAT, "chat", {"name": name, "body": body})
        if job is None:
            return self._json(503, {"error": "fila cheia (maxQueue)"})
        cid, created = "chatcmpl-" + uuid.uuid4().hex[:20], int(time.time())
        if body.get("stream"):
            self._stream_out(job, cid, created, name)
        else:
            self._collect_out(job, cid, created, name)

    def _stream_out(self, job, cid, created, name):
        first, data = job.out.get()
        if first == "error":
            return self._json(500, {"error": data})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": name}

        def send(o):
            self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n")
            self.wfile.flush()

        finish = "stop"
        try:
            send({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}}]})
            while True:
                kind, data = job.out.get()
                if kind == "delta":
                    if data:
                        send({**base, "choices": [{"index": 0, "delta": {"content": data}}]})
                elif kind == "toolcalls":
                    calls, content = data
                    if content:
                        send({**base, "choices": [{"index": 0, "delta": {"content": content}}]})
                    tcs = _openai_tool_calls(calls)
                    send(
                        {
                            **base,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {"index": i, **tc} for i, tc in enumerate(tcs)
                                        ]
                                    },
                                }
                            ],
                        }
                    )
                    finish = "tool_calls"
                elif kind == "done":
                    send({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
                    break
                elif kind == "error":
                    send(
                        {
                            **base,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                            "error": data,
                        }
                    )
                    break
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except BrokenPipeError:
            job.cancel.set()  # cliente caiu -> aborta geração

    def _collect_out(self, job, cid, created, name):
        text = ""
        tool_calls = None
        while True:
            kind, data = job.out.get()
            if kind == "delta":
                text += data
            elif kind == "toolcalls":
                calls, content = data
                tool_calls = _openai_tool_calls(calls)
                text += content or ""
            elif kind == "done":
                msg = {"role": "assistant", "content": text or None}
                finish = "stop"
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                    finish = "tool_calls"
                return self._json(
                    200,
                    {
                        "id": cid,
                        "object": "chat.completion",
                        "created": created,
                        "model": name,
                        "choices": [{"index": 0, "finish_reason": finish, "message": msg}],
                        "usage": {"completion_tokens": data},
                    },
                )
            elif kind == "error":
                return self._json(500, {"error": data})

    # ---- autocomplete FIM ----
    def _completions(self, body):
        job = _submit(P_SHORT, "fim", {"body": body})
        if job is None:
            return self._json(503, {"error": "fila cheia (maxQueue)"})
        kind, data = job.out.get()
        if kind == "error":
            return self._json(500, {"error": data})
        self._json(
            200,
            {
                "id": "cmpl-" + uuid.uuid4().hex[:20],
                "object": "text_completion",
                "created": int(time.time()),
                "model": body.get("model", AC_NAME),
                "choices": [{"index": 0, "text": data, "finish_reason": "stop"}],
            },
        )

    # ---- embeddings ----
    def _embeddings(self, body):
        inp = body.get("input", "")
        texts = inp if isinstance(inp, list) else [inp]
        job = _submit(P_SHORT, "embed", {"texts": texts})
        if job is None:
            return self._json(503, {"error": "fila cheia (maxQueue)"})
        kind, data = job.out.get()
        if kind == "error":
            return self._json(500, {"error": data})
        self._json(
            200,
            {
                "object": "list",
                "model": body.get("model", EM_NAME),
                "data": [
                    {"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(data)
                ],
            },
        )

    # ---- unload ----
    def _unload(self, body):
        target = body.get("target", "chat")
        before = _mem()
        job = _submit(P_SHORT, "unload", {"target": target})
        if job is None:
            return self._json(503, {"error": "fila cheia (maxQueue)"})
        kind, data = job.out.get()
        freed = data if kind == "result" else []
        self._json(
            200,
            {"target": target, "unloaded": freed, "memory_before": before, "memory_after": _mem()},
        )


def serve(host=None, port=None):
    """Sobe o servidor HTTP e bloqueia (serve_forever)."""
    if port is None:
        port = (
            int(sys.argv[1])
            if len(sys.argv) > 1
            else int(os.environ.get("MLX_ROUTER_PORT", "8000"))
        )
    if host is None:
        host = os.environ.get("MLX_ROUTER_HOST", "127.0.0.1")
    idle = f"{IDLE_TIMEOUT:.0f}s" if IDLE_TIMEOUT > 0 else "off"
    print(
        f"[ember] http://{host}:{port}/v1  (chat:{len(CFG)} + ac + embed)  "
        f"[runners<={MAX_RUNNERS}, min_free={MIN_FREE_GB:.1f}GB, idle={idle}, "
        f"queue<={MAX_QUEUE}]",
        flush=True,
    )
    _tune_memory()
    threading.Thread(target=_worker, daemon=True).start()
    threading.Thread(target=_watchdog, daemon=True).start()
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main():
    serve()


if __name__ == "__main__":
    main()
