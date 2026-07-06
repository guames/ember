"""Ember — OpenAI-compatible MLX inference server for Apple Silicon.

One process, three capabilities, one memory policy (multi-runner, keep_alive,
preemption, prompt cache). Built for local coding assistants (e.g. Continue).

Serves 3 capabilities, all on MLX:
  • /v1/chat/completions  -> chat/code model (multi-runner with a RAM budget;
                             accepts OpenAI `tools`/`tool_choice` and returns tool_calls)
  • /v1/completions       -> FIM autocomplete (Qwen2.5-Coder-1.5B base, pinned in RAM)
  • /v1/embeddings        -> embeddings (nomic-modernbert, pinned in RAM)
Operations/observability:
  • GET  /health          -> trivial 200 for process supervisors (unauthenticated)
  • GET  /status          -> hot models + memory + queue + policy
  • GET  /memory          -> MLX and system memory (in use / free)
  • GET  /metrics         -> request counters + latency histogram (Prometheus text)
  • POST /unload {target}  -> unload ('chat' | 'all' | '<model-name>')

Ollama-style robustness (1 GPU worker + priority queue):
  - autocomplete/embed have priority and jump the chat queue (they run BETWEEN the
    chat tokens -> typing doesn't stall during generation);
  - maxQueue rejects with 503 when overloaded; cancels if the client drops;
  - multi-runner: keeps >1 chat model hot while there's >= MLX_MIN_FREE_GB of free
    RAM (and <= MLX_MAX_RUNNERS, a safety ceiling); LRU evicts the rest;
  - keep_alive: per-model idle-unload (env MLX_IDLE_TIMEOUT, or per-request keep_alive
    field). The next call reloads it automatically.

Prompt cache (KV reuse, Ollama/llama.cpp-style): each runner keeps a small pool of KV
cache slots (MLX_PROMPT_CACHE_SLOTS, default 2); on every request it reuses whichever
slot has the longest common prefix of tokens (system+history) and only processes the new
suffix -> cuts TTFT in conversations/edits, and lets interleaved conversations on the same
model each keep their own slot instead of evicting each other every turn. Zero deepcopy.
MLX_PROMPT_CACHE=0 turns matching off.

Under RAM pressure (free < MLX_MIN_FREE_CACHE_GB, default 1GB) the router drops the KV
caches (LRU, oldest first) BEFORE evicting a whole model.

Quantized KV cache (more context in the same RAM): 8-bit by default (~2x smaller than fp16,
practically lossless); MLX_KV_BITS=4 for more aggressive quantization, MLX_KV_BITS=0 for fp16.
Compatible with the prompt cache (the QuantizedKVCache is trimmable).

Memory tuning at boot: wired_limit (resident weights, no OS compression near the
limit) + chunked prefill (MLX_PREFILL_STEP, peak RAM ↓ on a cold long prompt; with
the prompt cache the normal prefill is already just the suffix, so it barely costs anything).

Envs: MLX_ROUTER_PORT(8000) MLX_ROUTER_HOST(127.0.0.1) MLX_IDLE_TIMEOUT(300)
      MLX_MAX_RUNNERS(auto by RAM, 4 on 24GB) MLX_MIN_FREE_GB(auto by RAM, 2.0 on 24GB)
      MLX_MIN_FREE_CACHE_GB(1.0) MLX_DEFAULT_EST_GB(auto by RAM, 8.0 on 24GB)
      MLX_MAX_QUEUE(32) MLX_PROMPT_CACHE(1) MLX_PROMPT_CACHE_SLOTS(2) MLX_KV_BITS(8) MLX_KV_GROUP_SIZE(64)
      MLX_KV_QUANT_START(0) MLX_PREFILL_STEP(512) MLX_WIRED_LIMIT_GB(auto by RAM)
      MLX_CACHE_LIMIT_GB(off) MLX_EMBED_CACHE(1) MLX_EMBED_CACHE_PATH(~/.cache/ember/embeddings.sqlite3)
      EMBER_API_KEY(off) EMBER_SHUTDOWN_TIMEOUT(30)
      EMBER_METRICS_LOG(~/.cache/ember/metrics.jsonl, "0" to disable)

Ops: GET /health is an unauthenticated 200 for process supervisors (LaunchAgent, systemd).
     EMBER_API_KEY, when set, requires `Authorization: Bearer <key>` on /v1/* routes only
     (off by default; never required for the default localhost-only setup). SIGTERM stops
     accepting new requests, waits up to EMBER_SHUTDOWN_TIMEOUT s for the in-flight job to
     finish, then exits.

Observability: every chat/fim/embed request appends a JSON line (endpoint, model, latency,
     prompt/completion/cached tokens, status) to EMBER_METRICS_LOG — additive to the existing
     print(...) logging, not a replacement. GET /metrics exposes the same counters/latency
     histogram in Prometheus text format for scraping; it has no time-series memory of its
     own (restart resets it), so long-term history lives in the JSONL log instead.

    ember                          # CLI (port 8000 or env MLX_ROUTER_PORT)
    python -m ember
"""

import gc
import hashlib
import itertools
import json
import os
import queue
import re
import select
import signal
import socket
import sqlite3
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from . import memory_policy
from .registry import load_registry

try:
    import psutil
except Exception:  # noqa: BLE001
    psutil = None

# Model registry (config file; see registry.py). CFG = chat/code/vision.
CFG, _AC, _EM = load_registry()
AC_NAME, AC_REPO = _AC["name"], _AC["mlx"]  # FIM autocomplete (pinned in RAM)
EM_NAME, EM_REPO = _EM["name"], _EM["mlx"]  # embeddings (pinned in RAM)

# ---- policy (envs) ----
_TOTAL_GB = (psutil.virtual_memory().total / 1024**3) if psutil else 24.0
_SCALED = memory_policy.scale_defaults(_TOTAL_GB)  # RAM-scaled defaults; envs below still win

IDLE_TIMEOUT = float(os.environ.get("MLX_IDLE_TIMEOUT", "300"))  # s; 0/neg = never
MAX_RUNNERS = int(os.environ.get("MLX_MAX_RUNNERS", str(_SCALED["max_runners"])))  # safety ceiling
MIN_FREE_GB = float(
    os.environ.get("MLX_MIN_FREE_GB", str(_SCALED["min_free_gb"]))
)  # headroom to evict a model
MIN_FREE_CACHE_GB = float(os.environ.get("MLX_MIN_FREE_CACHE_GB", "1.0"))  # floor to drop KV cache
DEFAULT_EST_GB = float(
    os.environ.get("MLX_DEFAULT_EST_GB", str(_SCALED["default_est_gb"]))
)  # size guess when unknown
MAX_QUEUE = int(os.environ.get("MLX_MAX_QUEUE", "32"))
EMBED_CHUNK = int(os.environ.get("MLX_EMBED_CHUNK", "8"))  # texts per embed slice
EMBED_CACHE = os.environ.get("MLX_EMBED_CACHE", "1") not in ("0", "false", "")  # content-hash cache
EMBED_CACHE_PATH = os.environ.get(
    "MLX_EMBED_CACHE_PATH", os.path.expanduser("~/.cache/ember/embeddings.sqlite3")
)
PROMPT_CACHE = os.environ.get("MLX_PROMPT_CACHE", "1") not in ("0", "false", "")  # KV reuse
PROMPT_CACHE_SLOTS = max(1, int(os.environ.get("MLX_PROMPT_CACHE_SLOTS", "2")))  # KV slots/runner
_KVB = os.environ.get("MLX_KV_BITS", "8")  # 8/4 = quantize KV cache; 0 = fp16
KV_BITS = int(_KVB) if _KVB not in (None, "", "0") else None
KV_GROUP_SIZE = int(os.environ.get("MLX_KV_GROUP_SIZE", "64"))
KV_QUANT_START = int(os.environ.get("MLX_KV_QUANT_START", "0"))  # quantize from token N onward
PREFILL_STEP = int(os.environ.get("MLX_PREFILL_STEP", "512"))  # prefill chunk (peak RAM ↓)
WIRED_LIMIT_GB = float(
    os.environ.get("MLX_WIRED_LIMIT_GB", "0")
)  # 0 = auto (total-headroom, RAM-scaled)
CACHE_LIMIT_GB = float(os.environ.get("MLX_CACHE_LIMIT_GB", "0"))  # 0 = MLX default (no cap)
_DEFAULT_KA = IDLE_TIMEOUT if IDLE_TIMEOUT > 0 else -1  # -1 = never expires
API_KEY = os.environ.get("EMBER_API_KEY") or None  # unset = no auth (default, localhost-only)
SHUTDOWN_TIMEOUT = float(os.environ.get("EMBER_SHUTDOWN_TIMEOUT", "30"))  # s to drain on SIGTERM
METRICS_LOG_PATH = os.environ.get(
    "EMBER_METRICS_LOG", os.path.expanduser("~/.cache/ember/metrics.jsonl")
)
if METRICS_LOG_PATH in (
    "0",
    "false",
    "",
):  # opt out of the JSONL log (the /metrics counters still work)
    METRICS_LOG_PATH = None

# ---- shutdown state (set by the SIGTERM handler; read by do_POST and the worker) ----
_shutting_down = threading.Event()
_worker_busy = threading.Event()

# ---- metrics (issue #28): JSONL request log + in-memory /metrics (Prometheus text) ----
_METRICS_BUCKETS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)  # seconds
_metrics_lock = threading.Lock()
_metrics = {}  # (endpoint, model, status) -> aggregate dict


def _record_metrics(
    endpoint, model, latency_s, prompt_tokens=0, completion_tokens=0, cached_tokens=0, error=None
):
    """Records one finished request: appends a JSON line to EMBER_METRICS_LOG (best-effort,
    additive to the existing print(...) logging) and updates the in-memory counters/histogram
    that GET /metrics reports from."""
    status = "error" if error else "ok"
    entry = {
        "ts": time.time(),
        "endpoint": endpoint,
        "model": model,
        "status": status,
        "latency_ms": round(latency_s * 1000, 1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
    }
    if error:
        entry["error"] = error
    if METRICS_LOG_PATH:
        try:
            os.makedirs(os.path.dirname(METRICS_LOG_PATH), exist_ok=True)
            with open(METRICS_LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"[router] metrics log write failed (continuing): {e}", flush=True)
    with _metrics_lock:
        m = _metrics.setdefault(
            (endpoint, model, status),
            {
                "count": 0,
                "latency_sum": 0.0,
                "buckets": dict.fromkeys(_METRICS_BUCKETS, 0),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
            },
        )
        m["count"] += 1
        m["latency_sum"] += latency_s
        m["prompt_tokens"] += prompt_tokens
        m["completion_tokens"] += completion_tokens
        m["cached_tokens"] += cached_tokens
        for le in _METRICS_BUCKETS:
            if latency_s <= le:
                m["buckets"][le] += 1


def _metrics_text():
    """Renders the in-memory counters as Prometheus text exposition format."""
    with _metrics_lock:
        snapshot = {k: {**v, "buckets": dict(v["buckets"])} for k, v in _metrics.items()}
    lines = [
        "# HELP ember_requests_total Total requests handled.",
        "# TYPE ember_requests_total counter",
    ]
    for (endpoint, model, status), m in snapshot.items():
        lines.append(
            f'ember_requests_total{{endpoint="{endpoint}",model="{model}",status="{status}"}} {m["count"]}'
        )
    lines += [
        "# HELP ember_request_latency_seconds Request latency in seconds.",
        "# TYPE ember_request_latency_seconds histogram",
    ]
    for (endpoint, model, status), m in snapshot.items():
        labels = f'endpoint="{endpoint}",model="{model}",status="{status}"'
        # m["buckets"][le] is already the cumulative count of requests with latency <= le
        # (each request increments every bucket it qualifies for; see _record_metrics).
        for le in _METRICS_BUCKETS:
            lines.append(
                f'ember_request_latency_seconds_bucket{{{labels},le="{le}"}} {m["buckets"][le]}'
            )
        lines.append(f'ember_request_latency_seconds_bucket{{{labels},le="+Inf"}} {m["count"]}')
        lines.append(f"ember_request_latency_seconds_sum{{{labels}}} {m['latency_sum']:.6f}")
        lines.append(f"ember_request_latency_seconds_count{{{labels}}} {m['count']}")
    for field, help_text in (
        ("prompt_tokens", "Prompt tokens processed."),
        ("completion_tokens", "Completion tokens generated."),
        ("cached_tokens", "Prompt tokens served from the KV cache."),
    ):
        lines += [
            f"# HELP ember_{field}_total {help_text}",
            f"# TYPE ember_{field}_total counter",
        ]
        for (endpoint, model, status), m in snapshot.items():
            labels = f'endpoint="{endpoint}",model="{model}",status="{status}"'
            lines.append(f"ember_{field}_total{{{labels}}} {m[field]}")
    return "\n".join(lines) + "\n"


def _kv_kwargs():
    """Quantized KV-cache kwargs for generate_step (empty = KV in fp16, default)."""
    if KV_BITS is None:
        return {}
    return {
        "kv_bits": KV_BITS,
        "kv_group_size": KV_GROUP_SIZE,
        "quantized_kv_start": KV_QUANT_START,
    }


def _tune_memory():
    """Memory tuning at boot: wired_limit keeps the weights resident (so the OS doesn't
    compress/page near the RAM limit -> consistent speed); cache_limit (optional) caps the
    MLX buffer pool, returning RAM to the OS."""
    try:
        wl = WIRED_LIMIT_GB or max(
            4.0, _TOTAL_GB - _SCALED["wired_headroom_gb"]
        )  # auto: leaves headroom for the OS
        mx.set_wired_limit(int(wl * 1024**3))
        extra = ""
        if CACHE_LIMIT_GB > 0:
            mx.set_cache_limit(int(CACHE_LIMIT_GB * 1024**3))
            extra = f" cache_limit={CACHE_LIMIT_GB:.0f}GB"
        print(
            f"[router] mem: wired_limit={wl:.0f}GB prefill_step={PREFILL_STEP}{extra}", flush=True
        )
    except Exception as e:  # noqa: BLE001
        print(f"[router] mem tuning failed (continuing without): {e}", flush=True)


# ---- model state (mutated ONLY by the worker; _reg_lock guards the structure) ----
_reg_lock = threading.Lock()
_chat = {}  # name -> {model, tok, size_gb, last, ka}
_sizes = {}  # name -> measured resident size_gb from a prior load (for pre-load estimates)
_ac = {"model": None, "tok": None, "pc": None, "pctoks": None}  # autocomplete (fixed)
_em = {"model": None, "proc": None}  # embed (fixed)

# ---- GPU queue (1 worker) ----
P_SHORT, P_CHAT = 0, 1  # lower = higher priority
_q = queue.PriorityQueue(maxsize=MAX_QUEUE)
_seq = itertools.count()
JOB_WAIT_POLL_S = 0.5  # how often a handler blocked on job.out re-checks the client (issue #31)


class Job:
    __slots__ = ("kind", "payload", "out", "cancel")

    def __init__(self, kind, payload):
        self.kind = kind
        self.payload = payload
        self.out = queue.Queue()
        self.cancel = threading.Event()


def _submit(prio, kind, payload):
    """Enqueues a GPU job. Returns the Job, or None if the queue is full."""
    job = Job(kind, payload)
    try:
        _q.put_nowait((prio, next(_seq), job))
    except queue.Full:
        return None
    return job


def _gb(b):
    return round(b / 1024**3, 2)


# ---------------------------------------------------------------- models (worker)
def _evict(name):
    """Unloads a chat model (called only on the worker)."""
    with _reg_lock:
        m = _chat.pop(name, None)
    if m is None:
        return
    m["model"] = m["tok"] = m["slots"] = None  # also frees the pool's KV caches
    gc.collect()
    mx.clear_cache()
    print(f"[router] evict {name}", flush=True)


def _free_gb():
    return psutil.virtual_memory().available / 1024**3 if psutil else None


def _weights_dir(mlx_id):
    """Directory holding a model's weight files: a local path, or its HF cache snapshot."""
    if os.path.isdir(mlx_id):
        return mlx_id
    # HF hub layout: <cache>/hub/models--<org>--<name>/snapshots/<rev>/*.safetensors
    base = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    hub = base if base.rstrip("/").endswith("hub") else os.path.join(base, "hub")
    snaps = os.path.join(hub, "models--" + mlx_id.replace("/", "--"), "snapshots")
    if not os.path.isdir(snaps):
        return None
    best = None
    for rev in os.listdir(snaps):
        d = os.path.join(snaps, rev)
        if os.path.isdir(d) and any(f.endswith(".safetensors") for f in os.listdir(d)):
            best = d  # normally exactly one snapshot with weights
    return best


def _dir_weight_gb(d):
    """Sum of the *.safetensors sizes in a dir (GB), following symlinks to the HF blobs."""
    total = 0
    try:
        for f in os.listdir(d):
            if f.endswith(".safetensors"):
                try:
                    total += os.path.getsize(os.path.join(d, f))
                except OSError:
                    pass
    except OSError:
        return None
    return total / 1024**3 if total else None


def _estimate_size_gb(name):
    """Best estimate of a model's resident size (GB) BEFORE loading it. Gathers the inputs
    (prior measurement, on-disk weight size, hot-model sizes) and defers the choice to the
    shared, pure `memory_policy.estimate_size_gb`."""
    measured = _sizes.get(name)
    d = _weights_dir(CFG[name]["mlx"]) if name in CFG else None
    disk = _dir_weight_gb(d) if d else None
    with _reg_lock:
        hot = [m["size_gb"] for m in _chat.values()]
    return memory_policy.estimate_size_gb(measured, disk, hot, DEFAULT_EST_GB)


def _make_room(name, est):
    """Proactively evict LRU chat models BEFORE loading `name`, so the incoming model
    fits the budget: free-after-load >= MIN_FREE_GB and runners <= MAX_RUNNERS. This is
    the admission gate — it runs before the (memory-spiking) load, so a second big model
    can't overflow RAM during load and then get evicted too late. Runs on the worker.

    The eviction *decision* is the shared, pure `memory_policy.plan_make_room`; this wrapper
    snapshots the hot models, then carries out the evictions (and logging)."""
    free = _free_gb()
    with _reg_lock:
        models = {n: {"last": m["last"], "size_gb": m["size_gb"]} for n, m in _chat.items()}
    for victim in memory_policy.plan_make_room(name, est, free, models, MIN_FREE_GB, MAX_RUNNERS):
        vsize = models[victim]["size_gb"]
        print(
            f"[router] admission: evicting LRU {victim} (~{vsize:.1f}GB) to fit "
            f"{name} (~{est:.1f}GB, free {free:.1f}GB)"
            if free is not None
            else f"[router] admission: evicting LRU {victim} to fit {name}",
            flush=True,
        )
        _evict(victim)
        if free is not None:
            free += vsize  # expected recovery (keeps the log's free in step with the plan)


def _cache_bytes(pc):
    try:
        return sum(getattr(c, "nbytes", 0) for c in pc)
    except Exception:  # noqa: BLE001
        return 0


def _relieve_cache(keep):
    """Under RAM pressure (free < MIN_FREE_CACHE_GB) drops the runners' KV caches —
    from oldest (LRU) to newest, `keep` last. Much cheaper than evicting the model
    (the weights stay hot; it just reprocesses the prompt next turn). Runs on the worker.
    Uses the cache size (.nbytes) as the expected recovery (the OS is slow to reflect it)."""
    free = _free_gb()
    if free is None or free >= MIN_FREE_CACHE_GB:
        return free
    with _reg_lock:
        snap = {
            n: {"last": m["last"], "has_cache": any(s["pc"] is not None for s in m["slots"])}
            for n, m in _chat.items()
        }
    order = memory_policy.order_cache_relief(keep, snap)  # LRU first, `keep` last resort
    for n in order:
        with _reg_lock:
            e = _chat.get(n)
            if not e:
                continue
            freed = 0.0
            for s in e["slots"]:
                if s["pc"] is not None:
                    freed += _cache_bytes(s["pc"]) / 1024**3
                    s["pc"] = s["pctoks"] = None
            if freed == 0.0:
                continue
        gc.collect()
        mx.clear_cache()
        print(
            f"[router] low RAM (<{MIN_FREE_CACHE_GB:.1f}GB): dropped KV cache of "
            f"{n} (~{freed:.2f}GB)",
            flush=True,
        )
        free = free + freed if free is not None else None
        if free is None or free >= MIN_FREE_CACHE_GB:
            break
    return free


def _enforce_memory(keep):
    """RAM policy: 1) under critical pressure (<MIN_FREE_CACHE_GB) drop KV caches (LRU,
    cheap); 2) if still short (<MIN_FREE_GB or >MAX_RUNNERS) evict the LRU model (never
    `keep`). Uses measured size as expected recovery (the OS is slow to reflect free)."""
    _relieve_cache(keep)
    free = _free_gb()
    with _reg_lock:
        models = {n: {"last": m["last"], "size_gb": m["size_gb"]} for n, m in _chat.items()}
    for victim in memory_policy.plan_enforce(keep, free, models, MIN_FREE_GB, MAX_RUNNERS):
        _evict(victim)


def chat_model(name):
    """Ensures the chat model is resident; loads+measures size+applies the budget.
    Models with `vision: true` in the config load via mlx_vlm (model, processor).
    Returns (model, tok_or_processor, is_vlm)."""
    if name not in CFG:
        raise KeyError(name)
    with _reg_lock:
        m = _chat.get(name)
        if m is not None:
            m["last"] = time.monotonic()
            return m["model"], m["tok"], m["vlm"]
    vlm = bool(CFG[name].get("vision"))
    _make_room(name, _estimate_size_gb(name))  # admission control: evict BEFORE loading
    before = mx.get_active_memory()
    print(f"[router] {'vlm' if vlm else 'chat'}: loading {name} ...", flush=True)
    if vlm:
        import mlx_vlm

        model, tok = mlx_vlm.load(CFG[name]["mlx"])
    else:
        model, tok = load(CFG[name]["mlx"])
        # some tokenizers only list <|endoftext|> in eos_token_ids; the chat template's
        # terminator (e.g. <|im_end|>) gets left out and would leak as text at the end.
        eid = getattr(tok, "eos_token_id", None)
        if eid is not None:
            try:
                tok.eos_token_ids.add(eid)
            except (AttributeError, TypeError):
                pass
    mx.eval([v for _, v in tree_flatten(model.parameters())])  # materialize to measure
    size = _gb(mx.get_active_memory() - before)
    _sizes[name] = size  # remember the real size for the next admission estimate
    with _reg_lock:
        _chat[name] = {
            "model": model,
            "tok": tok,
            "size_gb": size,
            "last": time.monotonic(),
            "ka": _DEFAULT_KA,
            "vlm": vlm,
            "slots": [{"pc": None, "pctoks": None, "last": 0.0} for _ in range(PROMPT_CACHE_SLOTS)],
        }  # prompt cache pool (KV reuse, multi-slot)
    _enforce_memory(keep=name)
    return model, tok, vlm


def ac_model():
    if _ac["model"] is None:
        print("[router] autocomplete: loading 1.5B FIM ...", flush=True)
        _ac["model"], _ac["tok"] = load(AC_REPO)
    return _ac["model"], _ac["tok"]


def em_model():
    if _em["model"] is None:
        import mlx_embeddings

        print("[router] embed: loading modernbert ...", flush=True)
        _em["model"], _em["proc"] = mlx_embeddings.load(EM_REPO)
    return _em["model"], _em["proc"]


def _normalize_messages(messages):
    """Normalizes messages for the chat template. In assistant messages with tool_calls,
    converts function.arguments from a JSON string -> object (most templates expect a dict)
    and ensures content is present; leaves role:'tool' (result) intact."""
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


# ---------------------------------------------------------------- tools (Phase 2)
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json|tool_call)?\s*(.*?)```", re.DOTALL)


def _calls_from_obj(obj):
    """obj (dict|list) -> list of {name, arguments}. Empty if it's not a tool-call.
    Accepts formats: {name, arguments}, {name, parameters},
    {function:{name, arguments}}, {tool_calls:[...]}, and lists of those."""
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
    """Extracts tool-calls from the generated text. Returns (calls, remaining_content).
    1) <tool_call>...</tool_call> blocks (Qwen/Hermes/GLM);
    2) fallback: a ```json``` block or raw text that is a JSON call object/array."""
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
    if "<tool_call>" in text:  # opening without a close (prefill/truncated)
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
    """Converts [{name, arguments}] -> OpenAI format (arguments as a JSON string)."""
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
    """Extracts the 1st balanced JSON object {...} from s (respects strings/escapes)."""
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
    """Forced tool_choice -> prefill (the opening of <tool_call>) to append to the prompt.
    Only acts on Hermes-style models (the <tool_call> tag is present in the template's
    instructions): 'required' opens any call; {function:{name}} pins the name. 'auto'/'none'/
    empty = no prefill (the model decides)."""
    if not tool_choice or tool_choice in ("auto", "none"):
        return ""
    if "<tool_call>" not in prompt:  # non-Hermes template: can't force via prefill
        return ""
    if tool_choice == "required":
        return '<tool_call>\n{"name": "'  # open the object to avoid junk before the JSON
    name = None
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function", tool_choice)
        name = fn.get("name") if isinstance(fn, dict) else None
    if name:
        return '<tool_call>\n{"name": "' + name + '", "arguments":'
    return "<tool_call>\n"


def _num_ctx_error(num_ctx, n_prompt_tokens):
    """Checks a prompt's token count against a model's configured num_ctx (if any).
    Returns an error message if the prompt is too long, else None. Pure/testable —
    callers own reading the config and reporting the error."""
    if num_ctx and n_prompt_tokens > num_ctx:
        return (
            f"prompt has {n_prompt_tokens} tokens, which exceeds this model's "
            f"configured num_ctx ({num_ctx}); trim the conversation or raise num_ctx"
        )
    return None


def _sampler(name, body):
    p = CFG.get(name, {}).get("params", {})
    return make_sampler(
        temp=body.get("temperature", p.get("temperature", 0.0)),
        top_p=body.get("top_p", p.get("top_p", 0.0)) or 0.0,
        top_k=p.get("top_k", -1),
        min_p=p.get("min_p", 0.0),
    )


def _logits_processors(name, body):
    """Repetition penalties (OpenAI + Ollama aliases). The request overrides the model's
    params. Returns a list of processors for generate_step, or None if nothing is set."""
    p = CFG.get(name, {}).get("params", {})

    def g(*keys, default=None):
        for src in (body, p):
            for k in keys:
                if src.get(k) is not None:
                    return src[k]
        return default

    rep = g("repetition_penalty", "repeat_penalty")  # multiplicative (Ollama)
    pres = g("presence_penalty")  # additive (OpenAI)
    freq = g("frequency_penalty")  # additive proportional (OpenAI)
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
    """Streaming-safe stop-sequence detection. Holds back a tail (up to maxlen-1
    chars) before emitting, so a stop that crosses a token boundary doesn't leak."""

    def __init__(self, stops):
        self.stops = [s for s in stops if s]
        self.hold = max((len(s) for s in self.stops), default=0) - 1
        self.acc = ""
        self.sent = 0

    @staticmethod
    def earliest_stop(text, stops):
        """Returns the index of the earliest stop sequence in text, or -1."""
        cut = -1
        for s in stops:
            j = text.find(s)
            if j != -1 and (cut == -1 or j < cut):
                cut = j
        return cut

    def push(self, text):
        """Adds new text. Returns (text_to_emit, stopped)."""
        self.acc += text
        cut = self.earliest_stop(self.acc, self.stops)
        if cut != -1:  # stop found: cut at it
            emit = self.acc[self.sent : cut] if cut > self.sent else ""
            self.sent = len(self.acc)
            return emit, True
        safe = len(self.acc) - self.hold  # hold back the tail
        emit = self.acc[self.sent : safe] if safe > self.sent else ""
        self.sent += len(emit)
        return emit, False

    def flush(self):
        emit = self.acc[self.sent :]
        self.sent = len(self.acc)
        return emit


def _response_format_processor(name, tok, body):
    """OpenAI response_format -> CONSTRAINED-decoding logits processor (llguidance,
    via mlx_vlm.structured). `json_object` = any JSON object; `json_schema` = conforming
    to the given schema. Guarantees valid output (masks invalid tokens at each step)."""
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
        return None  # "text" or unknown = no constraint
    try:
        from mlx_vlm.structured import build_json_schema_logits_processor

        hf_tok = getattr(tok, "_tokenizer", tok)
        return build_json_schema_logits_processor(hf_tok, schema)
    except Exception as e:  # noqa: BLE001
        print(f"[router] response_format ignored ({name}): {e}", flush=True)
        return None


def gen_fim(body):
    """FIM autocomplete (Qwen2.5-Coder): <|fim_prefix|>pre<|fim_suffix|>suf<|fim_middle|>."""
    model, tok = ac_model()
    pre = body.get("prompt", "")
    suf = body.get("suffix", "") or ""
    prompt = f"<|fim_prefix|>{pre}<|fim_suffix|>{suf}<|fim_middle|>"
    ptoks = tok.encode(prompt, add_special_tokens=False)
    stops = body.get("stop") or []
    if isinstance(stops, str):
        stops = [stops]
    sampler = make_sampler(temp=body.get("temperature", 0.1))
    cache, suffix, reused = _reuse_ac_cache(model, ptoks)
    out, gen_ids = [], []
    for r in stream_generate(
        model,
        tok,
        mx.array(suffix),
        prompt_cache=cache,
        max_tokens=body.get("max_tokens") or 256,
        sampler=sampler,
        prefill_step_size=PREFILL_STEP,
        **_kv_kwargs(),
    ):
        out.append(r.text)
        gen_ids.append(int(r.token))
        text = "".join(out)
        if "<|" in text or any(s and s in text for s in stops):
            break
    _store_ac_cache(ptoks + gen_ids, cache)
    if reused:
        print(
            f"[router] cache autocomplete: reused {reused}/{len(ptoks)} prompt tokens", flush=True
        )
    text = "".join(out)
    for marker in ("<|endoftext|>", "<|fim_pad|>", "<|file_sep|>", "<|repo_name|>"):
        text = text.split(marker)[0]
    for s in stops:
        if s:
            text = text.split(s)[0]
    usage = {
        "prompt_tokens": len(ptoks),
        "completion_tokens": len(gen_ids),
        "cached_tokens": reused,
    }
    return text, usage


class _EmbedCache:
    """On-disk content-hash -> embedding cache (sqlite3, stdlib only). Repeated indexing runs
    (e.g. `ledger recall`) skip the forward pass entirely for text already embedded."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (hash TEXT PRIMARY KEY, vec TEXT)"
        )
        self._conn.commit()
        self._lock = threading.Lock()

    def get(self, h):
        with self._lock:
            row = self._conn.execute("SELECT vec FROM embeddings WHERE hash = ?", (h,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, h, vec):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings (hash, vec) VALUES (?, ?)", (h, json.dumps(vec))
            )
            self._conn.commit()


_embed_cache_singleton = None


def _embed_cache():
    global _embed_cache_singleton
    if _embed_cache_singleton is None:
        _embed_cache_singleton = _EmbedCache(EMBED_CACHE_PATH)
    return _embed_cache_singleton


def _embed_hash(text):
    return hashlib.sha256(f"{EM_REPO}\0{text}".encode()).hexdigest()


def embeddings(texts):
    import mlx_embeddings

    model, proc = em_model()
    if not texts:
        return [], 0

    tok_ids = [proc.encode(t, truncation=True, max_length=512) for t in texts]
    prompt_tokens = sum(len(ids) for ids in tok_ids)

    cache = _embed_cache() if EMBED_CACHE else None
    vecs = [None] * len(texts)
    pending = {}  # content hash -> indices sharing that exact text (dedups repeats in-batch too)
    for i, t in enumerate(texts):
        h = _embed_hash(t)
        cached = cache.get(h) if cache is not None else None
        if cached is not None:
            vecs[i] = cached
        else:
            pending.setdefault(h, []).append(i)

    if pending:
        # mlx_embeddings.generate pads to the longest text in the batch; pooling/normalization
        # over the padded positions yields all-NaN embeddings for the shorter texts (and
        # json.dumps then emits the bare literal `NaN`, invalid JSON for strict clients).
        # Bucketing by exact tokenized length before batching sidesteps the padding entirely —
        # every text in a bucket is the same length, so there is nothing to pad. See issue #5.
        buckets = {}
        for h, idxs in pending.items():
            buckets.setdefault(len(tok_ids[idxs[0]]), []).append(h)
        for hs in buckets.values():
            batch_texts = [texts[pending[h][0]] for h in hs]
            out = mlx_embeddings.generate(model, proc, batch_texts).text_embeds.tolist()
            for h, v in zip(hs, out, strict=True):
                for i in pending[h]:
                    vecs[i] = v
                if cache is not None:
                    cache.put(h, v)

    return vecs, prompt_tokens


# ---------------------------------------------------------------- keep_alive / mem
def _parse_ka(v):
    """keep_alive -> seconds. Accepts a number or a string '30s'/'5m'/'1h'. None = default."""
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
                "cached_tokens": sum(len(s["pctoks"]) for s in m["slots"] if s["pctoks"]),
                "idle_s": round(now - m["last"]),
                "keep_alive_s": m["ka"],
            }
            for n, m in _chat.items()
        ]
    return {
        "chat": chat,
        "autocomplete": AC_NAME if _ac["model"] is not None else None,
        "autocomplete_cached_tokens": len(_ac["pctoks"]) if _ac.get("pctoks") else 0,
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


# ---------------------------------------------------------------- multimodal (Phase 3)
def _extract_images(messages):
    """Collects the image sources from OpenAI multimodal messages (data URI or URL);
    mlx-vlm loads each one via load_image. Accepts type image_url/input_image/image."""
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
    """Multimodal generation via mlx-vlm. Streams deltas like the text path
    (no tools). skip_special_tokens avoids leaking template tokens into the output."""
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
        usage = {
            "prompt_tokens": getattr(last, "prompt_tokens", 0),
            "completion_tokens": getattr(last, "generation_tokens", 0),
            "cached_tokens": getattr(last, "cached_tokens", 0),
        }
        job.out.put(("done", usage))
        return usage
    except Exception as e:  # noqa: BLE001
        job.out.put(("error", str(e)))
        return None
    finally:
        with _reg_lock:
            if name in _chat:
                _chat[name]["last"] = time.monotonic()


# ---------------------------------------------------------------- prompt cache (KV reuse)
def _reuse_cache(name, model, ptoks):
    """KV cache reuse by longest-common-prefix, across a small pool of slots per runner
    (Ollama/llama.cpp-style matching, extended to N slots; zero deepcopy). Picking which
    slot to reuse/evict is the pure `memory_policy.select_prompt_cache_slot`; this just
    carries out the trim on the chosen slot. Returns (cache, tokens_to_process, reused,
    write_idx) — `write_idx` is where `_store_cache` should save the post-generation cache."""
    with _reg_lock:
        e = _chat.get(name)
        slots = e["slots"] if e else []
        snap = [{"tokens": s["pctoks"], "last": s["last"]} for s in slots]
    match_idx, common_len, write_idx = memory_policy.select_prompt_cache_slot(snap, ptoks)
    if PROMPT_CACHE and match_idx is not None:
        with _reg_lock:
            slot = _chat[name]["slots"][match_idx]
            slot_c, slot_t = slot["pc"], slot["pctoks"]
        if slot_c is not None and can_trim_prompt_cache(slot_c):
            extra = len(slot_t) - common_len  # slot tokens beyond the prefix
            if extra > 0:
                trim_prompt_cache(slot_c, extra)
            suffix = ptoks[common_len:]
            if not suffix:  # prompt == the cache's prefix
                trim_prompt_cache(slot_c, 1)  # ensure >=1 token to generate
                suffix = ptoks[-1:]
            return slot_c, suffix, common_len, write_idx
    return make_prompt_cache(model), ptoks, 0, write_idx


def _store_cache(name, all_toks, cache, slot_idx):
    with _reg_lock:
        if name in _chat:
            _chat[name]["slots"][slot_idx] = {
                "pc": cache,
                "pctoks": all_toks,
                "last": time.monotonic(),
            }


def _reuse_ac_cache(model, ptoks):
    """Like _reuse_cache, but for the fixed autocomplete slot (_ac), which isn't
    keyed by name."""
    if PROMPT_CACHE:
        with _reg_lock:
            slot_c = _ac["pc"]
            slot_t = _ac["pctoks"]
        if slot_c is not None and slot_t and can_trim_prompt_cache(slot_c):
            n = memory_policy.common_prefix(slot_t, ptoks)
            if n > 0:
                extra = len(slot_t) - n
                if extra > 0:
                    trim_prompt_cache(slot_c, extra)
                suffix = ptoks[n:]
                if not suffix:
                    trim_prompt_cache(slot_c, 1)
                    suffix = ptoks[-1:]
                return slot_c, suffix, n
    return make_prompt_cache(model), ptoks, 0


def _store_ac_cache(all_toks, cache):
    with _reg_lock:
        _ac["pc"] = cache
        _ac["pctoks"] = all_toks


# ---------------------------------------------------------------- GPU worker
def _run_chat(job):
    if job.cancel.is_set():  # client gone while this job was still queued (issue #31)
        return
    t0 = time.monotonic()
    name, body = job.payload["name"], job.payload["body"]
    messages = body.get("messages", [])
    images = _extract_images(messages)
    if images and not CFG.get(name, {}).get("vision"):  # reject before loading
        msg = (
            f"model '{name}' is not a vision model (config vision:true); got {len(images)} image(s)"
        )
        job.out.put(("error", msg))
        _record_metrics("chat", name, time.monotonic() - t0, error=msg)
        return
    meta_sent = False
    retried_load_oom = False
    while True:
        try:
            model, tok, vlm = chat_model(name)
        except Exception as e:  # noqa: BLE001
            if not retried_load_oom and memory_policy.is_oom_error(str(e)):
                retried_load_oom = True
                _enforce_memory(keep=name)  # drop caches / evict LRU, then retry once
                continue
            job.out.put(("error", str(e)))
            _record_metrics("chat", name, time.monotonic() - t0, error=str(e))
            return
        break
    ka = _parse_ka(body.get("keep_alive"))
    if ka is not None:
        with _reg_lock:
            if name in _chat:
                _chat[name]["ka"] = ka
    if not meta_sent:
        job.out.put(("meta", name))
        meta_sent = True
    if vlm:  # Phase 3: multimodal path (mlx-vlm)
        usage = _gen_vlm(job, name, model, tok, body, messages, images)
        if usage is None:
            _record_metrics("chat", name, time.monotonic() - t0, error="vlm generation failed")
        else:
            _record_metrics("chat", name, time.monotonic() - t0, **usage)
        return
    tc = body.get("tool_choice")
    tools = body.get("tools") if tc != "none" else None
    prompt = _fmt_chat(tok, messages, tools)
    prefill = _tool_prefill(tc, prompt) if tools else ""  # forced tool_choice
    if prefill:
        prompt += prefill
    ptoks = tok.encode(prompt, add_special_tokens=False)  # template already has the specials
    p = CFG.get(name, {}).get("params", {})
    err = _num_ctx_error(p.get("num_ctx"), len(ptoks))
    if err:
        job.out.put(("error", err))
        return
    retried_gen_oom = False
    while True:
        cache, suffix, reused, slot_idx = _reuse_cache(name, model, ptoks)
        seed = body.get("seed", p.get("seed"))
        if seed is not None:  # reproducibility (temp>0)
            mx.random.seed(int(seed))
        stops = body.get("stop", p.get("stop"))  # stop sequences (str or list)
        if isinstance(stops, str):
            stops = [stops]
        stopbuf = _StopBuf(stops) if stops else None
        lps = _logits_processors(name, body) or []  # repetition penalties
        rf = _response_format_processor(name, tok, body)  # constrained decoding (JSON/schema)
        if rf is not None:
            lps = lps + [rf]
        last = None
        buf = []  # with tools, buffer to parse at the end
        gen_ids = []
        stopped = False
        emitted = False  # any delta already on job.out -> unsafe to retry past this point
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
                    if stopbuf is not None:
                        full = "".join(buf)
                        cut = _StopBuf.earliest_stop(full, stopbuf.stops)
                        if cut != -1:
                            buf = [full[:cut]]
                            stopped = True
                            break
                elif stopbuf is not None:
                    emit, hit = stopbuf.push(r.text)
                    if emit:
                        job.out.put(("delta", emit))
                        emitted = True
                    if hit:
                        stopped = True
                        break
                else:
                    job.out.put(("delta", r.text))
                    emitted = True
                _drain_short()  # let autocomplete/embed cut in front
            if not tools and stopbuf is not None and not stopped:
                tail = stopbuf.flush()  # drain the held-back tail (natural end)
                if tail:
                    job.out.put(("delta", tail))
            _store_cache(name, ptoks + gen_ids, cache, slot_idx)  # slot reflects prompt+generation
            if reused:
                print(
                    f"[router] cache {name}: reused {reused}/{len(ptoks)} prompt tokens", flush=True
                )
            if tools:
                calls, content = _parse_tool_calls(prefill + "".join(buf))
                if calls:
                    job.out.put(("toolcalls", (calls, content)))
                elif content:
                    job.out.put(("delta", content))
            usage = {
                "prompt_tokens": len(ptoks),
                "completion_tokens": getattr(last, "generation_tokens", 0),
                "cached_tokens": reused,
            }
            job.out.put(("done", usage))
            _record_metrics("chat", name, time.monotonic() - t0, **usage)
            _relieve_cache(name)  # response already sent; relieve RAM if needed
        except Exception as e:  # noqa: BLE001
            if not emitted and not retried_gen_oom and memory_policy.is_oom_error(str(e)):
                retried_gen_oom = True
                _enforce_memory(keep=name)  # drop caches / evict LRU, then retry once
                continue
            job.out.put(("error", str(e)))
            _record_metrics("chat", name, time.monotonic() - t0, error=str(e))
        finally:
            with _reg_lock:
                if name in _chat:
                    _chat[name]["last"] = time.monotonic()
        break


def _run_fim(job):
    if job.cancel.is_set():  # client gone while this job was still queued (issue #31)
        return
    t0 = time.monotonic()
    retried_oom = False
    while True:
        try:
            text, usage = gen_fim(job.payload["body"])
            job.out.put(("result", (text, usage)))
            _record_metrics("fim", AC_NAME, time.monotonic() - t0, **usage)
        except Exception as e:  # noqa: BLE001
            if not retried_oom and memory_policy.is_oom_error(str(e)):
                retried_oom = True
                _enforce_memory(keep=None)  # drop caches / evict LRU, then retry once
                continue
            job.out.put(("error", str(e)))
            _record_metrics("fim", AC_NAME, time.monotonic() - t0, error=str(e))
        break


def _run_embed(job):
    """Embeds job.payload["texts"] in slices of EMBED_CHUNK. A large batch would otherwise
    hold the single worker (and thus an in-progress chat stream) for the whole job; instead,
    after each slice it re-queues itself and returns, letting _drain_short's caller run a
    chat step in between (issue #25). Also bails out early if the client is already gone,
    whether that's before the first slice (still queued) or between re-queued slices
    (issue #31)."""
    if job.cancel.is_set():
        return
    t0 = job.payload.setdefault("_t0", time.monotonic())
    try:
        texts = job.payload["texts"]
        vecs = job.payload.setdefault("_vecs", [])
        while len(vecs) < len(texts):
            chunk = texts[len(vecs) : len(vecs) + EMBED_CHUNK]
            try:
                chunk_vecs, chunk_tokens = embeddings(chunk)
            except Exception as e:  # noqa: BLE001
                if not job.payload.get("_oom_retried") and memory_policy.is_oom_error(str(e)):
                    job.payload["_oom_retried"] = True
                    _enforce_memory(keep=None)  # drop caches / evict LRU, then retry once
                    continue
                raise
            vecs.extend(chunk_vecs)
            job.payload["_tokens"] = job.payload.get("_tokens", 0) + chunk_tokens
            if len(vecs) < len(texts):
                try:
                    _q.put_nowait((P_SHORT, next(_seq), job))
                    return  # yield the remaining slices to the next _drain_short call
                except queue.Full:
                    continue  # queue momentarily full: keep going inline rather than drop it
        job.out.put(("result", (vecs, job.payload["_tokens"])))
        _record_metrics(
            "embed", EM_NAME, time.monotonic() - t0, prompt_tokens=job.payload["_tokens"]
        )
    except Exception as e:  # noqa: BLE001
        job.out.put(("error", str(e)))
        _record_metrics("embed", EM_NAME, time.monotonic() - t0, error=str(e))


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
            _ac["pc"] = _ac["pctoks"] = None
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
    print(f"[router] unload({target}) -> freed {freed or 'nothing'}", flush=True)
    job.out.put(("result", freed))


def _run_clear(job):
    """Clears context/cache WITHOUT unloading models (called only on the worker).
    'context' = drops the prompt cache (conversation KV) of all runners; the model stays
    hot and the next call reprocesses the prompt. 'cache' = empties the MLX buffer pool
    (mx.clear_cache) and resets the peak. 'all' = both."""
    target = job.payload["target"]
    cleared = []
    if target in ("context", "all"):
        with _reg_lock:
            names = [n for n, m in _chat.items() if any(s["pc"] is not None for s in m["slots"])]
            for n in names:
                for s in _chat[n]["slots"]:
                    s["pc"] = s["pctoks"] = None
            ac_cleared = _ac["pc"] is not None
            if ac_cleared:
                _ac["pc"] = _ac["pctoks"] = None
        if names:
            cleared.append("prompt-cache: " + ", ".join(names))
        if ac_cleared:
            cleared.append("prompt-cache: autocomplete")
        gc.collect()
    if target in ("cache", "all"):
        mx.clear_cache()
        mx.reset_peak_memory()
        cleared.append("mlx-buffer-pool")
    print(f"[ember] clear({target}) -> {cleared or 'nothing'}", flush=True)
    job.out.put(("result", cleared))


def _run_evict(job):
    now = time.monotonic()
    with _reg_lock:
        due = [
            n
            for n in job.payload["names"]
            if n in _chat and _chat[n]["ka"] >= 0 and now - _chat[n]["last"] > _chat[n]["ka"]
        ]
    for n in due:
        print(f"[router] idle: {n} exceeded keep_alive", flush=True)
        _evict(n)
    if due:
        mx.reset_peak_memory()


def _dispatch(job):
    {
        "chat": _run_chat,
        "fim": _run_fim,
        "embed": _run_embed,
        "unload": _run_unload,
        "clear": _run_clear,
        "evict": _run_evict,
    }[job.kind](job)


def _drain_short():
    """Runs at most one queued high-priority (short) job, then returns. Only one job (or, for
    a chunked embed job, one slice of it — see _run_embed) runs per call, so a large job can't
    monopolize the worker: control returns to the chat generation loop between calls."""
    try:
        item = _q.get_nowait()
    except queue.Empty:
        return
    prio, _seqn, job = item
    if prio <= P_SHORT:
        _dispatch(job)
    else:  # it's a chat job: put it back, nothing to drain
        try:
            _q.put_nowait(item)
        except queue.Full:
            _dispatch(job)
    _q.task_done()


def _worker():
    while True:
        item = _q.get()
        _worker_busy.set()
        try:
            _dispatch(item[2])
        except Exception as e:  # noqa: BLE001
            print(f"[router] worker error: {e}", flush=True)
        finally:
            _worker_busy.clear()
            _q.task_done()


def _wait_for_drain(timeout):
    """Blocks (up to `timeout` s) until the GPU queue is empty and the worker is idle.
    Used on SIGTERM to let an in-flight generation finish before the process exits."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _q.empty() and not _worker_busy.is_set():
            return True
        time.sleep(0.1)
    return False


def _watchdog():
    """Enqueues eviction of idle models (the removal itself runs on the worker)."""
    while True:
        time.sleep(10)
        now = time.monotonic()
        with _reg_lock:
            expired = [n for n, m in _chat.items() if m["ka"] >= 0 and now - m["last"] > m["ka"]]
        if expired:
            _submit(P_SHORT, "evict", {"names": expired})


def _error_obj(message, err_type="internal_error", err_code=None):
    """message -> OpenAI-shaped error object: {message, type, code}."""
    return {"message": message, "type": err_type, "code": err_code}


def _usage_obj(u):
    """{prompt_tokens, completion_tokens, cached_tokens} -> OpenAI-shaped usage object."""
    p, c = u["prompt_tokens"], u["completion_tokens"]
    return {
        "prompt_tokens": p,
        "completion_tokens": c,
        "total_tokens": p + c,
        "prompt_tokens_details": {"cached_tokens": u["cached_tokens"]},
    }


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code, message, err_type=None, err_code=None):
        """OpenAI-compatible error envelope: {"error": {message, type, code}}."""
        if err_type is None:
            err_type = "internal_error" if code >= 500 else "invalid_request_error"
        self._json(code, {"error": _error_obj(message, err_type, err_code)})

    def _authorized(self):
        """True if EMBER_API_KEY is unset (auth off, default) or the request carries a
        matching `Authorization: Bearer <key>` header."""
        if not API_KEY:
            return True
        return self.headers.get("Authorization") == f"Bearer {API_KEY}"

    def _reject_unauthorized(self):
        self._error(
            401,
            "invalid or missing API key",
            err_type="authentication_error",
            err_code="invalid_api_key",
        )

    def _client_gone(self):
        """Non-blocking check for whether the client's TCP connection is still open.
        Used by the job-wait loops below to cancel a job (streaming, non-streaming, or still
        queued -- job.out only ever produces its first message once the worker dispatches it)
        once nobody is left to receive the response (issue #31). A closed connection reads as
        EOF (b"") on a MSG_PEEK; a reset/broken one raises OSError. Data other than EOF (e.g. a
        pipelined next request under keep-alive) means the client is still there."""
        try:
            if not select.select([self.connection], [], [], 0)[0]:
                return False
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    def _wait_out(self, job):
        """Blocks until job.out has a message, polling `_client_gone` while it waits instead
        of blocking forever -- covers a job that hasn't been dispatched yet (job.out stays
        empty for the whole time it sits in the queue) as well as one that's running but
        producing nothing for a while (issue #31). Returns (None, None) once the client is
        confirmed gone, after marking the job cancelled so the worker (queued or running)
        stops without doing further work for nobody."""
        while True:
            try:
                return job.out.get(timeout=JOB_WAIT_POLL_S)
            except queue.Empty:
                if self._client_gone():
                    job.cancel.set()
                    return None, None

    def do_GET(self):
        path = self.path.rstrip("/")
        if path.endswith("/health"):
            return self._json(200, {"status": "ok"})
        if path.startswith("/v1") and not self._authorized():
            return self._reject_unauthorized()
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
                        "prompt_cache_slots": PROMPT_CACHE_SLOTS,
                        "kv_bits": KV_BITS,
                        "prefill_step": PREFILL_STEP,
                    },
                },
            )
        elif path.endswith("/memory"):
            self._json(200, _mem())
        elif path.endswith("/metrics"):
            data = _metrics_text().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._error(404, "not found", err_code="not_found")

    def do_POST(self):
        path = self.path.rstrip("/")
        if _shutting_down.is_set():
            return self._error(503, "server is shutting down", err_code="shutting_down")
        if path.startswith("/v1") and not self._authorized():
            return self._reject_unauthorized()
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
            elif path.endswith("/clear"):
                self._clear(body)
            else:
                self._error(404, "not found", err_code="not_found")
        except BrokenPipeError:
            pass
        except Exception as e:  # noqa: BLE001
            try:
                self._error(500, str(e))
            except Exception:
                pass

    # ---- chat (multi-runner) ----
    def _chat(self, body):
        name = body.get("model", "")
        if name not in CFG:
            return self._error(404, f"unknown model '{name}'", err_code="model_not_found")
        job = _submit(P_CHAT, "chat", {"name": name, "body": body})
        if job is None:
            return self._error(503, "queue full (maxQueue)", err_code="queue_full")
        cid, created = "chatcmpl-" + uuid.uuid4().hex[:20], int(time.time())
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        if body.get("stream"):
            self._stream_out(job, cid, created, name, include_usage)
        else:
            self._collect_out(job, cid, created, name)

    def _stream_out(self, job, cid, created, name, include_usage):
        first, data = self._wait_out(job)
        if first is None:  # client gone before the job ever produced anything (incl. queued)
            return
        if first == "error":
            return self._error(500, data)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        # SSE body has no Content-Length/chunked framing, so a keep-alive
        # connection would leave the client unable to tell where the
        # response ends -> force close for this response only.
        self.send_header("Connection", "close")
        self.end_headers()
        base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": name}

        def send(o):
            self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n")
            self.wfile.flush()

        finish = "stop"
        try:
            send({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}}]})
            while True:
                kind, data = self._wait_out(job)
                if kind is None:  # client gone mid-stream, no one left to send [DONE] to
                    return
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
                    if include_usage:
                        send({**base, "choices": [], "usage": _usage_obj(data)})
                    break
                elif kind == "error":
                    send(
                        {
                            **base,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                            "error": _error_obj(data),
                        }
                    )
                    break
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except BrokenPipeError:
            job.cancel.set()  # client dropped -> abort generation

    def _collect_out(self, job, cid, created, name):
        text = ""
        tool_calls = None
        while True:
            kind, data = self._wait_out(job)
            if kind is None:  # client gone (incl. while the job was still queued)
                return
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
                        "usage": _usage_obj(data),
                    },
                )
            elif kind == "error":
                return self._error(500, data)

    # ---- autocomplete FIM ----
    def _completions(self, body):
        job = _submit(P_SHORT, "fim", {"body": body})
        if job is None:
            return self._error(503, "queue full (maxQueue)", err_code="queue_full")
        kind, data = self._wait_out(job)
        if kind is None:  # client gone (incl. while the job was still queued)
            return
        if kind == "error":
            return self._error(500, data)
        text, usage = data
        self._json(
            200,
            {
                "id": "cmpl-" + uuid.uuid4().hex[:20],
                "object": "text_completion",
                "created": int(time.time()),
                "model": body.get("model", AC_NAME),
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                "usage": _usage_obj(usage),
            },
        )

    # ---- embeddings ----
    def _embeddings(self, body):
        inp = body.get("input", "")
        texts = inp if isinstance(inp, list) else [inp]
        job = _submit(P_SHORT, "embed", {"texts": texts})
        if job is None:
            return self._error(503, "queue full (maxQueue)", err_code="queue_full")
        kind, data = self._wait_out(job)
        if kind is None:  # client gone (incl. while the job was still queued)
            return
        if kind == "error":
            return self._error(500, data)
        vecs, prompt_tokens = data
        self._json(
            200,
            {
                "object": "list",
                "model": body.get("model", EM_NAME),
                "data": [
                    {"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)
                ],
                "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
            },
        )

    # ---- unload ----
    def _unload(self, body):
        target = body.get("target", "chat")
        before = _mem()
        job = _submit(P_SHORT, "unload", {"target": target})
        if job is None:
            return self._error(503, "queue full (maxQueue)", err_code="queue_full")
        kind, data = job.out.get()
        freed = data if kind == "result" else []
        self._json(
            200,
            {"target": target, "unloaded": freed, "memory_before": before, "memory_after": _mem()},
        )

    # ---- clear (context/cache, without unloading models) ----
    def _clear(self, body):
        target = body.get("target", "all")
        if target not in ("context", "cache", "all"):
            return self._error(400, "target must be context|cache|all", err_code="invalid_target")
        before = _mem()
        job = _submit(P_SHORT, "clear", {"target": target})
        if job is None:
            return self._error(503, "queue full (maxQueue)", err_code="queue_full")
        kind, data = job.out.get()
        cleared = data if kind == "result" else []
        self._json(
            200,
            {"target": target, "cleared": cleared, "memory_before": before, "memory_after": _mem()},
        )


def serve(host=None, port=None):
    """Starts the HTTP server and blocks until SIGTERM (or Ctrl-C).

    serve_forever() runs on a background thread so the main thread is free to receive
    the signal and drive a graceful shutdown: stop accepting new requests/jobs (see
    do_POST's _shutting_down check), wait up to SHUTDOWN_TIMEOUT s for the in-flight
    GPU job to finish, then close the socket and return."""
    if port is None:
        port = int(os.environ.get("MLX_ROUTER_PORT", "8000"))
    if host is None:
        host = os.environ.get("MLX_ROUTER_HOST", "127.0.0.1")
    idle = f"{IDLE_TIMEOUT:.0f}s" if IDLE_TIMEOUT > 0 else "off"
    print(
        f"[ember] http://{host}:{port}/v1  (chat:{len(CFG)} + ac + embed)  "
        f"[runners<={MAX_RUNNERS}, min_free={MIN_FREE_GB:.1f}GB, idle={idle}, "
        f"queue<={MAX_QUEUE}, auth={'on' if API_KEY else 'off'}]",
        flush=True,
    )
    _tune_memory()
    threading.Thread(target=_worker, daemon=True).start()
    threading.Thread(target=_watchdog, daemon=True).start()
    httpd = ThreadingHTTPServer((host, port), Handler)
    stop = threading.Event()

    def _on_sigterm(signum, frame):
        print("[ember] SIGTERM: draining in-flight job before exit", flush=True)
        _shutting_down.set()
        stop.set()

    signal.signal(signal.SIGTERM, _on_sigterm)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        while not stop.is_set():
            stop.wait(0.5)
    except KeyboardInterrupt:
        _shutting_down.set()
    if not _wait_for_drain(SHUTDOWN_TIMEOUT):
        print(
            f"[ember] shutdown: job still running after {SHUTDOWN_TIMEOUT:.0f}s, exiting anyway",
            flush=True,
        )
    httpd.shutdown()
    httpd.server_close()
    print("[ember] shutdown complete", flush=True)


def main():
    serve()


if __name__ == "__main__":
    main()
