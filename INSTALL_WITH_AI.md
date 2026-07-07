# Install Ember with an AI assistant

**Hand this file to an AI coding assistant** (Claude, ChatGPT, Cursor, Continue, …) and
say: *"Set up Ember for me following INSTALL_WITH_AI.md."* The assistant will install and
configure Ember on your Mac, **asking you** for the choices that are yours to make.

---

## Instructions for the AI assistant

You are setting up **Ember** — an MLX, OpenAI-compatible inference server for Apple
Silicon (chat, tools, vision, embeddings, autocomplete) — on the user's Mac. Work in the
real terminal, run the commands, check their output, and **stop to ask the user at every
decision point below. Do not assume their hardware, models, or preferences.** Ask one
group of questions at a time, wait for the answer, then proceed.

### Step 1 — Preflight (run, then report)

```bash
uname -m            # must be "arm64" (Apple Silicon). If "x86_64", STOP: Ember needs M-series.
python3 --version   # must be 3.10 or newer
sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB RAM"}'
```

Report the chip type, Python version and total RAM back to the user. If Python < 3.10 or
the Mac is Intel, stop and explain.

### Step 2 — Ask: install options

Ask the user:
1. **Vision / structured output?** (image input + guaranteed JSON schema) — this adds the
   `[vision]` extra (mlx-vlm + llguidance). yes/no.
2. **Where to install?** Recommend a dedicated virtualenv. Default: `~/.ember-venv`.

Then install (adjust for their answers):

```bash
python3 -m venv ~/.ember-venv && source ~/.ember-venv/bin/activate
pip install "ember-mlx[vision]"     # or just "ember-mlx" if they said no to vision
ember --help                        # confirm it installed
```

### Step 3 — Ask: which models (this is the user's call)

Tell the user their **total RAM** (from Step 1) and that models must fit alongside ~3–4 GB
of system/headroom. Show this menu and let them pick — recommend by RAM, but they decide.
Speeds below were **measured on one specific machine (Apple M5, 24 GB)** — tell the user
that tok/s on their chip will differ, but the RAM column and the relative ranking carry
over (tok/s · RAM):

| Pick | Model (`mlx` repo) | tok/s | RAM | Good for |
|---|---|--:|--:|---|
| 🟢 light | `mlx-community/Qwen3-8B-4bit` | 28 | 5 GB | fast general/code chat |
| 🟢 light | `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit` | ~110 | 1 GB | tiny/quick |
| 🟢 fast MoE | `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit` | 77 | 9 GB | code, very fast |
| 🟡 balanced | `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit-DWQ` | 53 | 19 GB | best quality/speed (MoE) |
| 🟡 quality | `mlx-community/Qwen2.5-Coder-32B-Instruct-3bit` | 8 | 15 GB | strong code, slower (dense) |
| 👁 vision | `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` | — | 2 GB | image input (needs `[vision]`) |

Rules of thumb to share: MoE models (DeepSeek-Coder, Qwen3-30B-A3B) are much faster per GB.
Recommend by the user's RAM tier:

- **8 GB** → one 🟢 light model only (Ember's defaults keep a single model hot on this tier).
- **16 GB** → 🟢 light models comfortably, or the 🟢 fast-MoE as the single main model.
- **24–32 GB** → one 🟡 model fits comfortably alongside the small always-on models.
- **64 GB+** → 🟡 models at higher quants (4/6-bit) and several models hot at once.

Multiple models can be listed — Ember loads them on demand, keeps as many hot as RAM
allows, and [scales its memory defaults to the machine](docs/memory.md#ram-profiles-auto-defaults),
so no manual tuning is needed at any tier.

Ask which model(s) they want (and whether they want vision and a code-autocomplete model).

### Step 4 — Generate `ember.yaml`

In the directory the user will run Ember from, write `ember.yaml` from their picks:

```yaml
models:
  - name: <short-name>           # e.g. qwen3-8b — what they'll type in requests
    mlx: <repo they chose>
    params: { temperature: 0.0, num_ctx: 16384 }
  # one block per chosen model; add `vision: true` for a VLM
# autocomplete and embed are optional and default to sensible models — only add if asked.
```

Then validate:

```bash
ember config                     # should list their models with no error
```

### Step 5 — Ask: runtime options

Ask the user (offer the defaults):
- **Port?** default `8000`.
- **8-bit KV cache?** (≈2× more context in the same RAM, near-lossless) — recommend **yes**
  → set `MLX_KV_BITS=8`.
- **Keep models warm how long when idle?** default 5 min (`MLX_IDLE_TIMEOUT=300`).
- **Start automatically at login?** If yes, copy and edit
  `examples/com.ember.server.plist` into `~/Library/LaunchAgents/` (fill in the absolute
  path to the `ember` binary and the config dir, set the env vars above) and
  `launchctl load` it.

### Step 6 — Start and smoke-test

Start it (foreground for now, or via the launch agent if they chose that):

```bash
MLX_KV_BITS=8 ember serve        # include the envs they chose
```

In a second terminal, confirm it works (first run downloads the model — can take a while):

```bash
ember run <short-name> "Say hello in one short sentence."
ember status                     # show hot models + memory + policy
```

Report success and the API base URL (`http://127.0.0.1:<port>/v1`).

### Step 7 — Offer editor integration

Ask if they use **Continue** (or another OpenAI-compatible client). If yes, generate a
config from [`examples/continue.config.yaml`](examples/continue.config.yaml) with their
model names and port, and tell them where it goes (`~/.continue/config.yaml`).

### Wrap up

Summarize for the user: what was installed, the `ember.yaml` models, the env options set,
how to start/stop (`ember serve`, Ctrl-C, or the launch agent), and the handy commands
(`ember ps`, `ember run`, `ember clear`, `ember unload`). Point them to the README for the
full command and configuration reference.
