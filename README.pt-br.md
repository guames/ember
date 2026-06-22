# 🔥 Ember

[![CI](https://github.com/guames/ember/actions/workflows/ci.yml/badge.svg)](https://github.com/guames/ember/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ember-mlx.svg)](https://pypi.org/project/ember-mlx/)
[![Python](https://img.shields.io/pypi/pyversions/ember-mlx.svg)](https://pypi.org/project/ember-mlx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[🇬🇧 English](README.md) · 🇧🇷 **Português**

**Um servidor de inferência MLX quentinho e econômico de memória, para Apple Silicon.**

Um único processo serve **chat, tool-calling, visão, embeddings e autocomplete de código** —
tudo em [MLX](https://github.com/ml-explore/mlx), tudo compatível com a API da OpenAI, com uma
única política de memória adaptativa que mantém seus modelos *quentes* e nunca estoura sua RAM.

Feito para assistentes de código locais (ex.: [Continue](https://continue.dev)) num único Mac.

> ⚠️ Status: **beta**. Roda em uso diário, mas a API ainda pode mudar antes da 1.0.

---

## O que é o Ember?

**O Ember é um servidor de IA local para o seu Mac.** Você aponta suas ferramentas (um
assistente de código, um script, o `curl`) para ele como se fosse a API da OpenAI, e ele roda
os modelos **na sua própria máquina** — privado, offline, sem assinatura. Ele cuida de chat e
código, autocomplete de código, embeddings de texto e entendimento de imagem, tudo a partir de
um único endereço.

**O que ele faz.** Um ambiente de código de verdade precisa de vários modelos ao mesmo tempo —
um grande para conversar sobre código, um pequeno e rápido para o autocomplete inline, um
embedder para busca no codebase, talvez um modelo de visão para screenshots. Rodar isso como
servidores separados significa três processos brigando pela mesma RAM. O Ember roda todos em
**um** processo, com um único cérebro que decide o que manter carregado ("quente"), o que
descarregar e como compartilhar a memória — para tudo seguir rápido sem nunca estourar a RAM.

**Por que ele existe.** O [Ollama](https://ollama.com) tornou os modelos locais genuinamente
agradáveis: carregar pelo nome, manter os modelos quentes, reusar o cache da conversa, gerenciar
por uma CLI simples. Mas o Ollama roda em llama.cpp/GGUF. No **Apple Silicon (os chips da linha
M)**, o próprio framework [MLX](https://github.com/ml-explore/mlx) da Apple costuma ser **mais
rápido e mais leve** para o mesmo modelo. O Ember pega as partes do Ollama que o tornam gostoso
de usar — os modelos quentes, o cache de prefixo, a CLI sem frescura — e as reconstrói
**nativamente em MLX, afinadas para os Macs da linha M**: consistentemente usa de 1 a 3 GB a
menos de RAM por modelo e empata ou supera a velocidade do Ollama (veja
[Benchmarks](#benchmarks)). Ele **não** é um fork do Ollama — não compartilha nada do código
dele; pega emprestada a *ergonomia* e a reimplementa para a stack da Apple, somando o que um
assistente de código quer de fábrica (tools, visão, saída em JSON-schema, autocomplete
cooperativo).

Em resumo: **a facilidade do Ollama, a velocidade do MLX, feito para a linha M.**

## O que o torna diferente

Existem outros servidores MLX compatíveis com a OpenAI (`mlx_lm.server`, FastMLX, o backend do
LM Studio…). O nicho do Ember é ser o **unificado e adaptativo de memória** para um único Mac:

- 🧩 **Um servidor, todos os papéis.** Chat/código, autocomplete FIM, embeddings e visão num
  único processo — em vez de fazer malabarismo com três servidores e três orçamentos de memória.
- ⏱️ **Preempção cooperativa.** Requisições de autocomplete e embedding *furam a fila e rodam
  entre os tokens do chat*, então digitar nunca trava uma geração longa.
- 🧠 **Memória adaptativa.** Vários modelos ficam quentes enquanto a RAM permite (eviction LRU,
  descarga por ociosidade, `keep_alive`). Sob pressão, ele **dropa os caches KV do mais antigo
  primeiro** antes de evictar um modelo inteiro.
- ⚡ **Prompt cache (reuso de prefixo).** Reuso de KV por maior-prefixo-comum no estilo
  llama.cpp/Ollama → TTFT muito menor ao continuar uma conversa. Sem cópia.
- 🎯 **Decodificação realmente restrita.** `response_format` com JSON schema é *garantido* via
  [llguidance](https://github.com/guidance-ai/llguidance) (máscara em nível de token), não um
  empurrãozinho no prompt.
- 🛠️ **Superfície completa da OpenAI.** Tools/function-calling (`tool_choice` incl. forçado),
  streaming, `stop`, `seed`, penalidades de repetição/presença/frequência, `logit_bias`.
- 💾 **Afinado para 24 GB.** Cache KV de 8 bits, prefill em blocos (menor pico de RAM), pinagem
  de memória wired para velocidade consistente perto do limite.

---

## Benchmarks

Medido num **Apple M5, 24 GB** (MLX). A geração é limitada pela banda de memória, então modelos
MoE voam enquanto os densos da classe 30B trocam velocidade por qualidade:

| Modelo | Quant | tok/s · MLX | tok/s · Ollama | MLX mais rápido | RAM · MLX | RAM · Ollama | MLX mais leve |
|---|---|--:|--:|--:|--:|--:|--:|
| DeepSeek-Coder-V2-16B (MoE) | 4-bit | **77** | 68 | **+13%** | **9 GB** | 11 GB | **−18%** |
| Qwen3-30B-A3B (MoE) | 3-bit | **68** | 56 | **+21%** | **13 GB** | 15 GB | **−13%** |
| Qwen3-8B | 3-bit | **36** | 30 | **+20%** | **4 GB** | 7 GB | **−43%** |
| Phi-4-14B | 3-bit | **19** | 16 | **+19%** | **6 GB** | 10 GB | **−40%** |
| Qwen2.5-Coder-32B | 3-bit | 8 | 8 | ±0% | **15 GB** | 16 GB | **−6%** |

Otimizações (medidas): o prompt cache corta o **TTFT ~5×** (396 → 80 ms num prompt de 1,3k
tokens); o prefill em blocos reduz o pico de RAM ~19%; o cache KV de 8 bits é ~2× menor.

➡️ Tabelas completas (todas as 18 configs, memória de cache KV por modelo, comparação com o
Ollama) em [docs/benchmarks.md](docs/benchmarks.md).

## Primeiros passos

> 🤖 **Quer um assistente de IA configurando isso para você?** Entregue a ele o
> [INSTALL_WITH_AI.md](INSTALL_WITH_AI.md) — ele conduz o assistente pela instalação e
> configuração do Ember enquanto *pergunta a você* quais modelos e opções você quer.

### 0. Requisitos

- Um Mac com **Apple Silicon** (M1 ou mais novo). O Ember não roda em Macs Intel.
- **Python 3.10+** — confira com `python3 --version`. (Pegue em [python.org](https://www.python.org/downloads/macos/) ou `brew install python`.)
- Disco + RAM livres para os modelos que você escolher (8 GB dá conta de modelos pequenos; 24 GB+
  para os grandes — veja [Benchmarks](#benchmarks)).

### 1. Instalação

```bash
# recomendado: um ambiente isolado
python3 -m venv ~/.ember-venv
source ~/.ember-venv/bin/activate

pip install ember-mlx                # núcleo: chat, autocomplete, embeddings
# ou, para também ter visão + saída em JSON-schema:
pip install "ember-mlx[vision]"
```

Confira se funcionou:

```bash
ember --help
```

### 2. Configure seus modelos

Crie um arquivo chamado **`ember.yaml`** na pasta onde você vai rodar o Ember (comece a partir
do [`examples/models.yaml`](examples/models.yaml)). Cada entrada tem um `name` (como você vai
chamá-lo nas requisições) e um repositório `mlx` do Hugging Face:

```yaml
models:
  - name: qwen3-8b                       # pequeno e rápido — boa primeira escolha
    mlx: mlx-community/Qwen3-8B-4bit
    params: { temperature: 0.0, num_ctx: 32768 }

  - name: qwen2.5-vl                     # opcional: visão (precisa do extra [vision])
    mlx: mlx-community/Qwen2.5-VL-3B-Instruct-4bit
    vision: true
```

Não sabe quais modelos? Veja os [Benchmarks](#benchmarks) para velocidade/RAM, depois valide seu
arquivo com `ember config`. Os modelos são baixados automaticamente na primeira vez que são
usados.

### 3. Inicie o servidor

```bash
ember serve                            # serve em http://127.0.0.1:8000/v1
```

Deixe rodando neste terminal (ou configure para iniciar no login — veja
[`examples/com.ember.server.plist`](examples/com.ember.server.plist)).

### 4. Use

De outro terminal — do jeito amigável:

```bash
ember run qwen3-8b "Write a haiku about Metal shaders."
ember ps          # o que está carregado agora
```

…ou como uma API normal da OpenAI:

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3-8b",
  "messages": [{"role": "user", "content": "Write a haiku about Metal shaders."}]
}'
```

É isso. Para conectar ao seu editor, veja [Usar com o Continue](#usar-com-o-continue).

## Gerencie pelo terminal

O Ember vem com uma CLI enxuta. Rode `ember --help` ou `ember <comando> --help` para detalhes.

| Comando | O que faz |
|---|---|
| `ember serve` | inicia o servidor (`--host` `--port` `--config`) |
| `ember ps` | lista os modelos **quentes** na RAM (tamanho, ociosidade, keep-alive, tokens em cache) |
| `ember list` | lista os modelos **configurados** e quais estão quentes |
| `ember status` | status completo: modelos + memória + fila + política |
| `ember memory` | detalhamento de memória (MLX + sistema) |
| `ember run <modelo> [prompt]` | chat avulso com streaming (prompt via argumento ou stdin) |
| `ember warm <modelo>` | pré-carrega um modelo na RAM (sem geração) |
| `ember unload [alvo]` | descarrega `chat` (padrão) / `all` / `<modelo>` |
| `ember config` | mostra o arquivo de config resolvido e valida os modelos |
| `ember version` | imprime a versão |

```console
$ ember ps
MODEL                            SIZE  VISION    IDLE   KEEP   CACHE
qwen3-8b                         3.3G      -      0s   5.0m      50

$ echo "refactor this loop" | ember run qwen3-8b
```

Os comandos de gerência falam com um servidor em execução (`--url`, padrão
`http://127.0.0.1:8000`).

## Endpoints

| Endpoint | Propósito |
|---|---|
| `POST /v1/chat/completions` | chat/código — com e sem streaming; `tools`, `response_format`, imagens |
| `POST /v1/completions` | autocomplete FIM (mantido quente) |
| `POST /v1/embeddings` | embeddings (mantido quente) |
| `GET /v1/models` | lista os modelos configurados |
| `GET /status` | modelos quentes, memória, fila, política |
| `GET /memory` | memória MLX + sistema |
| `POST /unload` | descarrega `chat` / `all` / `<modelo>` |

## Configuração (env)

| Variável | Padrão | Significado |
|---|---|---|
| `MLX_ROUTER_PORT` / `MLX_ROUTER_HOST` | `8000` / `127.0.0.1` | endereço de bind |
| `MLX_MAX_RUNNERS` | `4` | máximo de modelos quentes ao mesmo tempo |
| `MLX_MIN_FREE_GB` | `2.0` | evicta um modelo abaixo desta RAM livre |
| `MLX_MIN_FREE_CACHE_GB` | `1.0` | dropa caches KV abaixo desta RAM livre |
| `MLX_IDLE_TIMEOUT` | `300` | segundos de ociosidade antes de descarregar um modelo de chat |
| `MLX_MAX_QUEUE` | `32` | profundidade da fila antes de retornar 503 |
| `MLX_PROMPT_CACHE` | `1` | reuso de cache KV por prefixo |
| `MLX_KV_BITS` | off | `8`/`4` para quantizar o cache KV (~2× menor em 8 bits) |
| `MLX_PREFILL_STEP` | `512` | tamanho do bloco de prefill (menor pico de RAM) |
| `MLX_WIRED_LIMIT_GB` | auto | teto de memória wired (RAM−5 GB) |
| `EMBER_CONFIG` | — | caminho explícito para o arquivo de config dos modelos |

Veja [`docs/`](docs/) para detalhes de tools, visão, `response_format`, prompt cache e memória.

## Usar com o Continue

Aponte o Continue para o Ember como um provedor OpenAI — veja
[`examples/continue.config.yaml`](examples/continue.config.yaml). Modelos de visão recebem
`capabilities: [image_input]`.

## Roadmap

- [ ] Prompt cache para modelos de visão
- [ ] Context shifting (gerar além do `num_ctx`)
- [ ] Camada de compatibilidade `/api/*` nativa
- [ ] Batching opcional para requisições concorrentes ao mesmo modelo

## Licença

[MIT](LICENSE) © Gustavo Ames. Sem afiliação com a Apple ou com o projeto MLX.
