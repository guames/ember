#!/usr/bin/env bash
# Boots a real Ember server against a genuinely tiny MLX model and exercises the two
# generation endpoints end-to-end. Used by .github/workflows/nightly-smoke.yml, but
# runnable locally too: ./scripts/nightly_smoke.sh
#
# Catches the class of bug that has hit production twice (mlx-lm/Homebrew Cellar churn
# breaking a long-lived process) — CI otherwise never loads a model or calls mlx-lm.
set -euo pipefail

TINY_MODEL="mlx-community/SmolLM2-135M-Instruct-8bit"
PORT="${EMBER_SMOKE_PORT:-8123}"
HOST="127.0.0.1"
WORKDIR="$(mktemp -d)"
LOG="$WORKDIR/ember.log"
CONFIG="$WORKDIR/ember.yaml"

cat >"$CONFIG" <<YAML
models:
  - name: smoke
    mlx: $TINY_MODEL
    params: { temperature: 0.0 }
autocomplete:
  name: autocomplete
  mlx: $TINY_MODEL
YAML

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
on_failure() {
  echo "--- ember server log ---"
  cat "$LOG" || true
  echo "-------------------------"
}
trap cleanup EXIT
trap on_failure ERR

MLX_ROUTER_PORT="$PORT" MLX_ROUTER_HOST="$HOST" EMBER_CONFIG="$CONFIG" \
  ember serve >"$LOG" 2>&1 &
SERVER_PID=$!

BASE="http://$HOST:$PORT"
echo "waiting for $BASE/health ..."
for _ in $(seq 1 120); do
  if curl -sf "$BASE/health" >/dev/null 2>&1; then
    echo "server is up"
    break
  fi
  sleep 1
done
curl -sf "$BASE/health" >/dev/null || { echo "server never became healthy"; exit 1; }

echo "--- /v1/chat/completions ---"
chat_resp=$(curl -sf -X POST "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"smoke","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":8}')
echo "$chat_resp"
echo "$chat_resp" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d["choices"][0]["message"]["content"].strip(), "empty chat completion"'

echo "--- /v1/completions ---"
comp_resp=$(curl -sf -X POST "$BASE/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"autocomplete","prompt":"def add(a, b):\n    return","max_tokens":8}')
echo "$comp_resp"
echo "$comp_resp" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d["choices"][0]["text"].strip() != "", "empty completion"'

echo "smoke test passed"
