#!/usr/bin/env bash
# Confucius4-T3PO (14B) as GGUF on the Apple GPU, behind an OpenAI-compatible
# API.  T3PO's own "vllm" backend talks to this unchanged -- only the URL
# differs.  --alias must match VLLM_MODEL in vendor/Confucius4-T3PO/.env.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${T3PO_GGUF:-$ROOT/models/Confucius4-T3PO-GGUF/Confucius4-T3PO-Q5_K_M.gguf}"

exec llama-server \
    --model "$MODEL" \
    --alias Confucius4-T3PO \
    --host 127.0.0.1 --port "${LLM_PORT:-8010}" \
    --ctx-size "${LLM_CTX:-8192}" \
    --n-gpu-layers 99 \
    --parallel 1 \
    --jinja \
    --log-prefix
