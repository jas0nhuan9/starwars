#!/usr/bin/env bash
# Reinstates vLLM's min_tokens on top of llama-server; see the module docstring.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/.venv/bin/python" "$ROOT/llm_proxy/min_tokens_proxy.py"
