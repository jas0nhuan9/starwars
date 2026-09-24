#!/usr/bin/env bash
# R2T2 streaming ASR on the Apple GPU, speaking upstream's WebSocket protocol.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
exec "$ROOT/.venv/bin/python" "$ROOT/asr_server/ws_server_mac.py"
