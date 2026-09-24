#!/usr/bin/env bash
# T3PO's own FastAPI + WebSocket web UI; .env points it at the two local
# services.  web_localized serves the upstream app with its interface text
# converted to the display script -- upstream's files are untouched.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/vendor/Confucius4-T3PO"
# The configuration lives in the tracked tree, not inside the upstream
# checkout, so it survives a vendor update and is version controlled.
if [[ -f "$ROOT/config/t3po.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/config/t3po.env"
    set +a
fi
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
exec "$ROOT/.venv/bin/python" -m uvicorn web_localized:app \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
