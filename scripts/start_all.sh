#!/usr/bin/env bash
# Bring up the whole local interpreter: translation LLM, ASR, web UI.
# Logs land in logs/, pids in run/.  Stop everything with scripts/stop_all.sh.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/logs" "$ROOT/run"

start() {
    local name="$1" script="$2"
    local pidfile="$ROOT/run/$name.pid"
    if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        echo "  $name already running (pid $(cat "$pidfile"))"
        return 0
    fi
    nohup bash "$script" > "$ROOT/logs/$name.log" 2>&1 &
    echo $! > "$pidfile"
    echo "  $name started (pid $!), log: logs/$name.log"
}

wait_for() {
    local name="$1" url="$2" tries="${3:-180}"
    printf "  waiting for %s " "$name"
    for ((i = 0; i < tries; i++)); do
        if curl -sf "$url" > /dev/null 2>&1; then echo "ready"; return 0; fi
        printf "."
        sleep 1
    done
    echo " TIMED OUT -- see logs/$name.log"
    return 1
}

echo "starting services:"
start llm "$ROOT/scripts/start_llm.sh"
start proxy "$ROOT/scripts/start_proxy.sh"
start asr "$ROOT/scripts/start_asr.sh"
echo
wait_for llm "http://127.0.0.1:${LLM_PORT:-8010}/v1/models" || exit 1
wait_for proxy "http://127.0.0.1:8011/v1/models" || exit 1
wait_for asr "http://127.0.0.1:8093/health" || exit 1
echo
start web "$ROOT/scripts/start_web.sh"
wait_for web "http://127.0.0.1:8000/api/v1/health" || exit 1
echo
echo "open http://127.0.0.1:8000/"
