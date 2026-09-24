#!/usr/bin/env bash
# Stop everything scripts/start_all.sh started.
#
# The pid files cover the normal case; the pattern sweep afterwards catches a
# service that was started by hand (scripts/start_asr.sh on its own, say) and
# therefore never wrote one.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stopped=0

for pidfile in "$ROOT"/run/*.pid; do
    [[ -f "$pidfile" ]] || continue
    name="$(basename "$pidfile" .pid)"
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
        # The launchers exec their server, but kill the group in case a shell
        # wrapper survives.
        pkill -TERM -P "$pid" 2>/dev/null
        kill -TERM "$pid" 2>/dev/null
        echo "stopped $name (pid $pid)"
        stopped=$((stopped + 1))
    fi
    rm -f "$pidfile"
done

sleep 1
for pattern in "llama-server --model $ROOT" "min_tokens_proxy.py" "ws_server_mac.py" "uvicorn inference.server"; do
    while read -r pid; do
        [[ -n "$pid" ]] || continue
        kill -TERM "$pid" 2>/dev/null && {
            echo "stopped stray ${pattern%% *} (pid $pid)"
            stopped=$((stopped + 1))
        }
    done < <(pgrep -f "$pattern")
done

if [[ "$stopped" -eq 0 ]]; then
    echo "nothing was running"
else
    echo "$stopped process(es) stopped"
fi
