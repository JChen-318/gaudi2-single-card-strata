#!/usr/bin/env bash
# Start the Strata server (Gaudi2 backend) in the background.
#   CONFIG=/path/to/strata-hpu-production.json PORT=8080 bash hpu/deploy/run-production.sh
# S: this checkout (found from the script's location); RUN: where the pid file and the server log go.
set -euo pipefail
S=${S:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RUN=${RUN:-$S/run}
CONFIG=${CONFIG:-$S/hpu/deploy/strata-hpu-production.json}
PORT=${PORT:-8080}
mkdir -p "$RUN"
PIDFILE=$RUN/strata-hpu-prod.pid
SRVLOG=$RUN/strata-hpu-prod-server.log
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running: pid $(cat "$PIDFILE")"
  exit 0
fi
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "port $PORT is already in use"
  exit 1
fi
cd "$S"
export PYTHONPATH="$S/tools${PYTHONPATH:+:$PYTHONPATH}"
setsid nohup python3 -m serve.server --engine strata --config "$CONFIG" --host 0.0.0.0 --port "$PORT" \
  >> "$SRVLOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "started pid $(cat "$PIDFILE"); HTTP on 0.0.0.0:$PORT once the engine reports READY"
echo "server log: $SRVLOG    engine log: the config's \"log\""
