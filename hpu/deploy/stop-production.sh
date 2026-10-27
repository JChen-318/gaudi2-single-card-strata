#!/usr/bin/env bash
# Stop the Strata server started by run-production.sh (the server ends its engine).
set -u
S=${S:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RUN=${RUN:-$S/run}
PIDFILE=$RUN/strata-hpu-prod.pid
if [ ! -f "$PIDFILE" ]; then echo "not running (no $PIDFILE)"; exit 0; fi
pid=$(cat "$PIDFILE")
if ! kill -0 "$pid" 2>/dev/null; then echo "not running (stale pid $pid)"; rm -f "$PIDFILE"; exit 0; fi
kill -TERM "$pid"
for _ in $(seq 1 60); do
  kill -0 "$pid" 2>/dev/null || { rm -f "$PIDFILE"; echo "stopped"; exit 0; }
  sleep 1
done
echo "still running after 60 s: pid $pid (SIGKILL not sent)"
exit 1
