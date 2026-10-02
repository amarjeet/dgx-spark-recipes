#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
#
# Stop the server and its watchdog. Logs are preserved under $OUT_DIR/logs.
#
# Native, so the pidfile is the only handle. SIGTERM first so uvicorn shuts
# down cleanly; SIGKILL after STOP_TIMEOUT. The systemd scope goes with the
# process (--collect), so there is nothing else to clean up.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

if pid="$(server_pid 2>/dev/null)"; then
  printf 'stopping clef-flash (pid %s)\n' "${pid}"
  kill -TERM "${pid}" 2>/dev/null || true
  for _ in $(seq 1 "${STOP_TIMEOUT}"); do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "${pid}" 2>/dev/null; then
    printf 'still alive after %ss; sending SIGKILL\n' "${STOP_TIMEOUT}" >&2
    kill -KILL "${pid}" 2>/dev/null || true
    sleep 1
  fi
  printf 'stopped\n'
elif [[ -f "${PID_FILE}" ]]; then
  printf 'pidfile is stale (no such server); cleaning up\n'
else
  printf 'no clef-flash pidfile found\n'
  if ss -ltn "sport = :${PORT}" 2>/dev/null | tail -n +2 | grep -q .; then
    printf 'note: port %s is still in use by something this recipe did not start\n' "${PORT}"
  fi
fi

# The watchdog exits on its own once the pid is gone; this only covers a
# watchdog orphaned by a lost pidfile.
pkill -f "${EXPERIMENT_DIR}/scripts/memwatch.sh" 2>/dev/null || true
rm -f "${PID_FILE}"
printf 'logs preserved in %s\n' "${OUT_DIR}/logs"
