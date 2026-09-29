#!/usr/bin/env bash
#
# Stop the llama-server process. The log is preserved.
#
# Native, so there is no container to remove: the pidfile is the only handle.
# SIGTERM first, because llama-server unmaps and unlocks tens of GiB on the way
# out and killing it outright leaves that to the kernel.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

if pid="$(server_pid 2>/dev/null)"; then
  printf 'stopping llama-server (pid %s)\n' "${pid}"
  kill -TERM "${pid}" 2>/dev/null || true
  for _ in $(seq 1 60); do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "${pid}" 2>/dev/null; then
    printf 'still alive after 60s; sending SIGKILL\n' >&2
    kill -KILL "${pid}" 2>/dev/null || true
    sleep 1
  fi
  printf 'stopped\n'
elif [[ -f "${PID_FILE}" ]]; then
  printf 'pidfile is stale (no such process); cleaning up\n'
else
  printf 'no llama-server pidfile found\n'
  # A server started by hand, or a pidfile lost to a reboot, still holds the
  # port -- say so rather than reporting "nothing to do".
  if ss -ltn "sport = :${PORT}" 2>/dev/null | tail -n +2 | grep -q .; then
    printf 'note: port %s is still in use by something this recipe did not start:\n' "${PORT}"
    pgrep -a -f 'llama-server' | sed 's/^/  /' || true
  fi
fi

rm -f "${PID_FILE}" "${ACTIVE_PROFILE_FILE}"
printf 'log preserved at %s\n' "${LOG_FILE}"
