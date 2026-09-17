#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2026 amarjeet
#
# Written for this port. Part of the same AGPL-3.0-only combined work as the
# files it sits beside, which derive from
# vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe,
# Copyright (C) 2026 Victor Cruz.
#
# Stop the TabbyAPI server started by start.sh.
#
# There is no container to remove -- this is a native recipe -- so the process
# tree is taken down by pidfile, then the log is left where it is. A model
# holding ~107 GiB is worth confirming gone rather than assuming: the next
# launch refuses to start while MemAvailable is still low.
#
# Usage:
#   ./stop.sh              # graceful TERM, then KILL after STOP_TIMEOUT
#   STOP_TIMEOUT=60 ./stop.sh
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

case "${1:-}" in
  -h|--help) sed -n '11,20p' "$0" | sed 's/^# \?//'; exit 0 ;;
  "") ;;
  *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
esac

STOP_TIMEOUT="${STOP_TIMEOUT:-30}"

if ! pid="$(server_pid)"; then
  printf 'no server recorded as running\n'
  rm -f "${PID_FILE}" "${ACTIVE_PROFILE_FILE}"
  # A stale pidfile is not the only way this recipe can be left running.
  if pgrep -f "[e]xllamav3" >/dev/null 2>&1; then
    printf 'warning: an exllamav3 process is still alive:\n' >&2
    pgrep -af "[e]xllamav3" >&2 || true
  fi
  exit 0
fi

printf 'stopping pid %s\n' "${pid}"
# Negative pid: TabbyAPI is launched through a subshell, so signal the group.
kill -TERM -- "-$(ps -o pgid= "${pid}" | tr -d ' ')" 2>/dev/null \
  || kill -TERM "${pid}" 2>/dev/null || true

waited=0
while kill -0 "${pid}" 2>/dev/null; do
  if (( waited >= STOP_TIMEOUT )); then
    printf 'still alive after %ss; sending KILL\n' "${STOP_TIMEOUT}" >&2
    kill -KILL -- "-$(ps -o pgid= "${pid}" | tr -d ' ')" 2>/dev/null \
      || kill -KILL "${pid}" 2>/dev/null || true
    break
  fi
  sleep 1
  (( waited++ ))
done

rm -f "${PID_FILE}" "${ACTIVE_PROFILE_FILE}"
printf 'stopped after %ss\n' "${waited}"
printf 'MemAvailable now %s\n' "$(human_bytes "$(mem_available_bytes)")"
printf 'log preserved at %s\n' "${LOG_FILE}"
