#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
#
# One screen of "is it up, what does it hold, and is the shared pool healthy".
#
# Usage: ./status.sh
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

printf 'experiment : %s\n' "${EXPERIMENT_NAME}"
printf 'model      : %s @ %s\n' "${MODEL_ID}" "${MODEL_REVISION:0:12}"
printf 'snapshot   : %s\n' "${SNAPSHOT_DIR}"
printf 'venv       : %s\n' "${VENV}"

printf '\nprocess\n'
if pid="$(server_pid 2>/dev/null)"; then
  printf '  running  pid %s, up %s\n' "${pid}" "$(ps -o etime= -p "${pid}" | tr -d ' ')"
  printf '  rss      %s\n' "$(human_bytes "$(( $(ps -o rss= -p "${pid}" | tr -d ' ') * 1024 ))")"
  cur="$(systemctl --user show "${SCOPE_UNIT}.scope" -p MemoryCurrent --value 2>/dev/null || true)"
  [[ "${cur}" =~ ^[0-9]+$ ]] && printf '  cgroup   %s of %s GiB (%s.scope; host side only)\n' \
    "$(human_bytes "${cur}")" "${CGROUP_MEM_GIB}" "${SCOPE_UNIT}"
  if pgrep -f "${EXPERIMENT_DIR}/scripts/memwatch.sh ${pid}" >/dev/null; then
    printf '  watchdog running (%s)\n' "${MEMWATCH_LOG}"
  else
    printf '  watchdog NOT running -- restart with ./stop.sh && ./start.sh\n'
  fi
else
  printf '  not running\n'
  [[ -f "${PID_FILE}" ]] && printf '  (stale pidfile: %s)\n' "${PID_FILE}"
fi

printf '\nhealth\n'
if h="$(curl -fsS "http://127.0.0.1:${PORT}/health" 2>/dev/null)"; then
  printf '%s' "${h}" | python3 -c '
import json, sys
h = json.load(sys.stdin)
print("  OK       %s, max_length %s" % (h["model"], h["max_length"]))
print("  torch    %s GiB allocated, %s GiB reserved, %s GiB peak" % (
    h["torch_allocated_gib"], h["torch_reserved_gib"], h["torch_peak_reserved_gib"]))
print("  load     %s s, host cost %s GiB" % (h["load_seconds"], h["host_cost_gib"]))
'
else
  printf '  not responding on port %s\n' "${PORT}"
fi

printf '\nhost (one pool, shared)\n'
printf '  MemAvailable %s of %s, MemFree %s\n' \
  "$(human_bytes "$(mem_available_bytes)")" "$(human_bytes "$(mem_total_bytes)")" "$(human_bytes "$(mem_free_bytes)")"
printf '  this watchdog fires under %s GiB; TensorFold'"'"'s under 6 GiB\n' "${MEMWATCH_MIN_GIB}"
others="$(docker ps --format '{{.Names}}' 2>/dev/null | paste -sd, - || true)"
printf '  containers   %s\n' "${others:-none}"

if [[ -f "${MEMWATCH_LOG}" ]]; then
  printf '\nwatchdog log (%s)\n' "${MEMWATCH_LOG}"
  tail -n 4 "${MEMWATCH_LOG}" | sed 's/^/  /'
fi
