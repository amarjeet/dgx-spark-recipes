#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
# Copyright (C) 2026 amarjeet
#
# Derived from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark, which is
# Copyright (C) 2026 MiaAI Lab and licensed AGPL-3.0-or-later.
# Modified 2026-09-05 by amarjeet: sources profiles.sh for names and paths
#
# Stop the vLLM container and its memory watchdog. The log is preserved.
#
# Stops gracefully by default: vLLM gets SIGTERM and a chance to unlink the
# POSIX shared-memory segments the PLE offload handshake allocates. The
# container runs with --ipc host, so anything it leaves behind leaks onto the
# host's /dev/shm and survives until reboot. Use --force to skip the wait.
#
# Usage:
#   ./stop.sh            # graceful, up to STOP_TIMEOUT (30s)
#   ./stop.sh --force    # straight to docker rm -f
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

FORCE=false
for arg in "$@"; do
  case "${arg}" in
    -f|--force) FORCE=true ;;
    -h|--help)  sed -n '4,15p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *)          printf 'unknown option: %s (try --help)\n' "${arg}" >&2; exit 2 ;;
  esac
done

# Stop the watchdog first so it cannot race a slow, graceful shutdown and turn
# it into a kill.
if pkill -f "memwatch.sh ${CONTAINER_NAME}" 2>/dev/null; then
  printf 'watchdog stopped\n'
fi

if container_exists; then
  if [[ "${FORCE}" == false ]]; then
    printf 'stopping %s (SIGTERM, up to %ss)...\n' "${CONTAINER_NAME}" "${STOP_TIMEOUT}"
    docker stop -t "${STOP_TIMEOUT}" "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  printf 'stopped %s\n' "${CONTAINER_NAME}"
else
  printf 'no container %s found\n' "${CONTAINER_NAME}"
fi

rm -f "${PID_FILE}" "${ACTIVE_PROFILE_FILE}"

# Report, but never delete: other containers on this host also run --ipc host,
# so their segments live here too and are not ours to remove.
leaked="$(find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'sem.mp-*' \) 2>/dev/null | wc -l)"
if (( leaked > 0 )); then
  bytes="$(find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'sem.mp-*' \) -printf '%s\n' 2>/dev/null \
           | awk '{s+=$1} END {print s+0}')"
  printf 'note: %s multiprocessing segment(s) in /dev/shm (%s)\n' "${leaked}" "$(human_bytes "${bytes}")"
  printf '      inspect: ls -la /dev/shm\n'
  printf '      only remove them once no vLLM/sglang container is running.\n'
fi

printf 'log preserved at %s\n' "${LOG_FILE}"
