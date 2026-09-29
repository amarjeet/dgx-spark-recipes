#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 MiaAI-Lab (original)
# Copyright (c) 2026 amarjeet (port)
#
# Derived from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
# (stop.sh), MIT. Also stops the watchdog and archives the container log.
#
# Stop the server and remove its container, freeing its memory. TensorFold
# gets STOP_TIMEOUT seconds (default 30) to exit. Requests still running are
# cut off, not drained, so this says when there are any.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

pkill -f "memwatch.sh ${CONTAINER_NAME}" 2>/dev/null || true
if ! container_exists; then
  printf 'no container named %s: nothing to stop\n' "${CONTAINER_NAME}"
  exit 0
fi
if container_running; then
  busy="$(curl -s --max-time 3 "http://$(probe_host):${PORT}/health" 2>/dev/null |
          python3 -c 'import json,sys; print(json.load(sys.stdin).get("requests_running", 0))' 2>/dev/null || echo 0)"
  (( busy == 0 )) || printf 'warning: %s request(s) still running will be cut off\n' "${busy}"
  printf 'stopping %s (up to %ss)\n' "${CONTAINER_NAME}" "${STOP_TIMEOUT}"
  docker stop -t "${STOP_TIMEOUT}" "${CONTAINER_NAME}" >/dev/null
fi
mkdir -p "${OUT_DIR}/logs"
docker logs --tail 3000 "${CONTAINER_NAME}" \
  >"${OUT_DIR}/logs/${CONTAINER_NAME}-$(date '+%Y%m%dT%H%M%S')-stopped.log" 2>&1 || true
docker rm -f "${CONTAINER_NAME}" >/dev/null
rm -f "${OUT_DIR}/profile.active"
printf 'stopped and removed %s; MemAvailable now %s\n' "${CONTAINER_NAME}" "$(human_bytes "$(mem_available_bytes)")"
