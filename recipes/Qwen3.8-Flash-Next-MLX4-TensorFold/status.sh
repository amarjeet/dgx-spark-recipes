#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
#
# Is it up, what is it serving, and is the host healthy.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

PROBE="http://$(probe_host):${PORT}"
printf 'container  : '
if container_running; then
  printf '%s running since %s (profile %s)\n' "${CONTAINER_NAME}" \
    "$(docker inspect -f '{{.State.StartedAt}}' "${CONTAINER_NAME}" | cut -c1-19)" \
    "$(cat "${OUT_DIR}/profile.active" 2>/dev/null || echo unknown)"
elif container_exists; then
  printf '%s exited (code %s)\n' "${CONTAINER_NAME}" "$(docker inspect -f '{{.State.ExitCode}}' "${CONTAINER_NAME}")"
else
  printf 'not running\n'
fi
printf 'endpoint   : '
if models="$(curl -sf --max-time 5 "${PROBE}/v1/models" 2>/dev/null)"; then
  printf '%s/v1  model %s\n' "${PROBE}" \
    "$(python3 -c 'import json,sys; print(", ".join(m["id"] for m in json.load(sys.stdin)["data"]))' <<<"${models}")"
  curl -s --max-time 5 "${PROBE}/health" | python3 -c '
import json, sys
h = json.load(sys.stdin)
print(f"health     : running {h.get(\"requests_running\", \"?\")}, prompt tokens {h.get(\"prompt_tokens_total\", \"?\"):,}, "
      f"completion tokens {h.get(\"completion_tokens_total\", \"?\"):,}, prefill {h.get(\"prefill_seconds_total\", 0):.1f} s")
' 2>/dev/null || true
else
  printf 'not answering on %s\n' "${PROBE}"
fi
if container_running; then
  docker logs "${CONTAINER_NAME}" 2>&1 | grep -E 'streams of|startup estimate' | tail -2 | sed 's/^/server     : /' || true
  cg="$(docker stats --no-stream --format '{{.MemUsage}}' "${CONTAINER_NAME}" 2>/dev/null || true)"
  [[ -n "${cg}" ]] && printf 'cgroup     : %s (host side only; GPU allocations are not charged)\n' "${cg}"
fi
printf 'host       : MemAvailable %s, MemFree %s of %s, swap used %s\n' \
  "$(human_bytes "$(mem_available_bytes)")" "$(human_bytes "$(mem_free_bytes)")" \
  "$(human_bytes "$(mem_total_bytes)")" "$(human_bytes "$(host_swap_used_bytes)")"
if pgrep -f "memwatch.sh ${CONTAINER_NAME}" >/dev/null; then
  printf 'watchdog   : running, %s\n' "$(grep 'low mark' "${OUT_DIR}/logs/memwatch-${CONTAINER_NAME}.log" 2>/dev/null | tail -1 | cut -d' ' -f2-)"
else
  printf 'watchdog   : not running\n'
fi
