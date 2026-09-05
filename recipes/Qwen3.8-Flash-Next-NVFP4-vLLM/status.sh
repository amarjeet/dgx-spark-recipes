#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2026 amarjeet
#
# Written for this port. Part of the same AGPL-3.0-or-later combined work
# as the files it sits beside, which derive from MiaAI-Lab/
# Qwen3.8-Flash-Next-Single-DGX-Spark, Copyright (C) 2026 MiaAI Lab.
#
# One screen of "is it up, what is it serving, and is the box healthy".
#
# Usage: ./status.sh
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
select_profile "$(cat "${ACTIVE_PROFILE_FILE}" 2>/dev/null || printf '%s' "${DEFAULT_PROFILE}")"

printf 'experiment : %s\n' "${EXPERIMENT_NAME}"
printf 'profile    : %s (ctx %s, kv %s, mtp %s)\n' \
  "${PROFILE}" "${MAX_MODEL_LEN}" "${KV_CACHE_DTYPE}" "${MTP_NUM_SPECULATIVE_TOKENS}"
printf 'model      : %s @ %s (%s)\n' \
  "${MODEL_ID}" "${MODEL_REVISION:0:12}" "$(human_bytes "${MODEL_TOTAL_BYTES}")"
printf 'snapshot   : %s\n' "${SNAPSHOT_DIR}"

printf '\ncontainer\n'
if container_running; then
  docker ps --filter "name=^${CONTAINER_NAME}$" \
    --format '{{.Names}}  {{.Status}}  {{.Ports}}' | sed 's/^/  /'
  printf '  memory   %s\n' \
    "$(docker stats --no-stream --format '{{.MemUsage}}' "${CONTAINER_NAME}" 2>/dev/null || printf 'n/a')"
elif container_exists; then
  docker ps -a --filter "name=^${CONTAINER_NAME}$" --format '{{.Names}}  {{.Status}}' | sed 's/^/  /'
else
  printf '  not running\n'
fi

printf '\nwatchdog\n'
if pgrep -f "memwatch.sh ${CONTAINER_NAME}" >/dev/null 2>&1; then
  printf '  running (floors: MemAvailable %s GiB, MemFree %s GiB)\n' \
    "${MEMWATCH_MIN_GIB}" "${MEMWATCH_MIN_FREE_GIB}"
  tail -2 "${OUT_DIR}/logs/memwatch-${CONTAINER_NAME}.log" 2>/dev/null | sed 's/^/  /' || true
else
  printf '  not running\n'
fi

printf '\nhealth\n'
if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  printf '  OK http://127.0.0.1:%s/health\n' "${PORT}"
  printf '  models: '
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null \
    | python3 -c 'import json,sys; print(", ".join(m["id"] for m in json.load(sys.stdin)["data"]))' \
    2>/dev/null || printf '(unreadable)\n'
else
  printf '  not responding on port %s\n' "${PORT}"
fi

printf '\nhost\n'
printf '  memory available %s of %s\n' \
  "$(human_bytes "$(mem_available_bytes)")" "$(human_bytes "$(mem_total_bytes)")"
printf '  swap used        %s\n' "$(human_bytes "$(host_swap_used_bytes)")"
# The earliest signal this box gives that the driver is out of free pages.
nvrm="$(journalctl -k --since '10 min ago' 2>/dev/null | grep -c NV_ERR_NO_MEMORY || true)"
printf '  NV_ERR_NO_MEMORY %s in the last 10 min\n' "${nvrm:-0}"
others="$(docker ps --format '{{.Names}}' | grep -vx "${CONTAINER_NAME}" | paste -sd, - || true)"
printf '  other containers %s\n' "${others:-none}"

printf '\nstorage\n'
printf '  checkpoint  %s\n' \
  "$(du -sh --dereference "${SNAPSHOT_DIR}" 2>/dev/null | cut -f1 || printf 'absent')"
printf '  PLE table   %s\n' \
  "$(du -sh "${VLLM_CACHE_HOST}/ple_cache/${MODEL_ORG}--${MODEL_NAME}" 2>/dev/null | cut -f1 || printf 'not built')"
printf '  out dir     %s\n' "${OUT_DIR}"

printf '\nload log (%s)\n' "${LOG_FILE}"
if [[ -f "${LOG_FILE}" ]]; then
  # 'Route:' matches every registered endpoint and would crowd out everything
  # worth reading, so it is excluded rather than matched.
  grep -iE 'launching|GPU KV cache size|Available KV cache|Maximum concurrency|Loading weights took|Free memory on device|error|Traceback' \
    "${LOG_FILE}" | grep -vE 'Route:|min_frames|max_frames' | cut -c1-160 | tail -8 | sed 's/^/  /' \
    || printf '  (no summary lines yet)\n'
else
  printf '  (no log yet)\n'
fi
