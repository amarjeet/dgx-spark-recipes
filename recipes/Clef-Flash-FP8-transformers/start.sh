#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
#
# Serve Clef-Flash (Cloudflare; Qwen3.5-9B + joint schema head, BF16) on the
# DGX Spark's GB10, natively, behind a SystemOne-compatible API on :8012.
#
# Why native transformers and not vLLM: the model generates no text. It scores
# every option of every question in one forward pass through a custom head,
# and the only code that runs that head is joint_schema_model.py in the
# snapshot. `vllm serve` would load the backbone as a chat model and drop it.
#
# Why it can share the box: no KV cache, and a footprint that stops growing
# once the allocator has seen one MAX_LENGTH forward pass. This script runs
# that pass before reporting ready, so the high-water mark is reached under
# the watchdog and measured, rather than met later by a long request while the
# server beside it is busy.
#
# Guards, because a native process gets none for free:
#   * preflight.sh's admission: budget + HOST_FLOOR_GIB must fit MemAvailable
#   * a systemd user scope with MemoryMax=CGROUP_MEM_GIB (host-side memory)
#   * scripts/memwatch.sh, set to fire before TensorFold's watchdog does
#
# Usage:
#   ./start.sh
#   MAX_LENGTH=8192 ./start.sh        # smaller activation peak
#   PREFLIGHT=0 ./start.sh            # skip preflight (not recommended)
set -euo pipefail

case "${1:-}" in
  -h|--help) sed -n '4,27p' "$0" | sed 's/^# \?//'; exit 0 ;;
esac

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
require_aarch64
command -v curl >/dev/null || { printf 'curl is not on PATH\n' >&2; exit 1; }
command -v systemd-run >/dev/null || { printf 'systemd-run is not on PATH\n' >&2; exit 1; }

if server_running; then
  printf 'clef-flash is already running (pid %s) on port %s\n' "$(server_pid)" "${PORT}"
  printf 'stop it first: ./stop.sh\n'
  exit 0
fi
rm -f "${PID_FILE}"

if [[ "${PREFLIGHT:-1}" == 1 ]]; then
  "${EXPERIMENT_DIR}/preflight.sh" || { printf '\npreflight failed; nothing was started\n' >&2; exit 1; }
  printf '\n'
fi

mkdir -p "${OUT_DIR}/logs"
# Keep the previous run's logs rather than appending a new run to them.
archive_ts="$(date '+%Y%m%dT%H%M%S')"
for f in "${LOG_FILE}" "${MEMWATCH_LOG}"; do
  [[ -s "${f}" ]] && mv "${f}" "${f%.log}-${archive_ts}.log"
done

READY_URL="http://$(probe_host):${PORT}/health"
avail_before="$(mem_available_bytes)"

printf 'model     : %s @ %s (BF16 checkpoint, %s)\n' "${MODEL_ID}" "${MODEL_REVISION:0:12}" "$(human_bytes "${MODEL_TOTAL_BYTES}")"
printf 'weights   : %s on the GPU (WEIGHTS)\n' "${WEIGHTS}"
printf 'runtime   : %s\n' "${PYTHON}"
printf 'max input : %s tokens\n' "${MAX_LENGTH}"
printf 'budget    : %s expected resident\n' "$(human_bytes "$(budget_bytes)")"
printf 'cgroup    : %s.scope, MemoryMax %s GiB, no swap\n' "${SCOPE_UNIT}" "${CGROUP_MEM_GIB}"
printf 'listening : %s:%s\n' "${HOST}" "${PORT}"
printf 'log       : %s\n\n' "${LOG_FILE}"

log_event "launch weights=${WEIGHTS} max_length=${MAX_LENGTH} revision=${MODEL_REVISION} cgroup=${CGROUP_MEM_GIB}GiB" >"${LOG_FILE}"

# systemd-run --scope execs the command in place, so $! is the server's own
# pid and the scope lives exactly as long as it does. setsid detaches it from
# this shell, so a Ctrl-C here does not take the server with it.
#
# Environment: nothing cache-related is set. HF_HOME is passed only because
# the snapshot path is derived from it; flash-linear-attention's Triton
# kernels go to triton's default ~/.triton, shared with every other recipe.
# expandable_segments lets the allocator grow segments instead of stranding
# reserved blocks, which keeps the footprint close to what is actually used.
setsid nohup systemd-run --user --scope --quiet --collect \
    --unit "${SCOPE_UNIT}" \
    -p "MemoryMax=${CGROUP_MEM_GIB}G" -p MemorySwapMax=0 \
    env HF_HOME="${HF_HOME}" \
        HF_HUB_OFFLINE=1 \
        CLEF_SNAPSHOT="${SNAPSHOT_DIR}" \
        SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
        WEIGHTS="${WEIGHTS}" \
        MAX_LENGTH="${MAX_LENGTH}" MAX_IMAGES="${MAX_IMAGES}" MAX_VIDEOS="${MAX_VIDEOS}" \
        MAX_BODY_MIB="${MAX_BODY_MIB}" \
        HOST="${HOST}" PORT="${PORT}" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONUNBUFFERED=1 \
        "${PYTHON}" "${EXPERIMENT_DIR}/server/app.py" \
    >>"${LOG_FILE}" 2>&1 &
pid=$!
printf '%s' "${pid}" >"${PID_FILE}"
printf 'spawned server (pid %s)\n' "${pid}"

# The watchdog starts with the server, not after it is ready: the load and the
# warmup are where the peak is.
MEMWATCH_MIN_GIB="${MEMWATCH_MIN_GIB}" MEMWATCH_MIN_FREE_GIB="${MEMWATCH_MIN_FREE_GIB}" \
MEMWATCH_FREE_GATE_GIB="${MEMWATCH_FREE_GATE_GIB}" MEMWATCH_SAMPLES="${MEMWATCH_SAMPLES}" \
MEMWATCH_INTERVAL="${MEMWATCH_INTERVAL}" STOP_TIMEOUT="${STOP_TIMEOUT}" \
  setsid nohup bash "${EXPERIMENT_DIR}/scripts/memwatch.sh" "${pid}" >"${MEMWATCH_LOG}" 2>&1 &
printf 'watchdog  : MemAvailable < %s GiB or MemFree < %s GiB (under %s) stops it\n' \
  "${MEMWATCH_MIN_GIB}" "${MEMWATCH_MIN_FREE_GIB}" "${MEMWATCH_FREE_GATE_GIB}"

printf 'waiting for %s (load, then one %s-token warmup pass)\n' "${READY_URL}" "${MAX_LENGTH}"
started=${SECONDS}
while ! curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! kill -0 "${pid}" 2>/dev/null; then
    printf '\nserver exited before becoming ready\n' >&2
    tail -n 40 "${LOG_FILE}" >&2 || true
    grep -q STOPPING "${MEMWATCH_LOG}" 2>/dev/null \
      && printf '\nThe watchdog stopped it: %s\n' "$(grep STOPPING "${MEMWATCH_LOG}")" >&2
    if grep -qiE 'out of memory|OutOfMemoryError|cannot allocate' "${LOG_FILE}"; then
      printf '\nAllocation failed. Try a shorter input limit: MAX_LENGTH=8192 ./start.sh\n' >&2
    fi
    if [[ "$(systemctl --user show "${SCOPE_UNIT}.scope" -p Result --value 2>/dev/null)" == oom-kill ]]; then
      printf '\nOOM-killed by the %s GiB cgroup cap; the host is fine. Raise CGROUP_MEM_GIB.\n' "${CGROUP_MEM_GIB}" >&2
    fi
    rm -f "${PID_FILE}"
    exit 1
  fi
  printf '  %4ss  MemAvailable %s\n' "$(( SECONDS - started ))" "$(human_bytes "$(mem_available_bytes)")"
  sleep 5
done

printf '\nready in %ss.\n' "$(( SECONDS - started ))"
avail_after="$(mem_available_bytes)"
printf '  predicted      %s (budget_bytes)\n' "$(human_bytes "$(budget_bytes)")"
printf '  measured       %s (MemAvailable before - after)\n' "$(human_bytes "$(( avail_before - avail_after ))")"
printf '  host left      %s MemAvailable\n' "$(human_bytes "${avail_after}")"
curl -fsS "${READY_URL}" | python3 -c '
import json, sys
h = json.load(sys.stdin)
print("  torch          %s GiB allocated, %s GiB peak reserved" % (h["torch_allocated_gib"], h["torch_peak_reserved_gib"]))
print("  load+warmup    %s s" % h["load_seconds"])
' 2>/dev/null || true

printf '\nendpoint   : POST http://%s:%s/v1/systemone\n' "$(probe_host)" "${PORT}"
printf 'model id   : %s\n' "${SERVED_MODEL_NAME}"
printf 'smoke test : ./scripts/smoke.py\n'
printf 'status     : ./status.sh\n'
printf 'stop       : ./stop.sh\n'
