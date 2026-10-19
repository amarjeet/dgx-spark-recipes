#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
#
# memwatch.sh <container>
#
# Host-memory watchdog. On GB10's unified memory an exhausted pool hangs the
# kernel rather than raising an OOM, so this stops the container while the
# host still has margin. It backs up TensorFold's own admission and the cgroup
# cap; it is not a substitute for either, since a poller cannot outrun a
# GiB-per-second collapse.
#
# Triggers, each debounced over MEMWATCH_SAMPLES consecutive samples:
#   * MemAvailable < MEMWATCH_MIN_GIB
#   * MemFree < MEMWATCH_MIN_FREE_GIB while MemAvailable < MEMWATCH_FREE_GATE_GIB
#     (the NVIDIA driver refuses allocations when free pages run out even
#     though MemAvailable still counts reclaimable cache; the gate keeps a full
#     but reclaimable page cache from tripping it)
#
# On a trigger it archives the container log to MEMWATCH_ARCHIVE_DIR, then
# `docker stop`s the container. Exits when the container stops for any reason.

set -uo pipefail

CONTAINER="${1:?usage: memwatch.sh <container>}"
MIN_GIB="${MEMWATCH_MIN_GIB:-6}"
MIN_FREE_GIB="${MEMWATCH_MIN_FREE_GIB:-2}"
GATE_GIB="${MEMWATCH_FREE_GATE_GIB:-10}"
SAMPLES="${MEMWATCH_SAMPLES:-5}"
INTERVAL="${MEMWATCH_INTERVAL:-2}"
GRACE="${STOP_TIMEOUT:-30}"
ARCHIVE_DIR="${MEMWATCH_ARCHIVE_DIR:-.}"

kib() { awk -v k="$1" '$1 == k":" {print $2; exit}' /proc/meminfo; }
to_kib() { awk -v g="$1" 'BEGIN {printf "%d", g * 1048576}'; }
gib() { awk -v k="$1" 'BEGIN {printf "%.2f", k / 1048576}'; }
stamp() { date -Is; }

min_kib="$(to_kib "${MIN_GIB}")"
free_kib="$(to_kib "${MIN_FREE_GIB}")"
gate_kib="$(to_kib "${GATE_GIB}")"

printf '[%s] watching %s: MemAvailable < %s GiB, or MemFree < %s GiB while MemAvailable < %s GiB, %s samples x %ss\n' \
  "$(stamp)" "${CONTAINER}" "${MIN_GIB}" "${MIN_FREE_GIB}" "${GATE_GIB}" "${SAMPLES}" "${INTERVAL}"

low=0
lowest=""
while [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null)" == true ]]; do
  avail="$(kib MemAvailable)"
  free="$(kib MemFree)"
  [[ -z "${lowest}" || "${avail}" -lt "${lowest}" ]] && lowest="${avail}"
  # Log a new low only when it is another 0.5 GiB down, not on every sample of a load.
  if [[ -z "${logged:-}" ]] || (( avail < logged - 524288 )); then
    logged="${avail}"
    printf '[%s] low mark: MemAvailable %s GiB, MemFree %s GiB\n' "$(stamp)" "$(gib "${avail}")" "$(gib "${free}")"
  fi
  reason=""
  if (( avail < min_kib )); then
    reason="MemAvailable $(gib "${avail}") GiB < ${MIN_GIB} GiB"
  elif (( avail < gate_kib && free < free_kib )); then
    reason="MemFree $(gib "${free}") GiB < ${MIN_FREE_GIB} GiB (MemAvailable $(gib "${avail}") GiB)"
  fi
  if [[ -n "${reason}" ]]; then
    low=$(( low + 1 ))
    printf '[%s] low %d/%d: %s\n' "$(stamp)" "${low}" "${SAMPLES}" "${reason}"
    if (( low >= SAMPLES )); then
      log="${ARCHIVE_DIR}/${CONTAINER}-$(date '+%Y%m%dT%H%M%S')-memwatch-stop.log"
      docker logs --tail 3000 "${CONTAINER}" >"${log}" 2>&1 || true
      printf '[%s] STOPPING %s: %s. Container log: %s\n' "$(stamp)" "${CONTAINER}" "${reason}" "${log}"
      docker stop -t "${GRACE}" "${CONTAINER}" >/dev/null 2>&1 || true
      exit 2
    fi
  elif (( low > 0 )); then
    printf '[%s] recovered after %d low sample(s)\n' "$(stamp)" "${low}"
    low=0
  fi
  sleep "${INTERVAL}"
done
printf '[%s] %s is no longer running; lowest MemAvailable seen %s GiB\n' "$(stamp)" "${CONTAINER}" "$(gib "${lowest:-0}")"
