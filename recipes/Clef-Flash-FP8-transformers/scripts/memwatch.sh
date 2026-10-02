#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
#
# memwatch.sh <pid>
#
# Host-memory watchdog for a native server process, adapted from the
# TensorFold recipe's container watchdog. On GB10's unified memory an
# exhausted pool hangs the kernel rather than raising an OOM, so this stops
# the server while the host still has margin.
#
# Its thresholds are deliberately stricter and faster than TensorFold's (see
# profiles.sh, "co-tenancy"): when both servers share the pool and memory runs
# short, this one fires first and Clef is the one that goes.
#
# Triggers, each debounced over MEMWATCH_SAMPLES consecutive samples:
#   * MemAvailable < MEMWATCH_MIN_GIB
#   * MemFree < MEMWATCH_MIN_FREE_GIB while MemAvailable < MEMWATCH_FREE_GATE_GIB
#
# On a trigger it SIGKILLs the pid at once. Not SIGTERM: during a load uvicorn
# is still inside its startup hook and does not act on SIGTERM, so a grace
# period lets the load keep allocating -- measured here, an 18 s grace carried
# the pool past TensorFold's watchdog and cost the co-tenant instead. The
# server holds no state worth a clean shutdown; a reload is a minute.
# Exits when the process exits for any reason.

set -uo pipefail

PID="${1:?usage: memwatch.sh <pid>}"
MIN_GIB="${MEMWATCH_MIN_GIB:-7}"
MIN_FREE_GIB="${MEMWATCH_MIN_FREE_GIB:-2}"
GATE_GIB="${MEMWATCH_FREE_GATE_GIB:-10.5}"
SAMPLES="${MEMWATCH_SAMPLES:-3}"
INTERVAL="${MEMWATCH_INTERVAL:-1}"

kib() { awk -v k="$1" '$1 == k":" {print $2; exit}' /proc/meminfo; }
to_kib() { awk -v g="$1" 'BEGIN {printf "%d", g * 1048576}'; }
gib() { awk -v k="$1" 'BEGIN {printf "%.2f", k / 1048576}'; }
stamp() { date -Is; }

min_kib="$(to_kib "${MIN_GIB}")"
free_kib="$(to_kib "${MIN_FREE_GIB}")"
gate_kib="$(to_kib "${GATE_GIB}")"

printf '[%s] watching pid %s: MemAvailable < %s GiB, or MemFree < %s GiB while MemAvailable < %s GiB, %s samples x %ss\n' \
  "$(stamp)" "${PID}" "${MIN_GIB}" "${MIN_FREE_GIB}" "${GATE_GIB}" "${SAMPLES}" "${INTERVAL}"

low=0
lowest=""
while kill -0 "${PID}" 2>/dev/null; do
  avail="$(kib MemAvailable)"
  free="$(kib MemFree)"
  [[ -z "${lowest}" || "${avail}" -lt "${lowest}" ]] && lowest="${avail}"
  # Log a new low only when it is another 0.5 GiB down, not on every sample.
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
      printf '[%s] STOPPING pid %s (SIGKILL): %s\n' "$(stamp)" "${PID}" "${reason}"
      kill -KILL "${PID}" 2>/dev/null || true
      exit 2
    fi
  elif (( low > 0 )); then
    printf '[%s] recovered after %d low sample(s)\n' "$(stamp)" "${low}"
    low=0
  fi
  sleep "${INTERVAL}"
done
printf '[%s] pid %s is no longer running; lowest MemAvailable seen %s GiB\n' "$(stamp)" "${PID}" "$(gib "${lowest:-0}")"
