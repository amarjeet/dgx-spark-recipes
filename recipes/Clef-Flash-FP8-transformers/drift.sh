#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
#
# The FP8 gate: answer a fixed set of records in BF16, convert the decoder to
# FP8, answer them again, and compare (server/fp8_drift.py). Exit 1 if any
# decision BF16 made with >= 0.6 confidence changes. Results are tee'd to
# $OUT_DIR/drift/<timestamp>.txt.
#
# It loads its own copy of the model at the BF16 load peak, so the server must
# be stopped, the same admission as a start applies, and it runs under the
# same watchdog.
#
#   ./drift.sh
set -euo pipefail

case "${1:-}" in
  -h|--help) sed -n '4,14p' "$0" | sed 's/^# \?//'; exit 0 ;;
esac

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

server_running && { printf 'error: the server is running (pid %s); ./stop.sh first\n' "$(server_pid)" >&2; exit 1; }
WEIGHTS=bf16
need="$(( $(load_peak_bytes) + LOAD_FLOOR_GIB * 1073741824 ))"
(( $(mem_available_bytes) >= need )) || {
  printf 'error: need %s available for the BF16 load, have %s\n' \
    "$(human_bytes "${need}")" "$(human_bytes "$(mem_available_bytes)")" >&2
  exit 1
}
verify_snapshot >/dev/null

mkdir -p "${OUT_DIR}/drift" "${OUT_DIR}/logs"
out="${OUT_DIR}/drift/$(date '+%Y%m%dT%H%M%S').txt"
CLEF_SNAPSHOT="${SNAPSHOT_DIR}" HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PYTHON}" "${EXPERIMENT_DIR}/server/fp8_drift.py" >"${out}" 2>&1 &
pid=$!
bash "${EXPERIMENT_DIR}/scripts/memwatch.sh" "${pid}" >"${OUT_DIR}/logs/drift-memwatch.log" 2>&1 &
rc=0
wait "${pid}" || rc=$?
grep -v -e 'fast path is not available' -e 'fla.utils' "${out}" || true
grep -q STOPPING "${OUT_DIR}/logs/drift-memwatch.log" && printf 'the watchdog stopped it: %s\n' "$(grep STOPPING "${OUT_DIR}/logs/drift-memwatch.log")" >&2
printf '(saved %s)\n' "${out}"
exit "${rc}"
