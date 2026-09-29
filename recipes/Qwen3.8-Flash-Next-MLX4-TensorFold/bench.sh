#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
#
# Measurements against the running server. Results are tee'd to
# $OUT_DIR/bench/<timestamp>-<mode>.txt, never into the recipe directory.
#
#   ./bench.sh             prefill at ~0.85k/3.2k/12.6k/50k tokens + decode (scripts/bench.py)
#   ./bench.sh concurrent  aggregate decode at 1, 2, 4, 5 concurrent requests
#   ./bench.sh needle      passphrase recall at ~195k tokens
#   ./bench.sh all         all three

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

MODE="${1:-prefill}"
export PORT SERVED_MODEL_NAME
curl -sf --max-time 5 "http://$(probe_host):${PORT}/v1/models" >/dev/null \
  || { printf 'error: nothing answering on port %s (./start.sh)\n' "${PORT}" >&2; exit 1; }
mkdir -p "${OUT_DIR}/bench"
profile="$(cat "${OUT_DIR}/profile.active" 2>/dev/null || echo unknown)"

run() {  # run <name> <command...>
  local name="$1"; shift
  local out="${OUT_DIR}/bench/$(date '+%Y%m%dT%H%M%S')-${name}.txt"
  { printf '# %s  profile %s  image %s  %s\n' "${name}" "${profile}" "${IMAGE}" "$(date -Is)"
    "$@"; } 2>&1 | tee "${out}"
  printf '(saved %s)\n\n' "${out}"
}

case "${MODE}" in
  prefill)    run prefill python3 "${EXPERIMENT_DIR}/scripts/bench.py" "${profile}" ;;
  concurrent) run concurrent python3 "${EXPERIMENT_DIR}/scripts/bench_concurrent.py" ;;
  needle)     run needle python3 "${EXPERIMENT_DIR}/scripts/needle.py" "${profile}" ;;
  all)
    run prefill python3 "${EXPERIMENT_DIR}/scripts/bench.py" "${profile}"
    run concurrent python3 "${EXPERIMENT_DIR}/scripts/bench_concurrent.py"
    run needle python3 "${EXPERIMENT_DIR}/scripts/needle.py" "${profile}"
    ;;
  *) printf 'unknown mode: %s (prefill | concurrent | needle | all)\n' "${MODE}" >&2; exit 2 ;;
esac
