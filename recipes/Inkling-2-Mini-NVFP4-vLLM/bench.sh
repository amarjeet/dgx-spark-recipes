#!/usr/bin/env bash
#
# Decode throughput against the running server: scripts/bench_tps.py at each
# concurrency, results appended to OUT_DIR/bench-<profile>-<timestamp>.jsonl.
# One warm-up request first, so first-request JIT does not land in the numbers.
#
# Usage: ./bench.sh [CONCURRENCIES] [MAX_TOKENS]     default: 1,2,4,8 1024
#        EFFORT=none ./bench.sh 1 512
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
CONCS="${1:-1,2,4,8}"
MAX_TOKENS="${2:-1024}"

curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || {
  printf 'error: nothing answering on :%s. Run ./start.sh first.\n' "${PORT}" >&2; exit 1; }
profile="$(cat "${ACTIVE_PROFILE_FILE}" 2>/dev/null || printf unknown)"
mkdir -p "${OUT_DIR}"
out="${OUT_DIR}/bench-${profile}-$(date +%Y%m%dT%H%M%S).jsonl"

printf 'profile %s, concurrency %s, %s tokens each, effort %s\n' "${profile}" "${CONCS}" "${MAX_TOKENS}" "${EFFORT:-low}"
PORT="${PORT}" python3 "${EXPERIMENT_DIR}/scripts/bench_tps.py" 1 128 >/dev/null
PORT="${PORT}" python3 "${EXPERIMENT_DIR}/scripts/bench_tps.py" "${CONCS}" "${MAX_TOKENS}" | tee "${out}"
printf 'MemAvailable %s\nresults: %s\n' "$(human_bytes "$(mem_available_bytes)")" "${out}"
