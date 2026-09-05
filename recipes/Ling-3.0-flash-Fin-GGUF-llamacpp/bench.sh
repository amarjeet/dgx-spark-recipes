#!/usr/bin/env bash
#
# Throughput against context depth, plus an optional MTP A/B.
#
# Results go to $OUT_DIR (see profiles.sh), never into the recipe directory.
#
# Usage:
#   ./bench.sh                 # depth ladder against the running server
#   ./bench.sh depths          # same
#   ./bench.sh mtp             # restart twice and A/B MTP speculative decoding
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
select_profile "$(cat "${ACTIVE_PROFILE_FILE}" 2>/dev/null || printf '%s' "${DEFAULT_PROFILE}")"

MODE="${1:-depths}"
mkdir -p "${OUT_DIR}"

require_server() {
  curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || {
    printf 'server is not responding on port %s -- run ./start.sh first\n' "${PORT}" >&2
    exit 1
  }
}

run_depths() {  # run_depths <out.json>
  python3 -u "${EXPERIMENT_DIR}/scripts/bench_depths.py" --port "${PORT}" --out "$1"
}

case "${MODE}" in
  depths)
    require_server
    out="${OUT_DIR}/depths-${PROFILE}-$(date +%Y%m%dT%H%M%S).json"
    printf 'profile %s (%s), ctx %s\n\n' "${PROFILE}" "${QUANT}" "${CTX_SIZE}"
    run_depths "${out}"
    ;;

  mtp)
    # Whether MTP self-speculation is worth it is a property of this
    # checkpoint and this machine, not something to take on faith -- so measure
    # both. Each arm needs its own server, hence the restarts.
    stamp="$(date +%Y%m%dT%H%M%S)"
    for spec in draft-mtp none; do
      printf '\n=== spec-type: %s ===\n' "${spec}"
      "${EXPERIMENT_DIR}/stop.sh" >/dev/null 2>&1 || true
      SPEC_TYPE="${spec}" "${EXPERIMENT_DIR}/start.sh" "${PROFILE}" >/dev/null
      require_server
      run_depths "${OUT_DIR}/mtp-${spec}-${PROFILE}-${stamp}.json"
    done
    printf '\nA/B written to %s (mtp-*-%s.json)\n' "${OUT_DIR}" "${stamp}"
    printf 'decode tok/s is the column that should move; prefill should not.\n'
    ;;

  *)
    printf 'usage: %s {depths|mtp}\n' "$0" >&2
    exit 2
    ;;
esac
