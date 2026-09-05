#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2026 amarjeet
#
# Written for this port. Part of the same AGPL-3.0-or-later combined work
# as the files it sits beside, which derive from MiaAI-Lab/
# Qwen3.8-Flash-Next-Single-DGX-Spark, Copyright (C) 2026 MiaAI Lab.
#
# Throughput against context depth, plus optional A/B runs of the two knobs
# whose value is a property of this machine rather than a given.
#
# Results go to $OUT_DIR (see profiles.sh), never into the recipe directory.
#
# Usage:
#   ./bench.sh                 # depth ladder against the running server
#   ./bench.sh depths          # same
#   ./bench.sh mtp             # restart twice, A/B MTP speculative decoding
#   ./bench.sh kv              # restart twice, A/B fp8 vs bf16 KV
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
    printf 'profile %s (ctx %s, kv %s, mtp %s)\n\n' \
      "${PROFILE}" "${MAX_MODEL_LEN}" "${KV_CACHE_DTYPE}" "${MTP_NUM_SPECULATIVE_TOKENS}"
    run_depths "${out}"
    ;;

  mtp)
    # Whether MTP self-speculation pays for its 1.49 GiB is a property of this
    # checkpoint and this machine, not something to take on faith. Each arm
    # needs its own server, hence the restarts.
    stamp="$(date +%Y%m%dT%H%M%S)"
    for mtp in 3 0; do
      printf '\n=== MTP_NUM_SPECULATIVE_TOKENS=%s ===\n' "${mtp}"
      "${EXPERIMENT_DIR}/stop.sh" >/dev/null 2>&1 || true
      MTP_NUM_SPECULATIVE_TOKENS="${mtp}" "${EXPERIMENT_DIR}/start.sh" "${PROFILE}" >/dev/null
      require_server
      run_depths "${OUT_DIR}/mtp-${mtp}-${PROFILE}-${stamp}.json"
    done
    printf '\nA/B written to %s (mtp-*-%s.json)\n' "${OUT_DIR}" "${stamp}"
    printf 'decode tok/s is the column that should move; prefill should not.\n'
    ;;

  kv)
    # FP8 KV buys context and costs some speed and possibly some quality. The
    # speed half is measurable here; the quality half is not, and the README
    # says so.
    stamp="$(date +%Y%m%dT%H%M%S)"
    for prof in native bf16kv; do
      printf '\n=== profile %s ===\n' "${prof}"
      "${EXPERIMENT_DIR}/stop.sh" >/dev/null 2>&1 || true
      "${EXPERIMENT_DIR}/start.sh" "${prof}" >/dev/null
      require_server
      run_depths "${OUT_DIR}/kv-${prof}-${stamp}.json"
    done
    printf '\nA/B written to %s (kv-*-%s.json)\n' "${OUT_DIR}" "${stamp}"
    ;;

  *)
    printf 'usage: %s {depths|mtp|kv}\n' "$0" >&2
    exit 2
    ;;
esac
