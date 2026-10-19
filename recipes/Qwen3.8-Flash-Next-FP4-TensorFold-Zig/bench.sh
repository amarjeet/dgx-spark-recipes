#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
#
# Measurements against the running server, with upstream's tools/. Results are
# tee'd to $OUT_DIR/bench/<timestamp>-<mode>.txt, never into the recipe directory.
#
#   ./bench.sh             prefill at four sizes + single-stream decode (tools/bench.py)
#   ./bench.sh streams     decode (prose, code) at STREAMS=1,2,3,4,5 requests, prefill 4k..128k (tools/bench_streams.py)
#   ./bench.sh needle      passphrase recall at ~195k tokens
#   ./bench.sh exact       greedy replies equal with drafts on and off, alone and concurrent
#   ./bench.sh checks      tool calls, image input, a 262,144-token window boundary
#   ./bench.sh all         prefill, streams, needle, exact, checks

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

MODE="${1:-prefill}"
URL="http://$(probe_host):${PORT}"
export PORT MODEL="${SERVED_MODEL_NAME}" API_URL="${URL}"
curl -sf --max-time 5 "${URL}/health" >/dev/null \
  || { printf 'error: nothing answering on port %s (./start.sh)\n' "${PORT}" >&2; exit 1; }
mkdir -p "${OUT_DIR}/bench"
profile="$(cat "${OUT_DIR}/profile.active" 2>/dev/null || echo unknown)"
T="${EXPERIMENT_DIR}/tools"

run() {  # run <name> <command...>
  local name="$1"; shift
  local out="${OUT_DIR}/bench/$(date '+%Y%m%dT%H%M%S')-${name}.txt"
  { printf '# %s  profile %s  image %s  %s\n' "${name}" "${profile}" "${IMAGE}" "$(date -Is)"
    "$@"; } 2>&1 | tee "${out}"
  printf '(saved %s)\n\n' "${out}"
}

prefill() { run prefill python3 "${T}/bench.py" "${profile}"; }
streams() { run streams python3 "${T}/bench_streams.py" "${URL}" --streams "${STREAMS:-1,2,3,4,5}"; }
needle()  { run needle python3 "${T}/needle.py" "${profile}"; }
exact()   { run exact python3 "${T}/exact.py"; }
checks()  {
  run toolcheck python3 "${T}/toolcheck.py"
  run visioncheck python3 "${T}/visioncheck.py"
  run context-boundary python3 "${T}/context_boundary.py" --context "${CONTEXT:-262144}"
}

case "${MODE}" in
  prefill) prefill ;;
  streams) streams ;;
  needle)  needle ;;
  exact)   exact ;;
  checks)  checks ;;
  all)     prefill; streams; needle; exact; checks ;;
  *) printf 'unknown mode: %s (prefill | streams | needle | exact | checks | all)\n' "${MODE}" >&2; exit 2 ;;
esac
