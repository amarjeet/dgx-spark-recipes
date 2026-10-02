#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
#
# Latency by input length and throughput under concurrency, against the
# running server. Results are tee'd to $OUT_DIR/bench/<timestamp>.txt, never
# into the recipe directory.
#
#   ./bench.sh [REPEATS]     default 20 requests per row
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

export PORT
curl -sf --max-time 5 "http://$(probe_host):${PORT}/health" >/dev/null \
  || { printf 'error: nothing answering on port %s (./start.sh)\n' "${PORT}" >&2; exit 1; }
mkdir -p "${OUT_DIR}/bench"
out="${OUT_DIR}/bench/$(date '+%Y%m%dT%H%M%S').txt"
others="$(docker ps --format '{{.Names}}' 2>/dev/null | paste -sd, - || true)"
{ printf '# clef-flash @ %s  max_length %s  co-tenants: %s  %s\n' \
    "${MODEL_REVISION:0:12}" "${MAX_LENGTH}" "${others:-none}" "$(date -Is)"
  SMOKE_HOST="$(probe_host)" python3 "${EXPERIMENT_DIR}/scripts/bench.py" "${1:-20}"; } 2>&1 | tee "${out}"
printf '(saved %s)\n' "${out}"
