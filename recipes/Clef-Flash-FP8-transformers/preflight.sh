#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
#
# Assert everything start.sh depends on, before a load:
#
#   1. Platform. aarch64 GB10, the venv exists and its torch sees the GPU.
#   2. Weights. The pinned snapshot is complete and matches the manifest --
#      including joint_schema_model.py, which is code this server imports.
#   3. Memory, at both moments that matter: the load peak (BF16, before the
#      FP8 conversion) must leave LOAD_FLOOR_GIB, and the steady state (served
#      weights + host side + one MAX_LENGTH pass) must leave HOST_FLOOR_GIB.
#      This recipe shares the pool with a running server, and admitting it
#      must not push that server toward its own watchdog.
#   4. Tenancy and port. Co-tenants are listed, not refused; the port must be
#      free.
#
# Usage: ./preflight.sh
set -euo pipefail

case "${1:-}" in
  -h|--help) sed -n '4,17p' "$0" | sed 's/^# \?//'; exit 0 ;;
esac

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

fail=0
ok()   { printf '  ok    %s\n' "$*"; }
warn() { printf '  warn  %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; fail=1; }

printf 'Clef-Flash preflight (%s weights, max_length %s, port %s)\n' "${WEIGHTS}" "${MAX_LENGTH}" "${PORT}"

printf '\nplatform\n'
if require_aarch64 2>/dev/null; then ok "aarch64"; else bad "not aarch64"; fi
if [[ -x "${PYTHON}" ]]; then
  if versions="$("${PYTHON}" -c 'import torch, transformers; assert torch.cuda.is_available(); print(torch.__version__, transformers.__version__)' 2>/dev/null)"; then
    ok "venv ${VENV}: torch ${versions% *}, transformers ${versions#* }"
  else
    bad "venv ${VENV} cannot import torch/transformers or see the GPU; run ./setup.sh"
  fi
else
  bad "no venv at ${VENV}; run ./setup.sh"
fi
if ( set +o pipefail; nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | grep -q '^12\.1' ); then
  ok "GB10 (sm_121)"
else
  bad "GPU is not compute capability 12.1"
fi

printf '\nweights\n'
if out="$(verify_snapshot 2>&1)"; then
  ok "${MODEL_ID} @ ${MODEL_REVISION:0:12}: ${out}"
else
  bad "snapshot: ${out}"
  printf '        run: ./download.sh\n'
fi

printf '\nmemory\n'
avail="$(mem_available_bytes)"
need="$(admission_bytes)"
budget_table
printf '    MemAvailable now     %s, need %s\n' "$(human_bytes "${avail}")" "$(human_bytes "${need}")"
if server_running; then
  ok "already running (pid $(server_pid)); its memory is already counted in MemAvailable"
elif (( avail >= need )); then
  ok "fits: $(human_bytes "$(( avail - $(budget_bytes) ))") would remain once up (floor ${HOST_FLOOR_GIB} GiB)"
else
  bad "needs $(human_bytes "${need}") available, have $(human_bytes "${avail}")"
  [[ "${WEIGHTS}" == bf16 ]] && printf '        BF16 needs ~6.4 GiB more than FP8: WEIGHTS=fp8 ./start.sh\n'
  printf '        or free memory first -- e.g. fewer TensorFold streams:\n'
  printf '          ../Qwen3.8-Flash-Next-MLX4-TensorFold/start.sh restart int8x1\n'
  printf '        or a shorter input limit: MAX_LENGTH=8192 ./start.sh\n'
fi

printf '\ntenancy\n'
others="$(docker ps --format '{{.Names}}  {{.Image}}' 2>/dev/null || true)"
if [[ -n "${others}" ]]; then
  printf '  sharing the pool with:\n%s\n' "$(printf '%s\n' "${others}" | sed 's/^/    /')"
else
  ok "no containers running"
fi
foreign="$(pgrep -a -f 'llama-server|vllm|sglang|tabbyapi' 2>/dev/null || true)"
[[ -n "${foreign}" ]] && printf '  native servers:\n%s\n' "$(printf '%s\n' "${foreign}" | cut -c1-120 | sed 's/^/    /')"
if awk -v m="${MEMWATCH_MIN_GIB}" 'BEGIN {exit !(m <= 6)}'; then
  warn "MEMWATCH_MIN_GIB=${MEMWATCH_MIN_GIB} is not above TensorFold's 6 GiB; under pressure Qwen may be stopped before Clef"
else
  ok "this watchdog (${MEMWATCH_MIN_GIB} GiB) fires before TensorFold's (6 GiB)"
fi

printf '\nport\n'
if ss -ltn "sport = :${PORT}" 2>/dev/null | tail -n +2 | grep -q .; then
  if server_running; then ok "port ${PORT} held by this server (pid $(server_pid))"
  else bad "port ${PORT} is already in use by something else"; fi
else
  ok "port ${PORT} is free"
fi

printf '\n'
if (( fail )); then printf 'preflight FAILED\n'; exit 1; fi
printf 'preflight passed\n'
