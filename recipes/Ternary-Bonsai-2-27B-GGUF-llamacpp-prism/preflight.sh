#!/usr/bin/env bash
#
# Assert everything start.sh depends on, before a load is attempted.
#
# The checks that actually matter, in order of how badly they fail silently:
#   1. Is the binary the fork? Stock llama.cpp refuses PQ2_0 and PTQ1_0
#      outright -- fine -- but it loads the same family's plain Q2_0 without a
#      warning and emits fluent nonsense, having no Hadamard runtime.
#   2. Was it built for sm_121? A binary without it cannot run on GB10, and the
#      CUDA error you get does not say so.
#   3. Is the GGUF a rotated Bonsai pack? Same failure mode as (1), from the
#      other direction.
#   4. Memory. A native recipe gets no cgroup cap and no watchdog, so this
#      budget is the only thing standing between a too-large profile and a box
#      that stops responding.
#
# Usage: ./preflight.sh [PROFILE]
set -euo pipefail

case "${1:-}" in
  -h|--help) sed -n '3,18p' "$0" | sed 's/^# \?//'; exit 0 ;;
esac

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
select_profile "${1:-${DEFAULT_PROFILE}}"

fail=0
ok()   { printf '  OK    %s\n' "$*"; }
warn() { printf '  WARN  %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; fail=1; }

printf 'preflight: %s (%s, %s over %s slot(s) = %s per slot)\n\n' \
  "${PROFILE}" "${QUANT}" "${CTX_SIZE}" "${PARALLEL}" "${CTX_PER_SLOT}"

printf 'platform\n'
if require_aarch64 2>/dev/null; then ok "aarch64"; else bad "not aarch64 (found $(uname -m))"; fi
for tool in curl python3 nvidia-smi strings; do
  if command -v "${tool}" >/dev/null 2>&1; then ok "${tool} on PATH"; else bad "${tool} is not on PATH"; fi
done
if nvidia-smi >/dev/null 2>&1; then
  gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  if [[ "${gpu}" == *GB10* ]]; then
    ok "GPU: ${gpu}"
  else
    warn "GPU is ${gpu}, not GB10 -- CUDA_ARCHS=${CUDA_ARCHS} may be wrong for it"
  fi
else
  bad "nvidia-smi failed"
fi

printf '\nfork binary\n'
printf '  build   %s\n' "${BIN_DIR}"
if [[ -x "${BIN_DIR}/llama-server" ]]; then
  ok "llama-server present"
  # The check that catches "I built stock llama.cpp into this tree".
  if fork_has_ternary_kernels; then
    ok "ternary kernels present (pq2_0/ptq1_0 in libggml-base)"
  else
    bad "no ternary types in libggml-base -- this is stock llama.cpp, not the fork"
    printf '        run: ./build.sh --clean\n'
  fi
  archs="$(cuda_backend_archs || true)"
  if [[ -z "${archs}" ]]; then
    warn "could not read CUDA architectures (cuobjdump missing?)"
  elif [[ "${archs}" == *sm_121* ]]; then
    ok "CUDA backend has sm_121 (${archs})"
  else
    bad "CUDA backend lacks sm_121: ${archs} -- cannot run on GB10"
    printf '        run: CUDA_ARCHS=121a-real ./build.sh --clean\n'
  fi
  build="$(bin_build_number || true)"
  if [[ -z "${build}" ]]; then
    bad "could not read a build number from llama-server --version"
  elif (( build < MIN_LLAMA_BUILD )); then
    bad "llama.cpp build ${build} < ${MIN_LLAMA_BUILD}"
  else
    ok "llama.cpp build ${build} >= ${MIN_LLAMA_BUILD}"
  fi
else
  bad "llama-server not built -- run: ./build.sh"
fi

printf '\nweights\n'
printf '  store   %s\n' "${MODEL_ROOT}"
missing=0
while IFS=$'\t' read -r rel bytes; do
  path="${MODEL_ROOT}/${rel}"
  if [[ ! -f "${path}" ]]; then
    bad "missing file: ${rel}"
    missing=1
  else
    actual="$(stat -c %s "${path}")"
    if [[ "${actual}" != "${bytes}" ]]; then
      bad "wrong size: ${rel} (${actual} != ${bytes})"
      missing=1
    else
      ok "${rel} ($(human_bytes "${bytes}"))"
    fi
  fi
done < <(python3 -c '
import json, sys
for f in json.load(open(sys.argv[1]))["files"]:
    print(f["path"], f["bytes"], sep="\t")' "${MANIFEST}")

if (( missing )); then
  printf '\n  run: ./download.sh %s\n' "${PROFILE}"
else
  if verify_shards "${MANIFEST}" "${MODEL_ROOT}" "${QUANT} files" >/dev/null 2>&1; then
    ok "sha256 verified"
  else
    bad "sha256 verification failed (re-run: ./download.sh ${PROFILE} --verify-only)"
  fi
  # Filename is not evidence. The rotation metadata is.
  if gguf_has_hadamard "${MODEL_FILE}"; then
    ok "prism.hadamard.* present -- this is a rotated Bonsai pack"
  else
    bad "${MODEL_ENTRY_REL} has no prism.hadamard.* metadata"
    printf '        the fork will either refuse it or produce garbage.\n'
  fi
fi

printf '\ndisk\n'
# On a machine that has never run a llama.cpp recipe the store does not exist
# yet, and df would print nothing -- making the arithmetic below a syntax error
# under `set -e` rather than a useful message.
mkdir -p "${LLAMA_CACHE}"
free_disk="$(df --output=avail -B1 "${LLAMA_CACHE}" | tail -1)"
if (( free_disk >= DISK_RESERVE_BYTES )); then
  ok "free on $(df --output=target "${LLAMA_CACHE}" | tail -1): $(human_bytes "${free_disk}")"
else
  bad "only $(human_bytes "${free_disk}") free, want $(human_bytes "${DISK_RESERVE_BYTES}")"
fi

printf '\nmemory budget\n'
avail="$(mem_available_bytes)"
total="$(mem_total_bytes)"
swap_used="$(host_swap_used_bytes)"
needed="$(budget_bytes)"
printf '  total     %s\n' "$(human_bytes "${total}")"
printf '  available %s\n' "$(human_bytes "${avail}")"
printf '  swap used %s\n' "$(human_bytes "${swap_used}")"
budget_table
if (( avail >= needed )); then
  ok "available memory covers the budget"
else
  bad "need >= $(human_bytes "${needed}") available, have $(human_bytes "${avail}")"
  case "${PROFILE}" in
    wide) printf '        try: ./start.sh safe   (4 slots x 65536)\n' ;;
    deep) printf '        try: ./start.sh long   (131072 over 1 slot)\n' ;;
  esac
fi
# This is a native recipe: nothing external will stop it from taking the pool.
warn "native recipe: no cgroup cap and no watchdog -- this budget is the only guard"

printf '\ntenancy\n'
others="$(docker ps --format '{{.Names}}  {{.Image}}  {{.Status}}' 2>/dev/null || true)"
if [[ -n "${others}" ]]; then
  printf '  other containers are holding memory:\n'
  printf '%s\n' "${others}" | sed 's/^/    /'
  printf '  a DGX Spark has one pool; stop what you do not need.\n'
else
  ok "no other containers running"
fi
foreign="$(pgrep -a -f 'llama-server|vllm|sglang|tabbyapi' 2>/dev/null \
  | grep -v "^$(server_pid 2>/dev/null || printf 'none') " || true)"
if [[ -n "${foreign}" ]]; then
  printf '  other model servers:\n'
  printf '%s\n' "${foreign}" | sed 's/^/    /'
else
  ok "no other model server processes"
fi

printf '\nport\n'
if ss -ltn "sport = :${PORT}" 2>/dev/null | tail -n +2 | grep -q .; then
  if server_running; then
    ok "port ${PORT} held by our own llama-server (pid $(server_pid))"
  else
    bad "port ${PORT} is already in use by something else"
  fi
else
  ok "port ${PORT} is free"
fi

printf '\ncontext\n'
if (( CTX_PER_SLOT > 98304 )); then
  # Not a FAIL: the user may well want to run it anyway, and the memory really
  # is there. But it must not be discovered from an empty API response.
  warn "${CTX_PER_SLOT} per slot is above the depths llama.cpp #27756 reports failing"
  printf '        that issue is open against this exact architecture and the failure is\n'
  printf '        silent: clean prefill, then EOS as the first token, no error.\n'
  printf '        this build passed a needle test at 254032 tokens, but the issue\n'
  printf '        describes a prompt-dependent onset, so re-prove it for your workload:\n'
  printf '          ./scripts/smoke.py --needle-depth %s\n' "${CTX_PER_SLOT}"
  printf '        conservative alternative: ./start.sh safe\n'
else
  ok "${CTX_PER_SLOT} per slot is below the reported #27756 failure depths"
fi

printf '\n'
if (( fail )); then
  printf 'preflight FAILED\n' >&2
  exit 1
fi
printf 'preflight OK -- ./start.sh %s\n' "${PROFILE}"
