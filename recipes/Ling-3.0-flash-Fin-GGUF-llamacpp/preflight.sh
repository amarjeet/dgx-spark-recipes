#!/usr/bin/env bash
#
# Assert everything start.sh depends on, before a 60+ GiB load is attempted.
#
# The checks that actually matter, in order of how likely they are
# to bite:
#   1. llama.cpp build number -- bailingmoe3 landed in b10776. The
#      :server-cuda tag is mutable and an older pull will fail the load with
#      "unknown model architecture" only after reading the whole first shard.
#   2. Free memory -- if the host runs other model servers they will be
#      holding tens of GiB. The weights are mlock'd, so a load that does not
#      fit does not degrade gracefully.
#   3. Shard integrity -- a truncated shard surfaces as a confusing tensor
#      error mid-load.
#
# Usage: ./preflight.sh [PROFILE]
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
select_profile "${1:-${DEFAULT_PROFILE}}"

fail=0
ok()   { printf '  OK    %s\n' "$*"; }
warn() { printf '  WARN  %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; fail=1; }

printf 'preflight: %s (%s, %s)\n\n' "${PROFILE}" "${QUANT}" "$(human_bytes "${MODEL_TOTAL_BYTES}")"

printf 'platform\n'
if require_aarch64 2>/dev/null; then ok "aarch64"; else bad "not aarch64 (found $(uname -m))"; fi
for tool in docker curl python3 nvidia-smi; do
  if command -v "${tool}" >/dev/null 2>&1; then ok "${tool} on PATH"; else bad "${tool} is not on PATH"; fi
done
if nvidia-smi >/dev/null 2>&1; then
  ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
  bad "nvidia-smi failed"
fi

printf '\nruntime image\n'
if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  ok "image present: ${IMAGE}"
else
  bad "image not pulled: ${IMAGE} (run: docker pull ${IMAGE})"
fi
if (( ! fail )); then
  build="$(image_build_number || true)"
  if [[ -z "${build}" ]]; then
    bad "could not read a build number from ${IMAGE} --version"
  elif (( build < MIN_LLAMA_BUILD )); then
    bad "llama.cpp build ${build} < ${MIN_LLAMA_BUILD}: this image predates bailingmoe3 support. Run: docker pull ${IMAGE}"
  else
    ok "llama.cpp build ${build} >= ${MIN_LLAMA_BUILD} (bailingmoe3 supported)"
  fi
fi

printf '\nweights\n'
printf '  store   %s\n' "${MODEL_ROOT}"
missing=0
while IFS=$'\t' read -r rel bytes; do
  path="${MODEL_ROOT}/${rel}"
  if [[ ! -f "${path}" ]]; then
    bad "missing shard: ${rel}"
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
  # Full SHA-256 pass, skipped when nothing moved since the last one.
  if verify_shards "${MANIFEST}" "${MODEL_ROOT}" "${QUANT} shards" >/dev/null 2>&1; then
    ok "sha256 verified"
  else
    bad "sha256 verification failed (re-run: ./download.sh ${PROFILE} --verify-only)"
  fi
fi

printf '\ndisk\n'
# On a machine that has never run a llama.cpp recipe the store does not exist
# yet, and df would print nothing -- which makes the arithmetic below a syntax
# error under `set -e` rather than a useful message.
mkdir -p "${LLAMA_CACHE}"
free_disk="$(df --output=avail -B1 "${LLAMA_CACHE}" | tail -1)"
if (( free_disk >= DISK_RESERVE_BYTES )); then
  ok "free on $(df --output=target "${LLAMA_CACHE}" | tail -1): $(human_bytes "${free_disk}")"
else
  bad "only $(human_bytes "${free_disk}") free, want $(human_bytes "${DISK_RESERVE_BYTES}")"
fi

printf '\nmemory\n'
avail="$(mem_available_bytes)"
total="$(mem_total_bytes)"
swap_used="$(host_swap_used_bytes)"
needed=$(( MODEL_TOTAL_BYTES + MEM_HEADROOM_BYTES ))
printf '  total     %s\n' "$(human_bytes "${total}")"
printf '  available %s\n' "$(human_bytes "${avail}")"
printf '  swap used %s\n' "$(human_bytes "${swap_used}")"
printf '  weights   %s (+ %s headroom for KV, compute buffers and the OS)\n' \
  "$(human_bytes "${MODEL_TOTAL_BYTES}")" "$(human_bytes "${MEM_HEADROOM_BYTES}")"
if (( avail >= needed )); then
  ok "available memory covers weights + headroom"
else
  bad "need >= $(human_bytes "${needed}") available, have $(human_bytes "${avail}")"
  # Name the usual culprit rather than making the reader go looking.
  others="$(docker ps --format '{{.Names}}' | grep -vx "${CONTAINER_NAME}" || true)"
  if [[ -n "${others}" ]]; then
    printf '\n  other model containers are holding memory:\n'
    docker ps --format '{{.Names}}  {{.Image}}  {{.Status}}' \
      | grep -v "^${CONTAINER_NAME}  " | sed 's/^/    /' || true
    printf '  stop the one you do not need, then re-run preflight.\n'
  fi
fi
# KV and compute buffers are not in MODEL_TOTAL_BYTES; the headroom figure is a
# floor, not a budget. Flag the combinations most likely to be tight.
if (( avail >= needed )) && (( avail - needed < 16106127360 )); then
  warn "under 15 GiB of slack beyond the floor -- if the load OOMs, lower CTX_SIZE (currently ${CTX_SIZE})"
fi

printf '\nport\n'
if ss -ltn "sport = :${PORT}" 2>/dev/null | tail -n +2 | grep -q .; then
  if container_running; then
    ok "port ${PORT} held by our own container ${CONTAINER_NAME}"
  else
    bad "port ${PORT} is already in use by something else"
  fi
else
  ok "port ${PORT} is free"
fi

printf '\n'
if (( fail )); then
  printf 'preflight FAILED\n' >&2
  exit 1
fi
printf 'preflight OK -- ./start.sh %s\n' "${PROFILE}"
