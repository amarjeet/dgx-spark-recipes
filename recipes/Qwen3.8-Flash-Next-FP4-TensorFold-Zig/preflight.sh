#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
#
# Assert everything start.sh depends on, before a multi-minute load.
#
#   ./preflight.sh [profile]      profile: fp8x5 (default) | fp8x3 | bf16x3 | fp8x1
#
# Exit 0 when every check passes, 1 otherwise. Warnings do not fail.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
select_profile "${1:-${DEFAULT_PROFILE}}"
serve_args

fails=0
ok()   { printf '  [ OK ] %s\n' "$*"; }
warn() { printf '  [WARN] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; fails=$(( fails + 1 )); }

printf 'profile %s: %s x %s tokens, %s KV%s\n' "${PROFILE}" "${PARALLEL}" "${CONTEXT}" "${KV_DTYPE}" \
  "$([[ "${VISION}" == 1 ]] && printf ', image/video input')"

printf '\nhost\n'
if require_aarch64 2>/dev/null; then ok "aarch64"; else bad "not aarch64 ($(uname -m))"; fi
if command -v docker >/dev/null && docker info >/dev/null 2>&1; then ok "docker daemon reachable"
else bad "docker not installed or not reachable (is your user in the docker group?)"; fi
if docker info 2>/dev/null | grep -qi nvidia; then ok "nvidia container runtime"
else warn "docker lists no nvidia runtime; --gpus all may fail"; fi
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
[[ -n "${gpu}" ]] && ok "GPU: ${gpu}" || bad "nvidia-smi found no GPU"

printf '\nimage\n'
if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  ok "${IMAGE} present"
  label="$(image_patches_label)"
  [[ "${label}" == "${PATCHES_HASH}" ]] && ok "tf.patches ${label} matches patches/ and build.sh" \
    || bad "tf.patches is ${label:-missing}, the recipe hashes to ${PATCHES_HASH}: run ./build.sh"
  [[ "$(image_label tf.kernels)" == present ]] && ok "AOT kernel set present" \
    || bad "no tf.kernels label: run ./build.sh --rebuild"
  docker run --rm --network none --entrypoint test "${IMAGE}" -x /opt/tensorfold/native/bin/tensorfold-native \
    && ok "tensorfold-native" || bad "${IMAGE} has no tensorfold-native"
else
  bad "${IMAGE} missing: run ./build.sh"
fi

printf '\ncheckpoint\n'
if [[ -d "${SNAPSHOT_DIR}" ]]; then
  if out="$(verify_snapshot 2>&1)"; then ok "${MODEL_ID} @ ${MODEL_REVISION:0:12}: ${out##*$'\n'}"
  else bad "snapshot does not match the manifest: run ./download.sh (${out##*$'\n'})"; fi
else
  bad "${SNAPSHOT_DIR} missing: run ./download.sh"
fi

printf '\nport and tenancy\n'
if container_running; then
  warn "${CONTAINER_NAME} is already running; ./start.sh will leave it alone (./start.sh restart replaces it)"
elif ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
  bad "port ${PORT} is in use: $(ss -ltnp "sport = :${PORT}" 2>/dev/null | tail -n +2 | awk '{print $NF}')"
else
  ok "port ${PORT} free"
fi
others="$(docker ps --format '{{.Names}}' | grep -vx "${CONTAINER_NAME}" || true)"
gpu_apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
# nvidia-smi cannot name processes inside containers on GB10, so the running
# containers are the better witness.
if ! container_running && { [[ -n "${gpu_apps}" ]] || [[ -n "${others}" ]]; }; then
  [[ -n "${gpu_apps}" ]] && { bad "the GPU is already in use -- one model server at a time on this pool:"
    printf '         %s\n' "${gpu_apps//$'\n'/$'\n'         }"; }
  [[ -n "${others}" ]] && printf '         running containers: %s\n' "$(paste -sd' ' <<<"${others}")"
else
  ok "no other GPU process"
fi

printf '\nmemory\n'
total="$(mem_total_bytes)"; avail="$(mem_available_bytes)"
need_gib="$(mem_need_gib)"; need="$(gib_to_bytes "${need_gib}")"
printf '         MemTotal %s, MemAvailable %s, swap used %s\n' \
  "$(human_bytes "${total}")" "$(human_bytes "${avail}")" "$(human_bytes "$(host_swap_used_bytes)")"
printf '         a start needs ~%s GiB: %s weights + %s one %s-token window + %s scratch%s + %s reserve + 2\n' \
  "${need_gib}" "${WEIGHTS_GIB}" "$(kv_gib)" "${CONTEXT}" "${SCRATCH_GIB}" \
  "$([[ "${VISION}" == 1 ]] && printf ' + %s vision' "${VISION_WORKSPACE_GIB}")" "${TENSORFOLD_MEMORY_RESERVE_GIB}"
if container_running; then
  warn "skipping the budget check: ${CONTAINER_NAME} is running and holds its memory"
elif (( avail >= need )); then
  ok "MemAvailable covers it ($(human_bytes "$(( avail - need ))") spare)"
else
  bad "MemAvailable $(human_bytes "${avail}") is under ~${need_gib} GiB: stop the other model server first"
fi
# Under traffic the pool grows until MemAvailable is about the reserve. Below
# the watchdog's floor, the watchdog would stop the server first.
if (( $(gib_to_bytes "${TENSORFOLD_MEMORY_RESERVE_GIB}") > $(gib_to_bytes "${MEMWATCH_MIN_GIB}") )); then
  ok "the engine stops growing the pool at ~${TENSORFOLD_MEMORY_RESERVE_GIB} GiB free, above the watchdog's ${MEMWATCH_MIN_GIB} GiB"
else
  warn "reserve ${TENSORFOLD_MEMORY_RESERVE_GIB} GiB is not above the watchdog's ${MEMWATCH_MIN_GIB} GiB floor, which then stops the server"
fi
cg="$(gib_to_bytes "${CONTAINER_MEM_GIB}")"
(( cg < total / 2 )) && ok "container host-side cap ${CONTAINER_MEM_GIB} GiB" \
  || warn "CONTAINER_MEM_GIB=${CONTAINER_MEM_GIB} is over half the pool; the cap no longer protects the host"
full=$(( PARALLEL * $(kv_gib) ))
(( WEIGHTS_GIB + full + TENSORFOLD_MEMORY_RESERVE_GIB > 121 )) \
  && warn "${PARALLEL} full windows are ~${full} GiB of ${KV_DTYPE} cache beside ~${WEIGHTS_GIB} GiB of weights: the engine refuses a request that would not fit"

printf '\nserve arguments\n'
if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  # The engine's own parser in a throwaway container without the GPU: a typo in
  # EXTRA_SERVE_ARGS fails here rather than after a stop. A path that is not a
  # directory is the parser's "arguments fine" outcome.
  code=0
  err="$(docker run --rm --network none --entrypoint /opt/tensorfold/native/bin/tensorfold-native "${IMAGE}" \
      serve /nonexistent/argument-check "${SERVE_ARGS[@]}" 2>&1 >/dev/null)" || code=$?
  if (( code == 2 )) || ! grep -q 'neither a directory' <<<"${err}"; then
    bad "tensorfold-native serve rejects the arguments: $(grep -v '^usage:' <<<"${err}" | tail -1)"
  else
    ok "tensorfold-native serve accepts: ${SERVE_ARGS[*]}"
  fi
fi

printf '\n'
if (( fails )); then printf 'preflight: %d check(s) failed\n' "${fails}"; exit 1; fi
printf 'preflight: all checks passed\n'
