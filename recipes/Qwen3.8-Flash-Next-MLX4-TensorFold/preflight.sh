#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
#
# Assert everything start.sh depends on, before a ~2.5 minute load.
#
#   ./preflight.sh [profile]      profile: int8x4 (default) | int8x5 | int4x6 | bf16x3 | int8x1 (co-tenant)
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

printf 'profile %s: %s x %s tokens, %s KV\n' "${PROFILE}" "${PARALLEL}" "${CONTEXT}" "${KV_DTYPE}"

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
  [[ "${label}" == "${PATCHES_HASH}" ]] && ok "tf.patches ${label} matches patches/" \
    || bad "tf.patches is ${label:-missing}, patches/ hashes to ${PATCHES_HASH}: run ./build.sh"
  ver="$(image_tf_version)"
  [[ "v${ver}" == "${TF_VERSION}" ]] && ok "tensorfold ${ver}" \
    || bad "tensorfold reports ${ver:-nothing}, the patches are for ${TF_VERSION}: run ./build.sh --rebuild"
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
if [[ -n "${gpu_apps}" ]] && ! container_running; then
  bad "the GPU is already in use -- one model server at a time on this pool:"
  printf '         %s\n' "${gpu_apps//$'\n'/$'\n'         }"
  [[ -n "${others}" ]] && printf '         running containers: %s\n' "$(paste -sd' ' <<<"${others}")"
else
  ok "no other GPU process"
fi

printf '\nmemory\n'
total="$(mem_total_bytes)"; avail="$(mem_available_bytes)"; reserve="$(tf_reserve_bytes)"
budget=$(( avail - reserve ))
printf '         MemTotal %s, MemAvailable %s, swap used %s\n' \
  "$(human_bytes "${total}")" "$(human_bytes "${avail}")" "$(human_bytes "$(host_swap_used_bytes)")"
printf '         TensorFold budget = MemAvailable - %s reserve = %s\n' "$(human_bytes "${reserve}")" "$(human_bytes "${budget}")"
if container_running; then
  warn "skipping the budget check: ${CONTAINER_NAME} is running and holds its memory"
elif [[ -n "${EST_GIB}" ]]; then
  need="$(gib_to_bytes "${EST_GIB}")"
  if (( budget >= need )); then
    ok "budget covers ${PROFILE}'s ${EST_GIB} GiB estimate ($(human_bytes "$(( budget - need ))") spare)"
  else
    bad "budget $(human_bytes "${budget}") is under ${PROFILE}'s ${EST_GIB} GiB estimate: free memory, or try int8x4 / int4x6 (or stop the other GPU tenant)"
  fi
else
  warn "PARALLEL/CONTEXT/KV_DTYPE overridden: no table estimate; TensorFold's admission decides (it refuses before loading)"
fi
cg="$(gib_to_bytes "${CONTAINER_MEM_GIB}")"
(( cg < total / 2 )) && ok "container host-side cap ${CONTAINER_MEM_GIB} GiB" \
  || warn "CONTAINER_MEM_GIB=${CONTAINER_MEM_GIB} is over half the pool; the cap no longer protects the host"

printf '\nserve arguments\n'
if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  # TensorFold's own parser in a throwaway container without the GPU, so a
  # typo in EXTRA_SERVE_ARGS fails here rather than after a stop.
  if msg="$(docker run --rm --entrypoint python "${IMAGE}" -c \
      'import sys; from tensorfold.cli import build_parser; build_parser().parse_args(sys.argv[1:])' \
      serve "${SNAPSHOT_CTR}" "${SERVE_ARGS[@]}" 2>&1 >/dev/null)"; then
    ok "tensorfold serve accepts: ${SERVE_ARGS[*]}"
  else
    bad "tensorfold serve rejects the arguments: $(tail -1 <<<"${msg}")"
  fi
fi

printf '\n'
if (( fails )); then printf 'preflight: %d check(s) failed\n' "${fails}"; exit 1; fi
printf 'preflight: all checks passed\n'
