#!/usr/bin/env bash
#
# Assert everything start.sh depends on, before a five-minute load is attempted.
#
# Memory is the check that matters. On a unified-memory GB10 the KV pool comes
# out of the same 121.6 GiB the OS lives on, and a pool that is truly exhausted
# hangs the kernel rather than raising an OOM. The standard tier needs
# gmu x MemTotal free (vLLM refuses otherwise); the max tiers need KV_GIB plus
# their estimated overhead plus HOST_FLOOR_GIB. Neither fits beside another
# large model server -- the other containers are listed when the check fails.
#
# Usage: ./preflight.sh [standard|max|max-fp8]
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
select_profile "${1:-${DEFAULT_PROFILE}}"

fail=0
ok()   { printf '  OK    %s\n' "$*"; }
warn() { printf '  WARN  %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; fail=1; }

printf 'preflight: %s (window %s, %s, kv dtype %s, spec tokens %s)\n\n' "${PROFILE}" "${MAX_MODEL_LEN}" \
  "$([[ -n "${KV_GIB}" ]] && printf 'KV %s GiB' "${KV_GIB}" || printf 'gmu %s' "${GPU_MEMORY_UTILIZATION}")" \
  "${KV_CACHE_DTYPE}" "${SPEC_TOKENS}"

printf 'platform\n'
if require_aarch64 2>/dev/null; then ok "aarch64"; else bad "not aarch64 (found $(uname -m))"; fi
for tool in docker curl python3 nvidia-smi; do
  if command -v "${tool}" >/dev/null 2>&1; then ok "${tool} on PATH"; else bad "${tool} is not on PATH"; fi
done
if nvidia-smi >/dev/null 2>&1; then
  ok "GPU: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"
  # fp8 KV is allowed only on sm_12x in this build.
  if [[ "${KV_CACHE_DTYPE}" == fp8 ]]; then
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
    [[ "${cc}" == 12.* ]] && ok "compute capability ${cc} allows fp8 KV" || bad "fp8 KV needs sm_12x (found ${cc})"
  fi
else
  bad "nvidia-smi failed"
fi
(( SPEC_TOKENS == 0 || SPEC_TOKENS == 7 )) && ok "SPEC_TOKENS=${SPEC_TOKENS}" || bad "SPEC_TOKENS must be 7 (or 0 to disable)"
if [[ -n "${KV_GIB}" ]]; then
  (( KV_GIB <= KV_CAP_GIB )) && ok "KV_GIB ${KV_GIB} <= ${KV_CAP_GIB}" || bad "KV_GIB ${KV_GIB} is above the measured ceiling ${KV_CAP_GIB}"
fi

printf '\nruntime image\n'
if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  ok "image present: ${IMAGE:0:60}"
  # The flags, method and parsers start.sh passes, read from the vLLM source in
  # the image rather than assumed. Located without importing vllm, which pulls
  # in vllm._C and so needs the GPU runtime.
  missing="$(docker run --rm --entrypoint sh "${IMAGE}" -c '
    d=$(python3 -c "import importlib.util as u; print(u.find_spec(\"vllm\").submodule_search_locations[0])") || { printf UNREADABLE; exit 0; }
    cd "$d" || exit 0
    for k in kv_cache_memory_bytes speculative_config draft_sample_method; do
      grep -rqs --include=*.py "$k" engine config || printf "%s " "$k"
    done
    grep -rqs --include=*.py "\"dspark\"" config || printf "dspark-method "
    ls reasoning 2>/dev/null | grep -qi inkling || grep -rqs --include=*.py "\"inkling\"" reasoning || printf "inkling-reasoning-parser "
    grep -rqsi --include=*.py "inkling" tool_parsers entrypoints/openai/tool_parsers 2>/dev/null || printf "inkling-tool-parser "
    printf END' 2>/dev/null || printf 'UNREADABLE')"
  if [[ "${missing}" == "END" ]]; then
    ok "image has kv_cache_memory_bytes, dspark and the inkling parsers"
  else
    bad "image is missing: ${missing% END}"
  fi
else
  bad "image not pulled (run: docker pull ${IMAGE})"
fi

printf '\ncheckpoint\n'
repo="$(manifest_field "${MODEL_MANIFEST}" repo)"
revision="$(manifest_field "${MODEL_MANIFEST}" revision)"
snap="$(snapshot_dir "${MODEL_MANIFEST}")"
if [[ ! -d "${snap}" ]]; then
  bad "${repo}: pinned snapshot not in the HF cache (run: ./download.sh)"
elif verify_manifest "${MODEL_MANIFEST}" >/dev/null 2>&1; then
  ok "${repo} @ ${revision:0:12}, sha256 verified"
  [[ -f "${snap}/mtp.safetensors" ]] && ok "mtp.safetensors (DSpark drafter) present" || bad "mtp.safetensors missing"
else
  bad "${repo}: verification failed (run: ./download.sh --verify-only)"
fi

printf '\nmemory\n'
avail="$(mem_available_bytes)"; need="$(memory_need_bytes)"
if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
  what="gmu ${GPU_MEMORY_UTILIZATION} x MemTotal + $(human_bytes "${MEM_HEADROOM_BYTES}") headroom"
else
  what="KV ${KV_GIB} + overhead ~${OVERHEAD_GIB} (est.) + floor ${HOST_FLOOR_GIB} GiB"
fi
if (( avail >= need )); then
  ok "MemAvailable $(human_bytes "${avail}") >= $(human_bytes "${need}") (${what})"
else
  bad "MemAvailable $(human_bytes "${avail}") < $(human_bytes "${need}") (${what})"
  printf '        other containers:\n'
  docker ps --format '        {{.Names}}  {{.Image}}' | grep -v "^        ${CONTAINER_NAME} " || true
  printf '        stop the other model servers, drop the page cache, or lower KV_GIB\n'
fi
swap_used="$(awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print (t - f) * 1024}' /proc/meminfo)"
(( swap_used < GIB )) && ok "swap in use $(human_bytes "${swap_used}")" || warn "swap in use $(human_bytes "${swap_used}")"

printf '\nport and disk\n'
if ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then bad "port ${PORT} is already in use"; else ok "port ${PORT} free"; fi
mkdir -p "${HF_HOME}" "${VLLM_CACHE_HOST}" "${FLASHINFER_CACHE_HOST}" "${TRITON_CACHE_HOST}"
free_disk="$(df --output=avail -B1 "${VLLM_CACHE_HOST}" | tail -1 | tr -d ' ')"
(( free_disk > DISK_RESERVE_BYTES )) && ok "disk free $(human_bytes "${free_disk}")" || bad "disk free $(human_bytes "${free_disk}") < reserve"

printf '\n'
if (( fail )); then printf 'preflight FAILED\n' >&2; exit 1; fi
printf 'preflight passed\n'
