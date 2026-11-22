#!/usr/bin/env bash
#
# Serve Inkling 2 Mini (NVFP4/MXFP8) with vLLM and DSpark on the DGX Spark.
#
# Port of tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark at 300942d: its three
# launchers are the three profiles here, flag for flag. Differences, all
# deliberate: the checkpoint is the pinned revision, loaded offline from the
# shared HF cache (HF_HUB_OFFLINE=1, so a missing snapshot fails fast instead
# of downloading); the image is pinned by digest; the vLLM/flashinfer/triton
# caches are the shared defaults, so a restart does not re-JIT; and upstream's
# watch_start.sh is folded into the wait loop below.
#
# Usage: ./start.sh [standard|max|max-fp8]
#        KV_GIB=76 ./start.sh max          # step the BF16 pool down 4 GiB
#        DRY_RUN=1 ./start.sh max-fp8      # print the docker command and exit
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
select_profile "${1:-${DEFAULT_PROFILE}}"

mkdir -p "${OUT_DIR}" "${HF_HOME}" "${VLLM_CACHE_HOST}" "${FLASHINFER_CACHE_HOST}" "${TRITON_CACHE_HOST}"
resolve_hf_token

if [[ -n "${KV_GIB}" ]] && (( KV_GIB > KV_CAP_GIB )); then
  printf 'error: KV_GIB=%s is above %s GiB, the measured ceiling for %s. On GB10 an\n' "${KV_GIB}" "${KV_CAP_GIB}" "${PROFILE}" >&2
  printf '       out-of-memory can hang the box until a power cycle.\n' >&2
  exit 1
fi

MODEL_REVISION="$(manifest_field "${MODEL_MANIFEST}" revision)"

# Mount each host tree onto the tool's own in-container default, so no cache
# environment variable is needed inside.
DOCKER_ARGS=(
  --name "${CONTAINER_NAME}" --detach
  --restart "${RESTART_POLICY}"
  --gpus all --ipc host
  -p "${HOST}:${PORT}:8000"
  -e HF_HUB_OFFLINE=1
  -e HF_TOKEN="${HF_TOKEN}"
  -v "${HF_HOME}:/root/.cache/huggingface"
  -v "${VLLM_CACHE_HOST}:/root/.cache/vllm"
  -v "${FLASHINFER_CACHE_HOST}:/root/.cache/flashinfer"
  -v "${TRITON_CACHE_HOST}:/root/.triton"
)

VLLM_ARGS=(
  --revision "${MODEL_REVISION}"
  --served-model-name "${SERVED_MODEL_NAME}" "${MODEL_ID}"
  --host 0.0.0.0 --port 8000
  --tensor-parallel-size 1
  --max-model-len "${MAX_MODEL_LEN}"
)
if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
  VLLM_ARGS+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
else
  # An exact pool; ignores gpu-memory-utilization. (The log suggests
  # --kv-cache-memory=..., but the flag is --kv-cache-memory-bytes.)
  VLLM_ARGS+=(--kv-cache-memory-bytes "$(( KV_GIB * GIB ))")
fi
[[ "${KV_CACHE_DTYPE}" != auto ]] && VLLM_ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
VLLM_ARGS+=(
  --reasoning-parser inkling
  --enable-auto-tool-choice --tool-call-parser inkling
)
if (( SPEC_TOKENS > 0 )); then
  VLLM_ARGS+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_TOKENS},\"draft_sample_method\":\"probabilistic\"}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'docker run'
  for a in "${DOCKER_ARGS[@]}" "${IMAGE}" "${MODEL_ID}" "${VLLM_ARGS[@]}"; do
    case "${a}" in HF_TOKEN=*) printf ' %q' "HF_TOKEN=<redacted>" ;; *) printf ' %q' "${a}" ;; esac
  done
  printf '\n'; exit 0
fi

[[ -d "$(snapshot_dir "${MODEL_MANIFEST}")" ]] || {
  printf 'error: pinned snapshot missing. Run ./download.sh first.\n' >&2; exit 1; }
docker image inspect "${IMAGE}" >/dev/null 2>&1 || {
  printf 'error: image not pulled. Run: docker pull %s\n' "${IMAGE}" >&2; exit 1; }
if container_exists; then docker rm -f "${CONTAINER_NAME}" >/dev/null; fi

# GB10 page-cache trap: clean page cache counts against unified memory until
# the kernel reclaims it. Drop it once (clean pages only). Skipped without
# passwordless sudo, as upstream does.
sync
if sudo -n true 2>/dev/null; then echo 1 | sudo tee /proc/sys/vm/drop_caches >/dev/null; fi

need="$(memory_need_bytes)"; avail="$(mem_available_bytes)"
if (( avail < need )); then
  printf 'error: MemAvailable %s < %s needed for %s. Run ./preflight.sh %s for the breakdown.\n' \
    "$(human_bytes "${avail}")" "$(human_bytes "${need}")" "${PROFILE}" "${PROFILE}" >&2
  exit 1
fi

log_event "launching ${PROFILE} len=${MAX_MODEL_LEN} gmu=${GPU_MEMORY_UTILIZATION:-} kv=${KV_GIB:-auto}GiB dtype=${KV_CACHE_DTYPE} spec=${SPEC_TOKENS}" >>"${LOG_FILE}"
docker run "${DOCKER_ARGS[@]}" "${IMAGE}" "${MODEL_ID}" "${VLLM_ARGS[@]}" >/dev/null
printf '%s' "${PROFILE}" >"${ACTIVE_PROFILE_FILE}"
printf 'started %s (%s)\n' "${CONTAINER_NAME}" "${PROFILE}"

# Wait for /health, watching memory (upstream's watch_start.sh): kill the
# container if MemAvailable falls under KILL_FLOOR_GIB, so the box never hangs.
printf 'waiting for http://127.0.0.1:%s/health (about 5 to 6 min: load, JIT warmup, CUDA graphs)\n' "${PORT}"
floor=$(( KILL_FLOOR_GIB * GIB )); low="$(mem_available_bytes)"; t0=$SECONDS
while ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
  avail="$(mem_available_bytes)"; (( avail < low )) && low="${avail}"
  if (( avail < floor )); then
    docker logs --tail 3000 "${CONTAINER_NAME}" >"${OUT_DIR}/last-container.log" 2>&1 || true
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    log_event "KILLED ${PROFILE}: MemAvailable $(human_bytes "${avail}") under ${KILL_FLOOR_GIB} GiB" >>"${LOG_FILE}"
    printf '\nKILLED: MemAvailable %s under the %s GiB floor. Lower KV_GIB by 4 and retry.\n' \
      "$(human_bytes "${avail}")" "${KILL_FLOOR_GIB}" >&2
    rm -f "${ACTIVE_PROFILE_FILE}"; exit 2
  fi
  if ! container_running; then
    printf '\nvLLM exited before becoming ready:\n' >&2
    docker logs --tail 60 "${CONTAINER_NAME}" >&2 || true
    rm -f "${ACTIVE_PROFILE_FILE}"; exit 1
  fi
  printf '  %4ds  MemAvailable %s (low %s)\n' $(( SECONDS - t0 )) "$(human_bytes "${avail}")" "$(human_bytes "${low}")"
  sleep 5
done
log_event "ready ${PROFILE} after $(( SECONDS - t0 ))s, MemAvailable low $(human_bytes "${low}"), now $(human_bytes "$(mem_available_bytes)")" >>"${LOG_FILE}"
printf '\nready after %ss.  http://127.0.0.1:%s/v1   model: %s\n' $(( SECONDS - t0 )) "${PORT}" "${SERVED_MODEL_NAME}"
printf 'MemAvailable now %s, lowest during start %s\n' "$(human_bytes "$(mem_available_bytes)")" "$(human_bytes "${low}")"
docker logs "${CONTAINER_NAME}" 2>&1 | grep -E 'GPU KV cache size|Maximum concurrency' | tail -2 | sed 's/^/  /' || true
printf 'checks: ./scripts/sanity.py chat    stop: ./stop.sh\n'
