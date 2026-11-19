#!/usr/bin/env bash
#
# Serve Qwen/Qwen3.8-27B (BF16) with the harness-hub pi LoRA on vLLM.
#
# The `pi` profile is the adapter's model-card command, flag for flag: base +
# LoRA (rank 32), language model only, prefix caching with
# --mamba-cache-mode align, qwen3_xml tool calls, thinking off by default.
# Differences, all deliberate: the window is the base's native 262,144, not the
# card's 131,072 (MAX_MODEL_LEN=131072 restores it); both checkpoints are the
# pinned revisions, loaded offline from the shared HF cache (HF_HUB_OFFLINE=1,
# so a missing snapshot fails fast instead of downloading); the adapter is
# passed by its snapshot path, so the pinned revision is what is served; the
# card's adapter id `harness-hub/Qwen3.8-27B-pi` does not exist and is replaced
# by `-pi-adapted`; and the image is pinned by digest.
#
# Usage: ./start.sh [pi|pi-mtp|nvfp4|base]
#        DRY_RUN=1 ./start.sh      # print the docker command and exit
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
select_profile "${1:-${DEFAULT_PROFILE}}"

mkdir -p "${OUT_DIR}" "${HF_HOME}" "${VLLM_CACHE_HOST}" "${FLASHINFER_CACHE_HOST}" "${TRITON_CACHE_HOST}"
resolve_hf_token

MODEL_REVISION="$(manifest_field "${MODEL_MANIFEST}" revision)"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  while IFS= read -r manifest; do
    [[ -d "$(snapshot_dir "${manifest}")" ]] || {
      printf 'error: pinned snapshot of %s missing. Run ./download.sh first.\n' \
        "$(manifest_field "${manifest}" repo)" >&2; exit 1; }
  done < <(profile_manifests)
fi

# The adapter by snapshot path. It sits under HF_HOME by construction, so the
# HF_HOME bind mount reaches it; derive the container path, never retype it.
ADAPTER_SNAPSHOT="$(snapshot_dir "${ADAPTER_MANIFEST}")"
[[ "${ADAPTER_SNAPSHOT}" == "${HF_HOME}"/* ]] || {
  printf 'error: adapter snapshot is outside HF_HOME, so the bind mount cannot reach it\n' >&2; exit 1; }
ADAPTER_IN_CONTAINER="$(snapshot_dir "${ADAPTER_MANIFEST}" "${CONTAINER_HF_HOME}")"

# Mount each host tree onto the tool's own in-container default, so no cache
# environment variable is needed inside.
DOCKER_ARGS=(
  --name "${CONTAINER_NAME}" --detach
  --restart "${RESTART_POLICY}"
  --gpus all --ipc host
  -p "${HOST}:${PORT}:8000"
  -e HF_HUB_OFFLINE=1
  -e HF_TOKEN="${HF_TOKEN}"
  -v "${HF_HOME}:${CONTAINER_HF_HOME}"
  -v "${VLLM_CACHE_HOST}:/root/.cache/vllm"
  -v "${FLASHINFER_CACHE_HOST}:/root/.cache/flashinfer"
  -v "${TRITON_CACHE_HOST}:/root/.triton"
)

VLLM_ARGS=(
  --revision "${MODEL_REVISION}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host 0.0.0.0 --port 8000
  --language-model-only
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --enable-prefix-caching
  --mamba-cache-mode align
  --enable-auto-tool-choice
  --tool-call-parser qwen3_xml
  --reasoning-parser qwen3
  --default-chat-template-kwargs '{"enable_thinking": false}'
)
if (( ENABLE_LORA )); then
  VLLM_ARGS+=(
    --enable-lora
    --max-lora-rank "${ADAPTER_RANK}"
    --max-loras 1
    --lora-modules "${LORA_NAME}=${ADAPTER_IN_CONTAINER}"
  )
fi
if (( SPEC_TOKENS > 0 )); then
  # "mtp": vLLM 0.26 resolves it to Qwen3_5MTP and logs the older
  # "qwen3_5_mtp" name as deprecated.
  VLLM_ARGS+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${SPEC_TOKENS}}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'docker run'
  for a in "${DOCKER_ARGS[@]}" "${IMAGE}" "${MODEL_ID}" "${VLLM_ARGS[@]}"; do
    case "${a}" in HF_TOKEN=*) printf ' %q' "HF_TOKEN=<redacted>" ;; *) printf ' %q' "${a}" ;; esac
  done
  printf '\n'; exit 0
fi

docker image inspect "${IMAGE}" >/dev/null 2>&1 || {
  printf 'error: image not pulled. Run: docker pull %s\n' "${IMAGE}" >&2; exit 1; }
if container_exists; then docker rm -f "${CONTAINER_NAME}" >/dev/null; fi

log_event "launching ${PROFILE} lora=${ENABLE_LORA} spec=${SPEC_TOKENS} gmu=${GPU_MEMORY_UTILIZATION} len=${MAX_MODEL_LEN}" >>"${LOG_FILE}"
docker run "${DOCKER_ARGS[@]}" "${IMAGE}" "${MODEL_ID}" "${VLLM_ARGS[@]}" >/dev/null
printf '%s' "${PROFILE}" >"${ACTIVE_PROFILE_FILE}"
printf 'started %s\n' "${CONTAINER_NAME}"

printf 'waiting for http://127.0.0.1:%s/health (a cold load reads %s; first start also compiles)\n' \
  "${PORT}" "$(human_bytes "$(manifest_field "${MODEL_MANIFEST}" total_bytes)")"
while ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
  if ! container_running; then
    printf '\nvLLM exited before becoming ready:\n' >&2
    docker logs --tail 60 "${CONTAINER_NAME}" >&2 || true
    (( ENABLE_LORA )) && printf '\nTo rule out the adapter: ./start.sh base\n' >&2
    exit 1
  fi
  printf '  still loading (MemAvailable %s)\n' "$(human_bytes "$(mem_available_bytes)")"
  sleep 15
done
printf '\nready.  http://%s:%s/v1   models: %s%s\n' "127.0.0.1" "${PORT}" "${SERVED_MODEL_NAME}" \
  "$( (( ENABLE_LORA )) && printf ', %s (LoRA)' "${LORA_NAME}")"
printf 'smoke test: ./scripts/smoke.py    stop: ./stop.sh\n'
