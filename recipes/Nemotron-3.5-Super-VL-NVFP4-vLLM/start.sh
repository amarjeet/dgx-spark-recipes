#!/usr/bin/env bash
#
# Serve NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 with vLLM on the DGX Spark.
#
# The `spark` profile is the model card's own DGX Spark command, flag for flag:
# one stream, text + image, dflash speculative decoding with the LiLiCorr draft.
# Differences, all deliberate: both checkpoints are the pinned revisions, loaded
# offline from the shared HF cache (HF_HUB_OFFLINE=1, so a missing snapshot
# fails fast instead of downloading), and the image is pinned by digest.
#
# Usage: ./start.sh [spark|nodraft]
#        DRY_RUN=1 ./start.sh      # print the docker command and exit
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
select_profile "${1:-${DEFAULT_PROFILE}}"

mkdir -p "${OUT_DIR}" "${HF_HOME}" "${VLLM_CACHE_HOST}" "${FLASHINFER_CACHE_HOST}" "${TRITON_CACHE_HOST}"
resolve_hf_token

MODEL_REVISION="$(manifest_field "${MODEL_MANIFEST}" revision)"
[[ -d "$(snapshot_dir "${MODEL_MANIFEST}")" ]] || {
  printf 'error: pinned snapshot missing. Run ./download.sh first.\n' >&2; exit 1; }
if (( SPEC_TOKENS > 0 )); then
  [[ -d "$(snapshot_dir "${DRAFT_MANIFEST}")" ]] || {
    printf 'error: draft model snapshot missing. Run ./download.sh first.\n' >&2; exit 1; }
fi

# Mount each host tree onto the tool's own in-container default, so no cache
# environment variable is needed inside.
DOCKER_ARGS=(
  --name "${CONTAINER_NAME}" --detach
  --restart "${RESTART_POLICY}"
  --gpus all --ipc host
  -p "${PORT}:8000"
  -e HF_HUB_OFFLINE=1
  -e HF_TOKEN="${HF_TOKEN}"
  -v "${HF_HOME}:/root/.cache/huggingface"
  -v "${VLLM_CACHE_HOST}:/root/.cache/vllm"
  -v "${FLASHINFER_CACHE_HOST}:/root/.cache/flashinfer"
  -v "${TRITON_CACHE_HOST}:/root/.triton"
)

VLLM_ARGS=(
  --revision "${MODEL_REVISION}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host 0.0.0.0 --port 8000
  --max-num-seqs "${MAX_NUM_SEQS}"
  --trust-remote-code
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
)
[[ -n "${MAX_MODEL_LEN}" ]] && VLLM_ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
if (( SPEC_TOKENS > 0 )); then
  VLLM_ARGS+=(
    --speculative_config.model "${DRAFT_MODEL_ID}"
    --speculative_config.num_speculative_tokens "${SPEC_TOKENS}"
    --speculative_config.method dflash
  )
fi
VLLM_ARGS+=(
  --mamba-backend flashinfer
  --mamba-ssm-cache-dtype float16
  --enable-mamba-cache-stochastic-rounding
  --mamba-cache-philox-rounds 5
  --enable-mamba-shared-prefix-checkpoint
  --video-pruning-rate 0.5
  --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}'
  --block-size 64
  --prefix-match-unit 64
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser nemotron_v3
)

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

log_event "launching ${PROFILE} spec=${SPEC_TOKENS} gmu=${GPU_MEMORY_UTILIZATION}" >>"${LOG_FILE}"
docker run "${DOCKER_ARGS[@]}" "${IMAGE}" "${MODEL_ID}" "${VLLM_ARGS[@]}" >/dev/null
printf '%s' "${PROFILE}" >"${ACTIVE_PROFILE_FILE}"
printf 'started %s\n' "${CONTAINER_NAME}"

printf 'waiting for http://127.0.0.1:%s/health (a cold load reads %s; first start also compiles)\n' \
  "${PORT}" "$(human_bytes "$(manifest_field "${MODEL_MANIFEST}" total_bytes)")"
while ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
  if ! container_running; then
    printf '\nvLLM exited before becoming ready:\n' >&2
    docker logs --tail 60 "${CONTAINER_NAME}" >&2 || true
    (( SPEC_TOKENS > 0 )) && printf '\nTo rule out the draft model: ./start.sh nodraft\n' >&2
    exit 1
  fi
  printf '  still loading (MemAvailable %s)\n' "$(human_bytes "$(mem_available_bytes)")"
  sleep 15
done
printf '\nready.  http://%s:%s/v1   model: %s\n' "127.0.0.1" "${PORT}" "${SERVED_MODEL_NAME}"
printf 'smoke test: ./scripts/smoke.py    stop: ./stop.sh\n'
