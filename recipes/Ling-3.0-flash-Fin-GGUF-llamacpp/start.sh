#!/usr/bin/env bash
#
# Serve Ling-3.0-flash-Fin (bailingmoe3, 124B total / 5.1B active) on the
# DGX Spark's GB10 via llama.cpp's published server-cuda image.
#
# Why GGUF and not the usual SGLang/vLLM pattern on this hardware:
# inclusionAI released this checkpoint only in BF16 (~248 GB) and nobody has
# published an FP8/AWQ/NVFP4 repack, so there is nothing for those runtimes to
# load inside the Spark's 121.7 GiB of unified memory. bartowski's imatrix
# GGUFs fit.
#
# Why the stock image and not a source build: bailingmoe3 support landed
# upstream in b10776, and the currently published aarch64 :server-cuda image is
# newer than that. preflight.sh asserts the build number rather than trusting
# the mutable tag.
#
# Storage: the host GGUF store is bind-mounted onto llama.cpp's own
# in-container default (/root/.cache/llama.cpp), so no cache env var has to be
# set inside the container and the weights are shared with every other
# llama.cpp recipe on the host.
#
# Usage:
#   ./start.sh                # default profile (iq4xs)
#   ./start.sh q3kxl          # 256K-context profile
#   CTX_SIZE=262144 ./start.sh iq4xs
#   SPEC_TYPE=none ./start.sh # disable MTP speculative decoding
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
select_profile "${1:-${DEFAULT_PROFILE}}"

# What the host can actually reach depends on the publish address: 0.0.0.0
# covers loopback, but HOST=192.168.x.y or a tailnet address does not, and a
# readiness probe hardcoded to 127.0.0.1 would then wait forever on a server
# that is already up.
case "${HOST}" in
  0.0.0.0|::|"") PROBE_HOST=127.0.0.1 ;;
  *)             PROBE_HOST="${HOST}" ;;
esac
READY_URL="http://${PROBE_HOST}:${PORT}/health"

# This checkpoint ships a multi-token-prediction layer (config.json:
# num_nextn_predict_layers=1) and bartowski's GGUFs keep it, so llama.cpp can
# self-speculate with no draft model. Worth real decode throughput on a 5.1B-
# active MoE. If the server refuses to start with it, re-run with
# SPEC_TYPE=none -- the failure path below prints that command.
SPEC_TYPE="${SPEC_TYPE:-draft-mtp}"

command -v docker >/dev/null || { printf 'docker is not on PATH\n' >&2; exit 1; }
command -v curl   >/dev/null || { printf 'curl is not on PATH\n' >&2; exit 1; }

# MODEL_FILE_CONTAINER is derived by swapping the LLAMA_CACHE prefix for the
# bind-mount target, so a MODEL_STORE pointed outside LLAMA_CACHE would produce
# a container path that does not exist. Catch that here rather than four
# minutes into a load.
[[ "${MODEL_ROOT}" == "${LLAMA_CACHE}"/* ]] || {
  printf 'error: MODEL_ROOT is outside LLAMA_CACHE, so the bind mount cannot reach it\n' >&2
  printf '  MODEL_ROOT  : %s\n' "${MODEL_ROOT}" >&2
  printf '  LLAMA_CACHE : %s\n' "${LLAMA_CACHE}" >&2
  printf 'set LLAMA_CACHE to the parent of your store instead of overriding MODEL_STORE alone.\n' >&2
  exit 1
}

[[ -f "${MODEL_FILE}" ]] || {
  printf 'error: weights are missing: %s\n' "${MODEL_FILE}" >&2
  printf 'run: ./download.sh %s\n' "${PROFILE}" >&2
  exit 1
}

mkdir -p "${LLAMA_CACHE}" "${OUT_DIR}"

if container_exists; then
  if container_running; then
    printf 'container %s is already running (profile: %s)\n' \
      "${CONTAINER_NAME}" "$(cat "${ACTIVE_PROFILE_FILE}" 2>/dev/null || printf 'unknown')"
    printf 'log: %s\n' "${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

# --mlock is deprecated in current llama.cpp; mmap+mlock is the replacement
# that keeps 60+ GiB of weights off swap.
load_mode="mmap"
[[ "${MLOCK}" == "1" ]] && load_mode="mmap+mlock"

printf 'model     : inclusionAI/Ling-3.0-flash-Fin (bailingmoe3, 124B total / 5.1B active)\n'
printf 'weights   : %s %s @ %s\n' "${QUANT}" "$(human_bytes "${MODEL_TOTAL_BYTES}")" "${MODEL_REVISION_SHORT}"
printf 'image     : %s\n' "${IMAGE}"
printf 'context   : %s over %s slot(s)\n' "${CTX_SIZE}" "${PARALLEL}"
printf 'spec      : %s\n' "${SPEC_TYPE}"
printf 'load mode : %s\n' "${load_mode}"
printf 'listening : %s:%s\n' "${HOST}" "${PORT}"
printf 'log       : %s\n\n' "${LOG_FILE}"

log_event "launching ${PROFILE} (${QUANT}) ctx=${CTX_SIZE} parallel=${PARALLEL} spec=${SPEC_TYPE}" \
  >"${LOG_FILE}"

spec_args=()
[[ "${SPEC_TYPE}" != "none" ]] && spec_args=(--spec-type "${SPEC_TYPE}")

# The image's built-in HEALTHCHECK hardcodes port 8080, so a server on any
# other port shows as "unhealthy" in docker ps while serving perfectly well.
# Override it with the port we actually use, and give the start period enough
# room for a cold 64 GiB load (~4 min) so it is not marked unhealthy mid-load.
#
# On HOST: it is the *publish* address, and that is the only thing restricting
# reach. The server still binds 0.0.0.0 inside the container -- it has to, or
# the published port would have nothing to forward to -- so HOST=127.0.0.1 is
# what actually keeps this off the LAN. A bare -p "${PORT}:${PORT}" publishes
# on every interface no matter what HOST says, which is what this used to do.
DOCKER_ARGS=(
  -d
  --name "${CONTAINER_NAME}"
  --restart "${RESTART_POLICY}"
  --gpus all
  --ipc host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -p "${HOST}:${PORT}:${PORT}"
  --health-cmd "curl -fsS http://localhost:${PORT}/health || exit 1"
  --health-interval 30s
  --health-start-period 30m
  --health-retries 3
  -e LLAMA_CACHE=/root/.cache/llama.cpp
  -v "${LLAMA_CACHE}:/root/.cache/llama.cpp"
)

SERVER_ARGS=(
  --model "${MODEL_FILE_CONTAINER}"
  --alias "${SERVED_MODEL_NAME}"
  --host 0.0.0.0
  --port "${PORT}"
  --ctx-size "${CTX_SIZE}"
  --parallel "${PARALLEL}"
  --n-gpu-layers 999
  --flash-attn on
  --batch-size "${BATCH_SIZE}"
  --ubatch-size "${UBATCH_SIZE}"
  --load-mode "${load_mode}"
  --jinja
  --reasoning-format deepseek
  --temp "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  ${spec_args+"${spec_args[@]}"}
)

docker run "${DOCKER_ARGS[@]}" "${IMAGE}" "${SERVER_ARGS[@]}" >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
printf '%s' "${container_id}" >"${PID_FILE}"
printf '%s' "${PROFILE}" >"${ACTIVE_PROFILE_FILE}"
printf 'spawned %s (%s)\n' "${CONTAINER_NAME}" "${container_id:0:12}"

log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]] && kill -0 "${log_follow_pid}" 2>/dev/null; then
    kill "${log_follow_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

(docker logs -f "${CONTAINER_NAME}" >>"${LOG_FILE}" 2>&1) &
log_follow_pid=$!

printf 'waiting for %s\n' "${READY_URL}"
printf '(a cold %s load reads %s off NVMe; expect minutes)\n' "${QUANT}" "$(human_bytes "${MODEL_TOTAL_BYTES}")"
while ! curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! container_running; then
    printf '\nllama.cpp exited before becoming ready\n' >&2
    tail -n 60 "${LOG_FILE}" >&2 || true
    # The two failures worth naming, because their log lines are far from
    # obvious: an unsupported spec type, and an oversized context.
    if [[ "${SPEC_TYPE}" != "none" ]] && grep -qiE 'spec|draft|mtp' "${LOG_FILE}"; then
      printf '\nMTP speculative decoding may be the cause. Retry with:\n' >&2
      printf '  SPEC_TYPE=none ./start.sh %s\n' "${PROFILE}" >&2
    fi
    if grep -qiE 'failed to allocate|out of memory|cannot allocate' "${LOG_FILE}"; then
      printf '\nAllocation failed. Retry with a smaller context, e.g.:\n' >&2
      printf '  CTX_SIZE=%s ./start.sh %s\n' "$(( CTX_SIZE / 2 ))" "${PROFILE}" >&2
    fi
    exit 1
  fi
  printf '  still loading (avail %s, swap used %s)\n' \
    "$(human_bytes "$(mem_available_bytes)")" "$(human_bytes "$(host_swap_used_bytes)")"
  sleep 10
done

printf '\nready.\n'
# What the run actually cost, so CTX_SIZE can be tuned against a measurement
# rather than the profile table's guess. llama.cpp does not print a "KV self
# size" line at this verbosity, so resident memory minus the weights is the
# honest read on what KV plus compute buffers came to.
used_now="$(( $(mem_total_bytes) - $(mem_available_bytes) ))"
printf '  weights        %s\n' "$(human_bytes "${MODEL_TOTAL_BYTES}")"
printf '  host in use    %s of %s\n' \
  "$(human_bytes "${used_now}")" "$(human_bytes "$(mem_total_bytes)")"
printf '  overhead       ~%s (KV at n_ctx=%s, compute buffers, MTP draft ctx)\n' \
  "$(human_bytes "$(( used_now > MODEL_TOTAL_BYTES ? used_now - MODEL_TOTAL_BYTES : 0 ))")" \
  "${CTX_SIZE}"
grep -iE 'MTP draft|n_ctx_slot|model loaded' "${LOG_FILE}" \
  | sed 's/^[0-9.]* [A-Z] /  /' | tail -4 || true
printf '\nOpenAI base URL : http://%s:%s/v1\n' "${PROBE_HOST}" "${PORT}"
printf 'model id        : %s\n' "${SERVED_MODEL_NAME}"
printf 'smoke test      : ./scripts/smoke.py\n'
printf 'stop            : ./stop.sh\n'
