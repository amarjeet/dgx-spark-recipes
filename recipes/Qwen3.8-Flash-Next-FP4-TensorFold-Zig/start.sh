#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 MiaAI-Lab (original)
# Copyright (c) 2026 amarjeet (port)
#
# Derived from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
# (start.sh), MIT. Setup moved to build.sh / download.sh, checks to
# preflight.sh; adds a pinned snapshot, a host-side cgroup cap, a memory
# watchdog and bridge networking.
#
# Serve Qwen3.8-Flash-Next (NVIDIA FP4) with TensorFold's Zig engine on one
# DGX Spark.
#
#   ./start.sh [restart] [profile] [--no-launch]
#
#   ./start.sh                  fp8x5: 5 streams x 262,144 tokens, fp8 KV, image/video input
#   ./start.sh fp8x3            3 streams
#   ./start.sh bf16x3           3 streams with the exact bf16 cache
#   ./start.sh restart          replace the running server (checks pass first)
#   ./start.sh --no-launch      print the docker command, start nothing
#   PARALLEL=8 CONTEXT=131072 ./start.sh
#   CONTEXT=1048576 ./start.sh  past 262,144 the engine applies YaRN x4
#   EXTRA_SERVE_ARGS='--no-drafts' ./start.sh restart
#
# If the server is already running, ./start.sh says so and leaves it alone.
# See profiles.sh for every knob.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

MODE=start
DO_LAUNCH=1
PROFILE_ARG=""
for arg in "$@"; do
  case "${arg}" in
    restart) MODE=restart ;;
    --no-launch) DO_LAUNCH=0 ;;
    -h|--help) sed -n '12,26p' "$0" | sed 's/^# \?//'; exit 0 ;;
    -*) printf 'unknown option: %s (try --help)\n' "${arg}" >&2; exit 2 ;;
    *) PROFILE_ARG="${arg}" ;;
  esac
done
select_profile "${PROFILE_ARG:-${DEFAULT_PROFILE}}"
serve_args
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
PROBE="http://$(probe_host):${PORT}"

info() { printf '[INFO]  %s\n' "$*"; }
okm()  { printf '[ OK ]  %s\n' "$*"; }
warnm() { printf '[WARN]  %s\n' "$*"; }
die()  { printf '[ERR ]  %s\n' "$*" >&2; exit 1; }

if [[ "${MODE}" == start ]] && container_running; then
  info "${CONTAINER_NAME} is already running on port ${PORT}: nothing to do."
  info "./start.sh restart replaces it; ./stop.sh stops it."
  exit 0
fi

mkdir -p "${OUT_DIR}/logs" "${CUDA_CACHE_HOST}" "${TRITON_CACHE_HOST}" "${TENSORFOLD_CACHE_HOST}"

DOCKER_ARGS=(
  -d --name "${CONTAINER_NAME}"
  --restart "${RESTART_POLICY}"
  --gpus all --ipc host
  -p "${HOST}:${PORT}:${PORT}"
  --ulimit memlock=-1 --ulimit stack=67108864
  # Host-side cap. GPU allocations are not charged to it on GB10.
  --memory "${CONTAINER_MEM_GIB}g" --memory-swap "${CONTAINER_MEM_GIB}g"
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  -e "TENSORFOLD_CUDA_KERNELS=${KERNELS_PATH}"
  # Read-only: serving reads the snapshot and writes nothing into the HF cache.
  -v "${HF_HOME}:/root/.cache/huggingface:ro"
  -v "${CUDA_CACHE_HOST}:/root/.nv"
  -v "${TRITON_CACHE_HOST}:/root/.triton"
  -v "${TENSORFOLD_CACHE_HOST}:/root/.cache/tensorfold"
)
# Every TENSORFOLD_* and TF_FLASHNEXT_* variable reaches the engine, as upstream
# does: profiles.sh sets this recipe's, the environment adds or overrides.
# The API key goes by name only, so it never shows in a process listing.
while IFS='=' read -r name _; do
  [[ "${name}" == TENSORFOLD_API_KEY || "${name}" == TENSORFOLD_CUDA_KERNELS ]] && continue
  DOCKER_ARGS+=(-e "${name}=${!name}")
done < <(env | grep -E '^(TENSORFOLD|TF_FLASHNEXT)_[A-Z0-9_]+=' || true)
[[ -z "${TENSORFOLD_API_KEY:-}" ]] || { export TENSORFOLD_API_KEY; DOCKER_ARGS+=(-e TENSORFOLD_API_KEY); }

CMD=(tensorfold-native serve "${SNAPSHOT_CTR}" "${SERVE_ARGS[@]}")

info "Profile ${PROFILE}: ${PARALLEL} x ${CONTEXT} tokens, ${KV_DTYPE} KV, MTP drafts $([[ "${DRAFTS}" == 1 ]] && printf '%s' "${TF_FLASHNEXT_DEPTH}" || printf off)$([[ "${VISION}" == 1 ]] && printf ', image/video input')$([[ "${TF_FLASHNEXT_YARN}" != 0 ]] && printf ', YaRN x%s' "${TF_FLASHNEXT_YARN}")"
info "Reserve ${TENSORFOLD_MEMORY_RESERVE_GIB} GiB (TENSORFOLD_MEMORY_RESERVE_GIB); a start needs ~$(mem_need_gib) GiB of MemAvailable"
info "Model   ${MODEL_ID} @ ${MODEL_REVISION:0:12}"
info "Image   ${IMAGE}"
info "Listen  ${HOST}:${PORT}   model id ${SERVED_MODEL_NAME}${SERVED_ALIAS:+ (alias ${SERVED_ALIAS})}"

if (( DO_LAUNCH == 0 )); then
  printf 'docker run'; printf ' %q' "${DOCKER_ARGS[@]}" "${IMAGE}" "${CMD[@]}"; printf '\n'
  exit 0
fi

# --- checks, before anything is stopped -------------------------------------
if [[ "${PREFLIGHT:-1}" == 1 ]]; then
  "${EXPERIMENT_DIR}/preflight.sh" "${PROFILE}" || die "preflight failed; nothing was changed"
fi

archive_ts="$(date '+%Y%m%dT%H%M%S')"
if container_exists; then
  docker logs --tail 3000 "${CONTAINER_NAME}" \
    >"${OUT_DIR}/logs/${CONTAINER_NAME}-${archive_ts}-container.log" 2>&1 || true
  if container_running; then
    info "Stopping the running ${CONTAINER_NAME} (restart)"
    "${EXPERIMENT_DIR}/stop.sh" >/dev/null
    # Its memory is free again only now, so the budget check runs again.
    [[ "${PREFLIGHT:-1}" == 1 ]] && { "${EXPERIMENT_DIR}/preflight.sh" "${PROFILE}" >/dev/null \
      || die "preflight failed after stopping the old server; run ./preflight.sh ${PROFILE}"; }
  fi
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

# --- launch + watchdog -------------------------------------------------------
LOG_FILE="${OUT_DIR}/logs/${CONTAINER_NAME}.log"
[[ -s "${LOG_FILE}" ]] && mv "${LOG_FILE}" "${OUT_DIR}/logs/${CONTAINER_NAME}-${archive_ts}-previous.log"
log_event "launch ${PROFILE} parallel=${PARALLEL} context=${CONTEXT} kv=${KV_DTYPE} image=${IMAGE}" >"${LOG_FILE}"
docker run "${DOCKER_ARGS[@]}" "${IMAGE}" "${CMD[@]}" >/dev/null
printf '%s\n' "${PROFILE}" >"${OUT_DIR}/profile.active"
okm "Container ${CONTAINER_NAME} started"

pkill -f "memwatch.sh ${CONTAINER_NAME}" 2>/dev/null || true
MEMWATCH_LOG="${OUT_DIR}/logs/memwatch-${CONTAINER_NAME}.log"
[[ -s "${MEMWATCH_LOG}" ]] && mv "${MEMWATCH_LOG}" "${OUT_DIR}/logs/${CONTAINER_NAME}-${archive_ts}-memwatch.log"
MEMWATCH_MIN_GIB="${MEMWATCH_MIN_GIB}" MEMWATCH_MIN_FREE_GIB="${MEMWATCH_MIN_FREE_GIB}" \
MEMWATCH_FREE_GATE_GIB="${MEMWATCH_FREE_GATE_GIB}" MEMWATCH_SAMPLES="${MEMWATCH_SAMPLES}" \
MEMWATCH_INTERVAL="${MEMWATCH_INTERVAL}" STOP_TIMEOUT="${STOP_TIMEOUT}" \
MEMWATCH_ARCHIVE_DIR="${OUT_DIR}/logs" \
  nohup bash "${EXPERIMENT_DIR}/scripts/memwatch.sh" "${CONTAINER_NAME}" >"${MEMWATCH_LOG}" 2>&1 &
okm "Watchdog: MemAvailable < ${MEMWATCH_MIN_GIB} GiB or MemFree < ${MEMWATCH_MIN_FREE_GIB} GiB (under ${MEMWATCH_FREE_GATE_GIB}) stops it"
info "  ${MEMWATCH_LOG}"

follow_pid=""
trap '[[ -n "${follow_pid}" ]] && kill "${follow_pid}" 2>/dev/null || true' EXIT
(docker logs -f "${CONTAINER_NAME}" >>"${LOG_FILE}" 2>&1) &
follow_pid=$!

info "Loading ~${WEIGHTS_GIB} GiB of weights, the FP8 n-gram table included"
info "  log: ${LOG_FILE}"
started=${SECONDS}
next_beat=15
until curl -sf --max-time 5 "${PROBE}/health" >/dev/null 2>&1; do
  if ! container_running; then
    sleep 1
    printf '\n' >&2
    tail -n 30 "${LOG_FILE}" >&2 || true
    if [[ "$(docker inspect -f '{{.State.OOMKilled}}' "${CONTAINER_NAME}" 2>/dev/null)" == true ]]; then
      printf '\nOOM-killed by the %s GiB host-side cgroup cap; the host is fine. Raise CONTAINER_MEM_GIB.\n' \
        "${CONTAINER_MEM_GIB}" >&2
    fi
    grep -q STOPPING "${MEMWATCH_LOG}" 2>/dev/null && printf '\nThe watchdog stopped it: %s\n' "$(grep STOPPING "${MEMWATCH_LOG}")" >&2
    die "the server exited (code $(docker inspect -f '{{.State.ExitCode}}' "${CONTAINER_NAME}" 2>/dev/null)) before it was ready"
  fi
  (( SECONDS - started < WAIT_TIMEOUT )) || die "not ready after ${WAIT_TIMEOUT}s; still running: docker logs -f ${CONTAINER_NAME}"
  if (( SECONDS - started >= next_beat )); then
    printf '  %4ss  MemAvailable %s\n' "$(( SECONDS - started ))" "$(human_bytes "$(mem_available_bytes)")"
    next_beat=$(( next_beat + 15 ))
  fi
  sleep 3
done
okm "Server answered after $(( SECONDS - started ))s"
grep -iE 'pool|slots|sequence|window' "${LOG_FILE}" | tail -3 | sed 's/^/  /' || true
printf '  host in use %s of %s, MemAvailable %s\n' \
  "$(human_bytes "$(( $(mem_total_bytes) - $(mem_available_bytes) ))")" "$(human_bytes "$(mem_total_bytes)")" \
  "$(human_bytes "$(mem_available_bytes)")"

# --- smoke ---------------------------------------------------------------------
if smoke="$(curl -s --max-time 180 "${PROBE}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\": \"${SERVED_MODEL_NAME}\", \"max_tokens\": 32, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}]}" |
    python3 -c 'import json,sys; r = json.load(sys.stdin); c = r["choices"][0]["message"].get("content") or ""; assert c.strip(); print(repr(c.strip()[:40]) + ",", r["usage"]["completion_tokens"], "tokens")' 2>/dev/null)"; then
  okm "Smoke request: ${smoke}"
else
  warnm "the smoke request failed; the server is still up (docker logs -f ${CONTAINER_NAME})"
fi

printf '\nOpenAI base URL : http://%s:%s/v1\n' "$( [[ "${HOST}" == 0.0.0.0 ]] && hostname -I | awk '{print $1}' || printf '%s' "${HOST}")" "${PORT}"
printf 'model id        : %s\n' "${SERVED_MODEL_NAME}"
printf 'checks          : ./status.sh   ./bench.sh all   tools/toolcheck.py   tools/visioncheck.py\n'
printf 'stop            : ./stop.sh\n'
