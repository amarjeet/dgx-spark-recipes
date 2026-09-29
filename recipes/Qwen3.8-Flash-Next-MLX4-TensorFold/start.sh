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
# Serve Qwen3.8-Flash-Next (Vontra MLX 4-bit + MTP head) with TensorFold on
# one DGX Spark.
#
#   ./start.sh [restart] [profile] [--no-launch]
#
#   ./start.sh                  int8x4: 4 streams x 262,144 tokens, int8 KV
#   ./start.sh int8x5           upstream's 5 streams; tight on this host
#   ./start.sh restart          replace the running server (checks pass first)
#   ./start.sh --no-launch      print the docker command, start nothing
#   PARALLEL=6 CONTEXT=220000 ./start.sh
#   EXTRA_SERVE_ARGS='--thinking-budget 4096' ./start.sh restart
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
    -h|--help) sed -n '12,25p' "$0" | sed 's/^# \?//'; exit 0 ;;
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

mkdir -p "${OUT_DIR}/logs" "${TRITON_CACHE_HOST}" "${TORCH_EXTENSIONS_HOST}" "${TENSORFOLD_CACHE_HOST}"

DOCKER_ARGS=(
  -d --name "${CONTAINER_NAME}"
  --restart "${RESTART_POLICY}"
  --gpus all --ipc host
  -p "${HOST}:${PORT}:${PORT}"
  --ulimit memlock=-1 --ulimit stack=67108864
  # Host-side cap. GPU allocations are not charged to it on GB10.
  --memory "${CONTAINER_MEM_GIB}g" --memory-swap "${CONTAINER_MEM_GIB}g"
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  -e "TENSORFOLD_PREFILL_ROWS=${TENSORFOLD_PREFILL_ROWS}"
  -e "TENSORFOLD_MTP_COPY=${TENSORFOLD_MTP_COPY}"
  -e "TENSORFOLD_NO_UPDATE_CHECK=${TENSORFOLD_NO_UPDATE_CHECK}"
  # Read-only: serving reads the snapshot and writes nothing into the HF cache.
  -v "${HF_HOME}:/root/.cache/huggingface:ro"
  -v "${TRITON_CACHE_HOST}:/root/.triton"
  -v "${TORCH_EXTENSIONS_HOST}:/root/.cache/torch_extensions"
  -v "${TENSORFOLD_CACHE_HOST}:/root/.cache/tensorfold"
)
# Any other TENSORFOLD_* / TF_* switch in the environment reaches the server,
# as upstream does.
while IFS='=' read -r name _; do
  case "${name}" in
    TENSORFOLD_PREFILL_ROWS|TENSORFOLD_MTP_COPY|TENSORFOLD_NO_UPDATE_CHECK) ;;
    *) DOCKER_ARGS+=(-e "${name}") ;;
  esac
done < <(env | grep -E '^(TENSORFOLD|TF)_[A-Z0-9_]+=' | grep -vE '^TF_(VERSION|REPO|COMMIT)=' || true)

# The image sets TRITON_CACHE_DIR and TORCH_EXTENSIONS_DIR to /cache/...
# Unset rather than overridden: torch only adds its py<ver>_cu<ver> folder
# when TORCH_EXTENSIONS_DIR is unset, and that folder is what keeps builds
# from different torch versions apart in a shared cache. Both tools then use
# their own defaults, which are the mounts above.
CMD=(env -u TORCH_EXTENSIONS_DIR -u TRITON_CACHE_DIR tensorfold serve "${SNAPSHOT_CTR}" "${SERVE_ARGS[@]}")

info "Profile ${PROFILE}: ${PARALLEL} x ${CONTEXT} tokens, ${KV_DTYPE} KV$([[ "${PLE_ON_SSD}" == 1 ]] && printf ', n-gram tables on SSD')"
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

info "Loading ~75 GiB of weights (~2.5 min warm; the first start also compiles kernels)"
info "  log: ${LOG_FILE}"
started=${SECONDS}
next_beat=15
until curl -sf --max-time 5 "${PROBE}/v1/models" >/dev/null 2>&1; do
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
    estimate="$(sed -n 's/.*startup estimate \([0-9.]*\) GiB within \([0-9.]*\) GiB.*/\1 of \2/p' "${LOG_FILE}" | tail -1)"
    printf '  %4ss  MemAvailable %s%s\n' "$(( SECONDS - started ))" "$(human_bytes "$(mem_available_bytes)")" \
      "${estimate:+   startup estimate ${estimate} GiB}"
    next_beat=$(( next_beat + 15 ))
  fi
  sleep 3
done
okm "Server answered after $(( SECONDS - started ))s"
grep -E 'streams of|startup estimate' "${LOG_FILE}" | tail -2 | sed 's/^/  /' || true
printf '  host in use %s of %s, MemAvailable %s\n' \
  "$(human_bytes "$(( $(mem_total_bytes) - $(mem_available_bytes) ))")" "$(human_bytes "$(mem_total_bytes)")" \
  "$(human_bytes "$(mem_available_bytes)")"

# --- smoke ---------------------------------------------------------------------
if smoke="$(curl -s --max-time 180 "${PROBE}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\": \"${SERVED_MODEL_NAME}\", \"max_tokens\": 64, \"messages\": [{\"role\": \"user\", \"content\": \"Say hi.\"}]}" |
    python3 -c 'import json,sys; r = json.load(sys.stdin); print(r["usage"]["completion_tokens"], "tokens in", r.get("tensorfold", {}).get("decode_s"), "s")' 2>/dev/null)"; then
  okm "Smoke request: ${smoke}"
else
  warnm "the smoke request failed; the server is still up (docker logs -f ${CONTAINER_NAME})"
fi

printf '\nOpenAI base URL : http://%s:%s/v1\n' "$( [[ "${HOST}" == 0.0.0.0 ]] && hostname -I | awk '{print $1}' || printf '%s' "${HOST}")" "${PORT}"
printf 'model id        : %s\n' "${SERVED_MODEL_NAME}"
printf 'checks          : ./scripts/smoke.py   ./status.sh   ./bench.sh\n'
printf 'stop            : ./stop.sh\n'
