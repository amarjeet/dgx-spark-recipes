#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2026 Victor Cruz
# Copyright (C) 2026 amarjeet
#
# Derived from one-spark-tp1/scripts/run_tp1.sh and one-spark-tp1/tabbyapi/ of
# vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe, which is
# Copyright (C) 2026 Victor Cruz and licensed AGPL-3.0-only.
# Modified 2026-09-17 by amarjeet: upstream's launcher execs an unpublished
# ENTRY driver script, so this serves through TabbyAPI instead -- upstream's own
# documented API server for this stack. Config is rendered from the template at
# launch so no absolute path is committed, and the guards move to preflight.sh.
#
# Serve the re-laid EXL3 pack through TabbyAPI on an OpenAI-compatible endpoint.
#
# STATUS: TabbyAPI has NOT been run end-to-end against this pack on a DGX Spark,
# upstream or here. The measured numbers in BENCHMARKS.md come from driving
# ExLlamaV3 directly. Treat this as a starting point, not a qualified path, and
# see README.md -> Known unknowns before trusting it.
#
# The one thing that will break this: TabbyAPI has no knowledge of the ATS
# zero-copy environment variables, so they are exported HERE, before it starts.
# Without EXL3_ATS_COPY the model is copied wholesale -- 111.16 GiB of text plus
# a 7.39 GiB drafter against a 121.69 GiB pool -- and will not fit.
#
# Usage:
#   ./start.sh                 # default profile
#   ./start.sh aliased         # lowest memory, all tensors aliased
#   ./start.sh --no-launch     # print the command and the rendered config, exit
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

PROFILE_ARG=""
NO_LAUNCH=0
while (( $# )); do
  case "$1" in
    -h|--help) sed -n '15,30p' "$0" | sed 's/^# \?//'; exit 0 ;;
    --no-launch) NO_LAUNCH=1; shift ;;
    -*) printf 'error: unknown flag: %s\n' "$1" >&2; exit 2 ;;
    *) PROFILE_ARG="$1"; shift ;;
  esac
done
select_profile "${PROFILE_ARG:-${DEFAULT_PROFILE}}"

info() { printf '[INFO]  %s\n' "$*"; }
ok()   { printf '[ OK ]  %s\n' "$*"; }
warn() { printf '[WARN]  %s\n' "$*"; }
err()  { printf '[ERR ]  %s\n' "$*" >&2; exit 1; }

# --- 0. Refuse to start twice -------------------------------------------------

if pid="$(server_pid)"; then
  err "already serving (pid ${pid}); ./stop.sh first"
fi
if pgrep -f "[e]xllamav3" >/dev/null 2>&1; then
  err "an exllamav3 process is already running; never run two on one Spark"
fi

# --- 1. Resolve the re-laid pack. Never downloads, never re-lays. -------------

[[ -d "${MODEL_ROOT}" ]] || err "re-laid pack missing: ${MODEL_ROOT} (./relay.sh)"
dangling="$(relaid_dangling)"
[[ -z "${dangling}" || "${dangling}" == "missing" ]] \
  || err "dangling symlinks in ${MODEL_ROOT}; the hub cache was pruned (./relay.sh --force)"
[[ -x "${VENV}/bin/python" ]] || err "venv missing: ${VENV} (README.md -> Build)"
[[ -f "${TABBY_DIR}/main.py" ]] || err "TabbyAPI missing: ${TABBY_DIR} (README.md -> Serving)"

avail_kb="$(awk '/MemAvailable/{print $2}' /proc/meminfo)"
(( avail_kb >= MIN_AVAIL_KB )) || err \
  "MemAvailable $(human_bytes "$((avail_kb * 1024))") < $(human_bytes "$((MIN_AVAIL_KB * 1024))"); on unified memory an over-commit hangs the kernel rather than raising an OOM"

mode="$(addressing_mode)"
[[ "${mode}" == "ATS" ]] \
  || warn "addressing mode is '${mode}', expected ATS; aliasing will fall back to copies"

# --- 2. Render the TabbyAPI config -------------------------------------------

# The tracked template carries placeholders rather than absolute paths: this
# recipe exists partly because upstream committed someone's home directory into
# tabbyapi/config.yml. Rendering keeps MODEL_ROOT overridable and the repo clean.
mkdir -p "${OUT_DIR}"
RENDERED_CONFIG="${OUT_DIR}/tabbyapi-config.${PROFILE}.yml"

# `model_dir` is the PARENT directory; TabbyAPI joins it with `model_name`.
# Pointing it at MODEL_ROOT itself is an off-by-one-directory that fails late.
if [[ -n "${PROMPT_TEMPLATE}" ]]; then
  template_line="prompt_template: ${PROMPT_TEMPLATE}"
else
  template_line="# prompt_template: unset -- /v1/chat/completions will fail"
fi
sed -e "s|@MODEL_DIR@|$(dirname "${MODEL_ROOT}")|g" \
    -e "s|@MODEL_NAME@|$(basename "${MODEL_ROOT}")|g" \
    -e "s|@CTX@|${CTX}|g" \
    -e "s|@CHUNK@|${CHUNK}|g" \
    -e "s|@HOST@|${HOST}|g" \
    -e "s|@PORT@|${PORT}|g" \
    -e "s|@DISABLE_AUTH@|$( (( TABBY_DISABLE_AUTH )) && printf 'true' || printf 'false' )|g" \
    -e "s|@PROMPT_TEMPLATE_LINE@|${template_line}|g" \
    -e "s|@DRAFT_MODE@|$( (( DRAFT )) && printf 'mtp' || printf 'disabled' )|g" \
    "${EXPERIMENT_DIR}/tabbyapi/config.yml" >"${RENDERED_CONFIG}"
ok "config rendered: ${RENDERED_CONFIG}"

# TabbyAPI's --config returns early and discards every other CLI flag and env
# override, so the rendered file is the single source of truth. Nothing else
# passed on the command line would take effect.
if (( TABBY_DISABLE_AUTH )); then
  case "${HOST}" in
    127.0.0.1|localhost|::1) ok "auth disabled, bound to loopback only" ;;
    *) err "TABBY_DISABLE_AUTH=1 with HOST=${HOST} would expose an unauthenticated server; set HOST=127.0.0.1 or TABBY_DISABLE_AUTH=0" ;;
  esac
else
  warn "auth enabled: TabbyAPI writes api_tokens.yml in ${TABBY_DIR}; smoke.py will 401 without a key"
fi

if [[ -z "${PROMPT_TEMPLATE}" ]]; then
  warn "no prompt template: /v1/chat/completions will fail, /v1/completions will work"
fi

# --- 3. The ATS variables TabbyAPI cannot know about -------------------------

export EXL3_ATS_MMAP="${EXL3_ATS_MMAP}"
export EXL3_ATS_COPY="${EXL3_ATS_COPY}"
export EXL3_DSPARK_CONF="${EXL3_DSPARK_CONF}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR}"
export CUDA_HOME="${CUDA_HOME}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
export PATH="${VENV}/bin:${CUDA_HOME}/bin:${PATH}"
export HF_HOME="${HF_HOME}"

info "profile     ${PROFILE}"
info "model       ${MODEL_ROOT}"
info "copy regex  ${EXL3_ATS_COPY:-<empty: alias everything>}"
info "draft       ${DRAFT} (conf ${EXL3_DSPARK_CONF})"
info "ctx/chunk   ${CTX} / ${CHUNK}"
info "endpoint    http://${HOST}:${PORT}/v1"

if (( NO_LAUNCH )); then
  printf '\n--- rendered config ---\n'
  cat "${RENDERED_CONFIG}"
  printf '\n--- command ---\n'
  printf 'EXL3_ATS_MMAP=%q EXL3_ATS_COPY=%q EXL3_DSPARK_CONF=%q \\\n' \
    "${EXL3_ATS_MMAP}" "${EXL3_ATS_COPY}" "${EXL3_DSPARK_CONF}"
  printf '  %q %q --config %q\n' \
    "${VENV}/bin/python" "${TABBY_DIR}/main.py" "${RENDERED_CONFIG}"
  exit 0
fi

# --- 4. Launch ----------------------------------------------------------------

if [[ -s "${LOG_FILE}" ]]; then
  mkdir -p "${OUT_DIR}/logs"
  mv "${LOG_FILE}" "${OUT_DIR}/logs/$(date +%Y%m%dT%H%M%S)-tabby.log"
fi
log_event "launching TabbyAPI (load takes ~40 s and drives MemAvailable to ~5 GiB)" \
  >"${LOG_FILE}"

( cd "${TABBY_DIR}" && exec nohup "${VENV}/bin/python" main.py \
    --config "${RENDERED_CONFIG}" >>"${LOG_FILE}" 2>&1 ) &
server_launch_pid=$!
printf '%s' "${server_launch_pid}" >"${PID_FILE}"
printf '%s' "${PROFILE}" >"${ACTIVE_PROFILE_FILE}"
ok "spawned pid ${server_launch_pid}"

READY_URL="http://${HOST}:${PORT}/v1/models"
info "waiting for ${READY_URL}"
waited=0
until curl -fsS -m 5 "${READY_URL}" >/dev/null 2>&1; do
  if ! kill -0 "${server_launch_pid}" 2>/dev/null; then
    printf '\n'
    tail -n 60 "${LOG_FILE}" >&2 || true
    rm -f "${PID_FILE}" "${ACTIVE_PROFILE_FILE}"
    err "TabbyAPI exited before becoming ready -- see ${LOG_FILE}"
  fi
  # TabbyAPI does not fail on a busy port: it logs a warning and silently binds
  # PORT+1, which would leave this loop polling a port nothing will ever answer.
  if grep -qiE 'port.*(in use|unavailable).*(switch|trying)|switching to port' \
       "${LOG_FILE}" 2>/dev/null; then
    printf '\n'
    grep -iE 'port' "${LOG_FILE}" | tail -5 >&2
    rm -f "${PID_FILE}" "${ACTIVE_PROFILE_FILE}"
    kill "${server_launch_pid}" 2>/dev/null || true
    err "TabbyAPI moved off port ${PORT}; free it and retry rather than serving somewhere unexpected"
  fi
  sleep 5
  (( waited += 5 ))
  (( waited % 30 == 0 )) && info "  still loading (${waited}s)"
done

ok "server ready after ${waited}s"
printf '\nOpenAI base URL : http://%s:%s/v1\n' "${HOST}" "${PORT}"
printf 'model id        : %s\n' "${SERVED_MODEL_NAME}"
printf 'smoke test      : ./scripts/smoke.py\n'
printf 'log             : %s\n' "${LOG_FILE}"
printf 'stop            : ./stop.sh\n'
