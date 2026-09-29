#!/usr/bin/env bash
#
# Serve Ternary-Bonsai-2-27B (qwen35, 27.36B, ternary g128 at 1.72 bpw) on the
# DGX Spark's GB10, natively, via the PrismML llama.cpp fork.
#
# Why native and not the published llama.cpp container: these files need the
# fork's ternary kernels and its Hadamard activation runtime, and the fork
# ships prebuilt CUDA binaries for linux-x64 only. There is no aarch64 CUDA
# archive and no fork image, so the binary is built here -- see ./build.sh.
#
# Why context is the interesting knob: PQ2_0 is 6.66 GiB resident out of
# 121.7 GiB, and the hybrid backbone (48 of 64 blocks are linear attention)
# costs only 64 KiB/token of KV. So the full 262,144-token context fits four
# times over, which is what the default profile does. Every other recipe in
# this repo spends its memory on weights; this one spends it on context.
#
# Storage: weights come from llama.cpp's own standard cache (LLAMA_CACHE,
# default ~/.cache/llama.cpp), shared with every other llama.cpp recipe. This
# is a native recipe, so no cache variable needs setting -- the default is
# already the shared location.
#
# Usage:
#   ./start.sh                     # default profile (wide: 4 x 262144)
#   ./start.sh deep                # one 262144 slot
#   ./start.sh long                # one 131072 slot -- the conservative one
#   CTX_SIZE=524288 ./start.sh wide
#   REASONING_EFFORT=medium ./start.sh deep
set -euo pipefail

case "${1:-}" in
  -h|--help) sed -n '3,28p' "$0" | sed 's/^# \?//'; exit 0 ;;
esac

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
select_profile "${1:-${DEFAULT_PROFILE}}"

# Native, so HOST is the actual bind address rather than a docker publish
# address -- there is no container indirection to see through. 0.0.0.0 reaches
# loopback too, but a tailnet or LAN address does not, so a readiness probe
# hardcoded to 127.0.0.1 would wait forever on a server that is already up.
case "${HOST}" in
  0.0.0.0|::|"") PROBE_HOST=127.0.0.1 ;;
  *)             PROBE_HOST="${HOST}" ;;
esac
READY_URL="http://${PROBE_HOST}:${PORT}/health"

command -v curl >/dev/null || { printf 'curl is not on PATH\n' >&2; exit 1; }

[[ -x "${BIN_DIR}/llama-server" ]] || {
  printf 'error: llama-server not built: %s\n' "${BIN_DIR}/llama-server" >&2
  printf 'run: ./build.sh\n' >&2
  exit 1
}

# Cheap, and it is the one mistake whose symptom is not an error message: stock
# llama.cpp built into the same tree would serve this model as fluent nonsense.
fork_has_ternary_kernels || {
  printf 'error: %s has no PQ2_0 type -- this is stock llama.cpp, not the fork.\n' "${BIN_DIR}" >&2
  printf 'stock llama.cpp cannot apply the Hadamard transform these weights need.\n' >&2
  printf 'run: ./build.sh --clean\n' >&2
  exit 1
}

[[ -f "${MODEL_FILE}" ]] || {
  printf 'error: weights are missing: %s\n' "${MODEL_FILE}" >&2
  printf 'run: ./download.sh %s\n' "${PROFILE}" >&2
  exit 1
}

if server_running; then
  printf 'llama-server is already running (pid %s, profile: %s)\n' \
    "$(server_pid)" "$(cat "${ACTIVE_PROFILE_FILE}" 2>/dev/null || printf 'unknown')"
  printf 'log: %s\n' "${LOG_FILE}"
  printf 'stop it first: ./stop.sh\n'
  exit 0
fi
# A pidfile left by a crashed server would make server_running() false but
# still confuse status.sh, so clear it now rather than leaving it behind.
rm -f "${PID_FILE}"

mkdir -p "${LLAMA_CACHE}" "${OUT_DIR}"

# --mlock is deprecated in current llama.cpp; mmap+mlock is the replacement.
# It matters more here than the small weight footprint suggests: the KV cache
# for this profile is committed up front and dwarfs the weights.
load_mode="mmap"
[[ "${MLOCK}" == "1" ]] && load_mode="mmap+mlock"

mmproj_note="off"
[[ -n "${MMPROJ_FILE}" && -f "${MMPROJ_FILE}" ]] && mmproj_note="$(basename "${MMPROJ_FILE}")"

printf 'model     : prism-ml/Ternary-Bonsai-2-27B-gguf (qwen35, 27.36B, ternary g128)\n'
printf 'weights   : %s %s @ %s\n' "${QUANT}" "$(human_bytes "${MODEL_TOTAL_BYTES}")" "${MODEL_REVISION_SHORT}"
printf 'binary    : %s (build %s)\n' "${BIN_DIR}/llama-server" "$(bin_build_number || printf '?')"
printf 'context   : %s total over %s slot(s) = %s per slot\n' \
  "${CTX_SIZE}" "${PARALLEL}" "${CTX_PER_SLOT}"
printf 'budget    : %s expected resident\n' "$(human_bytes "$(budget_bytes)")"
printf 'vision    : %s\n' "${mmproj_note}"
printf 'thinking  : on (effort: %s)\n' "${REASONING_EFFORT:-xhigh (template default)}"
case "${PROFILE}" in
  wide|deep)
    printf '\n'
    printf 'NOTE: %s per slot sits above the depths llama.cpp #27756 reports\n' "${CTX_PER_SLOT}"
    printf '      failing with a silent instant-EOS. This build passed a needle test\n'
    printf '      at 254032 tokens, but that onset is prompt-dependent, so verify:\n'
    printf '        ./scripts/smoke.py --needle-depth %s\n' "${CTX_PER_SLOT}"
    printf '      Conservative alternative: ./start.sh safe\n'
    ;;
esac
printf 'load mode : %s\n' "${load_mode}"
printf 'listening : %s:%s\n' "${HOST}" "${PORT}"
printf 'log       : %s\n\n' "${LOG_FILE}"

log_event "launching ${PROFILE} (${QUANT}) ctx=${CTX_SIZE} parallel=${PARALLEL} per_slot=${CTX_PER_SLOT}" \
  >"${LOG_FILE}"

# Optional flags, assembled rather than inlined so an unset one contributes
# nothing at all (an empty string would be parsed as a positional argument).
mmproj_args=()
if [[ -n "${MMPROJ_FILE}" && -f "${MMPROJ_FILE}" ]]; then
  mmproj_args=(--mmproj "${MMPROJ_FILE}")
  # BONSAI_MMPROJ_CPU's equivalent: keep the projector in system RAM. Not
  # needed at this footprint, but it is the documented knob upstream exposes.
  [[ "${MMPROJ_CPU}" == "1" ]] && mmproj_args+=(--no-mmproj-offload)
fi

image_args=()
[[ -n "${IMAGE_MAX_TOKENS}" ]] && image_args=(--image-max-tokens "${IMAGE_MAX_TOKENS}")

# The template accepts xhigh|medium|low and raises on anything else;
# select_profile has already validated this.
effort_args=()
[[ -n "${REASONING_EFFORT}" ]] && \
  effort_args=(--chat-template-kwargs "{\"reasoning_effort\":\"${REASONING_EFFORT}\"}")

SERVER_ARGS=(
  --model "${MODEL_FILE}"
  --alias "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --ctx-size "${CTX_SIZE}"
  --parallel "${PARALLEL}"
  --n-gpu-layers 999
  --flash-attn on
  --batch-size "${BATCH_SIZE}"
  --ubatch-size "${UBATCH_SIZE}"
  --load-mode "${load_mode}"
  # A hybrid model cannot partially evict its recurrent state, so an edited
  # conversation re-prefills from a checkpoint or from zero. These bound what
  # that costs in host RAM; see profiles.sh for the arithmetic.
  --ctx-checkpoints "${CTX_CHECKPOINTS}"
  --cache-ram "${CACHE_RAM_MIB}"
  # --jinja uses the template embedded in the GGUF -- there is no template file
  # to ship here. It also selects llama.cpp's Qwen3-Coder tool-call parser,
  # which is the one that understands this template's XML
  # <tool_call><function=..><parameter=..> dialect.
  --jinja
  # auto is the default and means "thoughts in message.reasoning_content".
  # Stated explicitly because it is load-bearing: the template opens <think> in
  # the assistant prefix, so generation starts inside the block and only
  # </think> is ever emitted.
  --reasoning-format auto
  --temp "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  ${mmproj_args+"${mmproj_args[@]}"}
  ${image_args+"${image_args[@]}"}
  ${effort_args+"${effort_args[@]}"}
)

# Native: no container, no restart policy, no healthcheck. setsid detaches the
# server from this shell's process group so a Ctrl-C here does not take the
# server with it.
LD_LIBRARY_PATH="${BIN_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  setsid nohup "${BIN_DIR}/llama-server" "${SERVER_ARGS[@]}" \
  >>"${LOG_FILE}" 2>&1 &
server_pid_value=$!
printf '%s' "${server_pid_value}" >"${PID_FILE}"
printf '%s' "${PROFILE}" >"${ACTIVE_PROFILE_FILE}"
printf 'spawned llama-server (pid %s)\n' "${server_pid_value}"

printf 'waiting for %s\n' "${READY_URL}"
printf '(cold load reads %s off NVMe, then commits %s of KV)\n' \
  "$(human_bytes "${MODEL_TOTAL_BYTES}")" \
  "$(human_bytes "$(( CTX_SIZE * KV_BYTES_PER_TOKEN ))")"
while ! curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! kill -0 "${server_pid_value}" 2>/dev/null; then
    printf '\nllama-server exited before becoming ready\n' >&2
    tail -n 60 "${LOG_FILE}" >&2 || true
    # The failures worth naming, because their log lines are far from obvious.
    if grep -qiE 'failed to allocate|out of memory|cannot allocate|unable to allocate' "${LOG_FILE}"; then
      printf '\nAllocation failed -- this profile did not fit. Step down the ladder:\n' >&2
      case "${PROFILE}" in
        wide) printf '  ./start.sh safe      # 4 slots x 65536, the known-good profile\n' >&2 ;;
        deep) printf '  ./start.sh long      # 131072 over 1 slot\n' >&2 ;;
        *)    printf '  CTX_SIZE=%s ./start.sh %s\n' "$(( CTX_SIZE / 2 ))" "${PROFILE}" >&2 ;;
      esac
      printf '  PARALLEL=1 ./start.sh %s\n' "${PROFILE}" >&2
    fi
    # Upstream issue #28377: data-dependent cublasGemmEx failure during prefill
    # on GB10 (sm_121) for a sibling arch, avoided by a smaller micro-batch.
    # Cheap to try and hard to guess, so name it.
    if grep -qiE 'cublas|internal operation failed|GemmEx' "${LOG_FILE}"; then
      printf '\ncuBLAS failed during prefill. This family is batch-size fragile on GB10;\n' >&2
      printf 'upstream issue #28377 is avoided with a smaller micro-batch:\n' >&2
      printf '  UBATCH_SIZE=256 ./start.sh %s\n' "${PROFILE}" >&2
    fi
    if grep -qiE 'unknown model architecture|unsupported|unknown type' "${LOG_FILE}"; then
      printf '\nThe binary does not know this file. Rebuild the fork:\n' >&2
      printf '  ./build.sh --clean\n' >&2
    fi
    if grep -qiE 'no kernel image|invalid device function|CUDA error' "${LOG_FILE}"; then
      printf '\nCUDA refused the kernels: the build probably lacks sm_121. Rebuild:\n' >&2
      printf '  CUDA_ARCHS=121a-real ./build.sh --clean\n' >&2
    fi
    rm -f "${PID_FILE}" "${ACTIVE_PROFILE_FILE}"
    exit 1
  fi
  printf '  still loading (avail %s, swap used %s)\n' \
    "$(human_bytes "$(mem_available_bytes)")" "$(human_bytes "$(host_swap_used_bytes)")"
  sleep 10
done

printf '\nready.\n'

# What the run actually cost, so the profile table can be corrected against a
# measurement rather than left as arithmetic.
used_now="$(( $(mem_total_bytes) - $(mem_available_bytes) ))"
printf '  predicted      %s (budget_bytes)\n' "$(human_bytes "$(budget_bytes)")"
printf '  host in use    %s of %s\n' \
  "$(human_bytes "${used_now}")" "$(human_bytes "$(mem_total_bytes)")"

# The claim worth checking every single start: llama.cpp can reduce a requested
# context to make it fit (llama_params_fit), and it says so only in this line.
# A wide run that quietly became 4 x 65536 would otherwise look like a success.
slot_line="$(grep -oE 'n_ctx_slot = [0-9]+' "${LOG_FILE}" | tail -1 || true)"
reported_slot="${slot_line##* }"
if [[ -n "${reported_slot}" ]]; then
  if [[ "${reported_slot}" == "${CTX_PER_SLOT}" ]]; then
    printf '  context        %s per slot, as requested\n' "${reported_slot}"
  else
    printf '  context        WARNING: %s per slot, but %s was requested\n' \
      "${reported_slot}" "${CTX_PER_SLOT}"
    printf '                 llama.cpp reduced the context to make it fit.\n'
  fi
fi
grep -iE 'n_slots|n_ctx_slot|model loaded|main: server is listening' "${LOG_FILE}" \
  | sed 's/^[0-9.]* [A-Z] /  /' | tail -4 || true

printf '\nOpenAI base URL : http://%s:%s/v1\n' "${PROBE_HOST}" "${PORT}"
printf 'model id        : %s\n' "${SERVED_MODEL_NAME}"
printf 'smoke test      : ./scripts/smoke.py\n'
printf 'status          : ./status.sh\n'
printf 'stop            : ./stop.sh\n'
