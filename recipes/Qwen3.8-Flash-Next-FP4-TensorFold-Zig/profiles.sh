#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 MiaAI-Lab (original)
# Copyright (c) 2026 amarjeet (port)
#
# Derived from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
# (scripts/config.sh) at 656879f, MIT. Carries its knob names, defaults and
# measured constants; replaces its recipe-private kernel cache with the tools'
# own defaults. The shared helpers at the bottom follow this repo's other
# TensorFold recipe.
#
# Shared configuration for Qwen3.8-Flash-Next (NVIDIA FP4) on TensorFold's Zig
# engine. Sourced by build.sh / download.sh / preflight.sh / start.sh / stop.sh /
# status.sh / bench.sh -- the ONLY place paths, the profile table and helpers
# are defined.
#
# Usage:  source profiles.sh          # paths + defaults only
#         select_profile fp8x5        # additionally sets the profile vars
#
# Storage rule: every path is an env-overridable variable whose default is the
# tool's own standard location (CONVENTIONS.md). This is a DOCKER recipe, so
# each host cache is bind-mounted onto the tool's in-container default.

set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Pinned rather than derived from the directory name, so OUT_DIR does not move
# if the recipe is cloned or renamed.
EXPERIMENT_NAME="Qwen3.8-Flash-Next-FP4-TensorFold-Zig"

# --- storage -----------------------------------------------------------------

HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
# The engine's kernels are compiled ahead of time into the image. What is left
# to cache at run time: the CUDA driver's JIT cache (CUDA_CACHE_PATH, whose
# default is ~/.nv/ComputeCache), triton's, for the vision helper, and
# TensorFold's own (prompt-state snapshots, update check). Upstream keeps all
# of them under a recipe-private ~/.cache/tensorfold-qwen38.
CUDA_CACHE_HOST="${CUDA_CACHE_HOST:-${HOME}/.nv}"
TRITON_CACHE_HOST="${TRITON_CACHE_HOST:-${HOME}/.triton}"
TENSORFOLD_CACHE_HOST="${TENSORFOLD_CACHE_HOST:-${HOME}/.cache/tensorfold}"

OUT_DIR="${OUT_DIR:-${XDG_STATE_HOME:-${HOME}/.local/state}/dgx-spark-recipes/${EXPERIMENT_NAME}}"

# --- model -------------------------------------------------------------------

MODEL_ID="${MODEL_ID:-nvidia/Qwen3.8-Flash-Next-FP4-Local}"
# Pinned. Upstream resolves the repo id; start.sh passes this revision's
# snapshot directory instead.
MODEL_REVISION="${MODEL_REVISION:-b9ce18a6887e18c089fdea8509a33e63429a3300}"
MANIFEST="${MANIFEST:-${EXPERIMENT_DIR}/manifests/fp4.json}"
MODEL_PATH="${HF_HOME}/hub/models--${MODEL_ID//\//--}"
SNAPSHOT_DIR="${MODEL_PATH}/snapshots/${MODEL_REVISION}"
# The same snapshot as the container sees it through the HF_HOME mount.
SNAPSHOT_CTR="/root/.cache/huggingface/hub/models--${MODEL_ID//\//--}/snapshots/${MODEL_REVISION}"
# What loading takes of MemAvailable (upstream, measured on the Zig engine:
# 81 GiB with vision on). The n-gram table is part of it.
WEIGHTS_GIB="${WEIGHTS_GIB:-81}"

# The id the vLLM recipe for this model serves, so a client can move between
# the two unchanged. Upstream's name is answered too, as an alias.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next}"
SERVED_ALIAS="${SERVED_ALIAS:-Qwen3.8-Flash-Next}"

# --- runtime -----------------------------------------------------------------

# TensorFold's Zig engine (tensorfold-native), built from TF_REPO at TF_REF
# with patches/*.patch applied (git apply, in the checkout's root), with Zig
# ZIG_VERSION. Only the Zig build goes into the image.
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
TF_REF="${TF_REF:-db281878ddb836fd0df510d8771ecb7e0fe47d26}"   # zig-flashnext; the patches bring the gated Flash Next CUDA engine
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
KERNELS_PATH=/opt/tensorfold/share/tensorfold/cuda/sm121
# Triton's line info records each kernel source's mtime. Pinned so the AOT
# cubins match the reference kernel set.
KERNEL_SOURCE_MTIME="${KERNEL_SOURCE_MTIME:-1791318675}"
# One hash of everything the image is made of: patches, this recipe's
# Dockerfile (the heredoc in build.sh), and the build inputs. It is the tag
# suffix and the tf.patches label, so a changed patch is a different image.
image_hash() {
  ( cat "${EXPERIMENT_DIR}"/patches/*.patch
    sed -n "/<<'DOCKERFILE'/,/^DOCKERFILE\$/p" "${EXPERIMENT_DIR}/build.sh"
    echo "${TF_REPO} ${TF_REF} zig-${ZIG_VERSION} ${ZIG_SHA256} ${BASE_IMAGE} kernel-mtime-${KERNEL_SOURCE_MTIME} tp-1"
  ) | sha256sum | cut -c1-12
}
PATCHES_HASH="$(image_hash)"
IMAGE="${IMAGE:-tensorfold-qwen38:zig-${TF_REF:0:7}-${PATCHES_HASH}}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen3.8-flash-next-tensorfold-zig}"

HOST="${HOST:-0.0.0.0}"
# 8011 is where the other TensorFold recipe for this model serves, so a client
# does not change when this one replaces it. The vLLM recipe claims 8888.
PORT="${PORT:-8011}"
RESTART_POLICY="${RESTART_POLICY:-no}"

# --- memory ------------------------------------------------------------------
#
# The engine sizes its KV pool from MemAvailable read after the weights are
# loaded, less TENSORFOLD_MEMORY_RESERVE_GIB, and refuses a request that would
# not fit. Slots start small and grow on demand, so PARALLEL is a cap, not a
# promise that every stream can sit at a full window at once.
#
# On unified memory an exhausted pool hangs the kernel -- no OOM, no logs -- so
# two host-side guards sit around the engine's own:
#   * a cgroup cap on the container. GPU allocations are not charged to it on
#     GB10, so it bounds the host-side footprint, not the weights;
#   * scripts/memwatch.sh, which stops the container when MemAvailable or
#     MemFree stays under a floor.
# The reserve (10 GiB) is above the watchdog's floor (6), so the engine stops
# growing the pool before the watchdog would stop the server.

TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-10}"
CONTAINER_MEM_GIB="${CONTAINER_MEM_GIB:-24}"
MEMWATCH_MIN_GIB="${MEMWATCH_MIN_GIB:-6}"
MEMWATCH_MIN_FREE_GIB="${MEMWATCH_MIN_FREE_GIB:-2}"
# MemFree only counts while MemAvailable is under this: MemFree sits near zero
# whenever the page cache is full of reclaimable data.
MEMWATCH_FREE_GATE_GIB="${MEMWATCH_FREE_GATE_GIB:-10}"
MEMWATCH_SAMPLES="${MEMWATCH_SAMPLES:-5}"
MEMWATCH_INTERVAL="${MEMWATCH_INTERVAL:-2}"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"
# MemAvailable a start must have, beyond weights + one full window: prefill
# scratch slack (a 2,048-row chunk, not a measured 1M-prompt peak) and the
# vision helper's workspace.
SCRATCH_GIB="${SCRATCH_GIB:-8}"
VISION_WORKSPACE_GIB=2

DISK_RESERVE_BYTES="${DISK_RESERVE_BYTES:-10737418240}"   # 10 GiB

# --- serving defaults (upstream's, measured there) ---------------------------

# Native window 262,144. Past it the engine applies Qwen's YaRN (factor 4), up
# to 1,048,576; a full 1M-token fp8 cache is 17.46 GiB on top of the weights.
_ENV_PARALLEL="${PARALLEL:-}"
_ENV_CONTEXT="${CONTEXT:-}"
_ENV_KV_DTYPE="${KV_DTYPE:-}"
NATIVE_CONTEXT=262144
# MTP: at most this many drafts a round. Drafts are checked, so replies match
# serial decoding. 6, not the engine's 15: deeper chains cost verify rows and
# lose 5 to 20% at 3 to 5 streams (upstream, measured at 1 to 5 streams).
export TF_FLASHNEXT_DEPTH="${TF_FLASHNEXT_DEPTH:-6}"
# The running-product draft rule at every width the default serves.
export TF_FLASHNEXT_PRODUCT_STREAMS="${TF_FLASHNEXT_PRODUCT_STREAMS:-5}"
export TF_FLASHNEXT_PREFILL_TAIL="${TF_FLASHNEXT_PREFILL_TAIL:-512}"
# Prompt experts as a shared-memory tile (patch 0014): bit-identical output,
# cold prefill +14%. 2 = gate/up and down, 1 = gate/up only, 0 = older kernels.
export TF_FLASHNEXT_PROMPT_EXPERT_TILE="${TF_FLASHNEXT_PROMPT_EXPERT_TILE:-2}"
# Fair prefill (patch 0015) stays off: it costs ~20% of the total time of a
# mixed load. TF_FLASHNEXT_FILL_FAIR=1 opts in.
# KV cache: fp8 is about 1.84x the pool of bf16 and lossy (~98.8% top-1
# agreement with a bf16 cache); bf16 is exact.
MAX_TOKENS="${MAX_TOKENS:-32768}"
THINKING="${THINKING:-1}"
DRAFTS="${DRAFTS:-1}"
# Qwen's thinking-mode sampling. A request's own values win.
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
# Image and video input: rank 0 starts the Python helper (Pillow, PyAV, the
# 27-layer tower). VISION=0 is text only and returns the 2 GiB workspace to
# the KV budget.
VISION="${VISION:-1}"
VISION_URLS="${VISION_URLS:-0}"            # 1: also fetch https:// image/video URLs (default: data URLs only)
MAX_VIDEOS="${MAX_VIDEOS:-4}"              # per request
export TENSORFOLD_MAX_IMAGES="${TENSORFOLD_MAX_IMAGES:-50}"
export TENSORFOLD_IMAGE_TOKENS="${TENSORFOLD_IMAGE_TOKENS:-16384}"
export TENSORFOLD_VIDEO_TOKENS="${TENSORFOLD_VIDEO_TOKENS:-16384}"
export TENSORFOLD_VISION_WORKSPACE_MIB="${TENSORFOLD_VISION_WORKSPACE_MIB:-2048}"
export TENSORFOLD_MEMORY_RESERVE_GIB
export TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"

# --- profile table -----------------------------------------------------------
#
# One checkpoint; the profiles are KV-pool trade-offs. EST is what upstream
# measured (or this recipe, once run here); the server prints its own pool
# size at every start.
#
#   fp8x5   5 x 262,144, fp8 KV   default, upstream's; five full windows are 21.9 GiB of a 24.3 GiB pool
#   fp8x3   3 x 262,144, fp8 KV   a smaller pool
#   bf16x3  3 x 262,144, bf16 KV  exact cache; one window is 7.5 GiB, so 3 is 22.6 GiB
#   fp8x1   1 x 262,144, fp8 KV   one stream

DEFAULT_PROFILE="${DEFAULT_PROFILE:-fp8x5}"
KNOWN_PROFILES=(fp8x5 fp8x3 bf16x3 fp8x1)

select_profile() {
  PROFILE="${1:-${DEFAULT_PROFILE}}"
  case "${PROFILE}" in
    fp8x5)  PARALLEL=5; KV_DTYPE=fp8 ;;
    fp8x3)  PARALLEL=3; KV_DTYPE=fp8 ;;
    bf16x3) PARALLEL=3; KV_DTYPE=bf16 ;;
    fp8x1)  PARALLEL=1; KV_DTYPE=fp8 ;;
    *)
      printf 'error: unknown profile: %s (expected one of: %s)\n' \
        "${PROFILE}" "${KNOWN_PROFILES[*]}" >&2
      return 2
      ;;
  esac
  CONTEXT="${_ENV_CONTEXT:-${NATIVE_CONTEXT}}"
  PARALLEL="${_ENV_PARALLEL:-${PARALLEL}}"
  KV_DTYPE="${_ENV_KV_DTYPE:-${KV_DTYPE}}"
  [[ "${CONTEXT}" =~ ^[1-9][0-9]*$ && "${CONTEXT}" -le 1048576 ]] \
    || { printf 'error: CONTEXT is a token count up to 1048576, not %s\n' "${CONTEXT}" >&2; return 2; }
  [[ "${KV_DTYPE}" =~ ^(bf16|fp8)$ ]] \
    || { printf 'error: KV_DTYPE is bf16 or fp8, not %s\n' "${KV_DTYPE}" >&2; return 2; }
  # Past the native window the engine needs Qwen's YaRN (factor 4).
  if (( CONTEXT > NATIVE_CONTEXT )); then TF_FLASHNEXT_YARN="${TF_FLASHNEXT_YARN:-4}"
  else TF_FLASHNEXT_YARN="${TF_FLASHNEXT_YARN:-0}"; fi
  export TF_FLASHNEXT_YARN
  # Deliberately not exported: a child script (start.sh runs preflight.sh)
  # re-sources this file and would read its parent's profile values as
  # caller overrides.
}

# One sequence's cache at TP=1, in GiB rounded up (State.cacheBytes in the
# engine). bf16: 30,784 bytes a token. fp8: 17,888 (keys carry both scales,
# values are codes, indexer and pooled stay bf16).
kv_gib() {
  local ctx="${1:-${CONTEXT}}" bytes=30784
  [[ "${KV_DTYPE}" == fp8 ]] && bytes=17888
  printf '%s' "$(( (bytes * ctx + 1073741823) / 1073741824 ))"
}

# MemAvailable a start needs: the weights, one full window, scratch slack, the
# vision workspace when VISION=1, the engine's reserve, and 2 GiB.
mem_need_gib() {
  local vision=0
  [[ "${VISION}" == 1 ]] && vision="${VISION_WORKSPACE_GIB}"
  printf '%s' "$(( WEIGHTS_GIB + $(kv_gib) + SCRATCH_GIB + vision + TENSORFOLD_MEMORY_RESERVE_GIB + 2 ))"
}

# The tensorfold-native serve arguments for the selected profile. EXTRA_SERVE_ARGS
# go last, so they win.
serve_args() {
  SERVE_ARGS=(
    --host 0.0.0.0 --port "${PORT}"
    --name "${SERVED_MODEL_NAME}"
  )
  [[ -n "${SERVED_ALIAS}" && "${SERVED_ALIAS}" != "${SERVED_MODEL_NAME}" ]] \
    && SERVE_ARGS+=(--alias "${SERVED_ALIAS}")
  SERVE_ARGS+=(
    --context "${CONTEXT}" --parallel "${PARALLEL}" --kv-dtype "${KV_DTYPE}"
    --max-tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}" --top-p "${TOP_P}" --top-k "${TOP_K}"
  )
  if [[ "${THINKING}" == 1 ]]; then SERVE_ARGS+=(--thinking); else SERVE_ARGS+=(--no-thinking); fi
  [[ "${DRAFTS}" == 1 ]] || SERVE_ARGS+=(--no-drafts)
  if [[ "${VISION}" == 1 ]]; then
    SERVE_ARGS+=(--vision --vision-max-images "${TENSORFOLD_MAX_IMAGES}" --vision-max-videos "${MAX_VIDEOS}"
                 --vision-image-tokens "${TENSORFOLD_IMAGE_TOKENS}")
    [[ "${VISION_URLS}" == 1 ]] && SERVE_ARGS+=(--vision-urls)
  fi
  # shellcheck disable=SC2206
  [[ -n "${EXTRA_SERVE_ARGS}" ]] && SERVE_ARGS+=(${EXTRA_SERVE_ARGS})
  return 0
}

# --- shared helpers ----------------------------------------------------------

GIB=1073741824

mem_available_bytes() { awk '/^MemAvailable:/ {print $2 * 1024; exit}' /proc/meminfo; }
mem_free_bytes()      { awk '/^MemFree:/ {print $2 * 1024; exit}' /proc/meminfo; }
mem_total_bytes()     { awk '/^MemTotal:/ {print $2 * 1024; exit}' /proc/meminfo; }
host_swap_used_bytes() {
  awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print (t - f) * 1024}' /proc/meminfo
}

human_bytes() {
  local n="$1" sign=''
  if [[ "${n}" == -* ]]; then sign='-'; n="${n#-}"; fi
  printf '%s%s' "${sign}" "$(numfmt --to=iec-i --round=nearest --format='%.1f' --suffix=B "${n}" 2>/dev/null || printf '%s' "${n}")"
}

gib_to_bytes() { python3 -c "import sys; print(int(float(sys.argv[1]) * 2**30))" "$1"; }

log_event() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

require_aarch64() {
  [[ "$(uname -m)" == "aarch64" ]] || {
    printf 'error: this recipe requires aarch64 GB10 (found %s)\n' "$(uname -m)" >&2
    return 1
  }
}

container_running() { docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; }
container_exists()  { docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; }

image_label() {   # image_label <label>
  docker image inspect -f "{{index .Config.Labels \"$1\"}}" "${IMAGE}" 2>/dev/null || true
}
image_patches_label() { image_label tf.patches; }

probe_host() {
  if [[ "${HOST}" == 0.0.0.0 || "${HOST}" == "::" ]]; then printf '127.0.0.1'; else printf '%s' "${HOST}"; fi
}

resolve_hf_token() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    local token_file="${HF_TOKEN_PATH:-${HF_HOME}/token}"
    [[ -r "${token_file}" ]] && HF_TOKEN="$(tr -d '[:space:]' <"${token_file}")"
  fi
  export HF_TOKEN="${HF_TOKEN:-}"
}

# Verify the pinned snapshot against the manifest, skipping the re-hash when
# the files' (path, size, mtime) have not moved since the last full pass.
# Hashing 95 GiB costs minutes; FORCE_VERIFY=1 always re-hashes.
verify_snapshot() {
  local stamp fingerprint
  stamp="${OUT_DIR}/verified-$(basename "${MANIFEST}" .json)-${MODEL_REVISION:0:12}.stamp"
  fingerprint="$(python3 - "${MANIFEST}" "${SNAPSHOT_DIR}" <<'PYEOF'
import json, os, sys
manifest, root = sys.argv[1], sys.argv[2]
rows = []
for item in json.load(open(manifest))["files"]:
    try:
        st = os.stat(os.path.join(root, item["path"]))
    except OSError:
        print("MISSING"); raise SystemExit(0)
    rows.append(f'{item["path"]}:{st.st_size}:{int(st.st_mtime)}')
print("|".join(rows))
PYEOF
)"
  if [[ "${FORCE_VERIFY:-0}" != "1" && "${fingerprint}" != "MISSING" \
        && -f "${stamp}" && "$(cat "${stamp}")" == "${fingerprint}" ]]; then
    printf 'already verified since last change (FORCE_VERIFY=1 to re-hash)\n'
    return 0
  fi
  [[ "${fingerprint}" != "MISSING" ]] || {
    printf 'snapshot incomplete: %s\n' "${SNAPSHOT_DIR}" >&2
    return 1
  }
  python3 "${EXPERIMENT_DIR}/scripts/download_snapshot.py" \
    --manifest "${MANIFEST}" --hf-home "${HF_HOME}" --verify-only >/dev/null || return 1
  mkdir -p "${OUT_DIR}"
  printf '%s' "${fingerprint}" >"${stamp}"
  printf 'verified %s files against the manifest\n' \
    "$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["files"]))' "${MANIFEST}")"
}
