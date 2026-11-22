#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 MiaAI-Lab (original)
# Copyright (c) 2026 amarjeet (port)
#
# Derived from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
# (scripts/config.sh) at b956dba, MIT. Carries its knob names, defaults and
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
# The engine's kernels are embedded in the binary. What is left to cache at run
# time: the CUDA driver's JIT cache (CUDA_CACHE_PATH, whose default is
# ~/.nv/ComputeCache) and TensorFold's own (prompt-state snapshots, update
# check). Upstream keeps both under a recipe-private ~/.cache/tensorfold-qwen38.
CUDA_CACHE_HOST="${CUDA_CACHE_HOST:-${HOME}/.nv}"
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

# ABLIT=1 serves the same checkpoint with drowzeys's Keys "house-qsa"
# abliteration overlay grafted in: the self_attn.o_proj weight and scale of the
# 12 QSA layers (3, 7, ... 47; MXFP8 g32, requantized after a rank-1
# refusal-direction projection) come from the overlay's ~186 MiB sidecar.
# Every expert, the MTP head, the n-gram table, vision and the chat template
# stay NVIDIA's. The model has had its safety refusals removed. The overlay is
# gated: accept its Responsible Use Agreement on the Hub with the account of
# HF_TOKEN, then ./download.sh --ablit. It fits only the revision above.
ABLIT="${ABLIT:-0}"
[[ "${ABLIT}" =~ ^[01]$ ]] || { printf 'error: ABLIT is 0 or 1, not %s\n' "${ABLIT}" >&2; exit 2; }
ABLIT_ID=drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-FP4-QAD-house-qsa-L3-47
ABLIT_REVISION=c44f5b92ee3d307b82aab1c55f4cc4ae1d440344
ABLIT_SIDECAR=fp4-local/house-qsa-oproj-mxfp8-L3-47.safetensors
ABLIT_STOCK_REVISION=b9ce18a6887e18c089fdea8509a33e63429a3300   # the snapshot the sidecar was made from
ABLIT_MANIFEST="${EXPERIMENT_DIR}/manifests/ablit.json"
ABLIT_SNAPSHOT_DIR="${HF_HOME}/hub/models--${ABLIT_ID//\//--}/snapshots/${ABLIT_REVISION}"
# scripts/ablit_view.py's folder: relative symlinks to the stock snapshot's
# files plus the sidecar and a rewritten index. It sits under HF_HOME because
# that is all the container mounts. Upstream's location.
ABLIT_VIEW_SUB="tensorfold-qwen38/ablit-${MODEL_REVISION:0:8}-${ABLIT_REVISION:0:8}"
ABLIT_VIEW_DIR="${HF_HOME}/${ABLIT_VIEW_SUB}"
if [[ "${ABLIT}" == 1 && "${MODEL_REVISION}" != "${ABLIT_STOCK_REVISION}" ]]; then
  printf 'error: ABLIT=1 grafts onto %s @ %s only, not %s\n' "${MODEL_ID}" "${ABLIT_STOCK_REVISION:0:12}" "${MODEL_REVISION:0:12}" >&2
  exit 2
fi

# What the engine reads, as the container sees it through the HF_HOME mount.
if [[ "${ABLIT}" == 1 ]]; then
  MODEL_CTR="/root/.cache/huggingface/${ABLIT_VIEW_SUB}"
else
  MODEL_CTR="/root/.cache/huggingface/hub/models--${MODEL_ID//\//--}/snapshots/${MODEL_REVISION}"
fi
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
# ZIG_VERSION. The kernels are embedded in the binary. Beside it the image
# carries what the engine dlopens, each from a pinned, checksummed source:
# libjpeg-turbo and libwebp's decoder (image input), PDFium out of the
# pypdfium2 wheel (PDF=1), libtfvideo over a static FFmpeg (VIDEO=1), and
# libtfgrammar from TensorFold's own release archive (structured output,
# tool_choice "required").
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
TF_REF="${TF_REF:-93a2e93ecebff8aa52b88e58c5dd222516e69c5f}"   # release v1.2.7
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
LIBJPEG_TURBO_VERSION=3.1.4.1
LIBJPEG_TURBO_SHA256=ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022
LIBWEBP_VERSION=1.6.0
LIBWEBP_SHA256=e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564
PYPDFIUM2_VERSION=5.14.0
PYPDFIUM2_WHEEL_URL="https://files.pythonhosted.org/packages/92/b4/0c31aa51887cd6cd032191dfe010a6d01ed43cf03204cfbd2184ebe4b715/pypdfium2-5.14.0-py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
PYPDFIUM2_AARCH64_SHA256=f1b696e6901e16f114a2ec6332e5e3f8f5033a901614ead28499ab18ca6024f5
FFMPEG_VERSION=9.0.2       # the release TensorFold's tools/zig/build_ffmpeg.sh pins
FFMPEG_SHA256=8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e
TF_RELEASE_VERSION=1.2.7   # the release archive libtfgrammar.so comes from; matches TF_REF
TF_RELEASE_AARCH64_SHA256=250b7cd3d1aa2685cbf011f501c4f10bbff44c6ccd876dc63a7a4bbf337ed01b
# One hash of everything the image is made of: patches, this recipe's
# Dockerfile (the heredoc in build.sh), and the build inputs. It is the tag
# suffix and the tf.patches label, so a changed patch is a different image.
image_hash() {
  ( cat "${EXPERIMENT_DIR}"/patches/*.patch
    sed -n "/<<'DOCKERFILE'/,/^DOCKERFILE\$/p" "${EXPERIMENT_DIR}/build.sh"
    echo "${TF_REPO} ${TF_REF} zig-${ZIG_VERSION} ${ZIG_SHA256} ${BASE_IMAGE} jpeg-${LIBJPEG_TURBO_VERSION}-${LIBJPEG_TURBO_SHA256} webp-${LIBWEBP_VERSION}-${LIBWEBP_SHA256} pdfium-${PYPDFIUM2_VERSION}-${PYPDFIUM2_AARCH64_SHA256} ffmpeg-${FFMPEG_VERSION}-${FFMPEG_SHA256} grammar-${TF_RELEASE_VERSION}-${TF_RELEASE_AARCH64_SHA256}"
  ) | sha256sum | cut -c1-12
}
PATCHES_HASH="$(image_hash)"
IMAGE="${IMAGE:-tensorfold-qwen38:v${TF_RELEASE_VERSION}-${TF_REF:0:7}-${PATCHES_HASH}}"
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
# The reserve (9 GiB, upstream's since five full windows need 23.9 GiB of
# pool) is above the watchdog's floor (6), so the engine stops growing the
# pool before the watchdog would stop the server.

TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-9}"
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

# Native window 262,144, and also the maximum: the Flash Next CUDA engine of
# TensorFold 1.2.7 refuses a scaled rope, so there is no YaRN extension.
_ENV_PARALLEL="${PARALLEL:-}"
_ENV_CONTEXT="${CONTEXT:-}"
_ENV_KV_DTYPE="${KV_DTYPE:-}"
NATIVE_CONTEXT=262144
# Prompt rows a pass. v1.2.7's default is 1,536; 3,072 prefills ~4% faster
# for 0.8 GiB of the prompt cache, and a stream decoding beside a prompt waits
# up to one pass for its token.
export TF_FLASHNEXT_CHUNK="${TF_FLASHNEXT_CHUNK:-3072}"
# MTP: at most this many drafts a round. Drafts are checked, so replies match
# serial decoding.
export TF_FLASHNEXT_DRAFTS="${TF_FLASHNEXT_DRAFTS:-6}"
# Already v1.2.7's defaults, set so the environment can turn them off; none
# changes a reply's tokens: sampled streams in the batched draft pass, pending
# DeltaNet rows, the shared expert on a side stream, lane-batched DeltaNet.
export TF_FLASHNEXT_SAMPLED_HEADS="${TF_FLASHNEXT_SAMPLED_HEADS:-1}"
export TF_FLASHNEXT_KEEP_LATE="${TF_FLASHNEXT_KEEP_LATE:-1}"
export TF_FLASHNEXT_SHARED_SIDE="${TF_FLASHNEXT_SHARED_SIDE:-1}"
export TF_FLASHNEXT_GDN_LANES="${TF_FLASHNEXT_GDN_LANES:-1}"
# Opt-in in the engine, on here (upstream's choice): the next prompt chunk's
# n-gram rows hinted to the page cache during this one, and NVFP4 prompt
# experts on shared-memory K32 tiles (2: gate/up and down; 1: gate/up; 0 off).
export TF_FLASHNEXT_WARM_AHEAD="${TF_FLASHNEXT_WARM_AHEAD:-1}"
export TF_FLASHNEXT_PROMPT_EXPERT_TILE="${TF_FLASHNEXT_PROMPT_EXPERT_TILE:-2}"
# Patch 0004: prompt passes run MXFP8 projections 1,024 columns or wider on a
# 128 x 128 tile with the block scales folded into the weights (+7-9% prefill).
# It changes a prompt's numbers within the exact kernels' rounding bound; each
# row's bits do not depend on its chunk. 0 keeps the exact kernels.
export TF_FLASHNEXT_PROMPT_MX128="${TF_FLASHNEXT_PROMPT_MX128:-1024}"
# The draft chain stops when the running product of its draft probabilities
# falls under 0.4. Beside other streams v1.2.7 would switch greedy chains to a
# 0.6 floor a draft; SHARED_DRAFTS=0 keeps the product rule there too. Only
# draft rows change, never the reply.
export TF_FLASHNEXT_CONFIDENCE="${TF_FLASHNEXT_CONFIDENCE:--0.4}"
export TF_FLASHNEXT_SHARED_DRAFTS="${TF_FLASHNEXT_SHARED_DRAFTS:-0}"
# The MTP draft head's rows in 4 bits (a fourth of the bytes a chained draft
# reads). Drafts are only proposals; bf16 for the exact head.
export TF_FLASHNEXT_DRAFT_HEAD="${TF_FLASHNEXT_DRAFT_HEAD:-affine4}"
# KV cache: fp8 is about 1.84x the pool of bf16 and lossy (~98.8% top-1
# agreement with a bf16 cache); bf16 is exact.
MAX_TOKENS="${MAX_TOKENS:-32768}"
THINKING="${THINKING:-1}"
DRAFTS="${DRAFTS:-1}"
# Qwen's thinking-mode sampling. A request's own values win.
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
# Image input: the engine decodes JPEG and WebP itself and runs the Qwen-VL
# tower on the GPU. VISION=0 is text only and returns the 2 GiB workspace to
# the KV budget.
VISION="${VISION:-1}"
VISION_URLS="${VISION_URLS:-0}"            # 1: also fetch public https:// media URLs (default: data URLs only)
# VIDEO=1: video parts (patch 0005). Puts libtfvideo.so on the loader's path;
# with VIDEO=0 it is off every path the server probes, so a video part is
# refused. Sampled at 2 frames a second, two frames a group; each group counts
# as one image against TENSORFOLD_MAX_IMAGES, so 50 holds ~50 s of footage a
# request, images included. Needs VISION=1.
VIDEO="${VIDEO:-0}"
# PDF=1: PDF document parts (Anthropic's document, OpenAI's file/input_file, a
# data:application/pdf image_url), rendered to page images by the engine's own
# PDFium in a worker per document. 32 MiB and 16 pages a document; pages
# share the image budget. Needs VISION=1.
PDF="${PDF:-0}"
MAX_VIDEOS="${MAX_VIDEOS:-4}"              # per request
export TENSORFOLD_MAX_IMAGES="${TENSORFOLD_MAX_IMAGES:-50}"
export TENSORFOLD_IMAGE_TOKENS="${TENSORFOLD_IMAGE_TOKENS:-16384}"
export TENSORFOLD_VIDEO_TOKENS="${TENSORFOLD_VIDEO_TOKENS:-16384}"     # a request's video rows (VIDEO=1), 64 to 262,144
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
#   fp8x5   5 x 262,144, fp8 KV   default, upstream's; five full windows fit the pool with the 9 GiB reserve
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
  [[ "${CONTEXT}" =~ ^[1-9][0-9]*$ && "${CONTEXT}" -le "${NATIVE_CONTEXT}" ]] \
    || { printf 'error: CONTEXT is a token count up to %s (v1.2.7 has no YaRN extension), not %s\n' "${NATIVE_CONTEXT}" "${CONTEXT}" >&2; return 2; }
  [[ "${PARALLEL}" =~ ^([1-9]|1[0-6])$ ]] \
    || { printf 'error: PARALLEL is 1 to 16, not %s\n' "${PARALLEL}" >&2; return 2; }
  [[ "${KV_DTYPE}" =~ ^(bf16|fp8)$ ]] \
    || { printf 'error: KV_DTYPE is bf16 or fp8, not %s\n' "${KV_DTYPE}" >&2; return 2; }
  local v
  for v in THINKING DRAFTS VISION VISION_URLS VIDEO PDF; do
    [[ "${!v}" =~ ^[01]$ ]] || { printf 'error: %s is 0 or 1, not %s\n' "${v}" "${!v}" >&2; return 2; }
  done
  [[ "${VIDEO}" != 1 || "${VISION}" == 1 ]] || { printf 'error: VIDEO=1 decodes through the vision path: needs VISION=1\n' >&2; return 2; }
  [[ "${PDF}" != 1 || "${VISION}" == 1 ]] || { printf 'error: PDF=1 renders pages through the vision path: needs VISION=1\n' >&2; return 2; }
  [[ "${TENSORFOLD_VIDEO_TOKENS}" =~ ^[0-9]+$ ]] && (( TENSORFOLD_VIDEO_TOKENS >= 64 && TENSORFOLD_VIDEO_TOKENS <= 262144 )) \
    || { printf 'error: TENSORFOLD_VIDEO_TOKENS is 64 to 262144, not %s\n' "${TENSORFOLD_VIDEO_TOKENS}" >&2; return 2; }
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
    --context "${CONTEXT}" --parallel "${PARALLEL}" --kv-cache "${KV_DTYPE}"
    --max-tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}" --top-p "${TOP_P}" --top-k "${TOP_K}"
  )
  if [[ "${THINKING}" == 1 ]]; then SERVE_ARGS+=(--thinking); else SERVE_ARGS+=(--no-thinking); fi
  [[ "${DRAFTS}" == 1 ]] || SERVE_ARGS+=(--no-drafts)
  if [[ "${VISION}" == 1 ]]; then
    SERVE_ARGS+=(--vision --vision-max-images "${TENSORFOLD_MAX_IMAGES}" --vision-max-videos "${MAX_VIDEOS}"
                 --vision-image-tokens "${TENSORFOLD_IMAGE_TOKENS}")
    [[ "${VISION_URLS}" == 1 ]] && SERVE_ARGS+=(--vision-urls)
    [[ "${PDF}" == 1 ]] && SERVE_ARGS+=(--vision-pdf)
    [[ "${VIDEO}" == 1 ]] && SERVE_ARGS+=(--vision-video-tokens "${TENSORFOLD_VIDEO_TOKENS}")
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
# verify_snapshot [manifest snapshot_dir]: the model's by default.
verify_snapshot() {
  local manifest="${1:-${MANIFEST}}" snapdir="${2:-${SNAPSHOT_DIR}}" stamp fingerprint
  stamp="${OUT_DIR}/verified-$(basename "${manifest}" .json)-$(basename "${snapdir}" | cut -c1-12).stamp"
  fingerprint="$(python3 - "${manifest}" "${snapdir}" <<'PYEOF'
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
    printf 'snapshot incomplete: %s\n' "${snapdir}" >&2
    return 1
  }
  python3 "${EXPERIMENT_DIR}/scripts/download_snapshot.py" \
    --manifest "${manifest}" --hf-home "${HF_HOME}" --verify-only >/dev/null || return 1
  mkdir -p "${OUT_DIR}"
  printf '%s' "${fingerprint}" >"${stamp}"
  printf 'verified %s files against the manifest\n' \
    "$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["files"]))' "${manifest}")"
}
