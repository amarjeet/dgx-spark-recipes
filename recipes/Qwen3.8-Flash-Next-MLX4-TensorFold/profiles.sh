#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 MiaAI-Lab (original)
# Copyright (c) 2026 amarjeet (port)
#
# Derived from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
# (scripts/config.sh), MIT. Carries its knob names, defaults and measured
# constants; replaces its per-recipe kernel cache with the tools' own defaults.
# The shared helpers at the bottom follow this repo's MIT llama.cpp recipes.
#
# Shared configuration for Qwen3.8-Flash-Next (MLX 4-bit) on TensorFold.
# Sourced by build.sh / download.sh / preflight.sh / start.sh / stop.sh /
# status.sh / bench.sh -- the ONLY place paths, the profile table and helpers
# are defined.
#
# Usage:  source profiles.sh          # paths + defaults only
#         select_profile int8x5       # additionally sets the profile vars
#
# Storage rule: every path is an env-overridable variable whose default is the
# tool's own standard location (CONVENTIONS.md). This is a DOCKER recipe, so
# each host cache is bind-mounted onto the tool's in-container default.

set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Pinned rather than derived from the directory name, so OUT_DIR does not move
# if the recipe is cloned or renamed.
EXPERIMENT_NAME="Qwen3.8-Flash-Next-MLX4-TensorFold"

# --- storage -----------------------------------------------------------------

HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
# Compiled kernels. Upstream keeps both under a recipe-private
# ~/.cache/tensorfold-qwen38 -- the pattern CONVENTIONS.md exists to prevent,
# since triton's cache is content-hashed and torch's extension cache is keyed
# by python/CUDA version, so both are safe to share. The image sets
# TRITON_CACHE_DIR and TORCH_EXTENSIONS_DIR to /cache/..., so start.sh unsets
# them and each tool falls back to its default, which is the mount.
TRITON_CACHE_HOST="${TRITON_CACHE_HOST:-${HOME}/.triton}"
TORCH_EXTENSIONS_HOST="${TORCH_EXTENSIONS_HOST:-${HOME}/.cache/torch_extensions}"
# TensorFold's own cache (update check, prefix snapshots), at its default.
TENSORFOLD_CACHE_HOST="${TENSORFOLD_CACHE_HOST:-${HOME}/.cache/tensorfold}"

OUT_DIR="${OUT_DIR:-${XDG_STATE_HOME:-${HOME}/.local/state}/dgx-spark-recipes/${EXPERIMENT_NAME}}"

# --- model -------------------------------------------------------------------

MODEL_ID="${MODEL_ID:-Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP}"
# Pinned. Upstream resolves the repo id, which serves the newest snapshot on
# disk; start.sh passes this revision's snapshot directory instead.
MODEL_REVISION="${MODEL_REVISION:-dadefa8066e3be900a0d148d0f5a2f4eb1cf6534}"
MANIFEST="${MANIFEST:-${EXPERIMENT_DIR}/manifests/mlx4.json}"
MODEL_PATH="${HF_HOME}/hub/models--${MODEL_ID//\//--}"
SNAPSHOT_DIR="${MODEL_PATH}/snapshots/${MODEL_REVISION}"
# The same snapshot as the container sees it through the HF_HOME mount.
SNAPSHOT_CTR="/root/.cache/huggingface/hub/models--${MODEL_ID//\//--}/snapshots/${MODEL_REVISION}"

# The id the vLLM recipe for this model serves, so a client can move between
# the two unchanged. Upstream's id is answered too, as an alias.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next}"
SERVED_ALIAS="${SERVED_ALIAS:-Qwen3.8-Flash-Next}"

# --- runtime -----------------------------------------------------------------

# The patch and the serve flags are for TensorFold v0.6.1 exactly.
TF_VERSION="${TF_VERSION:-v0.6.1}"
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
# The release tag's commit, so a moved tag is caught rather than followed.
TF_COMMIT="${TF_COMMIT:-17c73e189f5e6a5304cda7ea37f086f9c49b4788}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
# sha256 of patches/*.patch in name order, first 12 hex digits: the tag suffix
# and the image's tf.patches label. preflight.sh recomputes it.
PATCHES_HASH_PINNED="797af1d9df4d"
# Upstream's prebuilt image for exactly this patch, pinned by digest: a tag
# can be re-pushed, a digest cannot. build.sh pulls it and tags it as IMAGE,
# or with PULL=0 builds the same thing locally from BASE_IMAGE.
GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold}"
PREBUILT_DIGEST="${PREBUILT_DIGEST:-sha256:667693ccd7eea95d365ee5a17eed97067e13deb131a6e61fbbbbe6f69d0ebc69}"
# The tag follows patches/, so an edited patch set is a different image.
PATCHES_HASH="$(cat "${EXPERIMENT_DIR}"/patches/*.patch | sha256sum | cut -c1-12)"
IMAGE="${IMAGE:-tensorfold-qwen38:${TF_VERSION}-${PATCHES_HASH}}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen3.8-flash-next-tensorfold}"

HOST="${HOST:-0.0.0.0}"
# Upstream uses 8888, which the vLLM recipe for this model already claims.
PORT="${PORT:-8011}"
RESTART_POLICY="${RESTART_POLICY:-no}"

# --- memory ------------------------------------------------------------------
#
# TensorFold sizes itself, from MemAvailable read once at start minus a
# reserve (tensorfold/cuda/capacity.py): TENSORFOLD_MEMORY_RESERVE_GIB, or
# max(4 GiB, a tenth of MemTotal) when unset -- 12.17 GiB on this host. It
# refuses a setting whose full windows do not fit before any weights load.
#
# Since v0.6 the KV caches are not allocated whole at load. Each stream's
# slot starts small and grows with its context, and before each growth a
# gate asks the live MemAvailable for the growth plus max(2 GiB, the vision
# workspace) on top of the same reserve. So the reserve is also the floor
# the server grows down to under traffic. A lone stream grows past the gate
# to finish its reply (multi.py: _grow(alone=True)); int8x1 always does.
#
# Upstream sets the reserve to 2 GiB, the knob's floor, to admit five full
# windows, and measured MemAvailable down to 2.9 GiB through five ~236k
# prompts. That is under this recipe's watchdog floors, so the watchdog would
# stop the server first. Left unset here (TensorFold's own 12.17 GiB) except
# where a profile needs less; see select_profile and README "Memory".
#
# Two host-side guards back that up, because on unified memory an exhausted
# pool hangs the kernel -- no OOM, no logs:
#   * a cgroup cap on the container. GPU allocations are not charged to it on
#     GB10, so it bounds the host-side footprint (Python, read buffers, page
#     cache the container faults in), not the weights;
#   * scripts/memwatch.sh, which stops the container when MemAvailable or
#     MemFree stays under a floor.

_ENV_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-}"

# Host-side cgroup cap, GiB. Measured: see README "Memory".
CONTAINER_MEM_GIB="${CONTAINER_MEM_GIB:-24}"
MEMWATCH_MIN_GIB="${MEMWATCH_MIN_GIB:-6}"
MEMWATCH_MIN_FREE_GIB="${MEMWATCH_MIN_FREE_GIB:-2}"
# MemFree only counts while MemAvailable is under this: MemFree sits near zero
# whenever the page cache is full of reclaimable data.
MEMWATCH_FREE_GATE_GIB="${MEMWATCH_FREE_GATE_GIB:-10}"
MEMWATCH_SAMPLES="${MEMWATCH_SAMPLES:-5}"
MEMWATCH_INTERVAL="${MEMWATCH_INTERVAL:-2}"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"

DISK_RESERVE_BYTES="${DISK_RESERVE_BYTES:-10737418240}"   # 10 GiB

# --- serving defaults (upstream's, measured there) ---------------------------

PLE_ON_SSD="${PLE_ON_SSD:-1}"      # 1: the 29.8 GiB n-gram tables are read from SSD, not held in RAM
# At most 6 MTP drafts a round, a chain stopping before a draft under 60%.
# Upstream swept these with identical output in every arm: 6/0.60 beat the
# stock 6/0.30 by ~3% on prose and ~4% on code.
MTP_DRAFTS="${MTP_DRAFTS:-6}"
MTP_CONFIDENCE="${MTP_CONFIDENCE:-0.60}"
# Qwen's thinking-mode sampling. A request's own values win. TensorFold has no
# min_p / presence / repetition penalty: those are always off.
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
THINKING="${THINKING:-1}"
# TensorFold's server default is medium, which adds no system-prompt text.
# Clients override it with chat_template_kwargs.reasoning_effort.
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
# Reply length for a request that sets no max_tokens. TensorFold's own 4,096
# can end a thinking reply before it answers (finish_reason "length", no
# content). Clamped to the room left in the stream's window and reserves no
# memory; a request's own max_tokens wins.
MAX_TOKENS="${MAX_TOKENS:-32768}"
# Image and video input: the checkpoint's own vision tower (0.84 GiB). Its
# ~0.8 GiB of scratch is borrowed while an image encodes, not reserved at
# start (TENSORFOLD_VISION_WORKSPACE_MIB=0). VISION=0: text only.
# The profile decides when unset: on, except int8x1 (TensorFold's Flash Next
# image input needs --parallel 2 or more).
_ENV_VISION="${VISION:-}"
VISION="${VISION:-1}"
VISION_URLS="${VISION_URLS:-0}"            # 1: also fetch https:// image/video URLs (default: data URLs only)
VISION_MAX_IMAGES="${VISION_MAX_IMAGES-50}"   # per request, a chat's turns all count; empty: TensorFold's 4
TENSORFOLD_VISION_WORKSPACE_MIB="${TENSORFOLD_VISION_WORKSPACE_MIB:-0}"
TENSORFOLD_IMAGE_TOKENS="${TENSORFOLD_IMAGE_TOKENS:-16384}"   # shared by a request's images, each at most 4,096
TENSORFOLD_VIDEO_TOKENS="${TENSORFOLD_VIDEO_TOKENS:-16384}"   # a whole video's budget
# Prompt piece rows. Unset, v0.6.1 picks 2,048 with vision and 4,096 without.
# With the n-gram tables on SSD upstream measured 4,096 10-30% slower from 5k
# to 16k tokens, so 2,048 is pinned there. Empty: TensorFold's choice.
if [[ "${PLE_ON_SSD}" == 1 ]]; then TENSORFOLD_PREFILL_ROWS="${TENSORFOLD_PREFILL_ROWS-2048}"
else TENSORFOLD_PREFILL_ROWS="${TENSORFOLD_PREFILL_ROWS:-}"; fi
# Prompt-lookup drafts ahead of MTP (patch 0002; needs >= 2 streams).
TENSORFOLD_MTP_COPY="${TENSORFOLD_MTP_COPY:-1}"
TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"
# The TENSORFOLD_* switches this file sets; start.sh passes each one that is
# non-empty, plus the profile's TENSORFOLD_MEMORY_RESERVE_GIB.
RECIPE_TF_VARS=(TENSORFOLD_PREFILL_ROWS TENSORFOLD_MTP_COPY TENSORFOLD_NO_UPDATE_CHECK
                TENSORFOLD_VISION_WORKSPACE_MIB TENSORFOLD_IMAGE_TOKENS TENSORFOLD_VIDEO_TOKENS)

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"

# --- profile table -----------------------------------------------------------
#
# One checkpoint; the profiles are KV-pool trade-offs. TensorFold gives every
# stream a full window, so the pool is streams x window. EST_GIB is
# TensorFold's own startup estimate, measured here on v0.6.1 (2026-10-02);
# the server prints the real one at every start ("startup estimate ...").
# Empty: not run here, and TensorFold's admission alone decides.
#
#   int8x5   5 x 262,144, int8 KV   1,310,720 tokens   85.39 GiB   default (upstream's too)
#   int8x4   4 x 262,144, int8 KV   1,048,576 tokens   85.18 GiB
#   int4x6   6 x 262,144, int4 KV   1,572,864 tokens   not run   int4 changes output; quality unmeasured
#   bf16x3   3 x 262,144, bf16 KV     786,432 tokens   not run   full-precision KV
#   int8x1   1 x 262,144, int8 KV     262,144 tokens   87.57 GiB   co-tenant; text only
#
# int8 and int4 KV use one fp16 scale per 32 values and change the output
# slightly against bf16; upstream measured int8 only.

# int8x5, upstream's. On v0.3.6.2 this recipe defaulted to int8x4, because
# int8x5 allocated every window at load and took the host to 7.7 GiB. On
# v0.6.1 the caches grow under the gate: five concurrent ~226k prompts all
# completed here with MemAvailable no lower than 14.4 GiB.
DEFAULT_PROFILE="${DEFAULT_PROFILE:-int8x5}"
KNOWN_PROFILES=(int8x5 int8x4 int4x6 bf16x3 int8x1)

_ENV_PARALLEL="${PARALLEL:-}"
_ENV_CONTEXT="${CONTEXT:-}"
_ENV_KV_DTYPE="${KV_DTYPE:-}"

select_profile() {
  PROFILE="${1:-${DEFAULT_PROFILE}}"
  RESERVE_GIB=""   # empty: TensorFold's own reserve
  case "${PROFILE}" in
    int8x5) PARALLEL=5; KV_DTYPE=int8; EST_GIB=85.39 ;;
    int8x4) PARALLEL=4; KV_DTYPE=int8; EST_GIB=85.18 ;;
    int4x6) PARALLEL=6; KV_DTYPE=int4; EST_GIB="" ;;
    bf16x3) PARALLEL=3; KV_DTYPE=bf16; EST_GIB="" ;;
    # One full-window stream, to share the pool with a second server (the
    # Clef-Flash recipe). A lone stream's window is allocated whole: measured
    # MemAvailable 33.0 GiB once loaded, unchanged after a 226k prompt.
    int8x1) PARALLEL=1; KV_DTYPE=int8; EST_GIB=87.57 ;;
    *)
      printf 'error: unknown profile: %s (expected one of: %s)\n' \
        "${PROFILE}" "${KNOWN_PROFILES[*]}" >&2
      return 2
      ;;
  esac
  CONTEXT="${_ENV_CONTEXT:-262144}"
  PARALLEL="${_ENV_PARALLEL:-${PARALLEL}}"
  KV_DTYPE="${_ENV_KV_DTYPE:-${KV_DTYPE}}"
  VISION="${_ENV_VISION:-$(( PARALLEL >= 2 ))}"
  # What start.sh passes as TENSORFOLD_MEMORY_RESERVE_GIB; the environment wins.
  MEMORY_RESERVE_GIB="${_ENV_MEMORY_RESERVE_GIB:-${RESERVE_GIB}}"
  # An override makes the table's estimate meaningless; TensorFold's own
  # admission is then the only estimate, and preflight says so.
  if [[ -n "${_ENV_PARALLEL}${_ENV_CONTEXT}${_ENV_KV_DTYPE}" ]]; then
    EST_GIB=""
  fi
  # Deliberately not exported: a child script (start.sh runs preflight.sh)
  # re-sources this file and would read its parent's profile values as
  # caller overrides.
}

# The tensorfold serve arguments for the selected profile. EXTRA_SERVE_ARGS go
# last, so they win (argparse keeps a flag's last value).
serve_args() {
  SERVE_ARGS=(
    --host 0.0.0.0 --port "${PORT}"
    --name "${SERVED_MODEL_NAME}"
  )
  [[ -n "${SERVED_ALIAS}" && "${SERVED_ALIAS}" != "${SERVED_MODEL_NAME}" ]] \
    && SERVE_ARGS+=(--alias "${SERVED_ALIAS}")
  SERVE_ARGS+=(
    --parallel "${PARALLEL}" --context "${CONTEXT}" --kv-dtype "${KV_DTYPE}"
    --mtp-drafts "${MTP_DRAFTS}" --mtp-confidence "${MTP_CONFIDENCE}"
    --temperature "${TEMPERATURE}" --top-p "${TOP_P}" --top-k "${TOP_K}"
    --reasoning-effort "${REASONING_EFFORT}" --max-tokens "${MAX_TOKENS}"
  )
  [[ "${PLE_ON_SSD}" == 1 ]] && SERVE_ARGS+=(--ple-on-ssd)
  if [[ "${VISION}" == 1 ]]; then
    SERVE_ARGS+=(--vision)
    [[ "${VISION_URLS}" == 1 ]] && SERVE_ARGS+=(--vision-urls)
    [[ -n "${VISION_MAX_IMAGES}" ]] && SERVE_ARGS+=(--vision-max-images "${VISION_MAX_IMAGES}")
  fi
  if [[ "${THINKING}" == 1 ]]; then SERVE_ARGS+=(--thinking); else SERVE_ARGS+=(--no-thinking); fi
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

# TensorFold's startup reserve: MEMORY_RESERVE_GIB when set (select_profile),
# else max(4 GiB, MemTotal / 10).
tf_reserve_bytes() {
  if [[ -n "${MEMORY_RESERVE_GIB:-}" ]]; then gib_to_bytes "${MEMORY_RESERVE_GIB}"; return; fi
  local total; total="$(mem_total_bytes)"
  (( total / 10 > 4 * GIB )) && printf '%s' "$(( total / 10 ))" || printf '%s' "$(( 4 * GIB ))"
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

image_patches_label() {
  docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "${IMAGE}" 2>/dev/null || true
}
image_tf_version() {
  docker run --rm --entrypoint python "${IMAGE}" \
    -c 'import tensorfold; print(tensorfold.__version__)' 2>/dev/null | tail -1 | tr -d '[:space:]'
}

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
# Hashing 105 GiB costs minutes; FORCE_VERIFY=1 always re-hashes.
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
