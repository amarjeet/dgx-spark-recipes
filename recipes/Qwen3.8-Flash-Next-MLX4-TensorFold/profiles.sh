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

# The patches and the serve flags are for TensorFold v0.3.6.2 exactly.
TF_VERSION="${TF_VERSION:-v0.3.6.2}"
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
# The release tag's commit, so a moved tag is caught rather than followed.
TF_COMMIT="${TF_COMMIT:-71377a5373ed7b394f1b480ba2a6a3986b03af1c}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
# sha256 of patches/*.patch in name order, first 12 hex digits: the tag suffix
# and the image's tf.patches label. preflight.sh recomputes it.
PATCHES_HASH_PINNED="82e893ed2bcc"
# Upstream's prebuilt image for exactly these patches, pinned by digest: a tag
# can be re-pushed, a digest cannot. build.sh pulls it and tags it as IMAGE,
# or with PULL=0 builds the same thing locally from BASE_IMAGE.
GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold}"
PREBUILT_DIGEST="${PREBUILT_DIGEST:-sha256:d7d76f866137463937929d46b34426712095163e7404a81bc2c8a544f9d142a6}"
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
# TensorFold sizes itself: its budget is MemAvailable minus max(4 GiB, a tenth
# of MemTotal), read at start (tensorfold/cuda/capacity.py), and it refuses a
# setting that does not fit before any weights load. There is no knob for
# that reserve. On this host a tenth is 12.17 GiB, so an idle Spark gives a
# ~103-104 GiB budget and the default profile's 102.6 GiB estimate leaves the
# host roughly its reserve and no more. Upstream measured the host keeping at
# least 9.7 GiB free through a 195k-token prompt and 5 concurrent requests.
#
# Two host-side guards back that up, because on unified memory an exhausted
# pool hangs the kernel -- no OOM, no logs:
#   * a cgroup cap on the container. GPU allocations are not charged to it on
#     GB10, so it bounds the host-side footprint (Python, read buffers, page
#     cache the container faults in), not the weights;
#   * scripts/memwatch.sh, which stops the container when MemAvailable or
#     MemFree stays under a floor.

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
# 4,096-row prompt chunks (patch 0007): prefill +2-5% from 3k tokens, +0.94 GiB.
TENSORFOLD_PREFILL_ROWS="${TENSORFOLD_PREFILL_ROWS:-4096}"
# Prompt-lookup drafts ahead of MTP (patch 0008; needs >= 2 streams).
TENSORFOLD_MTP_COPY="${TENSORFOLD_MTP_COPY:-1}"
TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"
export TENSORFOLD_PREFILL_ROWS TENSORFOLD_MTP_COPY TENSORFOLD_NO_UPDATE_CHECK

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"

# --- profile table -----------------------------------------------------------
#
# One checkpoint; the profiles are KV-pool trade-offs. TensorFold gives every
# stream a full window, so the pool is streams x window. EST_GIB is
# TensorFold's own startup estimate for the profile as upstream reported it;
# the server prints the real one at every start ("startup estimate ...").
#
#   int8x4   4 x 262,144, int8 KV   1,048,576 tokens    97.8 GiB   default here
#   int8x5   5 x 262,144, int8 KV   1,310,720 tokens   102.6 GiB   upstream's default
#   int4x6   6 x 262,144, int4 KV   1,572,864 tokens    97.7 GiB   int4 changes output; quality unmeasured
#   bf16x3   3 x 262,144, bf16 KV     786,432 tokens   102.1 GiB   full-precision KV
#
# int8 and int4 KV use one fp16 scale per 32 values and change the output
# slightly against bf16; upstream measured int8 only.

# int8x4, not upstream's int8x5. Measured on this host: int8x5 loads with
# MemAvailable down to 7.7 GiB and MemFree under 2 GiB for three consecutive
# watchdog samples of five -- within a gigabyte of where the NVIDIA driver
# starts refusing allocations. int8x4 bottoms out at 17.9 GiB with no low
# sample at all, for one fewer stream. int8x5 is one word away.
DEFAULT_PROFILE="${DEFAULT_PROFILE:-int8x4}"
KNOWN_PROFILES=(int8x5 int8x4 int4x6 bf16x3 int8x1)

_ENV_PARALLEL="${PARALLEL:-}"
_ENV_CONTEXT="${CONTEXT:-}"
_ENV_KV_DTYPE="${KV_DTYPE:-}"

select_profile() {
  PROFILE="${1:-${DEFAULT_PROFILE}}"
  case "${PROFILE}" in
    int8x5) PARALLEL=5; KV_DTYPE=int8; EST_GIB=102.6 ;;
    int8x4) PARALLEL=4; KV_DTYPE=int8; EST_GIB=97.8 ;;
    int4x6) PARALLEL=6; KV_DTYPE=int4; EST_GIB=97.7 ;;
    bf16x3) PARALLEL=3; KV_DTYPE=bf16; EST_GIB=102.1 ;;
    # One full-window stream, to share the pool with a second server (the
    # Clef-Flash recipe). Measured: host MemAvailable 33.7 GiB once loaded.
    int8x1) PARALLEL=1; KV_DTYPE=int8; EST_GIB=88.18 ;;
    *)
      printf 'error: unknown profile: %s (expected one of: %s)\n' \
        "${PROFILE}" "${KNOWN_PROFILES[*]}" >&2
      return 2
      ;;
  esac
  CONTEXT="${_ENV_CONTEXT:-262144}"
  PARALLEL="${_ENV_PARALLEL:-${PARALLEL}}"
  KV_DTYPE="${_ENV_KV_DTYPE:-${KV_DTYPE}}"
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
    --reasoning-effort "${REASONING_EFFORT}"
  )
  [[ "${PLE_ON_SSD}" == 1 ]] && SERVE_ARGS+=(--ple-on-ssd)
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

# TensorFold's host reserve: max(4 GiB, MemTotal / 10).
tf_reserve_bytes() {
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
