#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
# Copyright (C) 2026 amarjeet
#
# Derived from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark, which is
# Copyright (C) 2026 MiaAI Lab and licensed AGPL-3.0-or-later.
# Modified 2026-09-05 by amarjeet: replaces upstream's .env; carries its knob
# names, defaults and measured constants
#
# Shared configuration for the Qwen3.8-Flash-Next NVFP4 vLLM experiment.
# Sourced by download.sh / preflight.sh / start.sh / stop.sh / status.sh /
# bench.sh -- this is the ONLY place the profile table and the storage paths
# are defined.
#
# Usage:  source profiles.sh          # paths + defaults only
#         select_profile native       # additionally sets the profile vars
#
# Adapted from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark, which drives the
# same checkpoint from a .env file. Everything that lived in .env lives here
# instead, which is what lets a second select_profile call in one shell be
# correct and removes upstream's environment/.env precedence dance.
#
# Storage rule: every path is an env-overridable variable whose default is the
# tool's own standard location. See CONVENTIONS.md at the repo root.
# This is a DOCKER experiment, so each host cache is bind-mounted onto the path
# the tool already defaults to inside the container and no cache env var has to
# be set in the container.

set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Pinned rather than derived from the directory name, so OUT_DIR does not move
# if the recipe is cloned or renamed.
EXPERIMENT_NAME="Qwen3.8-Flash-Next-NVFP4-vLLM"

# --- storage -----------------------------------------------------------------

# The checkpoint is a Hugging Face repo, so it lives in the HF cache -- the
# variable the HF libraries themselves read, and the tree vLLM resolves
# MODEL_ID against under HF_HUB_OFFLINE=1.
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

# vLLM's own cache root (the env var vLLM reads is VLLM_CACHE_ROOT, default
# ~/.cache/vllm). Holds the torch.compile cache and, for this model, the packed
# PLE table built on first launch. Both are reusable across recipes, so neither
# belongs in the experiment directory.
VLLM_CACHE_HOST="${VLLM_CACHE_HOST:-${HOME}/.cache/vllm}"
# flashinfer JIT/autotune and triton kernel caches, at their own defaults.
# Mount the whole ~/.triton tree, not ~/.triton/cache, so triton's sub-layout
# applies inside the container.
FLASHINFER_CACHE_HOST="${FLASHINFER_CACHE_HOST:-${HOME}/.cache/flashinfer}"
TRITON_CACHE_HOST="${TRITON_CACHE_HOST:-${HOME}/.triton}"

# Bench results, verification stamps and archived logs -- never in the
# experiment dir.
OUT_DIR="${OUT_DIR:-${XDG_STATE_HOME:-${HOME}/.local/state}/dgx-spark-recipes/${EXPERIMENT_NAME}}"

# --- model -------------------------------------------------------------------

MODEL_ID="${MODEL_ID:-Mia-AiLab/Qwen3.8-Flash-Next-NVFP4}"
# Pinned. Upstream takes `ls snapshots | head -1`, which silently serves
# whichever snapshot happens to be on disk; manifests/nvfp4.json records this
# revision's every shard size and SHA-256.
MODEL_REVISION="${MODEL_REVISION:-925d7be6c14c6c9442ef83e8f05b5a3c39304f69}"
MANIFEST="${MANIFEST:-${EXPERIMENT_DIR}/manifests/nvfp4.json}"

MODEL_ORG="${MODEL_ID%%/*}"
MODEL_NAME="${MODEL_ID##*/}"
# Where huggingface_hub puts it, and what start.sh mounts.
MODEL_PATH="${HF_HOME}/hub/models--${MODEL_ORG}--${MODEL_NAME}"
SNAPSHOT_DIR="${MODEL_PATH}/snapshots/${MODEL_REVISION}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next}"

# --- runtime -----------------------------------------------------------------

# The image carries the qwen3_8_flash_next model code and the PLE offload
# machinery; stock vllm-openai tags do not. Mutable tag, so preflight.sh
# asserts the vLLM version and the presence of the module paths start.sh
# patches rather than trusting it.
IMAGE="${IMAGE:-vllm/vllm-openai:qwen38-flash-next}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen3.8-flash-next-nvfp4}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"


DISK_RESERVE_BYTES="${DISK_RESERVE_BYTES:-10737418240}"   # 10 GiB

PID_FILE="${EXPERIMENT_DIR}/.vllm.pid"
LOG_FILE="${EXPERIMENT_DIR}/.vllm.log"
ACTIVE_PROFILE_FILE="${EXPERIMENT_DIR}/.profile.active"

# Snapshot caller-supplied overrides at source time. select_profile() must read
# these, not the live variables: it exports MAX_MODEL_LEN etc., so a second
# call in the same shell would otherwise see its own previous values via
# "${VAR:-...}" and silently keep the first profile's settings.
_ENV_MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
_ENV_YARN="${YARN:-}"
_ENV_KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
_ENV_MTP="${MTP_NUM_SPECULATIVE_TOKENS:-}"
_ENV_MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
_ENV_MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
_ENV_KV_TARGET_GIB="${KV_TARGET_GIB:-}"

# --- memory budget -----------------------------------------------------------
#
# The GPU budget is capped FROM THE HOST SIDE. On GB10 the GPU and CPU share
# one pool and vLLM reads MemAvailable (page cache included) as "free GPU
# memory", then fills the GPU side to exactly GMU x MemTotal. So an uncapped
# KV wish comes straight out of the PLE page cache and the free pages the
# NVIDIA driver needs -- which is how upstream lost three servers in a day.
# The cap is MemTotal - HOST_RESERVE_GIB and it wins over KV_TARGET_GIB.

# Covers, in order: other containers and sessions, vLLM's own host-side
# processes (~6 GiB), PLE page cache (>=6), the driver's free-page reserve
# (>=3), and 2-3 GiB of per-request growth that is never returned. Raise it in
# 2 GiB steps if the watchdog log shows MemAvailable idling under ~9 GiB. Do
# not lower it to buy KV.
HOST_RESERVE_GIB="${HOST_RESERVE_GIB:-26}"
# Host-side memory the container needs beyond the GPU budget: three Python
# processes, pinned staging buffers, CPU-side torch, page cache slack.
HOST_SLACK_GIB="${HOST_SLACK_GIB:-5}"
# The container cgroup cap may never come within this much of the pool.
OS_RESERVE_GIB="${OS_RESERVE_GIB:-16.0}"
# Runtime overhead on top of weights, GiB (measured upstream at TP=1:
# non-torch 3.37 + activation 1.92 + graphs 0.12).
OVERHEAD_GIB="${OVERHEAD_GIB:-5.6}"
# The PLE n-gram table, served memory-mapped by the CPU offload worker rather
# than resident on the GPU. Subtracted from the checkpoint size to get the GPU
# weight footprint.
PLE_GIB="${PLE_GIB:-26.82}"
# Measured KV cost at bf16 for this architecture: 28.8 KiB/token.
KV_BYTES_PER_TOKEN="${KV_BYTES_PER_TOKEN:-29482}"
# Optional hard pins. Empty means derived in start.sh step 2.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-}"
CONTAINER_MEM_GIB="${CONTAINER_MEM_GIB:-}"

# --- watchdog ----------------------------------------------------------------

# Stop the container if host MemAvailable stays below this (GiB) ...
MEMWATCH_MIN_GIB="${MEMWATCH_MIN_GIB:-6}"
# ... or MemFree stays below this. The NVIDIA driver refuses allocations
# (NV_ERR_NO_MEMORY) at MemFree ~3 GiB while MemAvailable still reads 6+.
MEMWATCH_MIN_FREE_GIB="${MEMWATCH_MIN_FREE_GIB:-2}"
# The MemFree floor only counts while MemAvailable is under this: with stock
# kernel watermarks MemFree sits near zero whenever the page cache is full of
# reclaimable data (measured: 0.9 GiB free, 32 GiB available, during load).
MEMWATCH_FREE_GATE_GIB="${MEMWATCH_FREE_GATE_GIB:-10}"
# Seconds the watchdog gives vLLM to exit on SIGTERM before SIGKILL.
MEMWATCH_GRACE="${MEMWATCH_GRACE:-30}"
# Seconds stop.sh gives vLLM to unlink its POSIX shared-memory segments.
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"

# --- serving -----------------------------------------------------------------

CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"   # NONE for eager debug
# PLE offload is not optional at TP=1: 98.6 GiB of weights through UVM hangs
# the host. start.sh refuses to disable it.
PLE_OFFLOAD="${PLE_OFFLOAD:-true}"
REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-true}"
RESTART_POLICY="${RESTART_POLICY:-no}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"

# text_config.max_position_embeddings. Anything longer needs YaRN.
NATIVE_MAX_MODEL_LEN="${NATIVE_MAX_MODEL_LEN:-262144}"
# Validated ceiling for a YaRN context on one Spark. 1M needs ~28.8 GiB of KV,
# which drives the cgroup cap past the pool; raise only after re-doing the
# start.sh step 2 arithmetic.
YARN_CEILING_MODEL_LEN="${YARN_CEILING_MODEL_LEN:-524288}"

# --- profile table -----------------------------------------------------------
#
# One checkpoint, one quantization -- so unlike the GGUF recipes here, the
# profiles are serving trade-offs rather than weight files. All three load the
# same 98.66 GiB of NVFP4 weights and differ only in what the KV pool costs:
#
#   native   262,144 ctx, FP8 KV    the model's full native context. Default.
#   yarn512  524,288 ctx, FP8 KV    YaRN rope scaling, 2x native
#   bf16kv   262,144 ctx, BF16 KV   quality-conservative: no quantized keys
#
# FP8 KV is a capacity trade, not a free win. It roughly doubles the KV pool
# (~1.85x here: the 12 full-attention layers halve, but the QSA side and
# compressor caches stay BF16), which is what makes a 512K context reachable.
# It also perturbs which blocks the sparse-attention indexer selects --
# upstream's reference implementation measured a long-reasoning benchmark
# falling from 6/6 to 2/6, though upstream's own run scored 11/11, same as
# BF16. Neither number was measured on this host. Use bf16kv if you care about
# long-reasoning quality more than context length.

DEFAULT_PROFILE="${DEFAULT_PROFILE:-native}"
KNOWN_PROFILES=(native yarn512 bf16kv)

select_profile() {
  PROFILE="${1:-${DEFAULT_PROFILE}}"

  case "${PROFILE}" in
    native)
      YARN="${_ENV_YARN:-0}"
      MAX_MODEL_LEN="${_ENV_MAX_MODEL_LEN:-262144}"
      KV_CACHE_DTYPE="${_ENV_KV_CACHE_DTYPE:-fp8}"
      ;;
    yarn512)
      YARN="${_ENV_YARN:-1}"
      MAX_MODEL_LEN="${_ENV_MAX_MODEL_LEN:-524288}"
      KV_CACHE_DTYPE="${_ENV_KV_CACHE_DTYPE:-fp8}"
      ;;
    bf16kv)
      YARN="${_ENV_YARN:-0}"
      MAX_MODEL_LEN="${_ENV_MAX_MODEL_LEN:-262144}"
      KV_CACHE_DTYPE="${_ENV_KV_CACHE_DTYPE:-bfloat16}"
      ;;
    *)
      printf 'error: unknown profile: %s (expected one of: %s)\n' \
        "${PROFILE}" "${KNOWN_PROFILES[*]}" >&2
      return 2
      ;;
  esac

  # Shared across profiles: 0 disables MTP self-speculation and gives back
  # 1.49 GiB. 3 is upstream's measured sweet spot.
  MTP_NUM_SPECULATIVE_TOKENS="${_ENV_MTP:-3}"
  MAX_NUM_SEQS="${_ENV_MAX_NUM_SEQS:-4}"
  # Prefill chunk width. Raising to 8192 buys ~11% prefill and ~10% off TTFT
  # upstream, paid for out of the KV pool (~3%).
  MAX_NUM_BATCHED_TOKENS="${_ENV_MAX_NUM_BATCHED_TOKENS:-2048}"
  # KV the derived budget aims for. Capped by HOST_RESERVE_GIB; the cap wins.
  # 16 sits just under the cap. 22 idled the host at 6.9-8.8 GiB MemAvailable
  # against a 6 GiB watchdog floor and the driver refused allocations there.
  KV_TARGET_GIB="${_ENV_KV_TARGET_GIB:-16}"

  MODEL_TOTAL_BYTES="$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["total_bytes"])' \
    "${MANIFEST}")"

  export PROFILE YARN MAX_MODEL_LEN KV_CACHE_DTYPE MTP_NUM_SPECULATIVE_TOKENS
  export MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS KV_TARGET_GIB MODEL_TOTAL_BYTES
}

# --- shared helpers ----------------------------------------------------------

mem_available_bytes() {
  awk '/^MemAvailable:/ {print $2 * 1024; exit}' /proc/meminfo
}

mem_total_bytes() {
  awk '/^MemTotal:/ {print $2 * 1024; exit}' /proc/meminfo
}

host_swap_used_bytes() {
  awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print (t - f) * 1024}' /proc/meminfo
}

human_bytes() {
  local n="$1" sign=''
  if [[ "${n}" == -* ]]; then sign='-'; n="${n#-}"; fi
  printf '%s%s' "${sign}" "$(numfmt --to=iec-i --round=nearest --format='%.1f' --suffix=B "${n}" 2>/dev/null || printf '%s' "${n}")"
}

log_event() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

require_aarch64() {
  [[ "$(uname -m)" == "aarch64" ]] || {
    printf 'error: this recipe requires aarch64 GB10 (found %s)\n' "$(uname -m)" >&2
    return 1
  }
}

container_running() {
  docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"
}

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"
}

# The vLLM version inside IMAGE. The tag is mutable and the model code, the PLE
# offload package and the QSA kernels all have to be present for start.sh's
# patches to apply, so preflight.sh checks the module paths too.
image_vllm_version() {
  docker run --rm --entrypoint python3 "${IMAGE}" \
    -c 'import vllm; print(vllm.__version__)' 2>/dev/null | tr -d '[:space:]'
}

# HF_TOKEN if exported, else the token `hf auth login` writes to the standard
# location. This checkpoint is public, so an empty token is not an error -- it
# only lowers the Hub rate limit.
resolve_hf_token() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    local token_file="${HF_TOKEN_PATH:-${HF_HOME}/token}"
    [[ -r "${token_file}" ]] && HF_TOKEN="$(tr -d '[:space:]' <"${token_file}")"
  fi
  export HF_TOKEN="${HF_TOKEN:-}"
}

# Checksum-verify the pinned snapshot, skipping the (slow) re-hash when nothing
# has changed since the last successful verification.
#
# Hashing 98.66 GiB costs minutes of cold-cache disk reads, and paying it on
# every restart would make the experiment painful to iterate on. So: verify for
# real once, then stamp the file identities (path, size, mtime) and re-verify
# only if that fingerprint moves. FORCE_VERIFY=1 always re-hashes.
#
# The HF cache stores blobs once and links snapshot entries to them, so this
# stats through the symlinks deliberately -- what matters is what vLLM opens.
verify_snapshot() {
  local label="${1:-snapshot}"
  local stamp fingerprint
  stamp="${OUT_DIR}/verified-$(basename "${MANIFEST}" .json).stamp"

  fingerprint="$(python3 - "${MANIFEST}" "${SNAPSHOT_DIR}" <<'PYEOF'
import json, os, sys
manifest, root = sys.argv[1], sys.argv[2]
rows = []
for item in json.load(open(manifest))["files"]:
    path = os.path.join(root, item["path"])
    try:
        st = os.stat(path)
    except (FileNotFoundError, OSError):
        print("MISSING")
        raise SystemExit(0)
    rows.append(f'{item["path"]}:{st.st_size}:{int(st.st_mtime)}')
print("|".join(rows))
PYEOF
)"

  if [[ "${FORCE_VERIFY:-0}" != "1" && "${fingerprint}" != "MISSING" \
        && -f "${stamp}" && "$(cat "${stamp}")" == "${fingerprint}" ]]; then
    printf 'already verified since last change: %s\n' "${label}"
    printf '(FORCE_VERIFY=1 to re-hash)\n'
    return 0
  fi

  HF_HOME="${HF_HOME}" python3 "${EXPERIMENT_DIR}/scripts/download_snapshot.py" \
    --manifest "${MANIFEST}" --hf-home "${HF_HOME}" --verify-only
  mkdir -p "${OUT_DIR}"
  printf '%s' "${fingerprint}" >"${stamp}"
}
