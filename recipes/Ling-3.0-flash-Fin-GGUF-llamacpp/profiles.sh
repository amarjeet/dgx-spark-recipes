#!/usr/bin/env bash
#
# Shared configuration for the Ling-3.0-flash-Fin llama.cpp experiment.
# Sourced by download.sh / preflight.sh / start.sh / stop.sh / status.sh /
# bench.sh -- this is the ONLY place the profile table and the storage paths
# are defined.
#
# Usage:  source profiles.sh          # paths + defaults only
#         select_profile iq4xs        # additionally sets the profile vars
#
# Storage rule: every path is an env-overridable variable whose default is the
# tool's own standard location. See CONVENTIONS.md at the repo root.
# This is a DOCKER experiment, so the host-side GGUF store is bind-mounted onto
# llama.cpp's in-container default (/root/.cache/llama.cpp) and no cache env
# var has to be set inside the container.

set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Pinned rather than derived from the directory name, so OUT_DIR does not move
# if the recipe is cloned or renamed.
EXPERIMENT_NAME="Ling-3.0-flash-Fin-GGUF-llamacpp"

# --- storage -----------------------------------------------------------------

# GGUF weights live in llama.cpp's own standard cache (the env var llama.cpp
# actually reads is LLAMA_CACHE, default ~/.cache/llama.cpp), so every
# llama.cpp recipe on the host shares one store.
LLAMA_CACHE="${LLAMA_CACHE:-${HOME}/.cache/llama.cpp}"
MODEL_STORE="${MODEL_STORE:-${LLAMA_CACHE}/Ling-3.0-flash-Fin-GGUF}"

# Bench results and verification stamps -- never in the experiment dir.
OUT_DIR="${OUT_DIR:-${XDG_STATE_HOME:-${HOME}/.local/state}/dgx-spark-recipes/${EXPERIMENT_NAME}}"

# --- runtime -----------------------------------------------------------------

# No source build needed: bailingmoe3 landed upstream before the currently
# published aarch64 image was cut. The tag is mutable, so preflight.sh asserts
# the build number rather than trusting it.
IMAGE="${IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}"
CONTAINER_NAME="${CONTAINER_NAME:-ling-3.0-flash-fin-gguf}"

# Pick a port nothing else on the host is using; preflight.sh checks.
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8008}"

# Minimum llama.cpp build that knows the bailingmoe3 architecture (bartowski's
# model card pins b10776).
MIN_LLAMA_BUILD="${MIN_LLAMA_BUILD:-10776}"

# Disk reserve demanded on top of the remaining download.
DISK_RESERVE_BYTES="${DISK_RESERVE_BYTES:-10737418240}"  # 10 GiB

# Free-memory floor preflight.sh insists on before a load is attempted. The
# weights are mlock'd, so a load that does not fit does not degrade -- it OOMs
# or drives the box into swap.
MEM_HEADROOM_BYTES="${MEM_HEADROOM_BYTES:-6442450944}"   # 6 GiB

PID_FILE="${EXPERIMENT_DIR}/.llama.pid"
LOG_FILE="${EXPERIMENT_DIR}/.llama.log"
ACTIVE_PROFILE_FILE="${EXPERIMENT_DIR}/.profile.active"

# Snapshot caller-supplied overrides at source time. select_profile() must read
# these, not the live variables: it exports CTX_SIZE etc., so a second call in
# the same shell would otherwise see its own previous values via "${VAR:-...}"
# and silently keep the first profile's settings.
_ENV_CTX_SIZE="${CTX_SIZE:-}"
_ENV_PARALLEL="${PARALLEL:-}"
_ENV_BATCH_SIZE="${BATCH_SIZE:-}"
_ENV_UBATCH_SIZE="${UBATCH_SIZE:-}"
_ENV_MODEL_ROOT="${MODEL_ROOT:-}"
_ENV_SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"

# Sampling defaults are the model card's recommendation for general inference.
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"

# Docker restart policy. unless-stopped brings the server back after a docker
# daemon restart or a reboot, but honours a deliberate ./stop.sh (which removes
# the container outright). Set RESTART_POLICY=no for a one-off run.
RESTART_POLICY="${RESTART_POLICY:-unless-stopped}"

# --mlock keeps the weights off swap. Set MLOCK=0 if you are
# deliberately oversubscribing.
MLOCK="${MLOCK:-1}"

MODEL_REVISION="a792b6bd51d2cb47b5aa8687ff5dd4f2974acdb4"
MODEL_REVISION_SHORT="${MODEL_REVISION:0:12}"

# --- profile table -----------------------------------------------------------
#
# Ling-3.0-flash-Fin is a 124B-total / 5.1B-active bailingmoe3 MoE released
# only in BF16 (~248 GB), which does not fit the Spark's 121.7 GiB of unified
# memory -- hence GGUF rather than the NVFP4-on-SGLang pattern used elsewhere
# here. Sizes below are bartowski's imatrix quants at revision a792b6bd51d2.
#
#   q3kxl   Q3_K_XL   61.8 GB   headroom profile, 256K ctx or 4 slots
#   iq4xs   IQ4_XS    68.7 GB   default: best quality-per-byte at 4 bits
#   q4km    Q4_K_M    77.8 GB   quality profile, shorter default context
#
# Context defaults are deliberately below the model's 256K maximum. Ling-3.0
# is a HYBRID model (42 layers: KDA linear-attention layers plus a minority of
# full-attention layers), so its KV cache is far smaller than a dense 124B
# model's -- but exactly how llama.cpp materialises the full-attention KV is
# not something this recipe assumes. Read the real number off the
# "KV self size" line in .llama.log after the first load, then raise CTX_SIZE.
# See README.md "Context sizing".

DEFAULT_PROFILE="${DEFAULT_PROFILE:-iq4xs}"
KNOWN_PROFILES=(q3kxl iq4xs q4km)

select_profile() {
  PROFILE="${1:-${DEFAULT_PROFILE}}"

  case "${PROFILE}" in
    q3kxl)
      QUANT="Q3_K_XL"
      MANIFEST="${EXPERIMENT_DIR}/manifests/q3kxl.json"
      CTX_SIZE="${_ENV_CTX_SIZE:-262144}"
      PARALLEL="${_ENV_PARALLEL:-1}"
      BATCH_SIZE="${_ENV_BATCH_SIZE:-4096}"
      UBATCH_SIZE="${_ENV_UBATCH_SIZE:-2048}"
      SERVED_MODEL_NAME_DEFAULT="ling-3.0-flash-fin-q3-k-xl"
      ;;
    iq4xs)
      QUANT="IQ4_XS"
      MANIFEST="${EXPERIMENT_DIR}/manifests/iq4xs.json"
      # 262144 is the model's full context. Measured: 64.0 GiB weights plus
      # ~9 GiB overhead at 131072 on a 121.7 GiB box, so the headroom is real.
      CTX_SIZE="${_ENV_CTX_SIZE:-262144}"
      PARALLEL="${_ENV_PARALLEL:-1}"
      BATCH_SIZE="${_ENV_BATCH_SIZE:-4096}"
      UBATCH_SIZE="${_ENV_UBATCH_SIZE:-2048}"
      SERVED_MODEL_NAME_DEFAULT="ling-3.0-flash-fin-iq4-xs"
      ;;
    q4km)
      QUANT="Q4_K_M"
      MANIFEST="${EXPERIMENT_DIR}/manifests/q4km.json"
      CTX_SIZE="${_ENV_CTX_SIZE:-65536}"
      PARALLEL="${_ENV_PARALLEL:-1}"
      BATCH_SIZE="${_ENV_BATCH_SIZE:-4096}"
      UBATCH_SIZE="${_ENV_UBATCH_SIZE:-2048}"
      SERVED_MODEL_NAME_DEFAULT="ling-3.0-flash-fin-q4-k-m"
      ;;
    *)
      printf 'error: unknown profile: %s (expected one of: %s)\n' \
        "${PROFILE}" "${KNOWN_PROFILES[*]}" >&2
      return 2
      ;;
  esac

  MODEL_ROOT="${_ENV_MODEL_ROOT:-${MODEL_STORE}/${QUANT}-${MODEL_REVISION_SHORT}}"

  # First shard listed in the manifest is the entry point; llama.cpp finds the
  # siblings by name.
  MODEL_ENTRY_REL="$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["files"][0]["path"])' \
    "${MANIFEST}")"
  MODEL_FILE="${MODEL_ROOT}/${MODEL_ENTRY_REL}"
  MODEL_TOTAL_BYTES="$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["total_bytes"])' \
    "${MANIFEST}")"

  # Path as the container sees it: MODEL_STORE is bind-mounted onto
  # llama.cpp's in-container default cache, so only the prefix differs.
  MODEL_FILE_CONTAINER="/root/.cache/llama.cpp/Ling-3.0-flash-Fin-GGUF/${QUANT}-${MODEL_REVISION_SHORT}/${MODEL_ENTRY_REL}"

  # Without --alias the API model id is the full GGUF path, which every client
  # then has to send back as "model". Matches the --served-model-name
  # convention.
  SERVED_MODEL_NAME="${_ENV_SERVED_MODEL_NAME:-${SERVED_MODEL_NAME_DEFAULT}}"

  export PROFILE QUANT MANIFEST MODEL_ROOT MODEL_FILE MODEL_FILE_CONTAINER
  export MODEL_ENTRY_REL MODEL_TOTAL_BYTES SERVED_MODEL_NAME
  export CTX_SIZE PARALLEL BATCH_SIZE UBATCH_SIZE
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

# The llama.cpp build number inside IMAGE. Printed as "build NNNNN" by
# --version, which goes to stderr.
image_build_number() {
  docker run --rm --entrypoint /app/llama-server "${IMAGE}" --version 2>&1 \
    | sed -n 's/.*build \([0-9]\+\).*/\1/p' | head -1
}

# Checksum-verify a manifest's files, skipping the (slow) re-hash when nothing
# has changed since the last successful verification.
#
# Hashing 69 GB costs minutes of cold-cache disk reads, and paying it on every
# restart would make the experiment painful to iterate on. So: verify for real
# once, then stamp the shard identities (path, size, mtime) and re-verify only
# if that fingerprint moves. FORCE_VERIFY=1 always re-hashes.
verify_shards() {
  local manifest="$1" model_root="$2" label="${3:-shards}"
  local stamp fingerprint
  stamp="${OUT_DIR}/verified-$(basename "${manifest}" .json).stamp"

  fingerprint="$(python3 - "${manifest}" "${model_root}" <<'PYEOF'
import json, os, sys
manifest, root = sys.argv[1], sys.argv[2]
rows = []
for item in json.load(open(manifest))["files"]:
    path = os.path.join(root, item["path"])
    try:
        st = os.stat(path)
    except FileNotFoundError:
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

  python3 "${EXPERIMENT_DIR}/scripts/download_model.py" \
    --manifest "${manifest}" --destination "${model_root}" --verify-only
  mkdir -p "${OUT_DIR}"
  printf '%s' "${fingerprint}" >"${stamp}"
}

# HF_TOKEN if exported, else the token `hf auth login` writes to the standard
# location. The Hub is readable anonymously at a lower rate limit, so an empty
# token is a warning, not an error.
resolve_hf_token() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    local token_file="${HF_TOKEN_PATH:-${HF_HOME:-${HOME}/.cache/huggingface}/token}"
    [[ -r "${token_file}" ]] && HF_TOKEN="$(tr -d '[:space:]' <"${token_file}")"
  fi
  export HF_TOKEN="${HF_TOKEN:-}"
}
