#!/usr/bin/env bash
#
# Shared configuration for the Inkling 2 Mini NVFP4 vLLM experiment.
# Sourced by download.sh / preflight.sh / start.sh / stop.sh / status.sh --
# this is the ONLY place the profile table and the storage paths are defined.
#
# Usage:  source profiles.sh          # paths + defaults only
#         select_profile standard     # additionally sets the profile vars
#
# Storage rule: every path is an env-overridable variable whose default is the
# tool's own standard location. See CONVENTIONS.md at the repo root. This is a
# DOCKER experiment, so each host-side tree is bind-mounted onto the path the
# tool defaults to inside the container, and no cache env var is set there.
# Upstream downloads with --local-dir to ~/models; here the weights go in the
# shared HF cache like every other recipe's.

set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_NAME="Inkling-2-Mini-NVFP4-vLLM"

# --- storage -----------------------------------------------------------------

HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
VLLM_CACHE_HOST="${VLLM_CACHE_HOST:-${HOME}/.cache/vllm}"
FLASHINFER_CACHE_HOST="${FLASHINFER_CACHE_HOST:-${HOME}/.cache/flashinfer}"
# The whole .triton tree, so triton's own sub-layout applies inside.
TRITON_CACHE_HOST="${TRITON_CACHE_HOST:-${HOME}/.triton}"

OUT_DIR="${OUT_DIR:-${XDG_STATE_HOME:-${HOME}/.local/state}/dgx-spark-recipes/${EXPERIMENT_NAME}}"

# --- checkpoint (pinned; see manifests/) -------------------------------------

# Upstream's tested revision, 1fb4134b ("upload model"). The checkpoint and the
# DSpark drafter (mtp.safetensors) are one repo, one manifest.
MODEL_ID="thinkingmachines/Inkling-2-Mini"
MODEL_MANIFEST="${EXPERIMENT_DIR}/manifests/model.json"

# --- runtime -----------------------------------------------------------------

# vllm/vllm-openai:inkling-2-mini as upstream tested it (built 2026-11-20,
# vLLM 0.1.dev25855+g7cb5dae22). The tag is mutable, so pin the digest.
IMAGE="${IMAGE:-vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e}"
CONTAINER_NAME="${CONTAINER_NAME:-inkling-2-mini}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8015}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-inkling-2-mini}"
# "no", as upstream: after a memory kill on GB10 nothing should relaunch the
# load behind your back.
RESTART_POLICY="${RESTART_POLICY:-no}"

# DSpark: num_speculative_tokens must be 7 (checkpoint block size 8 minus 1,
# enforced by vLLM). 0 turns speculative decoding off.
SPEC_TOKENS="${SPEC_TOKENS:-7}"

# Memory. MEM_HEADROOM_BYTES is demanded on top of vLLM's budget (standard).
# For the max tiers, HOST_FLOOR_GIB is the MemAvailable that must be left after
# warmup by the estimate below; KILL_FLOOR_GIB is where start.sh's watch kills
# the container while it loads (upstream's watch_start.sh floor).
MEM_HEADROOM_BYTES="${MEM_HEADROOM_BYTES:-6442450944}"   # 6 GiB
HOST_FLOOR_GIB="${HOST_FLOOR_GIB:-6}"
KILL_FLOOR_GIB="${KILL_FLOOR_GIB:-3}"
DISK_RESERVE_BYTES="${DISK_RESERVE_BYTES:-10737418240}"  # 10 GiB

ACTIVE_PROFILE_FILE="${OUT_DIR}/active-profile"
LOG_FILE="${OUT_DIR}/events.log"

DEFAULT_PROFILE="standard"

# select_profile NAME -- sets the per-profile knobs. Upstream's three tiers:
#   standard   262,144 window, --gpu-memory-utilization 0.80, BF16 KV sized by
#              vLLM after warmup. ~20 GiB MemAvailable left. The default.
#   max        1,048,576 window, BF16 KV pool fixed at KV_GIB (80). ~7 GiB left.
#   max-fp8    1,048,576 window, FP8 (E4M3, unit-scale) KV at KV_GIB (74).
#              ~8 GiB left; 3.3M-token pool. FP8 KV splits the cache into 99
#              groups whose buffers cost ~5 GiB outside the pool, hence 74.
# KV_CAP_GIB is upstream's measured ceiling; preflight refuses KV_GIB above it.
# OVERHEAD_GIB is everything but the KV at 1M (weights 20.1, graphs ~3, the
# rest) as implied by upstream's MemAvailable figures on a box starting near
# 118 GiB free -- an estimate, to be corrected against a measured start.
select_profile() {
  PROFILE="${1:-${DEFAULT_PROFILE}}"
  case "${PROFILE}" in
    standard)
      MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
      GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
      KV_CACHE_DTYPE=auto; KV_GIB=""; KV_CAP_GIB=""; OVERHEAD_GIB=""
      ;;
    max)
      MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
      GPU_MEMORY_UTILIZATION=""
      KV_CACHE_DTYPE=auto; KV_GIB="${KV_GIB:-80}"; KV_CAP_GIB=80; OVERHEAD_GIB=31
      ;;
    max-fp8)
      MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
      GPU_MEMORY_UTILIZATION=""
      KV_CACHE_DTYPE=fp8; KV_GIB="${KV_GIB:-74}"; KV_CAP_GIB=74; OVERHEAD_GIB=36
      ;;
    *)
      printf 'error: unknown profile %q (standard|max|max-fp8)\n' "${PROFILE}" >&2
      return 1
      ;;
  esac
  export PROFILE MAX_MODEL_LEN GPU_MEMORY_UTILIZATION KV_CACHE_DTYPE KV_GIB KV_CAP_GIB OVERHEAD_GIB
}

# --- shared helpers ----------------------------------------------------------

GIB=1073741824
mem_available_bytes() { awk '/^MemAvailable:/ {print $2 * 1024; exit}' /proc/meminfo; }
mem_total_bytes()     { awk '/^MemTotal:/ {print $2 * 1024; exit}' /proc/meminfo; }

human_bytes() {
  local n="$1" sign=''
  if [[ "${n}" == -* ]]; then sign='-'; n="${n#-}"; fi
  printf '%s%s' "${sign}" "$(numfmt --to=iec-i --round=nearest --format='%.1f' --suffix=B "${n}" 2>/dev/null || printf '%s' "${n}")"
}

log_event() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

require_aarch64() {
  [[ "$(uname -m)" == "aarch64" ]] || {
    printf 'error: this recipe requires aarch64 GB10 (found %s)\n' "$(uname -m)" >&2
    return 1
  }
}

container_running() { docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; }
container_exists()  { docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; }

# manifest_field MANIFEST KEY -- one top-level field of a manifest.
manifest_field() {
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}

# snapshot_dir MANIFEST -- where the pinned revision sits in the HF cache.
snapshot_dir() {
  local repo revision
  repo="$(manifest_field "$1" repo)"
  revision="$(manifest_field "$1" revision)"
  printf '%s/hub/models--%s/snapshots/%s' "${HF_HOME}" "${repo//\//--}" "${revision}"
}

# Checksum-verify a manifest's files against the HF cache snapshot, skipping the
# slow re-hash when nothing has changed since the last pass (path, size, mtime).
# FORCE_VERIFY=1 always re-hashes.
verify_manifest() {
  local manifest="$1" stamp fingerprint
  stamp="${OUT_DIR}/verified-$(basename "${manifest}" .json).stamp"
  fingerprint="$(python3 - "${manifest}" "$(snapshot_dir "${manifest}")" <<'PYEOF'
import json, os, sys
manifest, root = sys.argv[1], sys.argv[2]
rows = []
for item in json.load(open(manifest))["files"]:
    try:
        st = os.stat(os.path.join(root, item["path"]))
    except FileNotFoundError:
        print("MISSING"); raise SystemExit(0)
    rows.append(f'{item["path"]}:{st.st_size}:{int(st.st_mtime)}')
print("|".join(rows))
PYEOF
)"
  if [[ "${FORCE_VERIFY:-0}" != "1" && "${fingerprint}" != "MISSING" \
        && -f "${stamp}" && "$(cat "${stamp}")" == "${fingerprint}" ]]; then
    return 0
  fi
  HF_HOME="${HF_HOME}" python3 "${EXPERIMENT_DIR}/scripts/download_snapshot.py" \
    --manifest "${manifest}" --hf-home "${HF_HOME}" --verify-only
  mkdir -p "${OUT_DIR}"
  printf '%s' "${fingerprint}" >"${stamp}"
}

# HF_TOKEN if exported, else the token file "hf auth login" writes. The repo is
# public, so empty is a note, not an error.
resolve_hf_token() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    local token_file="${HF_TOKEN_PATH:-${HF_HOME}/token}"
    [[ -r "${token_file}" ]] && HF_TOKEN="$(tr -d '[:space:]' <"${token_file}")"
  fi
  export HF_TOKEN="${HF_TOKEN:-}"
}

# memory_need_bytes -- MemAvailable start.sh needs now, per profile.
#   standard: vLLM refuses unless gmu x MemTotal is free; add the headroom.
#   max*:     KV + OVERHEAD_GIB + HOST_FLOOR_GIB.
memory_need_bytes() {
  if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
    python3 -c "print(int(${GPU_MEMORY_UTILIZATION} * $(mem_total_bytes)) + ${MEM_HEADROOM_BYTES})"
  else
    printf '%s' $(( (KV_GIB + OVERHEAD_GIB + HOST_FLOOR_GIB) * GIB ))
  fi
}
