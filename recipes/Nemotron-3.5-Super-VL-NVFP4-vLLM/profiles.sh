#!/usr/bin/env bash
#
# Shared configuration for the Nemotron-3.5-Super-VL NVFP4 vLLM experiment.
# Sourced by download.sh / preflight.sh / start.sh / stop.sh / status.sh --
# this is the ONLY place the profile table and the storage paths are defined.
#
# Usage:  source profiles.sh          # paths + defaults only
#         select_profile spark        # additionally sets the profile vars
#
# Storage rule: every path is an env-overridable variable whose default is the
# tool's own standard location. See CONVENTIONS.md at the repo root. This is a
# DOCKER experiment, so each host-side tree is bind-mounted onto the path the
# tool defaults to inside the container, and no cache env var is set there.

set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_NAME="Nemotron-3.5-Super-VL-NVFP4-vLLM"

# --- storage -----------------------------------------------------------------

HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
VLLM_CACHE_HOST="${VLLM_CACHE_HOST:-${HOME}/.cache/vllm}"
FLASHINFER_CACHE_HOST="${FLASHINFER_CACHE_HOST:-${HOME}/.cache/flashinfer}"
# The whole .triton tree, so triton's own sub-layout applies inside.
TRITON_CACHE_HOST="${TRITON_CACHE_HOST:-${HOME}/.triton}"

OUT_DIR="${OUT_DIR:-${XDG_STATE_HOME:-${HOME}/.local/state}/dgx-spark-recipes/${EXPERIMENT_NAME}}"

# --- checkpoints (pinned; see manifests/) ------------------------------------

MODEL_ID="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
MODEL_MANIFEST="${EXPERIMENT_DIR}/manifests/model.json"
# The dflash draft model: 5 files, 1.2 GiB. Pinned and verified like the target.
DRAFT_MODEL_ID="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr"
DRAFT_MANIFEST="${EXPERIMENT_DIR}/manifests/draft.json"

# --- runtime -----------------------------------------------------------------

# The nightly the model card names, by digest (multi-arch index; arm64 is
# 9.96 GiB). The card also lists v0.32.0, which was not on Docker Hub on
# 2026-10-16: the newest release tag then was v0.31.0.
IMAGE="${IMAGE:-vllm/vllm-openai@sha256:cb9bc64f0f038c2c2ebf13b31248fb87aca40c356c3d20f363be087cfcff1fcd}"
CONTAINER_NAME="${CONTAINER_NAME:-nemotron-3.5-super-vl}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8013}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nemotron-3.5-super-vl}"
RESTART_POLICY="${RESTART_POLICY:-unless-stopped}"

# vLLM sizes its budget as this fraction of total memory and refuses to start
# if that much is not free. On unified memory "free" is MemAvailable, so the
# fraction is also the memory this server takes from the host.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
# Free memory demanded on top of the budget, for the OS and the driver.
MEM_HEADROOM_BYTES="${MEM_HEADROOM_BYTES:-6442450944}"   # 6 GiB
DISK_RESERVE_BYTES="${DISK_RESERVE_BYTES:-10737418240}"  # 10 GiB

# Empty omits --max-model-len, as the card's Spark command does: the model's own
# window (1M tokens) applies. Few layers are attention, so the KV pool is small.
MAX_MODEL_LEN="${MAX_MODEL_LEN-}"

ACTIVE_PROFILE_FILE="${OUT_DIR}/active-profile"
LOG_FILE="${OUT_DIR}/events.log"

DEFAULT_PROFILE="spark"

# select_profile NAME -- sets the per-profile knobs.
#   spark    the model card's DGX Spark recipe: one stream, text + image,
#            dflash with the LiLiCorr draft at 7 speculative tokens.
#   nodraft  the same without speculative decoding, for a baseline and to
#            isolate a failure to the draft model. Skips the 1.2 GiB draft.
select_profile() {
  PROFILE="${1:-${DEFAULT_PROFILE}}"
  case "${PROFILE}" in
    spark)
      SPEC_TOKENS="${SPEC_TOKENS:-7}"
      ;;
    nodraft)
      SPEC_TOKENS=0
      ;;
    *)
      printf 'error: unknown profile %q (spark|nodraft)\n' "${PROFILE}" >&2
      return 1
      ;;
  esac
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
  export PROFILE SPEC_TOKENS MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS
}

# --- shared helpers ----------------------------------------------------------

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

# HF_TOKEN if exported, else the token `hf auth login` writes. Both repos are
# public, so empty is a note, not an error.
resolve_hf_token() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    local token_file="${HF_TOKEN_PATH:-${HF_HOME}/token}"
    [[ -r "${token_file}" ]] && HF_TOKEN="$(tr -d '[:space:]' <"${token_file}")"
  fi
  export HF_TOKEN="${HF_TOKEN:-}"
}
