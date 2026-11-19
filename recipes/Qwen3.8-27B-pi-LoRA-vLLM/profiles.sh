#!/usr/bin/env bash
#
# Shared configuration for the Qwen3.8-27B + pi LoRA vLLM experiment.
# Sourced by download.sh / preflight.sh / start.sh / stop.sh / status.sh --
# this is the ONLY place the profile table and the storage paths are defined.
#
# Usage:  source profiles.sh          # paths + defaults only
#         select_profile pi           # additionally sets the profile vars
#
# Storage rule: every path is an env-overridable variable whose default is the
# tool's own standard location. See CONVENTIONS.md at the repo root. This is a
# DOCKER experiment, so each host-side tree is bind-mounted onto the path the
# tool defaults to inside the container, and no cache env var is set there.

set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_NAME="Qwen3.8-27B-pi-LoRA-vLLM"

# --- storage -----------------------------------------------------------------

HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
VLLM_CACHE_HOST="${VLLM_CACHE_HOST:-${HOME}/.cache/vllm}"
FLASHINFER_CACHE_HOST="${FLASHINFER_CACHE_HOST:-${HOME}/.cache/flashinfer}"
# The whole .triton tree, so triton's own sub-layout applies inside.
TRITON_CACHE_HOST="${TRITON_CACHE_HOST:-${HOME}/.triton}"

OUT_DIR="${OUT_DIR:-${XDG_STATE_HOME:-${HOME}/.local/state}/dgx-spark-recipes/${EXPERIMENT_NAME}}"

# Where HF_HOME is mounted inside the container (huggingface_hub's own default).
CONTAINER_HF_HOME="/root/.cache/huggingface"

# --- checkpoints (pinned; see manifests/) ------------------------------------

# The base, chosen per profile (select_profile sets MODEL_ID/MODEL_MANIFEST).
# vLLM runs with --language-model-only, so a vision tower is never loaded.
#   BF16:  the base the adapter was trained on. 55.6 GB including a vision
#          tower and an MTP head.
#   NVFP4: NVIDIA's ModelOpt MIXED_PRECISION export, 21.9 GB: NVFP4 MLPs and
#          lm_head, FP8 attention and linear-attention projections, MTP head
#          left in BF16. The LoRA stays unmerged on top of it, so the adapter's
#          small deltas are applied in BF16 rather than rounded into 4 bits.
BF16_MODEL_ID="Qwen/Qwen3.8-27B"
BF16_MANIFEST="${EXPERIMENT_DIR}/manifests/base.json"
NVFP4_MODEL_ID="nvidia/Qwen3.8-27B-NVFP4"
NVFP4_MANIFEST="${EXPERIMENT_DIR}/manifests/base-nvfp4.json"
MODEL_ID="${BF16_MODEL_ID}"
MODEL_MANIFEST="${BF16_MANIFEST}"
# The adapter: PEFT LoRA, r=32, alpha=64, F32, 638 MB. The model card's vLLM
# command names `harness-hub/Qwen3.8-27B-pi`, which does not exist (HTTP 401
# on 2026-11-18); this is the repo that does.
ADAPTER_ID="harness-hub/Qwen3.8-27B-pi-adapted"
ADAPTER_MANIFEST="${EXPERIMENT_DIR}/manifests/adapter.json"
ADAPTER_RANK=32

# --- runtime -----------------------------------------------------------------

# v0.26.0 (arm64), by digest. Read-only checks on 2026-11-18 found the
# Qwen3_5ForConditionalGeneration architecture with LoRA support, the
# model.language_model.* -> language_model.model.* weight mapper the adapter's
# keys need, and every flag start.sh passes.
IMAGE="${IMAGE:-vllm/vllm-openai@sha256:3016f449f5bf9bbe171406b7715822acb49026a4a404ba830f482575c02302d6}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen3.8-27b-pi}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8014}"
# The base model's name. The adapter is served beside it as LORA_NAME, the id
# pi is configured with in the model card.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b}"
LORA_NAME="${LORA_NAME:-qwen3.8-pi}"
RESTART_POLICY="${RESTART_POLICY:-unless-stopped}"

# vLLM sizes its budget as this fraction of total memory and refuses to start
# if that much is not free. On unified memory "free" is MemAvailable, so the
# fraction is also the memory this server takes from the host. 0.75 is about
# 91 GiB: ~51 GiB of BF16 weights, the rest KV, Mamba state and activations.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
# Free memory demanded on top of the budget, for the OS and the driver.
MEM_HEADROOM_BYTES="${MEM_HEADROOM_BYTES:-6442450944}"   # 6 GiB
DISK_RESERVE_BYTES="${DISK_RESERVE_BYTES:-10737418240}"  # 10 GiB

# The base model's native window. The adapter card evaluated at 131,072, so
# past that the adapter is untested; MAX_MODEL_LEN=131072 restores the card's
# setting. Only 16 of 64 layers keep a KV cache (4 KV heads x 256), so a full
# 262,144-token sequence costs about 16 GiB of BF16 KV: two fit in the pool.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"

ACTIVE_PROFILE_FILE="${OUT_DIR}/active-profile"
LOG_FILE="${OUT_DIR}/events.log"

DEFAULT_PROFILE="pi"

# select_profile NAME -- sets the per-profile knobs.
#   pi       the model card's evaluation command: base + LoRA, thinking off,
#            qwen3_xml tool calls, prefix caching with --mamba-cache-mode align.
#            Serves both SERVED_MODEL_NAME (base) and LORA_NAME (adapter).
#   pi-mtp   pi plus MTP speculative decoding from the base's own MTP head.
#            Lossless, but the head was not trained with the adapter, so
#            acceptance may be lower. Not in the model card.
#   nvfp4    pi-mtp on the NVFP4 base (LoRA unmerged, MTP on). Reads ~20 GiB
#            per token instead of ~51, so decode is faster; the adapter was
#            trained against BF16 weights, so its behaviour may drift.
#   base     the BF16 base alone, no LoRA: a baseline, and a way to isolate
#            a failure to the adapter. Does not need the adapter downloaded.
PROFILES="pi|pi-mtp|nvfp4|base"
select_profile() {
  PROFILE="${1:-${DEFAULT_PROFILE}}"
  case "${PROFILE}" in
    pi)     ENABLE_LORA=1; SPEC_TOKENS="${SPEC_TOKENS:-0}"; BASE_FORMAT=bf16 ;;
    pi-mtp) ENABLE_LORA=1; SPEC_TOKENS="${SPEC_TOKENS:-2}"; BASE_FORMAT=bf16 ;;
    nvfp4)  ENABLE_LORA=1; SPEC_TOKENS="${SPEC_TOKENS:-2}"; BASE_FORMAT=nvfp4 ;;
    base)   ENABLE_LORA=0; SPEC_TOKENS="${SPEC_TOKENS:-0}"; BASE_FORMAT=bf16 ;;
    *)
      printf 'error: unknown profile %q (%s)\n' "${PROFILE}" "${PROFILES}" >&2
      return 1
      ;;
  esac
  if [[ "${BASE_FORMAT}" == nvfp4 ]]; then
    MODEL_ID="${NVFP4_MODEL_ID}"; MODEL_MANIFEST="${NVFP4_MANIFEST}"
  else
    MODEL_ID="${BF16_MODEL_ID}"; MODEL_MANIFEST="${BF16_MANIFEST}"
  fi
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
  export PROFILE ENABLE_LORA SPEC_TOKENS BASE_FORMAT MODEL_ID MODEL_MANIFEST \
    MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS
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

# snapshot_dir MANIFEST [HF_HOME] -- where the pinned revision sits in the HF
# cache. The optional second argument gives the container-side path.
snapshot_dir() {
  local repo revision root="${2:-${HF_HOME}}"
  repo="$(manifest_field "$1" repo)"
  revision="$(manifest_field "$1" revision)"
  printf '%s/hub/models--%s/snapshots/%s' "${root}" "${repo//\//--}" "${revision}"
}

# The manifests this profile needs.
profile_manifests() {
  printf '%s\n' "${MODEL_MANIFEST}"
  (( ENABLE_LORA )) && printf '%s\n' "${ADAPTER_MANIFEST}"
  return 0
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
