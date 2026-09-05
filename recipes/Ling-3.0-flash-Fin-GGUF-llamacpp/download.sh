#!/usr/bin/env bash
#
# Download and checksum-verify the pinned GGUF shards.
#
# Storage: weights land in llama.cpp's own standard cache
# (LLAMA_CACHE, default ~/.cache/llama.cpp) under
#   Ling-3.0-flash-Fin-GGUF/<QUANT>-<revision12>/
# so they are downloaded once and shared with every other llama.cpp recipe.
# Override LLAMA_CACHE or MODEL_STORE to relocate.
#
# scripts/download_model.py (pure stdlib) pins the Hub revision, resumes
# partial downloads, verifies every size and SHA-256, and refuses to start
# without a 10 GiB disk reserve.
#
# Usage:
#   ./download.sh iq4xs                 # 64 GiB  (default profile)
#   ./download.sh q3kxl                 # 58 GiB
#   ./download.sh q4km                  # 72 GiB
#   ./download.sh all                   # all three (194 GiB)
#   ./download.sh iq4xs --verify-only   # re-check on disk, no network
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

TARGET="${1:-${DEFAULT_PROFILE}}"
shift || true
EXTRA_ARGS=("$@")

RETRIES="${RETRIES:-20}"
VERIFY_ONLY=0
for arg in ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}; do
  [[ "${arg}" == "--verify-only" ]] && VERIFY_ONLY=1
done

resolve_hf_token
[[ -n "${HF_TOKEN}" ]] || printf 'note: HF_TOKEN is empty; anonymous Hub access has lower rate limits\n' >&2

fetch() {  # fetch <manifest> <destination> <label>
  local manifest="$1" destination="$2" label="$3" i
  printf '\n=== %s ===\n' "${label}"
  printf 'manifest    : %s\n' "${manifest}"
  printf 'destination : %s\n' "${destination}"
  for (( i = 1; i <= RETRIES; i++ )); do
    printf '\nattempt %d/%d %s\n' "${i}" "${RETRIES}" "$(date -Is)"
    if python3 "${EXPERIMENT_DIR}/scripts/download_model.py" \
        --manifest "${manifest}" \
        --destination "${destination}" \
        ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}; then
      printf 'OK %s\n' "${label}"
      return 0
    fi
    # --verify-only failures are terminal: retrying will not change the bytes
    # already on disk.
    if (( VERIFY_ONLY )); then
      printf 'FAILED %s (verification, not retrying)\n' "${label}" >&2
      return 1
    fi
    printf 'retry in 10s\n' >&2
    sleep 10
  done
  printf 'FAILED %s after %d attempts\n' "${label}" "${RETRIES}" >&2
  return 1
}

download_profile() {
  local profile="$1"
  ( select_profile "${profile}"
    fetch "${MANIFEST}" "${MODEL_ROOT}" "${profile} (${QUANT}, $(human_bytes "${MODEL_TOTAL_BYTES}"))" )
}

case "${TARGET}" in
  q3kxl|iq4xs|q4km) download_profile "${TARGET}" ;;
  all)
    download_profile iq4xs
    download_profile q3kxl
    download_profile q4km
    ;;
  *)
    printf 'usage: %s {iq4xs|q3kxl|q4km|all} [--verify-only]\n' "$0" >&2
    exit 2
    ;;
esac

printf '\ndone.\n'
