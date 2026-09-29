#!/usr/bin/env bash
#
# Download and checksum-verify the pinned GGUF files.
#
# Storage: weights land in llama.cpp's own standard cache
# (LLAMA_CACHE, default ~/.cache/llama.cpp) under
#   Ternary-Bonsai-2-27B-gguf/<QUANT>-<revision12>/
# so they are downloaded once and shared with every other llama.cpp recipe.
# Override LLAMA_CACHE or MODEL_STORE to relocate.
#
# scripts/download_model.py (pure stdlib) pins the Hub revision, resumes
# partial downloads, verifies every size and SHA-256, and refuses to start
# without a 10 GiB disk reserve.
#
# Each profile's manifest also lists the Q8_0 vision projector, so a single
# fetch gets a servable set. Naming it in both manifests costs nothing: the
# downloader skips a file that already verifies.
#
# The F16 reference pack (53.8 GB) and the BF16 projector are deliberately not
# in any manifest. Nothing here loads them.
#
# Usage:
#   ./download.sh pq2                 # 7.8 GB  (default: PQ2_0 + projector)
#   ./download.sh ptq1                # 6.6 GB  (PTQ1_0 + projector)
#   ./download.sh all                 # both packs
#   ./download.sh pq2 --verify-only   # re-check on disk, no network
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

TARGET="${1:-pq2}"
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

# Subshell: select_profile exports, and a second call in the same shell would
# otherwise inherit the first profile's values.
download_pack() {
  local profile="$1"
  ( select_profile "${profile}"
    fetch "${MANIFEST}" "${MODEL_ROOT}" \
      "${profile} (${QUANT}, $(human_bytes "${MODEL_TOTAL_BYTES}"))" )
}

case "${TARGET}" in
  # pq2 is the pack, and wide/deep/long/safe all use it -- accept either
  # spelling so `./download.sh <profile>` works for whatever profile you are
  # about to run, which is what preflight.sh tells you to do on a miss.
  pq2|wide|deep|long|safe) download_pack deep ;;
  ptq1)               download_pack ptq1 ;;
  all)
    download_pack deep
    download_pack ptq1
    ;;
  *)
    printf 'usage: %s {pq2|ptq1|all} [--verify-only]\n' "$0" >&2
    printf '       (profile names wide|deep|long|safe are accepted for pq2)\n' >&2
    exit 2
    ;;
esac

printf '\ndone.\n'
