#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
# Copyright (C) 2026 amarjeet
#
# Derived from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark, which is
# Copyright (C) 2026 MiaAI Lab and licensed AGPL-3.0-or-later.
# Modified 2026-09-05 by amarjeet: fetches a pinned, checksum-verified
# revision via scripts/download_snapshot.py instead of huggingface_hub
#
# Download and checksum-verify the pinned NVFP4 checkpoint.
#
# Storage: the checkpoint is a Hugging Face repo, so it lands in the Hugging
# Face cache (HF_HOME, default ~/.cache/huggingface) in the ordinary
# hub/models--<org>--<name>/{blobs,snapshots,refs} layout. That is the tree
# vLLM resolves MODEL_ID against under HF_HUB_OFFLINE=1, and it is shared with
# every other recipe on the host. Override HF_HOME to relocate.
#
# scripts/download_snapshot.py (pure stdlib) pins the revision, resumes partial
# transfers, verifies every size and SHA-256 before linking a file into the
# snapshot, and refuses to start without a 10 GiB disk reserve.
#
# start.sh deliberately never downloads: it resolves the pinned snapshot and
# fails fast if it is absent. This is the script it points you at.
#
# Usage:
#   ./download.sh                  # 98.66 GiB, resumable -- rerun to continue
#   ./download.sh --verify-only    # re-hash what is on disk, no network
#   ./download.sh --workers 8      # more parallelism (the Hub throttles above ~8)
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
select_profile "${DEFAULT_PROFILE}"

EXTRA_ARGS=("$@")
RETRIES="${RETRIES:-20}"

VERIFY_ONLY=0
for arg in ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}; do
  case "${arg}" in
    --verify-only) VERIFY_ONLY=1 ;;
    -h|--help) sed -n '4,22p' "$0" | sed 's/^# \?//'; exit 0 ;;
  esac
done

resolve_hf_token
[[ -n "${HF_TOKEN}" ]] || printf 'note: HF_TOKEN is empty; this repo is public, but anonymous Hub access has lower rate limits\n' >&2

mkdir -p "${HF_HOME}"

printf 'model     : %s\n' "${MODEL_ID}"
printf 'revision  : %s\n' "${MODEL_REVISION}"
printf 'manifest  : %s\n' "${MANIFEST}"
printf 'HF_HOME   : %s\n' "${HF_HOME}"
printf 'size      : %s\n' "$(human_bytes "${MODEL_TOTAL_BYTES}")"

for (( i = 1; i <= RETRIES; i++ )); do
  printf '\nattempt %d/%d %s\n' "${i}" "${RETRIES}" "$(date -Is)"
  if HF_HOME="${HF_HOME}" python3 "${EXPERIMENT_DIR}/scripts/download_snapshot.py" \
      --manifest "${MANIFEST}" \
      --hf-home "${HF_HOME}" \
      ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}; then
    printf '\nOK %s @ %s\n' "${MODEL_ID}" "${MODEL_REVISION:0:12}"
    printf 'next: ./preflight.sh\n'
    exit 0
  fi
  # --verify-only failures are terminal: retrying will not change the bytes
  # already on disk.
  if (( VERIFY_ONLY )); then
    printf 'FAILED verification (not retrying)\n' >&2
    exit 1
  fi
  printf 'retry in 10s\n' >&2
  sleep 10
done

printf 'FAILED after %d attempts\n' "${RETRIES}" >&2
exit 1
