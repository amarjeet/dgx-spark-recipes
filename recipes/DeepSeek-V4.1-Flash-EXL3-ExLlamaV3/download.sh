#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2026 amarjeet
#
# Written for this port. Part of the same AGPL-3.0-only combined work as the
# files it sits beside, which derive from
# vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe,
# Copyright (C) 2026 Victor Cruz.
#
# Fetch the pinned EXL3 pack into the Hugging Face cache, verified by SHA-256.
#
# 307.72 GiB across 23 files. Shards 16 and 17 are 94.6 GiB each and hold only
# Engram tensors; they are 61% of the download and are NOT part of the ~107 GiB
# the server keeps resident. Whether this native ExLlamaV3 path reads them at
# all is not established upstream, so they are fetched rather than guessed away.
#
# Resumable: a partial blob is continued with a Range request, and nothing is
# linked into the snapshot until its SHA-256 matches the manifest.
#
# Usage:
#   ./download.sh                 # fetch and verify
#   ./download.sh --verify-only   # re-hash what is on disk, no network
#   FORCE_VERIFY=1 ./download.sh  # ignore the stamp and re-hash everything
#   RETRIES=5 ./download.sh       # fewer attempts before giving up
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
select_profile "${DEFAULT_PROFILE}"

EXTRA_ARGS=()
while (( $# )); do
  case "$1" in
    -h|--help) sed -n '11,25p' "$0" | sed 's/^# \?//'; exit 0 ;;
    --verify-only) EXTRA_ARGS+=(--verify-only); shift ;;
    --workers) EXTRA_ARGS+=(--workers "$2"); shift 2 ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

printf 'model     : %s\n' "${MODEL_ID}"
printf 'revision  : %s\n' "${MODEL_REVISION}"
printf 'manifest  : %s\n' "${MANIFEST}"
printf 'HF_HOME   : %s\n' "${HF_HOME}"
printf 'size      : %s\n\n' "$(human_bytes "${MODEL_TOTAL_BYTES}")"

resolve_hf_token
[[ -n "${HF_TOKEN}" ]] || printf 'note: HF_TOKEN is empty; this pack is public, but anonymous Hub access has lower rate limits\n' >&2

mkdir -p "${HF_HOME}" "${OUT_DIR}"

attempt=1
retries="${RETRIES:-20}"
while (( attempt <= retries )); do
  if HF_HOME="${HF_HOME}" python3 "${EXPERIMENT_DIR}/scripts/download_snapshot.py" \
       --manifest "${MANIFEST}" --hf-home "${HF_HOME}" \
       ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}; then
    break
  fi
  # A --verify-only failure means the bytes on disk are wrong. Retrying the
  # same check cannot change that, so fail now rather than twenty times.
  for arg in ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}; do
    [[ "${arg}" == "--verify-only" ]] && {
      printf '\nverification failed -- re-run without --verify-only to repair\n' >&2
      exit 1
    }
  done
  (( attempt++ ))
  if (( attempt > retries )); then
    printf '\ngave up after %s attempts\n' "${retries}" >&2
    exit 1
  fi
  printf '\nattempt %s/%s failed; retrying\n\n' "${attempt}" "${retries}" >&2
  sleep 5
done

printf '\nnext: ./relay.sh   (64-byte re-lay; start.sh loads the re-laid pack)\n'
