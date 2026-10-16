#!/usr/bin/env bash
#
# Download and checksum-verify the pinned checkpoint and its dflash draft model.
#
# Both land in the Hugging Face cache (HF_HOME, default ~/.cache/huggingface),
# in the ordinary hub/models--<org>--<name>/ layout, shared with every other
# recipe on the host. scripts/download_snapshot.py (stdlib only) resumes
# partial transfers and verifies every size and SHA-256 before linking a file
# into the snapshot. start.sh never downloads; this is the script it points at.
#
# Usage:
#   ./download.sh                  # 69.9 GiB + 1.2 GiB, resumable
#   ./download.sh --verify-only    # re-hash what is on disk, no network
#   ./download.sh --workers 8
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
RETRIES="${RETRIES:-20}"

case "${1:-}" in -h|--help) sed -n '2,13p' "$0" | sed 's/^# \?//'; exit 0 ;; esac

resolve_hf_token
[[ -n "${HF_TOKEN}" ]] || printf 'note: HF_TOKEN is empty; both repos are public, but anonymous access has lower rate limits\n' >&2
mkdir -p "${HF_HOME}"
printf 'HF_HOME : %s\n' "${HF_HOME}"

for manifest in "${DRAFT_MANIFEST}" "${MODEL_MANIFEST}"; do
  printf '\n%s @ %s  (%s)\n' "$(manifest_field "${manifest}" repo)" \
    "$(manifest_field "${manifest}" revision | cut -c1-12)" \
    "$(human_bytes "$(manifest_field "${manifest}" total_bytes)")"
  done_ok=0
  for (( i = 1; i <= RETRIES; i++ )); do
    printf 'attempt %d/%d %s\n' "${i}" "${RETRIES}" "$(date -Is)"
    if HF_HOME="${HF_HOME}" python3 "${EXPERIMENT_DIR}/scripts/download_snapshot.py" \
        --manifest "${manifest}" --hf-home "${HF_HOME}" "$@"; then
      done_ok=1; break
    fi
    # Verification failures are terminal: retrying cannot change bytes on disk.
    [[ " $* " == *" --verify-only "* ]] && break
    printf 'retry in 10s\n' >&2; sleep 10
  done
  (( done_ok )) || { printf 'FAILED: %s\n' "${manifest}" >&2; exit 1; }
done

printf '\nOK\nnext: ./preflight.sh\n'
