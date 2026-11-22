#!/usr/bin/env bash
#
# Download and checksum-verify the pinned checkpoint (with its DSpark drafter).
#
# It lands in the Hugging Face cache (HF_HOME, default ~/.cache/huggingface),
# in the ordinary hub/models--<org>--<name>/ layout, shared with every other
# recipe on the host. scripts/download_snapshot.py (stdlib only) resumes
# partial transfers and verifies every size and SHA-256 before linking a file
# into the snapshot. start.sh never downloads; this is the script it points at.
#
# Usage:
#   ./download.sh                  # 18.8 GiB, resumable
#   ./download.sh --verify-only    # re-hash what is on disk, no network
#   ./download.sh --workers 8
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
RETRIES="${RETRIES:-20}"

case "${1:-}" in -h|--help) sed -n '2,13p' "$0" | sed 's/^# \?//'; exit 0 ;; esac

resolve_hf_token
[[ -n "${HF_TOKEN}" ]] || printf 'note: HF_TOKEN is empty; the repo is public, but anonymous access has lower rate limits\n' >&2
mkdir -p "${HF_HOME}"
printf 'HF_HOME : %s\n' "${HF_HOME}"
# GB10 page-cache trap: a big download fills unified memory with page cache.
if docker ps --format '{{.Names}}' | grep -q .; then
  printf 'note: containers are running; if one is a model server, page cache from this download\n' >&2
  printf '      competes with it. After: sync; echo 1 | sudo tee /proc/sys/vm/drop_caches\n' >&2
fi

manifest="${MODEL_MANIFEST}"
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

printf '\nOK\nnext: docker pull %s && ./preflight.sh\n' "${IMAGE}"
