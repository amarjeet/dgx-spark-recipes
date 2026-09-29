#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
#
# Download the pinned checkpoint (105.46 GiB) into the Hugging Face cache and
# verify every file against manifests/mlx4.json. Resumable and safe to re-run.
#
#   ./download.sh                 download what is missing, verify, stamp
#   ./download.sh --verify-only   re-hash what is on disk, download nothing
#
# Standard library only: no hf CLI or huggingface_hub needed on the host. The
# files land in the standard HF cache layout, so any other tool reading the
# cache shares them.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

VERIFY_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --verify-only) VERIFY_ONLY=1 ;;
    -h|--help) sed -n '4,13p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) printf 'unknown argument: %s\n' "${arg}" >&2; exit 2 ;;
  esac
done
RETRIES="${RETRIES:-20}"

if (( VERIFY_ONLY )); then
  FORCE_VERIFY=1 verify_snapshot
  exit
fi

# Disk: what is still missing, plus the reserve.
need="$(python3 - "${MANIFEST}" "${MODEL_PATH}" <<'PYEOF'
import json, os, sys
m, root = json.load(open(sys.argv[1])), sys.argv[2]
left = 0
for f in m["files"]:
    blob = os.path.join(root, "blobs", f["blob"])
    have = os.path.getsize(blob) if os.path.exists(blob) else 0
    part = blob + ".incomplete"
    have = max(have, os.path.getsize(part) if os.path.exists(part) else 0)
    left += max(0, f["bytes"] - have)
print(left)
PYEOF
)"
mkdir -p "${HF_HOME}/hub"
avail="$(df -B1 --output=avail "${HF_HOME}/hub" | tail -1 | tr -d ' ')"
printf 'remaining download %s, free under %s: %s\n' "$(human_bytes "${need}")" "${HF_HOME}" "$(human_bytes "${avail}")"
(( avail >= need + DISK_RESERVE_BYTES )) || {
  printf 'error: need %s free (download + %s reserve)\n' \
    "$(human_bytes "$(( need + DISK_RESERVE_BYTES ))")" "$(human_bytes "${DISK_RESERVE_BYTES}")" >&2
  exit 1
}

resolve_hf_token
[[ -n "${HF_TOKEN}" ]] || printf 'note: no HF token; the repo is public, but anonymous access has a lower rate limit\n' >&2

for (( i = 1; i <= RETRIES; i++ )); do
  printf '\nattempt %d/%d %s\n' "${i}" "${RETRIES}" "$(date -Is)"
  if python3 -u "${EXPERIMENT_DIR}/scripts/download_snapshot.py" \
      --manifest "${MANIFEST}" --hf-home "${HF_HOME}"; then
    # download_snapshot.py verified every file it fetched or found; stamp that
    # so preflight.sh does not re-hash 105 GiB.
    verify_snapshot
    exit 0
  fi
  printf 'retry in 10s\n' >&2
  sleep 10
done
printf 'FAILED after %d attempts\n' "${RETRIES}" >&2
exit 1
