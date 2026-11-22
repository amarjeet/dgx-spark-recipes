#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
#
# Download the pinned checkpoint (95.24 GiB) into the Hugging Face cache and
# verify every file against manifests/fp4.json. Resumable and safe to re-run.
#
#   ./download.sh                 download what is missing, verify, stamp
#   ./download.sh --verify-only   re-hash what is on disk, download nothing
#   ./download.sh --ablit         also the ABLIT=1 overlay's sidecar (~186 MiB,
#                                 gated, manifests/ablit.json) and the grafted
#                                 checkpoint folder scripts/ablit_view.py builds
#                                 (ABLIT=1 ./download.sh is the same)
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
    --ablit) ABLIT=1 ;;
    -h|--help) sed -n '/^# Download the pinned/,/^[^#]/{/^#/p}' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) printf 'unknown argument: %s\n' "${arg}" >&2; exit 2 ;;
  esac
done
RETRIES="${RETRIES:-20}"

# The ABLIT=1 overlay: its pinned sidecar, then the view that grafts it onto
# the verified stock snapshot. The view is symlinks only; nothing in either
# snapshot is touched.
ablit() {
  [[ -f "${ABLIT_MANIFEST}" ]] || {
    printf 'error: %s is missing. The overlay is gated and the Hub withholds the sidecar'"'"'s SHA-256 until\n' "${ABLIT_MANIFEST}" >&2
    printf '       your account accepts its terms at https://huggingface.co/%s ; then:\n' "${ABLIT_ID}" >&2
    printf '       scripts/make_manifest.py %s %s manifests/ablit.json --include %s\n' "${ABLIT_ID}" "${ABLIT_REVISION}" "${ABLIT_SIDECAR}" >&2
    return 1
  }
  if (( VERIFY_ONLY )); then
    FORCE_VERIFY=1 verify_snapshot "${ABLIT_MANIFEST}" "${ABLIT_SNAPSHOT_DIR}"
  else
    printf '\nABLIT=1: %s @ %s, %s\n' "${ABLIT_ID}" "${ABLIT_REVISION:0:12}" "${ABLIT_SIDECAR}"
    printf 'note: this overlay removes the model'"'"'s safety refusals\n'
    [[ -n "${HF_TOKEN}" ]] || { printf 'error: the overlay is gated: set HF_TOKEN (or hf auth login) for an account that accepted its terms\n' >&2; return 1; }
    python3 -u "${EXPERIMENT_DIR}/scripts/download_snapshot.py" --manifest "${ABLIT_MANIFEST}" --hf-home "${HF_HOME}" \
      || { printf 'error: the sidecar download failed (HTTP 403: accept its terms at https://huggingface.co/%s with the account of HF_TOKEN)\n' "${ABLIT_ID}" >&2; return 1; }
    verify_snapshot "${ABLIT_MANIFEST}" "${ABLIT_SNAPSHOT_DIR}"
  fi
  python3 -I "${EXPERIMENT_DIR}/scripts/ablit_view.py" "${SNAPSHOT_DIR}" \
    "${ABLIT_SNAPSHOT_DIR}/${ABLIT_SIDECAR}" "${ABLIT_VIEW_DIR}"
}

resolve_hf_token

if (( VERIFY_ONLY )); then
  FORCE_VERIFY=1 verify_snapshot
  [[ "${ABLIT}" != 1 ]] || ablit
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

[[ -n "${HF_TOKEN}" ]] || printf 'note: no HF token; the repo is public, but anonymous access has a lower rate limit\n' >&2

for (( i = 1; i <= RETRIES; i++ )); do
  printf '\nattempt %d/%d %s\n' "${i}" "${RETRIES}" "$(date -Is)"
  if python3 -u "${EXPERIMENT_DIR}/scripts/download_snapshot.py" \
      --manifest "${MANIFEST}" --hf-home "${HF_HOME}"; then
    # download_snapshot.py verified every file it fetched or found; stamp that
    # so preflight.sh does not re-hash 95 GiB.
    verify_snapshot
    [[ "${ABLIT}" != 1 ]] || ablit
    exit
  fi
  printf 'retry in 10s\n' >&2
  sleep 10
done
printf 'FAILED after %d attempts\n' "${RETRIES}" >&2
exit 1
