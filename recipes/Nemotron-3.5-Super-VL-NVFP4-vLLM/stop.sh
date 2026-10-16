#!/usr/bin/env bash
#
# Stop the vLLM container, keeping its log in OUT_DIR.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
mkdir -p "${OUT_DIR}"

if container_exists; then
  docker logs --tail 3000 "${CONTAINER_NAME}" >"${OUT_DIR}/last-container.log" 2>&1 || true
  docker rm -f "${CONTAINER_NAME}" >/dev/null
  printf 'stopped %s; log: %s/last-container.log\n' "${CONTAINER_NAME}" "${OUT_DIR}"
else
  printf 'no container %s\n' "${CONTAINER_NAME}"
fi
rm -f "${ACTIVE_PROFILE_FILE}"
