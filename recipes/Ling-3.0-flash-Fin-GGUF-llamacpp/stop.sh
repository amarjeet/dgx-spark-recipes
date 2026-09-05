#!/usr/bin/env bash
#
# Stop the llama.cpp server container. The log is preserved.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

if container_exists; then
  printf 'stopping container %s\n' "${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
else
  printf 'no container %s found\n' "${CONTAINER_NAME}"
fi

rm -f "${PID_FILE}" "${ACTIVE_PROFILE_FILE}"
printf 'stopped. log preserved at %s\n' "${LOG_FILE}"
