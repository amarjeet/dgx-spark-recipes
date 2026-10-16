#!/usr/bin/env bash
#
# One screen: is it up, what is it serving, is the host healthy.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
select_profile "$(cat "${ACTIVE_PROFILE_FILE}" 2>/dev/null || printf '%s' "${DEFAULT_PROFILE}")"

printf 'experiment : %s\nprofile    : %s (spec tokens %s)\n' "${EXPERIMENT_NAME}" "${PROFILE}" "${SPEC_TOKENS}"
printf 'model      : %s @ %s\n' "${MODEL_ID}" "$(manifest_field "${MODEL_MANIFEST}" revision | cut -c1-12)"

printf '\ncontainer\n'
if container_running; then
  docker ps --filter "name=^${CONTAINER_NAME}$" --format '  {{.Names}}  {{.Status}}  {{.Ports}}'
  printf '  memory   %s\n' "$(docker stats --no-stream --format '{{.MemUsage}}' "${CONTAINER_NAME}" 2>/dev/null || printf n/a)"
elif container_exists; then
  docker ps -a --filter "name=^${CONTAINER_NAME}$" --format '  {{.Names}}  {{.Status}}'
else
  printf '  not running\n'
fi

printf '\nendpoint\n'
if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  printf '  /health ok on :%s, serving %s\n' "${PORT}" \
    "$(curl -fsS "http://127.0.0.1:${PORT}/v1/models" | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || printf '?')"
else
  printf '  not answering on :%s\n' "${PORT}"
fi

printf '\nhost\n'
printf '  MemAvailable %s of %s\n' "$(human_bytes "$(mem_available_bytes)")" "$(human_bytes "$(mem_total_bytes)")"
