#!/usr/bin/env bash
#
# One screen of "is it up, what is it serving, and is the box healthy".
#
# Usage: ./status.sh
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
select_profile "$(cat "${ACTIVE_PROFILE_FILE}" 2>/dev/null || printf '%s' "${DEFAULT_PROFILE}")"

printf 'experiment : %s\n' "${EXPERIMENT_NAME}"
printf 'profile    : %s (%s, %s)\n' "${PROFILE}" "${QUANT}" "$(human_bytes "${MODEL_TOTAL_BYTES}")"
printf 'store      : %s\n' "${MODEL_ROOT}"

printf '\ncontainer\n'
if container_running; then
  docker ps --filter "name=^${CONTAINER_NAME}$" \
    --format '{{.Names}}  {{.Status}}  {{.Ports}}' | sed 's/^/  /'
  printf '  memory   %s\n' \
    "$(docker stats --no-stream --format '{{.MemUsage}}' "${CONTAINER_NAME}" 2>/dev/null || printf 'n/a')"
elif container_exists; then
  docker ps -a --filter "name=^${CONTAINER_NAME}$" --format '{{.Names}}  {{.Status}}' | sed 's/^/  /'
else
  printf '  not running\n'
fi

printf '\nhealth\n'
if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  printf '  OK http://127.0.0.1:%s/health\n' "${PORT}"
  printf '  models: '
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null \
    | python3 -c 'import json,sys; print(", ".join(m["id"] for m in json.load(sys.stdin)["data"]))' \
    2>/dev/null || printf '(unreadable)\n'
  # /slots exposes what each server slot is doing and how much of its context
  # is consumed -- the number you actually want when tuning CTX_SIZE.
  curl -fsS "http://127.0.0.1:${PORT}/slots" 2>/dev/null \
    | python3 -c '
import json, sys
try:
    slots = json.load(sys.stdin)
except Exception:
    raise SystemExit(0)
for s in slots:
    print(f"  slot {s.get(\"id\")}: state={s.get(\"is_processing\") and \"busy\" or \"idle\"} "
          f"n_ctx={s.get(\"n_ctx\")} prompt_tokens={s.get(\"prompt_n\", 0)}")
' 2>/dev/null || true
else
  printf '  not responding on port %s\n' "${PORT}"
fi

printf '\nhost\n'
printf '  memory available %s of %s\n' \
  "$(human_bytes "$(mem_available_bytes)")" "$(human_bytes "$(mem_total_bytes)")"
printf '  swap used        %s\n' "$(human_bytes "$(host_swap_used_bytes)")"
others="$(docker ps --format '{{.Names}}' | grep -vx "${CONTAINER_NAME}" | paste -sd, - || true)"
printf '  other containers %s\n' "${others:-none}"

printf '\nload log (%s)\n' "${LOG_FILE}"
if [[ -f "${LOG_FILE}" ]]; then
  # llama.cpp's verbose model/KV tables are not emitted at this log level, so
  # match the lines it does print rather than silently showing nothing.
  grep -iE 'launching|MTP draft|n_ctx_slot|model loaded|listening on|error|failed' "${LOG_FILE}" \
    | sed 's/^[0-9.]* [A-Z] /  /' | sed 's/^\[/  [/' | tail -12 \
    || printf '  (no summary lines yet)\n'
else
  printf '  (no log yet)\n'
fi
