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
printf 'context    : %s over %s slot(s) = %s per slot\n' \
  "${CTX_SIZE}" "${PARALLEL}" "${CTX_PER_SLOT}"
printf 'store      : %s\n' "${MODEL_ROOT}"
printf 'binary     : %s\n' "${BIN_DIR}"

printf '\nprocess\n'
if pid="$(server_pid 2>/dev/null)"; then
  printf '  running  pid %s, up %s\n' "${pid}" "$(ps -o etime= -p "${pid}" | tr -d ' ')"
  printf '  rss      %s\n' \
    "$(human_bytes "$(( $(ps -o rss= -p "${pid}" | tr -d ' ') * 1024 ))")"
else
  printf '  not running\n'
  [[ -f "${PID_FILE}" ]] && printf '  (stale pidfile: %s)\n' "${PID_FILE}"
fi

printf '\nhealth\n'
if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  printf '  OK http://127.0.0.1:%s/health\n' "${PORT}"
  printf '  models: '
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null \
    | python3 -c 'import json,sys; print(", ".join(m["id"] for m in json.load(sys.stdin)["data"]))' \
    2>/dev/null || printf '(unreadable)\n'
  # The number that matters: llama.cpp can reduce a requested context to make
  # it fit, so compare what it is serving against what the profile asked for
  # rather than reporting the request back as if it were the answer.
  curl -fsS "http://127.0.0.1:${PORT}/props" 2>/dev/null \
    | CTX_PER_SLOT="${CTX_PER_SLOT}" PARALLEL="${PARALLEL}" python3 -c '
import json, os, sys
try:
    props = json.load(sys.stdin)
except Exception:
    raise SystemExit(0)
settings = props.get("default_generation_settings") or {}
slot_ctx = settings.get("n_ctx") or props.get("n_ctx")
slots = props.get("total_slots")
want_ctx, want_slots = os.environ["CTX_PER_SLOT"], os.environ["PARALLEL"]
flag = "" if str(slot_ctx) == want_ctx else f"  <-- asked for {want_ctx}"
print(f"  serving: {slots} slot(s), {slot_ctx} tokens per slot{flag}")
if str(slots) != want_slots:
    print(f"  WARNING: {slots} slots, profile asked for {want_slots}")
' 2>/dev/null || true
  # No escaped quotes inside an f-string here: this Python is inside a
  # single-quoted shell string, so a \" would reach the interpreter literally
  # and make it a syntax error that `|| true` would hide.
  curl -fsS "http://127.0.0.1:${PORT}/slots" 2>/dev/null \
    | python3 -c '
import json, sys
try:
    slots = json.load(sys.stdin)
except Exception:
    raise SystemExit(0)
if not isinstance(slots, list):
    raise SystemExit(0)
for s in slots:
    state = "busy" if s.get("is_processing") else "idle"
    print("  slot %s: %s n_ctx=%s prompt=%s/%s" % (
        s.get("id"), state, s.get("n_ctx"),
        s.get("n_prompt_tokens_processed", 0), s.get("n_prompt_tokens", 0)))
' 2>/dev/null || true
else
  printf '  not responding on port %s\n' "${PORT}"
fi

printf '\nhost\n'
printf '  memory available %s of %s\n' \
  "$(human_bytes "$(mem_available_bytes)")" "$(human_bytes "$(mem_total_bytes)")"
printf '  in use           %s (budget for this profile: %s)\n' \
  "$(human_bytes "$(( $(mem_total_bytes) - $(mem_available_bytes) ))")" \
  "$(human_bytes "$(budget_bytes)")"
printf '  swap used        %s\n' "$(human_bytes "$(host_swap_used_bytes)")"
others="$(docker ps --format '{{.Names}}' 2>/dev/null | paste -sd, - || true)"
printf '  other containers %s\n' "${others:-none}"

printf '\nload log (%s)\n' "${LOG_FILE}"
if [[ -f "${LOG_FILE}" ]]; then
  grep -iE 'launching|n_slots|n_ctx_slot|model loaded|listening|error|failed|warn' "${LOG_FILE}" \
    | sed 's/^[0-9.]* [A-Z] /  /' | sed 's/^\[/  [/' | tail -12 \
    || printf '  (no summary lines yet)\n'
else
  printf '  (no log yet)\n'
fi

if (( CTX_PER_SLOT > 98304 )); then
  printf '\nnote: %s per slot is above the depths llama.cpp #27756 reports failing\n' "${CTX_PER_SLOT}"
  printf '      silently. Passed here at 254032, but the onset is prompt-dependent:\n'
  printf '        ./scripts/smoke.py --needle-depth %s\n' "${CTX_PER_SLOT}"
fi
