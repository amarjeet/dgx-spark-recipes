#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
#
# Shared configuration for Clef-Flash (Cloudflare) on transformers.
# Sourced by setup.sh / download.sh / preflight.sh / start.sh / stop.sh /
# status.sh / bench.sh -- the ONLY place paths, the budget and helpers are
# defined.
#
# Storage rule: every path is an env-overridable variable whose default is the
# tool's own standard location (CONVENTIONS.md). This is a NATIVE recipe -- no
# container, nothing bind-mounted -- so the only variables that matter are the
# ones the tools themselves read, and their defaults are already the shared
# locations: HF_HOME for the weights, ~/.triton for flash-linear-attention's
# Triton kernels, ~/.cache/uv for uv's package cache. Nothing is exported.
#
# This recipe is built to run BESIDE another server (by default the
# Qwen3.8-Flash-Next TensorFold recipe), not instead of one. Its memory
# guards are therefore set to give way first: see "co-tenancy" below.

set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Pinned rather than derived from the directory name, so OUT_DIR does not move
# if the recipe is cloned or renamed.
EXPERIMENT_NAME="Clef-Flash-FP8-transformers"

# --- storage -----------------------------------------------------------------

HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
OUT_DIR="${OUT_DIR:-${XDG_STATE_HOME:-${HOME}/.local/state}/dgx-spark-recipes/${EXPERIMENT_NAME}}"

# The serving environment. pyproject.toml and uv.lock live here; the venv is
# several GB of torch and lives outside the recipe (CONVENTIONS.md, VENV).
# uv reads UV_PROJECT_ENVIRONMENT to put a project's venv somewhere other than
# ./.venv, so that is the one variable this recipe does export, and only to uv.
VENV="${VENV:-${HOME}/venvs/clef-flash}"
PYTHON="${VENV}/bin/python"

# --- model -------------------------------------------------------------------

MODEL_ID="${MODEL_ID:-Cloudflare/clef-flash}"
MANIFEST="${MANIFEST:-${EXPERIMENT_DIR}/manifests/bf16.json}"
# Pinned. The revision is the manifest's, so the two cannot disagree.
MODEL_REVISION="${MODEL_REVISION:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "${MANIFEST}")}"
MODEL_PATH="${HF_HOME}/hub/models--${MODEL_ID//\//--}"
SNAPSHOT_DIR="${MODEL_PATH}/snapshots/${MODEL_REVISION}"
MODEL_TOTAL_BYTES="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["total_bytes"])' "${MANIFEST}")"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-clef-flash}"

# --- runtime -----------------------------------------------------------------

# 8008 Ling, 8888 Qwen vLLM, 8009 DeepSeek, 8010 Bonsai, 8011 Qwen TensorFold.
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8012}"

# Longest input accepted, in tokens: the model card's default for
# encode_record. Longer states are truncated by the model's own encoder, not
# rejected. The largest single lever on this recipe's memory, because start.sh
# warms up at exactly this length -- see budget_bytes().
MAX_LENGTH="${MAX_LENGTH:-16384}"
# Per-request media limits. Vision tokens are prefill and count against
# MAX_LENGTH, but the vision encoder's activations do not, so they are bounded
# separately.
MAX_IMAGES="${MAX_IMAGES:-8}"
MAX_VIDEOS="${MAX_VIDEOS:-1}"
MAX_BODY_MIB="${MAX_BODY_MIB:-64}"

PID_FILE="${OUT_DIR}/server.pid"
LOG_FILE="${OUT_DIR}/logs/server.log"
MEMWATCH_LOG="${OUT_DIR}/logs/memwatch.log"
# The systemd scope the server runs in, so the cgroup cap has a name.
SCOPE_UNIT="${SCOPE_UNIT:-clef-flash-${PORT}}"

DISK_RESERVE_BYTES="${DISK_RESERVE_BYTES:-10737418240}"   # 10 GiB

# --- weights -----------------------------------------------------------------
#
# fp8 (default): the BF16 checkpoint is loaded, then the decoder's 248 linear
#   layers are converted to FP8 e4m3 in place (server/fp8.py). Embeddings,
#   lm_head, the vision tower and the joint schema head stay BF16. Measured
#   against BF16 on 89 questions (server/fp8_drift.py): the same winner on 87,
#   the two others one ambiguous question BF16 itself put at 0.54. 25% faster.
# bf16: the checkpoint exactly as released. Does NOT fit beside TensorFold
#   int8x1 with this recipe's floors -- preflight says so. See the README.
WEIGHTS="${WEIGHTS:-fp8}"
case "${WEIGHTS}" in
  fp8|bf16) ;;
  *) printf 'error: WEIGHTS=%s (expected fp8 or bf16)\n' "${WEIGHTS}" >&2; return 2 2>/dev/null || exit 2 ;;
esac

# --- memory ------------------------------------------------------------------
#
# Native, so nothing caps this process from outside unless start.sh adds it.
# Two moments matter, and preflight checks both:
#
#   load peak   BF16 weights on the GPU, before the FP8 conversion, plus the
#               host side. No activations yet: the warmup runs after.
#   steady      weights as served, plus the host side, plus the activations of
#               one MAX_LENGTH forward pass. There is no KV cache (one forward
#               pass, use_cache=False), so the footprint does not grow with
#               traffic once the allocator has seen its peak -- and start.sh
#               makes it see that peak by warming up at MAX_LENGTH before it
#               reports ready, under the watchdog.
#
# Measured on this host (GB10, torch 2.11, transformers 5.10.2), beside the
# TensorFold server:
#   BF16 weights on the GPU         17.76 GiB allocated
#   FP8 weights on the GPU          11.32 GiB allocated (6.44 GiB given back)
#   host side                       ~5 GiB: CUDA context and cuBLAS (~0.6),
#                                   Python heap (~2.2), allocation overhead
#   16,384-token forward pass       2.22 GiB of allocator high-water
# BF16 measured a 24.86 GiB MemAvailable fall against 25.0 predicted.
FP8_SAVED_BYTES="${FP8_SAVED_BYTES:-6914897920}"                    # 6.44 GiB, measured
RUNTIME_RESERVE_BYTES="${RUNTIME_RESERVE_BYTES:-5368709120}"        # 5 GiB
ACTIVATION_BYTES_PER_TOKEN="${ACTIVATION_BYTES_PER_TOKEN:-147456}"  # 144 KiB

# What must remain available after each moment.
#
# Steady: above the 10.5 GiB at which this recipe's watchdog (and, at 10,
# TensorFold's) starts counting a low MemFree as danger. Below that line the
# MemFree trigger is armed all the time, and the co-tenant's ordinary SSD
# reads churn MemFree under 2 GiB within minutes: measured with BF16, which
# left 8.3 GiB, the first 50k-token Qwen prefill had this watchdog kill Clef.
# Load: a few seconds' transient, so it only has to stay above both
# MemAvailable floors (7 and 6 GiB).
HOST_FLOOR_GIB="${HOST_FLOOR_GIB:-12}"
LOAD_FLOOR_GIB="${LOAD_FLOOR_GIB:-8}"

# The cgroup cap on the host-side footprint (Python heap, mapped libraries,
# page cache from reading the shards). Verified here: CUDA allocations are not
# charged to the cgroup on GB10, so it bounds the host side only -- measured
# 4.5 GiB in steady state. 8 GiB leaves room for image requests.
CGROUP_MEM_GIB="${CGROUP_MEM_GIB:-8}"

# --- co-tenancy --------------------------------------------------------------
#
# Two watchdogs on one pool: whichever fires first is the one that stops its
# server. This one must always be first, so that a memory squeeze costs the
# small, ten-second-reload Clef rather than the 100 GiB Qwen server beside it.
# Each condition below CONTAINS TensorFold's, and it reacts faster:
#
#   this watchdog   MemAvailable < 7 GiB,  or MemFree < 2 GiB while MemAvailable < 10.5 GiB,
#                   3 samples x 1 s, then SIGKILL
#   TensorFold's    MemAvailable < 6 GiB,  or MemFree < 2 GiB while MemAvailable < 10 GiB,
#                   5 samples x 2 s, then docker stop
#
# Any state that fires TensorFold's fires this one, ~7 s sooner. The MemFree
# condition is gated on MemAvailable exactly as TensorFold's is, and must be:
# loading Clef allocates 18 GiB on the GPU in one burst, faster than the
# kernel reclaims Qwen's SSD page cache, so MemFree touches ~1 GiB for a few
# seconds while MemAvailable is still 17-20 GiB. An ungated, or more loosely
# gated, MemFree trigger kills every load for nothing (measured: a 12 GiB gate
# did exactly that). The thresholds are fractional, which memwatch.sh handles.
MEMWATCH_MIN_GIB="${MEMWATCH_MIN_GIB:-7}"
MEMWATCH_MIN_FREE_GIB="${MEMWATCH_MIN_FREE_GIB:-2}"
MEMWATCH_FREE_GATE_GIB="${MEMWATCH_FREE_GATE_GIB:-10.5}"
MEMWATCH_SAMPLES="${MEMWATCH_SAMPLES:-3}"
MEMWATCH_INTERVAL="${MEMWATCH_INTERVAL:-1}"
STOP_TIMEOUT="${STOP_TIMEOUT:-20}"

activation_bytes() {
  printf '%s' "$(( MAX_LENGTH * ACTIVATION_BYTES_PER_TOKEN ))"
}

served_weight_bytes() {
  if [[ "${WEIGHTS}" == fp8 ]]; then
    printf '%s' "$(( MODEL_TOTAL_BYTES - FP8_SAVED_BYTES ))"
  else
    printf '%s' "${MODEL_TOTAL_BYTES}"
  fi
}

load_peak_bytes() {
  printf '%s' "$(( MODEL_TOTAL_BYTES + RUNTIME_RESERVE_BYTES ))"
}

budget_bytes() {
  printf '%s' "$(( $(served_weight_bytes) + RUNTIME_RESERVE_BYTES + $(activation_bytes) ))"
}

budget_table() {
  printf '    %-21s%s\n' "weights (${WEIGHTS^^})" "$(human_bytes "$(served_weight_bytes)")"
  printf '    host side            %s  (Python, CUDA context, allocation overhead)\n' "$(human_bytes "${RUNTIME_RESERVE_BYTES}")"
  printf '    activations          %s  (%s tokens x %s KiB)\n' \
    "$(human_bytes "$(activation_bytes)")" "${MAX_LENGTH}" "$(( ACTIVATION_BYTES_PER_TOKEN / 1024 ))"
  printf '    ------------------------------\n'
  printf '    steady               %s  + %s GiB floor (HOST_FLOOR_GIB)\n' "$(human_bytes "$(budget_bytes)")" "${HOST_FLOOR_GIB}"
  printf '    load peak            %s  + %s GiB floor (LOAD_FLOOR_GIB; BF16 before conversion)\n' \
    "$(human_bytes "$(load_peak_bytes)")" "${LOAD_FLOOR_GIB}"
}

# Bytes MemAvailable must hold before a start: the stricter of the two moments.
admission_bytes() {
  local steady load
  steady="$(( $(budget_bytes) + HOST_FLOOR_GIB * 1073741824 ))"
  load="$(( $(load_peak_bytes) + LOAD_FLOOR_GIB * 1073741824 ))"
  printf '%s' "$(( steady > load ? steady : load ))"
}

# --- shared helpers ----------------------------------------------------------

mem_available_bytes() { awk '/^MemAvailable:/ {print $2 * 1024; exit}' /proc/meminfo; }
mem_free_bytes()      { awk '/^MemFree:/ {print $2 * 1024; exit}' /proc/meminfo; }
mem_total_bytes()     { awk '/^MemTotal:/ {print $2 * 1024; exit}' /proc/meminfo; }

human_bytes() {
  local n="$1" sign=''
  if [[ "${n}" == -* ]]; then sign='-'; n="${n#-}"; fi
  printf '%s%s' "${sign}" "$(numfmt --to=iec-i --round=nearest --format='%.1f' --suffix=B "${n}" 2>/dev/null || printf '%s' "${n}")"
}

log_event() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

require_aarch64() {
  [[ "$(uname -m)" == "aarch64" ]] || {
    printf 'error: this recipe requires aarch64 GB10 (found %s)\n' "$(uname -m)" >&2
    return 1
  }
}

resolve_hf_token() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    local token_file="${HF_TOKEN_PATH:-${HF_HOME}/token}"
    [[ -r "${token_file}" ]] && HF_TOKEN="$(tr -d '[:space:]' <"${token_file}")"
  fi
  export HF_TOKEN="${HF_TOKEN:-}"
}

# Verify the pinned snapshot against the manifest, skipping the re-hash when
# the files' (path, size, mtime) have not moved since the last full pass.
# FORCE_VERIFY=1 always re-hashes.
verify_snapshot() {
  local stamp fingerprint
  stamp="${OUT_DIR}/verified-$(basename "${MANIFEST}" .json)-${MODEL_REVISION:0:12}.stamp"
  fingerprint="$(python3 - "${MANIFEST}" "${SNAPSHOT_DIR}" <<'PYEOF'
import json, os, sys
manifest, root = sys.argv[1], sys.argv[2]
rows = []
for item in json.load(open(manifest))["files"]:
    try:
        st = os.stat(os.path.join(root, item["path"]))
    except OSError:
        print("MISSING"); raise SystemExit(0)
    rows.append(f'{item["path"]}:{st.st_size}:{int(st.st_mtime)}')
print("|".join(rows))
PYEOF
)"
  if [[ "${FORCE_VERIFY:-0}" != "1" && "${fingerprint}" != "MISSING" \
        && -f "${stamp}" && "$(cat "${stamp}")" == "${fingerprint}" ]]; then
    printf 'already verified since last change (FORCE_VERIFY=1 to re-hash)\n'
    return 0
  fi
  [[ "${fingerprint}" != "MISSING" ]] || {
    printf 'snapshot incomplete: %s\n' "${SNAPSHOT_DIR}" >&2
    return 1
  }
  python3 "${EXPERIMENT_DIR}/scripts/download_snapshot.py" \
    --manifest "${MANIFEST}" --hf-home "${HF_HOME}" --verify-only >/dev/null || return 1
  mkdir -p "${OUT_DIR}"
  printf '%s' "${fingerprint}" >"${stamp}"
  printf 'verified %s files against the manifest\n' \
    "$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["files"]))' "${MANIFEST}")"
}

# --- native process helpers --------------------------------------------------
#
# No container, so the pidfile is the only handle, and a stale one is the
# normal case after a crash.

server_pid() {
  [[ -f "${PID_FILE}" ]] || return 1
  local pid
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  # A recycled pid would otherwise look like a live server.
  tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null | grep -q 'server/app.py' || return 1
  printf '%s' "${pid}"
}

server_running() { server_pid >/dev/null 2>&1; }

probe_host() {
  case "${HOST}" in
    0.0.0.0|::|"") printf '127.0.0.1' ;;
    *)             printf '%s' "${HOST}" ;;
  esac
}
