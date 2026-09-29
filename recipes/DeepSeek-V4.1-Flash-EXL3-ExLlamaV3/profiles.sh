#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2026 Victor Cruz
# Copyright (C) 2026 amarjeet
#
# Derived from vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe (one-spark-tp1),
# which is Copyright (C) 2026 Victor Cruz and licensed AGPL-3.0-only.
# Modified 2026-09-17 by amarjeet: replaces upstream's scripts/run_tp1.sh
# preamble; carries its knob names, defaults and measured constants, adds the
# pinned revision and manifest, and moves every storage path to the location the
# owning tool defaults to.
#
# Shared configuration for the DeepSeek-V4.1-Flash EXL3 native-ExLlamaV3 recipe.
# Sourced by download.sh / relay.sh / preflight.sh / start.sh / stop.sh -- this
# is the ONLY place the profile table and the storage paths are defined.
#
# Usage:  source profiles.sh          # paths + defaults only
#         select_profile measured     # additionally sets the profile vars
#
# Storage rule: every path is an env-overridable variable whose default is the
# tool's own standard location. See CONVENTIONS.md at the repo root.
# This is a NATIVE experiment, not a Docker one, so there is no bind mount to
# redirect anything: each variable below must be the one the tool itself reads.

set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Pinned rather than derived from the directory name, so OUT_DIR does not move
# if the recipe is cloned or renamed.
EXPERIMENT_NAME="DeepSeek-V4.1-Flash-EXL3-ExLlamaV3"

# --- storage -----------------------------------------------------------------

# Workspace root for data no tool has an opinion about. Only the re-laid pack
# and third-party source checkouts live under it.
DGX_SPARK_ROOT="${DGX_SPARK_ROOT:-${HOME}/dgx-spark}"

# The pack is a Hugging Face repo, so it lives in the HF cache -- the variable
# the HF libraries themselves read.
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

# PyTorch's own default, replacing upstream's recipe-private
# ~/.cache/torch_extensions_v41.
#
# Honest caveat: on the documented build path this variable is INERT. ExLlamaV3
# is built ahead-of-time by setup.py (CUDAExtension + BuildExtension), not
# through torch.utils.cpp_extension.load(), and only the JIT path reads
# TORCH_EXTENSIONS_DIR. It is set to the shared default anyway because a
# recipe-private cache default is the pattern this repository's conventions
# exist to prevent -- not because the _v41 suffix was costing rebuilds.
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${HOME}/.cache/torch_extensions}"

# Third-party source checkouts. None is vendored into this repository; all are
# cloned at a pinned commit by the user, and shared by any recipe needing them,
# so none belongs in the recipe directory. ~/src is this host's existing
# convention for third-party checkouts, so follow it rather than invent a
# second one under the workspace root.
SRC_ROOT="${SRC_ROOT:-${HOME}/src}"
EXL3_SRC="${EXL3_SRC:-${SRC_ROOT}/exllamav3}"
TABBY_DIR="${TABBY_DIR:-${SRC_ROOT}/tabbyAPI}"
# Supplies the aarch64 build patch, which does NOT exist in the ExLlamaV3 fork
# at the pinned commit. AGPL-3.0-only; see README.md -> License.
VLLM_EXL3_SRC="${VLLM_EXL3_SRC:-${SRC_ROOT}/vllm-exl3}"

# The virtualenv holding the fork build. Kept outside the recipe directory: it
# is several GB of torch, and a second recipe on this host should reuse it
# rather than build the aarch64 extension again.
VENV="${VENV:-${HOME}/venvs/exl3-v41}"

# Verification stamps and archived logs -- never in the experiment dir.
OUT_DIR="${OUT_DIR:-${XDG_STATE_HOME:-${HOME}/.local/state}/dgx-spark-recipes/${EXPERIMENT_NAME}}"

# --- model -------------------------------------------------------------------

MODEL_ID="${MODEL_ID:-vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw}"
# Pinned. manifests/exl3-1.59bpw.json records this revision's every file size
# and SHA-256; the two 94.6 GiB Engram shards make an unverified transfer an
# expensive thing to discover late.
MODEL_REVISION="${MODEL_REVISION:-0c29707b4fbcdd2e7ce61bc336dd355d6a9e1994}"
MANIFEST="${MANIFEST:-${EXPERIMENT_DIR}/manifests/exl3-1.59bpw.json}"

MODEL_ORG="${MODEL_ID%%/*}"
MODEL_REPO_NAME="${MODEL_ID##*/}"
# Where download_snapshot.py puts it, and what relay.sh reads.
MODEL_PATH="${HF_HOME}/hub/models--${MODEL_ORG}--${MODEL_REPO_NAME}"
SNAPSHOT_DIR="${MODEL_PATH}/snapshots/${MODEL_REVISION}"

# The 64-byte re-laid pack relay.sh writes and the server actually loads. A
# derived, non-HF artifact, so it goes under the workspace root rather than in
# the hub cache, which huggingface_hub owns and may prune.
MODEL_ROOT="${MODEL_ROOT:-${DGX_SPARK_ROOT}/base-models/${MODEL_REPO_NAME}-a64}"
RELAY_ALIGN="${RELAY_ALIGN:-64}"
RELAY_MIN_BYTES="${RELAY_MIN_BYTES:-1048576}"
# ".engram." and NOT upstream's ".engram.embed.". align_safetensors.py decides
# per SHARD -- needs_rewrite() returns true if ANY tensor in the shard is over
# --min-bytes, unmatched by --skip and off the grid. Shards 16 and 17 hold only
# Engram tensors, but each also holds `engram.wkv.weight`: 157 MB, F8_E4M3, and
# off the 64-byte grid (offset %64 = 24 and 48 respectively). ".engram.embed."
# does not match it, so upstream's own skip pattern rewrites both shards in
# full -- 189 GiB of copying, and 616 GiB of peak disk instead of 427 GiB.
#
# Widening to ".engram." is safe on the grounds that the alias grid is the
# dtype's item size, and every tensor in those shards is 1-byte (F8_E4M3 /
# F8_E8M0) or 2-byte (BF16), never the int16 the 64-byte trellis rule exists
# for. That reasoning comes from the fork's own doc/gb10_ats_loading.md, not
# from reading the loader, so treat it as verified-by-observation after the
# first load rather than as fact -- README.md -> Known unknowns says how.
RELAY_SKIP="${RELAY_SKIP:-.engram.}"
RELAY_JOBS="${RELAY_JOBS:-4}"

# TabbyAPI derives the served id from the pack directory name and has no
# --served-model-name, so this is what clients will actually see. Kept as a
# variable because scripts/smoke.py falls back to whatever /v1/models reports.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_ROOT}")}"

# The pack ships NO chat template: tokenizer_config.json for this revision is
# 801 bytes and has no `chat_template` key, and TabbyAPI ships only alpaca,
# chatml and lfm2. Without one, /v1/chat/completions fails while /v1/completions
# works.
#
# DeepSeek publish none either -- the V4.1-Flash model card says outright that
# "this release does not include a Jinja-format chat template", shipping a
# Python reference encoder instead. tabbyapi/deepseek-v4.1-chat.jinja is derived
# from that encoder for the chat (non-thinking) path and is verified against it
# by differential test; see the header of that file for what it does not cover.
# Set PROMPT_TEMPLATE= (empty) to serve /v1/completions only.
PROMPT_TEMPLATE="${PROMPT_TEMPLATE-${EXPERIMENT_DIR}/tabbyapi/deepseek-v4.1-chat.jinja}"

# --- runtime pins ------------------------------------------------------------

# The branch is required and the commit is what is pinned: upstream ExLlamaV3
# registers DeepseekV4ForCausalLM, while DeepseekV41ForCausalLM and the GB10 ATS
# zero-copy loader exist only here. The branch tip moves; this commit does not.
EXL3_REPO="${EXL3_REPO:-https://github.com/vcruz305/exllamav3.git}"
EXL3_BRANCH="${EXL3_BRANCH:-feat/gb10-ats-load}"
EXL3_COMMIT="${EXL3_COMMIT:-954a8ca6e59d48c3e3462068ecf083fe9990f4dc}"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"

# --- runtime state -----------------------------------------------------------

PID_FILE="${EXPERIMENT_DIR}/.tabby.pid"
LOG_FILE="${EXPERIMENT_DIR}/.tabby.log"
ACTIVE_PROFILE_FILE="${EXPERIMENT_DIR}/.profile.active"

# --- serving -----------------------------------------------------------------

# 8008 is Ling-3.0-flash-Fin, 8888 is Qwen3.8-Flash-Next. Neither can be running
# at the same time as this anyway -- preflight checks that -- but a distinct
# port keeps a stale client from talking to the wrong server.
PORT="${PORT:-8009}"
# Bind address. 127.0.0.1 keeps the server off the LAN; there is no API key.
HOST="${HOST:-127.0.0.1}"

# There is no API key by default; see TABBY_DISABLE_AUTH below and README.md.
TABBY_DISABLE_AUTH="${TABBY_DISABLE_AUTH:-1}"

# start.sh refuses to serve without auth on anything but loopback. Set this to
# 1 to override that and bind a keyless server to a routable address -- only
# sane on a network you control, and never on an untrusted one.
TABBY_ALLOW_INSECURE_BIND="${TABBY_ALLOW_INSECURE_BIND:-0}"

# The load drives MemAvailable to roughly 5 GiB by design. Refuse to start below
# this, rather than taking the host down: on unified memory exhausting the pool
# hangs the kernel with no OOM and no logs.
#
# Derived rather than hardcoded, because this repository's premise is derived
# budgets. The resident portion is the text+drafter shards (1-15); the two
# Engram shards are read from disk by row, not made resident -- the model card
# is explicit: "~101 GiB of routed experts on device; the Engram tables are read
# from disk". Upstream's flat MIN_AVAIL_KB=100 GiB happens to land close on a
# 121.69 GiB host, but only by coincidence.
MEM_HEADROOM_BYTES="${MEM_HEADROOM_BYTES:-6442450944}"   # 6 GiB
DISK_RESERVE_BYTES="${DISK_RESERVE_BYTES:-10737418240}"

# Per-family byte counts, measured from every shard's safetensors header and
# recorded in the manifest. Used to size the budget per profile rather than
# trusting upstream's prose, which describes a different, unpublished pack
# build ("18 shards", "110 GiB", "~14 GiB drafter" -- none of which match this
# 17-shard pack, whose drafter is 7.39 GiB).
tensor_bytes() {
  python3 -c '
import json, sys
print(json.load(open(sys.argv[1]))["tensor_bytes"][sys.argv[2]])' \
    "${MANIFEST}" "$1"
}

_ENV_MIN_AVAIL_KB="${MIN_AVAIL_KB:-}"

# --- caller-override snapshot ------------------------------------------------

# Snapshot caller-supplied overrides at source time. select_profile() must read
# these, not the live variables: it exports EXL3_ATS_COPY etc., so a second call
# in the same shell would otherwise see its own previous values via
# "${VAR:-...}" and silently keep the first profile's settings.
#
# EXL3_ATS_COPY needs "unset" and "set to empty" told apart, because EMPTY IS A
# MEANINGFUL VALUE: an empty copy-regex means "alias every tensor, copy none",
# which is the `aliased` profile and a documented way to trade throughput for
# reclaimable memory. "${VAR:-default}" cannot express that.
if [[ -v EXL3_ATS_COPY ]]; then
  _ENV_EXL3_ATS_COPY_SET=1
  _ENV_EXL3_ATS_COPY="${EXL3_ATS_COPY}"
else
  _ENV_EXL3_ATS_COPY_SET=0
  _ENV_EXL3_ATS_COPY=""
fi
_ENV_EXL3_ATS_MMAP="${EXL3_ATS_MMAP:-}"
_ENV_EXL3_DSPARK_CONF="${EXL3_DSPARK_CONF:-}"
_ENV_DRAFT="${DRAFT:-}"
_ENV_CTX="${CTX:-}"
_ENV_CHUNK="${CHUNK:-}"

_ats_copy() {
  # $1 is the profile's default regex. Honour an explicitly empty override.
  if (( _ENV_EXL3_ATS_COPY_SET )); then
    printf '%s' "${_ENV_EXL3_ATS_COPY}"
  else
    printf '%s' "$1"
  fi
}

# --- profiles ----------------------------------------------------------------
#
# The main model and the drafter cannot both be resident: this pack measures
# 111.16 GiB of text and a 7.39 GiB drafter against a 121.69 GiB pool. Every
# profile below is a different answer to that. The tok/s figures come from
# upstream's BENCHMARKS.md and were measured on a different, smaller build.
#
#   name      copies                         decode tok/s   why
#   measured  everything except mtp.*        17.53 median   upstream's recommendation
#   aliased   nothing (all tensors aliased)  11.46 median   lowest memory, reclaimable
#   nodraft   everything                     15.13-15.22    no speculation, steadiest
#
DEFAULT_PROFILE="${DEFAULT_PROFILE:-measured}"
KNOWN_PROFILES=(measured aliased nodraft)

select_profile() {
  PROFILE="${1:-${DEFAULT_PROFILE}}"

  case "${PROFILE}" in
    measured)
      # Copy every tensor whose name does NOT start with "mtp." into CUDA
      # memory; the drafter stays aliased in page cache. The negative lookahead
      # is the whole trick, and this is the configuration that fits.
      EXL3_ATS_COPY="$(_ats_copy '^(?!mtp\.)')"
      EXL3_ATS_MMAP="${_ENV_EXL3_ATS_MMAP:-1}"
      DRAFT="${_ENV_DRAFT:-1}"
      CHUNK="${_ENV_CHUNK:-2048}"
      ;;
    aliased)
      EXL3_ATS_COPY="$(_ats_copy '')"
      EXL3_ATS_MMAP="${_ENV_EXL3_ATS_MMAP:-1}"
      DRAFT="${_ENV_DRAFT:-1}"
      # 4096 measured faster while the weights are still aliased, and fits here
      # precisely because nothing has been copied into CUDA memory.
      CHUNK="${_ENV_CHUNK:-4096}"
      ;;
    nodraft)
      EXL3_ATS_COPY="$(_ats_copy '.*')"
      EXL3_ATS_MMAP="${_ENV_EXL3_ATS_MMAP:-1}"
      DRAFT="${_ENV_DRAFT:-0}"
      CHUNK="${_ENV_CHUNK:-2048}"
      ;;
    *)
      printf 'error: unknown profile: %s (expected one of: %s)\n' \
        "${PROFILE}" "${KNOWN_PROFILES[*]}" >&2
      return 2
      ;;
  esac

  # Shared across profiles. CTX=6144 is the context every published number was
  # measured at; raising it has not been measured on this hardware.
  CTX="${_ENV_CTX:-6144}"
  EXL3_DSPARK_CONF="${_ENV_EXL3_DSPARK_CONF:-0.7}"

  MODEL_TOTAL_BYTES="$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["total_bytes"])' \
    "${MANIFEST}")"

  # What this profile makes resident. Engram never counts: the model card is
  # explicit that those tables are read from disk. The drafter counts only when
  # the copy regex pulls it in.
  local _text _mtp
  _text="$(tensor_bytes text)"
  _mtp="$(tensor_bytes mtp)"
  case "${PROFILE}" in
    measured) RESIDENT_BYTES="${_text}" ;;                  # drafter aliased
    aliased)  RESIDENT_BYTES=0 ;;                           # everything aliased
    nodraft)  RESIDENT_BYTES="$(( _text + _mtp ))" ;;       # drafter copied too
  esac

  if [[ -n "${_ENV_MIN_AVAIL_KB}" ]]; then
    MIN_AVAIL_KB="${_ENV_MIN_AVAIL_KB}"
  else
    MIN_AVAIL_KB="$(( ( RESIDENT_BYTES + MEM_HEADROOM_BYTES ) / 1024 ))"
  fi

  # A budget larger than the whole pool is not a tight fit, it is impossible.
  # `nodraft` is exactly that on this pack: 111.16 GiB of text plus a 7.39 GiB
  # drafter plus headroom exceeds a 121.69 GiB machine. Upstream measured
  # "whole model copied" on a 110 GiB build, which this is not.
  PROFILE_FITS=1
  if (( RESIDENT_BYTES + MEM_HEADROOM_BYTES > $(mem_total_bytes) )); then
    PROFILE_FITS=0
  fi

  export PROFILE EXL3_ATS_COPY EXL3_ATS_MMAP EXL3_DSPARK_CONF DRAFT CTX CHUNK
  export MODEL_TOTAL_BYTES RESIDENT_BYTES MIN_AVAIL_KB PROFILE_FITS
}

# --- helpers -----------------------------------------------------------------

mem_available_bytes() { awk '/MemAvailable/{print $2 * 1024}' /proc/meminfo; }
mem_total_bytes()     { awk '/MemTotal/{print $2 * 1024}' /proc/meminfo; }

human_bytes() {
  local n="${1:-0}"
  numfmt --to=iec-i --suffix=B --format='%.2f' -- "${n}" 2>/dev/null \
    || printf '%s B' "${n}"
}

log_event() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

require_aarch64() {
  local arch
  arch="$(uname -m)"
  [[ "${arch}" == "aarch64" ]] || {
    printf 'error: this recipe targets DGX Spark (GB10, aarch64); found %s\n' \
      "${arch}" >&2
    return 1
  }
}

# ATS addressing mode is what makes zero-copy aliasing possible at all. Without
# it every tensor is copied and the pack does not fit.
addressing_mode() {
  command -v nvidia-smi >/dev/null 2>&1 || { printf 'unknown'; return 0; }
  local mode
  mode="$(nvidia-smi -q 2>/dev/null \
    | awk -F: '/Addressing Mode/{gsub(/ /,"",$2); print $2; exit}')"
  printf '%s' "${mode:-unknown}"
}

# Where the interpreter imports exllamav3 from. Informational only: after
# `pip install .` this is site-packages whether the source was the fork or a
# PyPI wheel, so the PATH alone proves nothing.
exllamav3_home() {
  "${VENV}/bin/python" -c 'import exllamav3, os; print(os.path.dirname(exllamav3.__file__))' \
    2>/dev/null || printf ''
}

# Whether the INSTALLED package is actually the fork. This is a content check
# for the two things only the fork has -- the V4.1 architecture class and the
# ATS loader -- because TabbyAPI "enforces the latest Exllamav3 version for
# compatibility purposes" and a stock wheel would satisfy any path-based test
# while being unable to load this model at all.
exllamav3_is_fork() {
  "${VENV}/bin/python" - <<'PYEOF' 2>/dev/null
import os, sys
try:
    import exllamav3, exllamav3_ext                       # extension built?
    from exllamav3.architecture import deepseek_v41       # fork-only arch
    from exllamav3.architecture import deepseek_v41_mtp   # fork-only drafter
except Exception as exc:
    print(f"missing: {exc}")
    raise SystemExit(1)
# The ATS zero-copy loader is the other thing the fork adds; a V4.1 class
# without it would load the model and then not fit.
root = os.path.dirname(exllamav3.__file__)
for dirpath, _, names in os.walk(root):
    for name in names:
        if not name.endswith(".py"):
            continue
        try:
            with open(os.path.join(dirpath, name), encoding="utf-8", errors="ignore") as fh:
                if "EXL3_ATS_COPY" in fh.read():
                    print("ok")
                    raise SystemExit(0)
        except OSError:
            pass
print("missing: EXL3_ATS_COPY not found in the installed package")
raise SystemExit(1)
PYEOF
}

server_pid() {
  [[ -f "${PID_FILE}" ]] || return 1
  local pid
  pid="$(cat "${PID_FILE}" 2>/dev/null)"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null && printf '%s' "${pid}"
}

# The re-laid pack is mostly symlinks back into the hub cache, which is itself
# symlinks into blobs/. A pruned cache leaves a directory that looks complete
# and fails 40 s into a load, so resolve the chain rather than test -e.
relaid_dangling() {
  [[ -d "${MODEL_ROOT}" ]] || { printf 'missing'; return 0; }
  find "${MODEL_ROOT}" -maxdepth 1 -name '*.safetensors' -type l \
    -exec test ! -e {} \; -print 2>/dev/null | head -5
}

verify_snapshot() {
  local label="${1:-snapshot}"
  local stamp fingerprint
  stamp="${OUT_DIR}/verified-$(basename "${MANIFEST}" .json).stamp"

  fingerprint="$(python3 - "${MANIFEST}" "${SNAPSHOT_DIR}" <<'PYEOF'
import json, os, sys
manifest, snapshot = json.load(open(sys.argv[1])), sys.argv[2]
parts = []
for item in manifest["files"]:
    path = os.path.join(snapshot, item["path"])
    if not os.path.exists(path):
        print("MISSING")
        raise SystemExit(0)
    st = os.stat(path)
    parts.append(f"{item['path']}:{st.st_size}:{int(st.st_mtime)}")
print("|".join(parts))
PYEOF
)"

  if [[ "${FORCE_VERIFY:-0}" != "1" && "${fingerprint}" != "MISSING" \
        && -f "${stamp}" && "$(cat "${stamp}")" == "${fingerprint}" ]]; then
    printf 'already verified since last change: %s\n' "${label}"
    printf '(FORCE_VERIFY=1 to re-hash)\n'
    return 0
  fi

  HF_HOME="${HF_HOME}" python3 "${EXPERIMENT_DIR}/scripts/download_snapshot.py" \
    --manifest "${MANIFEST}" --hf-home "${HF_HOME}" --verify-only
  mkdir -p "${OUT_DIR}"
  printf '%s' "${fingerprint}" >"${stamp}"
}

# HF_TOKEN if exported, else the token `hf auth login` writes to the standard
# location. This pack is public and ungated, so an empty token is not an error
# -- it only lowers the Hub rate limit.
resolve_hf_token() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    local token_file="${HF_TOKEN_PATH:-${HF_HOME}/token}"
    [[ -r "${token_file}" ]] && HF_TOKEN="$(tr -d '[:space:]' <"${token_file}")"
  fi
  export HF_TOKEN="${HF_TOKEN:-}"
}
