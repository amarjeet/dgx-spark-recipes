#!/usr/bin/env bash
#
# Shared configuration for the Ternary-Bonsai-2-27B llama.cpp experiment.
# Sourced by build.sh / download.sh / preflight.sh / start.sh / stop.sh /
# status.sh / bench.sh -- this is the ONLY place the profile table and the
# storage paths are defined.
#
# Usage:  source profiles.sh          # paths + defaults only
#         select_profile wide         # additionally sets the profile vars
#
# Storage rule: every path is an env-overridable variable whose default is the
# tool's own standard location. See CONVENTIONS.md at the repo root.
# This is a NATIVE experiment -- there is no container and nothing is
# bind-mounted -- so every cache variable here has to be the one the tool
# itself reads. For llama.cpp that is LLAMA_CACHE, and the default already
# points at the shared store, so nothing needs exporting.

set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Pinned rather than derived from the directory name, so OUT_DIR does not move
# if the recipe is cloned or renamed.
EXPERIMENT_NAME="Ternary-Bonsai-2-27B-GGUF-llamacpp-prism"

# --- storage -----------------------------------------------------------------

# GGUF weights live in llama.cpp's own standard cache (the env var llama.cpp
# actually reads is LLAMA_CACHE, default ~/.cache/llama.cpp), so every
# llama.cpp recipe on the host shares one store.
LLAMA_CACHE="${LLAMA_CACHE:-${HOME}/.cache/llama.cpp}"
MODEL_STORE="${MODEL_STORE:-${LLAMA_CACHE}/Ternary-Bonsai-2-27B-gguf}"

# Bench results and verification stamps -- never in the experiment dir.
OUT_DIR="${OUT_DIR:-${XDG_STATE_HOME:-${HOME}/.local/state}/dgx-spark-recipes/${EXPERIMENT_NAME}}"

# --- the fork, and why it has to be built here -------------------------------
#
# These GGUFs do not load on stock llama.cpp. PQ2_0 (ggml type id 142) and
# PTQ1_0 sit past upstream's GGML_TYPE_COUNT so it rejects them outright, and
# the weights are stored in a blockwise Hadamard-rotated basis whose matching
# activation transform only exists in the PrismML fork. The fork publishes
# prebuilt CUDA binaries for linux-x64 only, so on this aarch64 box there is
# nothing to download -- hence build.sh.
#
# Source checkouts follow this host's existing ~/src/llama.cpp-<slug>
# convention (CONVENTIONS.md, SRC_ROOT) rather than living in the recipe dir.
SRC_ROOT="${SRC_ROOT:-${HOME}/src}"
FORK_REPO="${FORK_REPO:-https://github.com/PrismML-Eng/llama.cpp.git}"
FORK_BRANCH="${FORK_BRANCH:-prism}"
# The commit behind the tagged release prism-b10687-5d80cff. Pinned rather than
# tracking the branch head, per CONVENTIONS.md "pin the runtime".
FORK_COMMIT="${FORK_COMMIT:-5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6}"
FORK_DIR="${FORK_DIR:-${SRC_ROOT}/llama.cpp-prism}"
BUILD_DIR="${BUILD_DIR:-${FORK_DIR}/build-cuda}"
# Binaries are run straight out of the build tree: libllama/libggml* sit beside
# them there, so LD_LIBRARY_PATH is enough and patchelf is not needed.
BIN_DIR="${BIN_DIR:-${BUILD_DIR}/bin}"

CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
# GB10 is compute capability 12.1. The arch-specific "a" suffix is load-bearing:
# sm_120a is NOT forwards-compatible to sm_121, and there is no virtual/PTX
# entry to JIT from, so a 120a-only binary will not run here at all. This is
# exactly what ggml's own CMake picks on this host (ggml/src/ggml-cuda/
# CMakeLists.txt appends 121a-real at CUDA >= 12.9); naming it explicitly makes
# the build reproducible without needing the GPU visible at configure time.
CUDA_ARCHS="${CUDA_ARCHS:-121a-real}"

# Minimum llama.cpp build number accepted. 10660 is the build upstream used for
# the published GB10 measurement of this model, so it is the earliest one proven
# on this hardware. Read out of the binary, not trusted from a tag.
MIN_LLAMA_BUILD="${MIN_LLAMA_BUILD:-10660}"

# --- runtime -----------------------------------------------------------------

# Pick a port nothing else on the host is using; preflight.sh checks.
# 8008 is Ling-3.0-flash-Fin, 8888 is Qwen3.8-Flash-Next, 8009 is
# DeepSeek-V4.1-Flash. None of them can be up at the same time as the default
# profile here anyway -- preflight checks that -- but a distinct port keeps a
# stale client from talking to the wrong server.
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8010}"

# Disk reserve demanded on top of the remaining download.
DISK_RESERVE_BYTES="${DISK_RESERVE_BYTES:-10737418240}"  # 10 GiB

# Compute and graph buffers, plus slack for the OS and the NVIDIA driver. This
# is the one term in budget_bytes() that is a guess rather than arithmetic, so
# it is generous. Correct it once start.sh has printed a real measurement.
COMPUTE_RESERVE_BYTES="${COMPUTE_RESERVE_BYTES:-5368709120}"  # 5 GiB

# Context checkpoints, and why they are set here rather than left at default.
#
# A hybrid model cannot partially evict its recurrent state: llama.cpp can only
# drop a sequence whole, so a client that edits its history forces a full
# re-prefill -- minutes at these depths -- instead of reusing a prefix. The
# mitigation is context checkpoints, which snapshot the recurrent state so a
# rollback lands on a checkpoint rather than at zero.
#
# They are not free. Each checkpoint is about a slot's worth of recurrent state
# (~150 MiB here), and the default is 32 per slot -- nominally ~4.8 GiB per
# slot, which with 4 slots would be a term far too large to leave out of a
# budget. -cram bounds the whole cache globally in MiB, so that is the number
# the budget actually has to carry.
CTX_CHECKPOINTS="${CTX_CHECKPOINTS:-8}"
CACHE_RAM_MIB="${CACHE_RAM_MIB:-8192}"    # 8 GiB, llama.cpp's own default

PID_FILE="${EXPERIMENT_DIR}/.llama.pid"
LOG_FILE="${EXPERIMENT_DIR}/.llama.log"
ACTIVE_PROFILE_FILE="${EXPERIMENT_DIR}/.profile.active"

# Snapshot caller-supplied overrides at source time. select_profile() must read
# these, not the live variables: it exports CTX_SIZE etc., so a second call in
# the same shell would otherwise see its own previous values via "${VAR:-...}"
# and silently keep the first profile's settings.
_ENV_CTX_SIZE="${CTX_SIZE:-}"
_ENV_PARALLEL="${PARALLEL:-}"
_ENV_BATCH_SIZE="${BATCH_SIZE:-}"
_ENV_UBATCH_SIZE="${UBATCH_SIZE:-}"
_ENV_MODEL_ROOT="${MODEL_ROOT:-}"
_ENV_SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"

# Sampling defaults are the model card's thinking-mode recommendation, which is
# also what the GGUF carries in general.sampling.* -- so a client that reads
# model defaults and one that sends nothing both get the same thing.
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"

# Thinking depth, pinned server-wide through the chat template. Unset leaves
# the template's own default, which is NOT the cheap one -- this model defaults
# to xhigh. Clients override per request with chat_template_kwargs.
# The template raises an exception on anything outside this set, so validate
# here rather than failing on the first request. Note the model card says low
# is not really honoured and behaves close to xhigh.
REASONING_EFFORT="${REASONING_EFFORT:-}"

# Vision: the 27B is a VLM and the projector is ~0.6 GiB. Loaded by default.
# MMPROJ_CPU=1 keeps it in system RAM (--no-mmproj-offload) at the cost of a
# slower image prefill. Upstream leaves the image-token cap off on CUDA, so
# IMAGE_MAX_TOKENS is unset by default; a number caps it, 0 disables capping.
MMPROJ_CPU="${MMPROJ_CPU:-0}"
IMAGE_MAX_TOKENS="${IMAGE_MAX_TOKENS:-}"

# mmap+mlock keeps the weights off swap. Set MLOCK=0 if you are deliberately
# oversubscribing.
MLOCK="${MLOCK:-1}"

MODEL_REVISION="${MODEL_REVISION:-6ed5e12bf84b7a63069882c91dd9e9218647d17b}"
MODEL_REVISION_SHORT="${MODEL_REVISION:0:12}"

# --- memory model ------------------------------------------------------------
#
# Every number here is read out of the GGUF header, not estimated:
#
#   block_count 64, full_attention_interval 4  -> 16 full-attention layers
#   head_count_kv 4, key_length 256, value_length 256
#   => KV per token = 16 * 4 * (256 + 256) * 2 B = 65536 B, exactly 64 KiB.
#
# That matches upstream's own KV-CACHE.md ("64 KiB per token on the 27B") and is
# roughly a quarter of what a dense 27B of this shape would cost. It is the
# whole reason a 262,144-token context is affordable here.
#
# The other 48 layers are linear attention: a fixed-size recurrent state per
# sequence, independent of how full the context is.
#
#   ssm.inner_size 6144, ssm.state_size 128, ssm.conv_kernel 4, fp32
#   => per layer 6144*128*4 + 6144*4*4 = 3244032 B
#   => 48 layers = 155713536 B, about 148 MiB per slot.
KV_BYTES_PER_TOKEN=65536
RECURRENT_BYTES_PER_SLOT=155713536

# --- profile table -----------------------------------------------------------
#
# Weights are the cheap part here: PQ2_0 is 6.70 GiB resident out of 121.7 GiB,
# so unlike every other recipe in this repo the interesting axis is context,
# not quantization. llama.cpp's -c is the TOTAL context divided across slots
# (llama-context.cpp: n_ctx_seq = n_ctx / n_seq_max), and per-slot context is
# hard-capped at the model's 262144 training context.
#
# Passing -np explicitly is load-bearing. With -np left at its default of -1
# the server picks 4 slots AND turns on kv_unified, which makes -c a single
# shared pool that is NOT divided. Every profile here sets both.
#
#   wide   PQ2_0   1048576 over 4 slots = 262144 each   ~85 GiB   default
#   deep   PQ2_0    262144 over 1 slot                  ~36 GiB
#   long   PQ2_0    131072 over 1 slot                  ~28 GiB
#   safe   PQ2_0    262144 over 4 slots =  65536 each   ~37 GiB   known-good
#   ptq1   PTQ1_0   262144 over 1 slot                  ~35 GiB
#
# (Budgets include an 8 GiB -cram checkpoint cache and a 5 GiB compute reserve,
# so they are floors with slack, not tight fits. ./preflight.sh prints the
# itemised version.)
#
# ===========================================================================
# READ THIS BEFORE TRUSTING A LONG CONTEXT
# ===========================================================================
# llama.cpp issue #27756 is open against exactly this architecture -- Qwen3.8-
# 27B, 48 gated-DeltaNet layers plus 16 full-attention layers -- and reports
# SILENT FAILURE at long context:
#
#     https://github.com/ggml-org/llama.cpp/issues/27756
#
# The prefill completes cleanly and then the model emits EOS as its very first
# token: tokens_predicted=1, empty content, stop_type "eos", no error anywhere.
# Reported depths: 132375 passes, ~129864 FAILS, and everything from 174495 up
# fails, including 243077. It is NON-MONOTONIC -- a depth that works can sit
# directly above one that does not -- and it reproduces on the CUDA and the CPU
# backend alike. A 30-GDN-layer control model passes 243077 on the same build,
# which points at per-layer recurrent-state accumulation rather than at a
# kernel.
#
# IT DID NOT REPRODUCE ON THIS BUILD. A needle buried mid-prompt was recalled
# at all seven swept depths -- 8202, 32358, 64629, 96837, 129104, 193583 and
# 254032 tokens -- on fork build 10687 (commit 5d80cff0). Note that 129104 is
# within a few hundred tokens of a depth the issue reports failing, and 193583
# is inside the band where it reports everything failing. The likeliest reason
# is that this fork postdates the build the issue was filed against.
#
# That is evidence, not proof: the issue describes a PROMPT-DEPENDENT onset, and
# one prompt shape per depth can confirm a depth works but never that every
# prompt at that depth works. So `wide` stays the default, and the recipe keeps
# testing rather than assuming -- scripts/smoke.py runs the needle test at the
# server's own per-slot context and tells silent EOS apart from a lost needle.
# `safe` stays under every reported failure depth for when an answer has to be
# trustworthy without re-testing.
# ===========================================================================
#
# On the packings: PQ2_0 (2.13 bpw) and PTQ1_0 (1.75 bpw) are a genuine trade,
# not an ordering. PTQ1_0 moves 17% less weight data per decode step but pays
# arithmetic to unpack dense trits, so it wins where memory bandwidth binds and
# loses where instruction throughput does. The model card measured PQ2_0 ahead
# on RTX 5090 and RTX PRO 6000 and calls them "the Blackwell cards", but those
# parts have roughly 6x GB10's bandwidth: inheriting their answer here is a
# category error. The closest analogue in upstream's own table is the L4, at a
# similar ~300 GB/s and a near-identical 29.8 t/s PQ2_0, where PTQ1_0 won
# decode by 7.7% -- and halved prompt processing. Since prefill is the binding
# cost at any real depth, PQ2_0 stays the default. ./bench.sh packs measures it
# here rather than transferring the card's answer.

DEFAULT_PROFILE="${DEFAULT_PROFILE:-wide}"
KNOWN_PROFILES=(wide deep long safe ptq1)

select_profile() {
  PROFILE="${1:-${DEFAULT_PROFILE}}"

  case "${PROFILE}" in
    wide)
      QUANT="PQ2_0"
      MANIFEST="${EXPERIMENT_DIR}/manifests/pq2.json"
      CTX_SIZE="${_ENV_CTX_SIZE:-1048576}"
      PARALLEL="${_ENV_PARALLEL:-4}"
      BATCH_SIZE="${_ENV_BATCH_SIZE:-2048}"
      UBATCH_SIZE="${_ENV_UBATCH_SIZE:-512}"
      SERVED_MODEL_NAME_DEFAULT="ternary-bonsai-2-27b-pq2-0"
      ;;
    deep)
      QUANT="PQ2_0"
      MANIFEST="${EXPERIMENT_DIR}/manifests/pq2.json"
      CTX_SIZE="${_ENV_CTX_SIZE:-262144}"
      PARALLEL="${_ENV_PARALLEL:-1}"
      BATCH_SIZE="${_ENV_BATCH_SIZE:-2048}"
      UBATCH_SIZE="${_ENV_UBATCH_SIZE:-512}"
      SERVED_MODEL_NAME_DEFAULT="ternary-bonsai-2-27b-pq2-0"
      ;;
    long)
      QUANT="PQ2_0"
      MANIFEST="${EXPERIMENT_DIR}/manifests/pq2.json"
      # 131072 is what upstream's own launcher auto-picks for a 27B on a box
      # this size, so it is the closest thing to a path someone has run.
      CTX_SIZE="${_ENV_CTX_SIZE:-131072}"
      PARALLEL="${_ENV_PARALLEL:-1}"
      BATCH_SIZE="${_ENV_BATCH_SIZE:-2048}"
      UBATCH_SIZE="${_ENV_UBATCH_SIZE:-512}"
      SERVED_MODEL_NAME_DEFAULT="ternary-bonsai-2-27b-pq2-0"
      ;;
    safe)
      # Four slots, each well under the lowest depth reported failing in
      # #27756. This is the profile to reach for when a long-context answer
      # has to be trustworthy rather than merely allocated.
      QUANT="PQ2_0"
      MANIFEST="${EXPERIMENT_DIR}/manifests/pq2.json"
      CTX_SIZE="${_ENV_CTX_SIZE:-262144}"
      PARALLEL="${_ENV_PARALLEL:-4}"
      BATCH_SIZE="${_ENV_BATCH_SIZE:-2048}"
      UBATCH_SIZE="${_ENV_UBATCH_SIZE:-512}"
      SERVED_MODEL_NAME_DEFAULT="ternary-bonsai-2-27b-pq2-0"
      ;;
    ptq1)
      QUANT="PTQ1_0"
      MANIFEST="${EXPERIMENT_DIR}/manifests/ptq1.json"
      CTX_SIZE="${_ENV_CTX_SIZE:-262144}"
      PARALLEL="${_ENV_PARALLEL:-1}"
      BATCH_SIZE="${_ENV_BATCH_SIZE:-2048}"
      UBATCH_SIZE="${_ENV_UBATCH_SIZE:-512}"
      SERVED_MODEL_NAME_DEFAULT="ternary-bonsai-2-27b-ptq1-0"
      ;;
    *)
      printf 'error: unknown profile: %s (expected one of: %s)\n' \
        "${PROFILE}" "${KNOWN_PROFILES[*]}" >&2
      return 2
      ;;
  esac

  case "${REASONING_EFFORT}" in
    ''|xhigh|medium|low) ;;
    *)
      printf 'error: REASONING_EFFORT=%s is not one of xhigh|medium|low\n' \
        "${REASONING_EFFORT}" >&2
      printf 'the chat template raises on anything else, so this would fail per-request.\n' >&2
      return 2
      ;;
  esac

  MODEL_ROOT="${_ENV_MODEL_ROOT:-${MODEL_STORE}/${QUANT}-${MODEL_REVISION_SHORT}}"

  # First file listed in the manifest is the language model; the projector is
  # matched by name so the order of the remaining entries does not matter.
  MODEL_ENTRY_REL="$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["files"][0]["path"])' \
    "${MANIFEST}")"
  MMPROJ_REL="$(python3 -c \
    'import json,sys
files = json.load(open(sys.argv[1]))["files"]
print(next((f["path"] for f in files if "mmproj" in f["path"]), ""))' \
    "${MANIFEST}")"
  MODEL_FILE="${MODEL_ROOT}/${MODEL_ENTRY_REL}"
  MMPROJ_FILE=""
  [[ -n "${MMPROJ_REL}" ]] && MMPROJ_FILE="${MODEL_ROOT}/${MMPROJ_REL}"
  MODEL_TOTAL_BYTES="$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["total_bytes"])' \
    "${MANIFEST}")"

  CTX_PER_SLOT=$(( CTX_SIZE / PARALLEL ))

  # Without --alias the API model id is the full GGUF path, which every client
  # then has to send back as "model".
  SERVED_MODEL_NAME="${_ENV_SERVED_MODEL_NAME:-${SERVED_MODEL_NAME_DEFAULT}}"

  export PROFILE QUANT MANIFEST MODEL_ROOT MODEL_FILE MMPROJ_FILE
  export MODEL_ENTRY_REL MMPROJ_REL MODEL_TOTAL_BYTES SERVED_MODEL_NAME
  export CTX_SIZE PARALLEL BATCH_SIZE UBATCH_SIZE CTX_PER_SLOT
}

# Resident bytes this profile is expected to need. A native recipe gets no
# cgroup cap, so preflight.sh has to refuse a configuration that cannot fit
# rather than discovering it during a load (CONVENTIONS.md, "Storage").
budget_bytes() {
  printf '%s' "$(( MODEL_TOTAL_BYTES \
    + CTX_SIZE * KV_BYTES_PER_TOKEN \
    + PARALLEL * RECURRENT_BYTES_PER_SLOT \
    + CACHE_RAM_MIB * 1048576 \
    + COMPUTE_RESERVE_BYTES ))"
}

# The itemised version, for a preflight failure message that explains itself.
budget_table() {
  printf '    weights + projector  %s\n' "$(human_bytes "${MODEL_TOTAL_BYTES}")"
  printf '    KV cache             %s  (%s tokens x 64 KiB)\n' \
    "$(human_bytes "$(( CTX_SIZE * KV_BYTES_PER_TOKEN ))")" "${CTX_SIZE}"
  printf '    recurrent state      %s  (%s slot(s) x 148 MiB)\n' \
    "$(human_bytes "$(( PARALLEL * RECURRENT_BYTES_PER_SLOT ))")" "${PARALLEL}"
  printf '    checkpoint cache     %s  (-cram, %s MiB)\n' \
    "$(human_bytes "$(( CACHE_RAM_MIB * 1048576 ))")" "${CACHE_RAM_MIB}"
  printf '    compute reserve      %s  (COMPUTE_RESERVE_BYTES)\n' \
    "$(human_bytes "${COMPUTE_RESERVE_BYTES}")"
  printf '    ------------------------------\n'
  printf '    total                %s\n' "$(human_bytes "$(budget_bytes)")"
}

# --- shared helpers ----------------------------------------------------------

mem_available_bytes() {
  awk '/^MemAvailable:/ {print $2 * 1024; exit}' /proc/meminfo
}

mem_total_bytes() {
  awk '/^MemTotal:/ {print $2 * 1024; exit}' /proc/meminfo
}

host_swap_used_bytes() {
  awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print (t - f) * 1024}' /proc/meminfo
}

human_bytes() {
  local n="$1" sign=''
  if [[ "${n}" == -* ]]; then sign='-'; n="${n#-}"; fi
  printf '%s%s' "${sign}" "$(numfmt --to=iec-i --round=nearest --format='%.1f' --suffix=B "${n}" 2>/dev/null || printf '%s' "${n}")"
}

log_event() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

require_aarch64() {
  [[ "$(uname -m)" == "aarch64" ]] || {
    printf 'error: this recipe requires aarch64 GB10 (found %s)\n' "$(uname -m)" >&2
    return 1
  }
}

# --- native process helpers --------------------------------------------------
#
# No container, so there is no `docker ps` to ask and no restart policy. The
# pidfile is the only handle, and a stale one is the normal case after a crash.

server_pid() {
  [[ -f "${PID_FILE}" ]] || return 1
  local pid
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  printf '%s' "${pid}"
}

server_running() {
  server_pid >/dev/null 2>&1
}

# The llama.cpp build number in the built binary. Printed as "build NNNNN" by
# --version, which goes to stderr.
bin_build_number() {
  # Subshell with pipefail off, for the same SIGPIPE reason as
  # fork_has_ternary_kernels: `head -1` closes the pipe early.
  ( set +o pipefail
    LD_LIBRARY_PATH="${BIN_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
      "${BIN_DIR}/llama-server" --version 2>&1 \
      | sed -n 's/.*build \([0-9]\+\).*/\1/p' | head -1 )
}

# Is this build the fork, or did someone build stock llama.cpp into the same
# tree? Worth an explicit check because the failure mode is not an error: stock
# llama.cpp rejects PQ2_0/PTQ1_0 outright, but it loads a plain Q2_0 of this
# family happily and emits fluent nonsense, having no Hadamard runtime.
# The ggml type names are lowercase in the symbol table (quantize_row_pq2_0
# and friends), so match those rather than the uppercase names the model card
# and the GGUF file_type use.
#
# The `set +o pipefail` is load-bearing, not tidying: `grep -q` exits the
# moment it matches, `strings` then dies of SIGPIPE, and under pipefail the
# pipeline reports 141 on success. Without this the check fails on a perfectly
# good build -- which is exactly how it failed the first time.
fork_has_ternary_kernels() {
  local lib
  for lib in "${BIN_DIR}"/libggml-base.so*; do
    [[ -f "${lib}" ]] || continue
    if ( set +o pipefail; strings "${lib}" 2>/dev/null | grep -q 'quantize_row_pq2_0' ); then
      return 0
    fi
  done
  return 1
}

# Which CUDA architectures the built CUDA backend actually contains. A binary
# without sm_121 will not run on GB10, and the error you get is unhelpful.
cuda_backend_archs() {
  local lib
  for lib in "${BIN_DIR}"/libggml-cuda.so*; do
    [[ -f "${lib}" ]] || continue
    ( set +o pipefail
      "${CUDA_PATH}/bin/cuobjdump" --list-elf "${lib}" 2>/dev/null \
        | sed -n 's/.*\.sm_\([0-9a-z]\+\)\..*/sm_\1/p' | sort -u | paste -sd, - )
    return 0
  done
  return 1
}

# Assert a GGUF really is a rotated Bonsai pack, by reading the header rather
# than trusting the filename. A file without prism.hadamard.* metadata will
# either be refused or -- worse -- produce garbage.
gguf_has_hadamard() {
  python3 "${EXPERIMENT_DIR}/scripts/gguf_probe.py" --require-hadamard "$1" >/dev/null 2>&1
}

# Checksum-verify a manifest's files, skipping the (slow) re-hash when nothing
# has changed since the last successful verification.
#
# FORCE_VERIFY=1 always re-hashes.
verify_shards() {
  local manifest="$1" model_root="$2" label="${3:-shards}"
  local stamp fingerprint
  stamp="${OUT_DIR}/verified-$(basename "${manifest}" .json).stamp"

  fingerprint="$(python3 - "${manifest}" "${model_root}" <<'PYEOF'
import json, os, sys
manifest, root = sys.argv[1], sys.argv[2]
rows = []
for item in json.load(open(manifest))["files"]:
    path = os.path.join(root, item["path"])
    try:
        st = os.stat(path)
    except FileNotFoundError:
        print("MISSING")
        raise SystemExit(0)
    rows.append(f'{item["path"]}:{st.st_size}:{int(st.st_mtime)}')
print("|".join(rows))
PYEOF
)"

  if [[ "${FORCE_VERIFY:-0}" != "1" && "${fingerprint}" != "MISSING" \
        && -f "${stamp}" && "$(cat "${stamp}")" == "${fingerprint}" ]]; then
    printf 'already verified since last change: %s\n' "${label}"
    printf '(FORCE_VERIFY=1 to re-hash)\n'
    return 0
  fi

  python3 "${EXPERIMENT_DIR}/scripts/download_model.py" \
    --manifest "${manifest}" --destination "${model_root}" --verify-only
  mkdir -p "${OUT_DIR}"
  printf '%s' "${fingerprint}" >"${stamp}"
}

# HF_TOKEN if exported, else the token `hf auth login` writes to the standard
# location. This repo is public and the Hub is readable anonymously at a lower
# rate limit, so an empty token is a warning, not an error.
resolve_hf_token() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    local token_file="${HF_TOKEN_PATH:-${HF_HOME:-${HOME}/.cache/huggingface}/token}"
    [[ -r "${token_file}" ]] && HF_TOKEN="$(tr -d '[:space:]' <"${token_file}")"
  fi
  export HF_TOKEN="${HF_TOKEN:-}"
}
