#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
# Copyright (C) 2026 amarjeet
#
# Derived from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark, which is
# Copyright (C) 2026 MiaAI Lab and licensed AGPL-3.0-or-later.
# Modified 2026-09-05 by amarjeet: config moved to profiles.sh, budget
# arithmetic to scripts/budget.py, pinned revision, bridge networking, caches
# at tool defaults
#
# Serve Qwen3.8-Flash-Next (NVFP4, multimodal) on one DGX Spark GB10 at TP=1,
# via vLLM.
#
# Adapted from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark. What changed:
# the .env file became profiles.sh, the budget arithmetic moved into
# scripts/budget.py so preflight.sh checks the same numbers this launches with,
# the checkpoint revision is pinned and checksum-verified rather than resolved
# with `ls snapshots | head -1`, every cache is bind-mounted onto the path the
# tool already defaults to inside the container, and nothing heavy or
# run-generated is written into the recipe directory.
#
# ---------------------------------------------------------------------------
# HOW IT FITS (121.69 GiB unified pool on this host)
#
#   checkpoint on disk ......  98.66 GiB
#     of which PLE table ....  26.82 GiB   -> NOT on the GPU (see below)
#   weights on GPU ..........  71.84 GiB
#   runtime overhead ........   5.6  GiB   (non-torch 3.37 + activation 1.92
#                                          + graphs 0.12, measured at TP=1)
#   MTP draft model .........   1.49 GiB   (MTP_NUM_SPECULATIVE_TOKENS > 0)
#   KV cache ................  what the host-side cap leaves (~16 GiB at the
#                              default HOST_RESERVE_GIB=26; FP8 => ~1M tokens)
#
# The PLE n-gram table is served by vLLM's CPU-offload worker from a
# MEMORY-MAPPED pre-packed file (files/build_ple_packed_table.py, built here on
# first launch, ~40 s). File-backed pages are evictable page cache, so the
# non-evictable footprint is ~78 GiB + KV instead of ~104 GiB + KV. That margin
# is what keeps the host alive: exhausting a unified pool hangs the kernel --
# no OOM, no logs.
#
# Two GB10-specific bugs in vLLM's offload path are patched at launch by
# files/patch_ple_offload.py (CUDA stream memory ops are unsupported on GB10
# and deadlocked the GPU worker after graph capture) and
# files/patch_ple_layer.py (offload rows must carry codes AND scales).
#
# SAFETY (no sudo needed):
#   * The GPU budget is capped FROM THE HOST SIDE at MemTotal -
#     HOST_RESERVE_GIB. vLLM treats this integrated GPU's "free memory" as
#     MemAvailable (page cache included) and fills the GPU side to exactly the
#     budget, so an uncapped KV wish comes straight out of the PLE page cache
#     and the free pages the NVIDIA driver needs. See scripts/budget.py.
#   * The container runs under a hard cgroup memory cap. GPU parameter
#     allocations are not charged to it on GB10, so the cap bounds the
#     host-side footprint while the GPU budget bounds the GPU side.
#   * A background watchdog (files/memwatch.sh) stops the container if host
#     MemAvailable or MemFree stays below its floors, archiving logs first.
#
# Usage:
#   ./start.sh                      # default profile (native: 262144, fp8 KV)
#   ./start.sh yarn512              # 524288 context via YaRN
#   ./start.sh bf16kv               # BF16 KV, no quantized keys
#   ./start.sh --no-launch          # patch + print the command, don't start
#   MAX_MODEL_LEN=65536 ./start.sh
#   MTP_NUM_SPECULATIVE_TOKENS=0 ./start.sh   # give back 1.49 GiB
#   HOST_RESERVE_GIB=28 ./start.sh            # more host margin, less KV
# ---------------------------------------------------------------------------
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

DO_LAUNCH=true
PROFILE_ARG=""
for arg in "$@"; do
  case "${arg}" in
    --no-launch) DO_LAUNCH=false ;;
    -h|--help)   sed -n '1,62p' "$0" | sed 's/^# \?//'; exit 0 ;;
    -*)          printf 'unknown option: %s (try --help)\n' "${arg}" >&2; exit 2 ;;
    *)           PROFILE_ARG="${arg}" ;;
  esac
done
select_profile "${PROFILE_ARG:-${DEFAULT_PROFILE}}"

info() { printf '[INFO]  %s\n' "$*"; }
ok()   { printf '[ OK ]  %s\n' "$*"; }
warn() { printf '[WARN]  %s\n' "$*"; }
err()  { printf '[ERR ]  %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null || err "docker is not on PATH"
command -v curl   >/dev/null || err "curl is not on PATH"

# What the host can actually reach depends on the publish address: 0.0.0.0
# covers loopback, but HOST=192.168.x.y or a tailnet address does not, and a
# readiness probe hardcoded to 127.0.0.1 would then wait forever on a server
# that is already up.
case "${HOST}" in
  0.0.0.0|::|"") PROBE_HOST=127.0.0.1 ;;
  *)             PROBE_HOST="${HOST}" ;;
esac
READY_URL="http://${PROBE_HOST}:${PORT}/health"

# ---------------------------------------------------------------------------
# 0. Validate the profile
# ---------------------------------------------------------------------------
[[ "${MAX_MODEL_LEN}" =~ ^[1-9][0-9]*$ ]] || err "MAX_MODEL_LEN must be a positive integer (got '${MAX_MODEL_LEN}')"
[[ "${YARN}" == "0" || "${YARN}" == "1" ]] || err "YARN must be 0 or 1 (got '${YARN}')"
# 98.66 GiB of weights through UVM hung the host upstream. Not negotiable at TP=1.
[[ "${PLE_OFFLOAD}" == "true" ]] || err "PLE_OFFLOAD=false cannot fit one Spark. Refusing."

case "${KV_CACHE_DTYPE}" in
  auto|bfloat16) ;;
  fp8|fp8_e4m3)
    warn "KV_CACHE_DTYPE=${KV_CACHE_DTYPE}: FP8 KV is a CAPACITY trade, not a free win."
    warn "      ~1.85x more KV tokens, which is what makes 512K reachable. It also"
    warn "      perturbs which blocks the sparse-attention indexer selects; neither"
    warn "      upstream's 11/11 nor the reference implementation's 6/6 -> 2/6 was"
    warn "      measured on this host. Use ./start.sh bf16kv if that matters."
    ;;
  *) err "KV_CACHE_DTYPE must be auto, bfloat16, fp8 or fp8_e4m3 (got '${KV_CACHE_DTYPE}')" ;;
esac

# YARN_FACTOR stays empty unless YaRN is actually applied; it is the single
# flag the rest of the script keys off.
YARN_FACTOR=""
if [[ "${YARN}" == "1" ]]; then
  (( MAX_MODEL_LEN <= YARN_CEILING_MODEL_LEN )) || err \
    "MAX_MODEL_LEN=${MAX_MODEL_LEN} is above YARN_CEILING_MODEL_LEN=${YARN_CEILING_MODEL_LEN}, the
       validated ceiling for one Spark. A 1M context needs ~28.8 GiB of KV, which drives
       the cgroup cap past the pool and hangs the host. Raise the ceiling only after
       re-doing the scripts/budget.py arithmetic."
  if (( MAX_MODEL_LEN <= NATIVE_MAX_MODEL_LEN )); then
    warn "YARN=1 but MAX_MODEL_LEN=${MAX_MODEL_LEN} is within the native ${NATIVE_MAX_MODEL_LEN};"
    warn "      serving it with native rope (nothing to scale)."
  else
    # Rounded UP so original_max * factor >= the served length and vLLM's own
    # derived-length check passes.
    YARN_FACTOR="$(python3 -c "import math
print(round(math.ceil(${MAX_MODEL_LEN} / ${NATIVE_MAX_MODEL_LEN} * 10000) / 10000, 4))")"
  fi
elif (( MAX_MODEL_LEN > NATIVE_MAX_MODEL_LEN )); then
  err "MAX_MODEL_LEN=${MAX_MODEL_LEN} exceeds the native ${NATIVE_MAX_MODEL_LEN} and YARN=0.
       Use ./start.sh yarn512, or set YARN=1, or lower MAX_MODEL_LEN."
fi

# ---------------------------------------------------------------------------
# 1. Resolve the pinned checkpoint. Never downloads.
# ---------------------------------------------------------------------------
info "=== Step 1: Resolve checkpoint ==="

# MODEL_PATH is bind-mounted by swapping the HF_HOME prefix for the in-container
# default, so an HF_HOME that does not contain the snapshot would produce a
# container path that does not exist. Catch that here, not four minutes into a
# load.
[[ "${MODEL_PATH}" == "${HF_HOME}"/* ]] || err \
  "MODEL_PATH is outside HF_HOME, so the bind mount cannot reach it.
       MODEL_PATH : ${MODEL_PATH}
       HF_HOME    : ${HF_HOME}"

[[ -d "${SNAPSHOT_DIR}" ]] || err \
  "pinned snapshot is not in the HF cache: ${SNAPSHOT_DIR}
       fetch it first:  ./download.sh"
[[ -f "${SNAPSHOT_DIR}/config.json" ]] || err "no config.json under ${SNAPSHOT_DIR}"

# Every file the manifest names, at the manifest's size. Cheap next to a load,
# and it turns an interrupted download into a clear message rather than a
# tensor error deep in weight loading.
python3 - "${MANIFEST}" "${SNAPSHOT_DIR}" <<'PY' || err "checkpoint is incomplete. Resume it with: ./download.sh"
import json, os, sys
manifest, root = sys.argv[1], sys.argv[2]
bad = []
for item in json.load(open(manifest))["files"]:
    path = os.path.join(root, item["path"])
    try:
        if os.stat(path).st_size != item["bytes"]:
            bad.append(item["path"])
    except OSError:
        bad.append(item["path"])
if bad:
    print("missing or wrong size: " + ", ".join(bad[:5]), file=sys.stderr)
raise SystemExit(1 if bad else 0)
PY
ok "${MODEL_ID} @ ${MODEL_REVISION:0:12}  ($(human_bytes "${MODEL_TOTAL_BYTES}"))"

# ---------------------------------------------------------------------------
# 2. Memory budget
# ---------------------------------------------------------------------------
info "=== Step 2: Memory budget ==="
mem_total_gib="$(python3 -c "print(f'{$(mem_total_bytes)/2**30:.2f}')")"
budget_json="$("${EXPERIMENT_DIR}/scripts/budget.py" \
  --mem-total-gib "${mem_total_gib}" \
  --model-bytes "${MODEL_TOTAL_BYTES}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --mtp "${MTP_NUM_SPECULATIVE_TOKENS}" \
  --ple-gib "${PLE_GIB}" \
  --overhead-gib "${OVERHEAD_GIB}" \
  --kv-bytes-per-token "${KV_BYTES_PER_TOKEN}" \
  --kv-target-gib "${KV_TARGET_GIB}" \
  --host-reserve-gib "${HOST_RESERVE_GIB}" \
  --host-slack-gib "${HOST_SLACK_GIB}" \
  --os-reserve-gib "${OS_RESERVE_GIB}" \
  --gmu "${GPU_MEMORY_UTILIZATION}" \
  --container-mem-gib "${CONTAINER_MEM_GIB}")"

read -r WEIGHTS_GPU_GIB KV_NEED_GIB BUDGET_CAP_GIB BUDGET_GIB DERIVED_GMU \
        KV_EXPECT_GIB KV_EXPECT_TOK CONTAINER_MEM_GIB MAX_CONTAINER_GIB \
        CAP_BINDS KV_FITS CONTAINER_FITS PINNED_ABOVE_CAP <<<"$(
  printf '%s' "${budget_json}" | python3 -c '
import json, sys
d = json.load(sys.stdin)
print(d["weights_gpu_gib"], d["kv_need_gib"], d["budget_cap_gib"], d["budget_gib"],
      d["gmu"], d["kv_expect_gib"], d["kv_expect_tokens"], d["container_mem_gib"],
      d["max_container_gib"], int(d["cap_binds"]), int(d["kv_fits"]),
      int(d["container_fits"]), int(d["pinned_above_cap"]))')"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${DERIVED_GMU}}"

mem_avail_gib="$(python3 -c "print(f'{$(mem_available_bytes)/2**30:.1f}')")"
mem_used_gib="$(python3 -c "print(f'{($(mem_total_bytes)-$(mem_available_bytes))/2**30:.1f}')")"

info "  unified pool ............. ${mem_total_gib} GiB total, ${mem_avail_gib} GiB available now"
info "  weights on GPU ........... ${WEIGHTS_GPU_GIB} GiB  (checkpoint minus ${PLE_GIB} GiB PLE table)"
info "  PLE table ................ ${PLE_GIB} GiB  memory-mapped in the CPU offload worker"
info "  runtime overhead ......... ${OVERHEAD_GIB} GiB"
(( MTP_NUM_SPECULATIVE_TOKENS > 0 )) && info "  MTP draft model .......... 1.49 GiB"
info "  KV needed for ${MAX_MODEL_LEN} ...... ${KV_NEED_GIB} GiB  (kv dtype ${KV_CACHE_DTYPE})"
info "  host reserve ............. ${HOST_RESERVE_GIB} GiB => GPU budget cap ${BUDGET_CAP_GIB} GiB"
(( CAP_BINDS )) && warn "  KV target ${KV_TARGET_GIB} GiB reduced to ${KV_EXPECT_GIB} GiB by HOST_RESERVE_GIB=${HOST_RESERVE_GIB}"
info "  GPU budget (gmu ${GPU_MEMORY_UTILIZATION}) .. ${BUDGET_GIB} GiB => ~${KV_EXPECT_GIB} GiB KV (~${KV_EXPECT_TOK} tokens)"
info "  container cgroup cap ..... ${CONTAINER_MEM_GIB} GiB  (hard ceiling ${MAX_CONTAINER_GIB}; bounds host-side memory only)"

if container_running; then
  info "  host footprint now ....... ${mem_used_gib} GiB used, INCLUDING the running ${CONTAINER_NAME}"
else
  info "  host footprint now ....... ${mem_used_gib} GiB used by everything else (MemTotal - MemAvailable)"
  if python3 -c "import sys; sys.exit(0 if ${mem_used_gib} > 9 else 1)"; then
    warn "  co-tenants already spend ${mem_used_gib} GiB of the ${HOST_RESERVE_GIB} GiB host reserve (~7 is normal)."
    warn "  Find them: docker stats --no-stream; ps -eo rss,cmd --sort=-rss | head"
  fi
fi

(( PINNED_ABOVE_CAP )) && warn \
  "  pinned budget ${BUDGET_GIB} GiB is ABOVE the host-side cap ${BUDGET_CAP_GIB} GiB. You asked for it; the watchdog will end it."
(( KV_FITS )) || err \
  "budget leaves ${KV_EXPECT_GIB} GiB for KV but ${MAX_MODEL_LEN} tokens need ${KV_NEED_GIB} GiB.
       Lower MAX_MODEL_LEN, or use a fp8 KV profile."
(( CONTAINER_FITS )) || err \
  "container cap ${CONTAINER_MEM_GIB} GiB exceeds the hard ceiling ${MAX_CONTAINER_GIB} GiB
       (pool minus OS_RESERVE_GIB=${OS_RESERVE_GIB}). On unified memory this is the line
       between a killed container and a hung host."
if ${DO_LAUNCH} && python3 -c "import sys; sys.exit(0 if ${mem_avail_gib} < ${CONTAINER_MEM_GIB}+4 else 1)"; then
  err "only ${mem_avail_gib} GiB available but the container may use ${CONTAINER_MEM_GIB} GiB.
       Something else is holding memory (docker ps; ps -eo rss,cmd --sort=-rss | head)."
fi
ok "  budget fits."

vm_min_free_kb="$(cat /proc/sys/vm/min_free_kbytes 2>/dev/null || printf 0)"
vm_wsf="$(cat /proc/sys/vm/watermark_scale_factor 2>/dev/null || printf 0)"
if (( vm_min_free_kb < 1048576 || vm_wsf < 100 )); then
  warn "  kernel VM tunables at defaults (min_free_kbytes=${vm_min_free_kb}, watermark_scale_factor=${vm_wsf}):"
  warn "  no free-page reserve for the NVIDIA driver. Not applied here (needs sudo)."
  warn "  See files/sysctl-spark3.conf, then: sudo sysctl -p files/sysctl-spark3.conf"
fi

# ---------------------------------------------------------------------------
# 3. GPU preflight
# ---------------------------------------------------------------------------
if ${DO_LAUNCH} && [[ "${REQUIRE_IDLE_GPU}" == "true" ]]; then
  info "=== Step 3: GPU preflight ==="
  tenants="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
             --format=csv,noheader 2>/dev/null | sed '/^$/d' || true)"
  if [[ -n "${tenants}" ]] && ! container_running; then
    printf '%s\n' "${tenants}"
    err "GPU is in use. Stop the other server first, or set REQUIRE_IDLE_GPU=false."
  fi
  ok "GPU idle."
fi

# ---------------------------------------------------------------------------
# 4. Patches + packed PLE table
# ---------------------------------------------------------------------------
info "=== Step 4: Prepare patches ==="
VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm
PLE_PKG="${VLLM_PKG}/models/qwen3_8_flash_next/nvidia/ple_layer.py"
MODELOPT_PKG="${VLLM_PKG}/model_executor/layers/quantization/modelopt.py"
QSA_OPS_PKG="${VLLM_PKG}/models/qwen3_8_flash_next/nvidia/ops/qsa.py"
QSA_NVIDIA_PKG="${VLLM_PKG}/models/qwen3_8_flash_next/nvidia/qsa.py"

docker image inspect "${IMAGE}" &>/dev/null || {
  info "Pulling ${IMAGE} ..."
  docker pull "${IMAGE}"
}

extract() {  # extract <path-in-image> <dest>
  if [[ ! -f "$2" ]]; then
    info "Extracting $(basename "$1") from image..."
    local tmp; tmp="$(docker create "${IMAGE}" /bin/true)"
    docker cp "${tmp}:$1" "$2"
    docker rm "${tmp}" >/dev/null 2>&1
  fi
}

PATCHED_PLE="${EXPERIMENT_DIR}/files/ple_layer_patched.py"
extract "${PLE_PKG}" "${PATCHED_PLE}.orig"
python3 "${EXPERIMENT_DIR}/files/patch_ple_layer.py"
[[ -f "${PATCHED_PLE}" ]] || err "PLE patch missing after patch_ple_layer.py"

PATCHED_MODELOPT="${EXPERIMENT_DIR}/files/modelopt_patched.py"
extract "${MODELOPT_PKG}" "${PATCHED_MODELOPT}.orig"
python3 "${EXPERIMENT_DIR}/files/patch_modelopt_mxfp8.py"
[[ -f "${PATCHED_MODELOPT}" ]] || err "modelopt patch missing after patch_modelopt_mxfp8.py"

# FP8 KV support for the QSA kernels. The patch compiles out when the KV cache
# is BF16, so it is applied unconditionally and costs nothing on the bf16kv
# profile.
PATCHED_QSA_OPS="${EXPERIMENT_DIR}/files/qsa_ops_patched.py"
PATCHED_QSA_NVIDIA="${EXPERIMENT_DIR}/files/qsa_nvidia_patched.py"
extract "${QSA_OPS_PKG}"    "${PATCHED_QSA_OPS}.orig"
extract "${QSA_NVIDIA_PKG}" "${PATCHED_QSA_NVIDIA}.orig"
python3 "${EXPERIMENT_DIR}/files/patch_qsa_fp8_kv.py"
[[ -f "${PATCHED_QSA_OPS}" && -f "${PATCHED_QSA_NVIDIA}" ]] || err "QSA fp8 patch missing after patch_qsa_fp8_kv.py"

OFFLOAD_DIR="${EXPERIMENT_DIR}/files/ple_offload"
mkdir -p "${OFFLOAD_DIR}/orig"
extract "${VLLM_PKG}/model_executor/layers/ple_offload_layer.py" "${OFFLOAD_DIR}/orig/ple_offload_layer.py"
for f in connector worker protocol; do
  extract "${VLLM_PKG}/v1/ple_offload/${f}.py" "${OFFLOAD_DIR}/orig/${f}.py"
done
python3 "${EXPERIMENT_DIR}/files/patch_ple_offload.py"
for f in ple_offload_layer connector worker protocol; do
  [[ -f "${OFFLOAD_DIR}/${f}.py" ]] || err "offload patch missing: ${f}.py"
done
ok "Patches ready."

# The packed PLE table is a ~27 GiB derived artifact reused across every launch,
# so it lives in vLLM's own cache root rather than beside the recipe.
PLE_CACHE_HOST="${VLLM_CACHE_HOST}/ple_cache/${MODEL_ORG}--${MODEL_NAME}"
PLE_CACHE_CTR="/root/.cache/vllm/ple_cache/${MODEL_ORG}--${MODEL_NAME}"
if ! ls "${PLE_CACHE_HOST}"/*.packed_u8 >/dev/null 2>&1; then
  info "Building packed PLE table (one-time, ~40 s, <1 GiB RAM, no GPU)..."
  mkdir -p "${PLE_CACHE_HOST}"
  docker run --rm --name "${CONTAINER_NAME}-plebuild" --memory 6g --cpus 8 \
    -v "${MODEL_PATH}:/m:ro" \
    -v "${VLLM_CACHE_HOST}/ple_cache:/out" \
    -v "${EXPERIMENT_DIR}/files/build_ple_packed_table.py:/b.py:ro" \
    --entrypoint python3 "${IMAGE}" -u /b.py \
      "/m/snapshots/${MODEL_REVISION}" "/out/${MODEL_ORG}--${MODEL_NAME}"
fi
ok "Packed PLE table: $(du -sh "${PLE_CACHE_HOST}" | cut -f1) in ${PLE_CACHE_HOST}"

# ---------------------------------------------------------------------------
# 5. Build the vLLM command
# ---------------------------------------------------------------------------
VLLM_ARGS=()
VLLM_ARGS+=(--served-model-name "${SERVED_MODEL_NAME}")
VLLM_ARGS+=(--tensor-parallel-size 1)
VLLM_ARGS+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
VLLM_ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
VLLM_ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
VLLM_ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
VLLM_ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
if [[ -n "${YARN_FACTOR}" ]]; then
  # Deep-merged into text_config.rope_parameters, which is what this model reads
  # (nvidia/qsa.py) and what vLLM's own max-len check scales by. The existing
  # mrope_section / rope_theta / partial_rotary_factor survive.
  VLLM_ARGS+=(--hf-overrides "$(printf '{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":%s,"original_max_position_embeddings":%s}}}' \
    "${YARN_FACTOR}" "${NATIVE_MAX_MODEL_LEN}")")
fi
VLLM_ARGS+=(--load-format safetensors)
VLLM_ARGS+=(--safetensors-load-strategy lazy)
VLLM_ARGS+=(--enable-chunked-prefill)
VLLM_ARGS+=(--reasoning-parser qwen3)
VLLM_ARGS+=(--enable-auto-tool-choice)
VLLM_ARGS+=(--tool-call-parser qwen3_coder)
# REQUIRED for PLE offload: only multiproc_executor spawns the offload worker.
VLLM_ARGS+=(--distributed-executor-backend mp)
[[ -n "${KV_CACHE_MEMORY}" ]] && VLLM_ARGS+=(--kv-cache-memory "${KV_CACHE_MEMORY}")
if (( MTP_NUM_SPECULATIVE_TOKENS > 0 )); then
  VLLM_ARGS+=(--speculative-config "$(printf '{"method":"mtp","num_speculative_tokens":%s}' "${MTP_NUM_SPECULATIVE_TOKENS}")")
fi
VLLM_ARGS+=(--compilation-config "$(printf '{"mode":0,"cudagraph_mode":"%s"}' "${CUDAGRAPH_MODE}")")
VLLM_ARGS+=(--host 0.0.0.0 --port "${PORT}")
# Word-split deliberately: this is the documented escape hatch for extra flags.
# shellcheck disable=SC2206
[[ -n "${EXTRA_VLLM_ARGS}" ]] && VLLM_ARGS+=(${EXTRA_VLLM_ARGS})

# Each host cache is mounted onto the path the tool already defaults to inside
# the container, so no cache environment variable has to be set in the
# container and every recipe on this host shares one copy of each.
mkdir -p "${HF_HOME}" "${VLLM_CACHE_HOST}" "${FLASHINFER_CACHE_HOST}" \
         "${TRITON_CACHE_HOST}" "${OUT_DIR}/logs"

DOCKER_ARGS=(
  -d --name "${CONTAINER_NAME}"
  --restart "${RESTART_POLICY}"
  --gpus all --ipc host
  -p "${HOST}:${PORT}:${PORT}"
  --cap-add SYS_NICE --cap-add SYS_PTRACE
  --ulimit memlock=-1 --ulimit stack=67108864
  --memory "${CONTAINER_MEM_GIB}g" --memory-swap "${CONTAINER_MEM_GIB}g"
  -e HF_HUB_OFFLINE=1
  -e TRANSFORMERS_OFFLINE=1
  -e VLLM_PLE_CPU_OFFLOAD=1
  -e VLLM_PLE_PACKED_TABLE_DIR="${PLE_CACHE_CTR}"
  -e VLLM_PLE_OFFLOAD_STEP_TIMEOUT=300
  -v "${PATCHED_PLE}:${PLE_PKG}:ro"
  -v "${PATCHED_MODELOPT}:${MODELOPT_PKG}:ro"
  -v "${PATCHED_QSA_OPS}:${QSA_OPS_PKG}:ro"
  -v "${PATCHED_QSA_NVIDIA}:${QSA_NVIDIA_PKG}:ro"
  -v "${OFFLOAD_DIR}/ple_offload_layer.py:${VLLM_PKG}/model_executor/layers/ple_offload_layer.py:ro"
  -v "${OFFLOAD_DIR}/connector.py:${VLLM_PKG}/v1/ple_offload/connector.py:ro"
  -v "${OFFLOAD_DIR}/worker.py:${VLLM_PKG}/v1/ple_offload/worker.py:ro"
  -v "${OFFLOAD_DIR}/protocol.py:${VLLM_PKG}/v1/ple_offload/protocol.py:ro"
  -v "${HF_HOME}:/root/.cache/huggingface"
  -v "${VLLM_CACHE_HOST}:/root/.cache/vllm"
  -v "${FLASHINFER_CACHE_HOST}:/root/.cache/flashinfer"
  -v "${TRITON_CACHE_HOST}:/root/.triton"
)
# HF_HUB_OFFLINE=1 means the token is not needed to serve, but pass it when
# there is one so an accidental online path still authenticates. Appended
# separately: an empty ${VAR:+...} inside the array literal above would add an
# empty argument rather than none.
resolve_hf_token
[[ -n "${HF_TOKEN}" ]] && DOCKER_ARGS+=(-e "HF_TOKEN=${HF_TOKEN}")
# shellcheck disable=SC2206
[[ -n "${EXTRA_DOCKER_ARGS}" ]] && DOCKER_ARGS+=(${EXTRA_DOCKER_ARGS})

printf '\n'
info "Config (single Spark, TP=1):"
info "  Profile:    ${PROFILE}"
info "  Model:      ${MODEL_ID} @ ${MODEL_REVISION:0:12}"
info "  Image:      ${IMAGE}"
if [[ -n "${YARN_FACTOR}" ]]; then
info "  Context:    ${MAX_MODEL_LEN} tokens (YaRN factor ${YARN_FACTOR} over native ${NATIVE_MAX_MODEL_LEN})"
else
info "  Context:    ${MAX_MODEL_LEN} tokens (native rope, no YaRN)"
fi
info "  Max seqs:   ${MAX_NUM_SEQS}   Batched tokens: ${MAX_NUM_BATCHED_TOKENS}   KV dtype: ${KV_CACHE_DTYPE}"
info "  MTP:        ${MTP_NUM_SPECULATIVE_TOKENS}$( (( MTP_NUM_SPECULATIVE_TOKENS == 0 )) && printf ' (disabled)')"
info "  Graphs:     ${CUDAGRAPH_MODE}"
info "  Listening:  ${HOST}:${PORT}"
info "  Log:        ${LOG_FILE}"
printf '\n'

if ! ${DO_LAUNCH}; then
  info "--no-launch: the command that would run"
  printf 'docker run'
  printf ' %q' "${DOCKER_ARGS[@]}" "${IMAGE}" "${MODEL_ID}" "${VLLM_ARGS[@]}"
  printf '\n'
  exit 0
fi

# ---------------------------------------------------------------------------
# 6. Launch + watchdog
# ---------------------------------------------------------------------------
info "=== Step 6: Launch ==="
archive_ts="$(date '+%Y%m%dT%H%M%S')"
if container_exists; then
  # The old container is removed below; keep its log for the post-mortem first.
  docker logs --tail 3000 "${CONTAINER_NAME}" \
    >"${OUT_DIR}/logs/${CONTAINER_NAME}-${archive_ts}-container.log" 2>&1 || true
  info "Previous container log archived: ${OUT_DIR}/logs/${CONTAINER_NAME}-${archive_ts}-container.log"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

log_event "launching ${PROFILE} ctx=${MAX_MODEL_LEN} kv=${KV_CACHE_DTYPE} mtp=${MTP_NUM_SPECULATIVE_TOKENS} gmu=${GPU_MEMORY_UTILIZATION}" \
  >"${LOG_FILE}"

docker run "${DOCKER_ARGS[@]}" "${IMAGE}" "${MODEL_ID}" "${VLLM_ARGS[@]}" >/dev/null
container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
printf '%s' "${container_id}" >"${PID_FILE}"
printf '%s' "${PROFILE}" >"${ACTIVE_PROFILE_FILE}"
ok "Container ${CONTAINER_NAME} started (${container_id:0:12})."

# Kill any previous watchdog, then start one for this container. Its logs go to
# OUT_DIR, not the recipe directory -- memwatch.sh reads both paths from the
# environment, so it stays byte-identical to upstream.
pkill -f "memwatch.sh ${CONTAINER_NAME}" 2>/dev/null || true
MEMWATCH_LOG="${OUT_DIR}/logs/memwatch-${CONTAINER_NAME}.log"
if [[ -s "${MEMWATCH_LOG}" ]]; then
  mv "${MEMWATCH_LOG}" "${OUT_DIR}/logs/${CONTAINER_NAME}-${archive_ts}-memwatch.log"
fi
MEMWATCH_MIN_FREE_GIB="${MEMWATCH_MIN_FREE_GIB}" \
MEMWATCH_FREE_GATE_GIB="${MEMWATCH_FREE_GATE_GIB}" \
MEMWATCH_GRACE="${MEMWATCH_GRACE}" \
MEMWATCH_LOG="${MEMWATCH_LOG}" \
MEMWATCH_ARCHIVE_DIR="${OUT_DIR}/logs" \
  nohup bash "${EXPERIMENT_DIR}/files/memwatch.sh" "${CONTAINER_NAME}" "${MEMWATCH_MIN_GIB}" \
  >"${MEMWATCH_LOG}" 2>&1 &
ok "Watchdog running (MemAvailable < ${MEMWATCH_MIN_GIB} GiB, or MemFree < ${MEMWATCH_MIN_FREE_GIB} GiB while MemAvailable < ${MEMWATCH_FREE_GATE_GIB} GiB, for 5 samples)"
info "  ${MEMWATCH_LOG}"

log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]] && kill -0 "${log_follow_pid}" 2>/dev/null; then
    kill "${log_follow_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

(docker logs -f "${CONTAINER_NAME}" >>"${LOG_FILE}" 2>&1) &
log_follow_pid=$!

info "Loading weights. A cold read of $(human_bytes "${MODEL_TOTAL_BYTES}") off NVMe plus graph capture"
info "takes ~10-12 minutes; a warm page cache is much faster."
printf 'waiting for %s\n' "${READY_URL}"
while ! curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! container_running; then
    printf '\n' >&2
    reason="$(grep -oE '(ValueError|RuntimeError|TimeoutError|torch\.[A-Za-z]*Error): .*' "${LOG_FILE}" \
              | grep -viE 'min_frames|max_frames' | tail -1 | cut -c1-400 || true)"
    [[ -n "${reason}" ]] && printf 'vLLM reported:\n  %s\n\n' "${reason}" >&2
    tail -n 40 "${LOG_FILE}" >&2 || true
    if docker inspect "${CONTAINER_NAME}" --format '{{.State.OOMKilled}}' 2>/dev/null | grep -q true; then
      printf '\nContainer was OOM-killed by its cgroup cap (%s GiB) -- the host survived as designed.\n' \
        "${CONTAINER_MEM_GIB}" >&2
      printf 'Retry with a smaller context, e.g.:  MAX_MODEL_LEN=%s ./start.sh %s\n' \
        "$(( MAX_MODEL_LEN / 2 ))" "${PROFILE}" >&2
    fi
    err "container exited before becoming ready. Full log: ${LOG_FILE}"
  fi
  printf '  still loading (avail %s, swap used %s)\n' \
    "$(human_bytes "$(mem_available_bytes)")" "$(human_bytes "$(host_swap_used_bytes)")"
  sleep 15
done

printf '\n'
ok "vLLM ready on port ${PORT} (TP=1, single Spark)."
grep -iE 'GPU KV cache size|Available KV cache|Maximum concurrency' "${LOG_FILE}" | tail -3 || true
used_now="$(( $(mem_total_bytes) - $(mem_available_bytes) ))"
printf '  host in use    %s of %s\n' \
  "$(human_bytes "${used_now}")" "$(human_bytes "$(mem_total_bytes)")"
printf '\nOpenAI base URL : http://%s:%s/v1\n' "${PROBE_HOST}" "${PORT}"
printf 'model id        : %s\n' "${SERVED_MODEL_NAME}"
printf 'smoke test      : ./scripts/smoke.py\n'
printf 'status          : ./status.sh\n'
printf 'stop            : ./stop.sh\n'
