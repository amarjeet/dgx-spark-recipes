#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
# Copyright (C) 2026 amarjeet
#
# Derived from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark, which is
# Copyright (C) 2026 MiaAI Lab and licensed AGPL-3.0-or-later.
# Modified 2026-09-05 by amarjeet: new script; asserts upstream's documented
# preconditions before a long load
#
# Assert everything start.sh depends on, before a 98.66 GiB load is attempted.
#
# The checks that actually matter, in order of how likely they are to bite:
#   1. Image contents -- the tag is mutable and start.sh bind-mounts patched
#      copies of five specific vLLM modules over the ones in the image. If a
#      re-pull moves or renames any of them the mount silently shadows nothing
#      (or the container fails to start), so the paths are checked by name.
#   2. Free memory -- other model servers on this box hold tens of GiB, and on
#      unified memory an exhausted pool hangs the kernel instead of raising an
#      OOM. The budget arithmetic is checked here rather than four minutes into
#      a load.
#   3. Checkpoint integrity -- a truncated shard surfaces as a confusing tensor
#      error mid-load.
#   4. Kernel VM tunables -- stock values give the NVIDIA driver no free-page
#      reserve on a 121 GiB box.
#
# Usage: ./preflight.sh [PROFILE]
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
select_profile "${1:-${DEFAULT_PROFILE}}"

fail=0
ok()   { printf '  OK    %s\n' "$*"; }
warn() { printf '  WARN  %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; fail=1; }

printf 'preflight: %s (ctx %s, kv %s, mtp %s)\n\n' \
  "${PROFILE}" "${MAX_MODEL_LEN}" "${KV_CACHE_DTYPE}" "${MTP_NUM_SPECULATIVE_TOKENS}"

printf 'platform\n'
if require_aarch64 2>/dev/null; then ok "aarch64"; else bad "not aarch64 (found $(uname -m))"; fi
for tool in docker curl python3 nvidia-smi; do
  if command -v "${tool}" >/dev/null 2>&1; then ok "${tool} on PATH"; else bad "${tool} is not on PATH"; fi
done
if nvidia-smi >/dev/null 2>&1; then
  ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
  bad "nvidia-smi failed"
fi

printf '\nruntime image\n'
if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  ok "image present: ${IMAGE}"

  version="$(image_vllm_version || true)"
  if [[ -z "${version}" ]]; then
    bad "could not read vllm.__version__ from ${IMAGE}"
  else
    ok "vLLM ${version}"
  fi

  # Every path start.sh patches, checked by name inside the image. A stock
  # vllm-openai tag has none of the qwen3_8_flash_next or ple_offload paths,
  # which is the single most likely way to get this recipe wrong.
  missing_modules="$(docker run --rm --entrypoint python3 "${IMAGE}" -c '
import os, sys
base = "/usr/local/lib/python3.12/dist-packages/vllm"
paths = [
    "models/qwen3_8_flash_next/nvidia/ple_layer.py",
    "models/qwen3_8_flash_next/nvidia/ops/qsa.py",
    "models/qwen3_8_flash_next/nvidia/qsa.py",
    "model_executor/layers/quantization/modelopt.py",
    "model_executor/layers/ple_offload_layer.py",
    "v1/ple_offload/connector.py",
    "v1/ple_offload/worker.py",
    "v1/ple_offload/protocol.py",
]
print(" ".join(p for p in paths if not os.path.isfile(os.path.join(base, p))))
' 2>/dev/null || printf 'UNREADABLE')"
  if [[ "${missing_modules}" == "UNREADABLE" ]]; then
    bad "could not list vLLM module paths inside ${IMAGE}"
  elif [[ -n "${missing_modules}" ]]; then
    bad "image is missing modules start.sh patches: ${missing_modules}"
    printf '        this is almost certainly a stock vllm-openai tag rather than the\n'
    printf '        qwen38-flash-next build. Run: docker pull %s\n' "${IMAGE}"
  else
    ok "all 8 patched module paths present in the image"
  fi
else
  bad "image not pulled: ${IMAGE} (run: docker pull ${IMAGE})"
fi

printf '\ncheckpoint\n'
printf '  repo     %s @ %s\n' "${MODEL_ID}" "${MODEL_REVISION:0:12}"
printf '  snapshot %s\n' "${SNAPSHOT_DIR}"
if [[ ! -d "${SNAPSHOT_DIR}" ]]; then
  bad "pinned snapshot is not in the HF cache"
  printf '\n  run: ./download.sh\n'
else
  # 51 files, so report a count and a handful of names rather than a wall of
  # one line per file.
  problems="$(python3 - "${MANIFEST}" "${SNAPSHOT_DIR}" <<'PYEOF'
import json, os, sys
manifest, root = sys.argv[1], sys.argv[2]
bad = []
for item in json.load(open(manifest))["files"]:
    path = os.path.join(root, item["path"])
    try:
        size = os.stat(path).st_size
    except OSError:
        bad.append((item["path"], "missing"))
        continue
    if size != item["bytes"]:
        bad.append((item["path"], f"{size} != {item['bytes']} bytes"))
for name, why in bad[:5]:
    print(f"{name}: {why}")
if len(bad) > 5:
    print(f"... and {len(bad) - 5} more")
print(f"COUNT {len(bad)}")
PYEOF
)"
  missing="$(printf '%s' "${problems}" | sed -n 's/^COUNT //p')"
  # An empty COUNT means the check itself failed, not that the files are fine.
  if [[ ! "${missing}" =~ ^[0-9]+$ ]]; then
    bad "could not inspect the snapshot"
    printf '%s\n' "${problems}" | sed 's/^/          /'
    missing=0
  elif (( missing )); then
    bad "${missing} of 51 files missing or the wrong size:"
    printf '%s\n' "${problems}" | grep -v '^COUNT ' | sed 's/^/          /'
    printf '\n  run: ./download.sh\n'
  else
    ok "all 51 files present at the manifest sizes ($(human_bytes "${MODEL_TOTAL_BYTES}"))"
    # Full SHA-256 pass, skipped when nothing moved since the last one.
    if verify_snapshot "${MODEL_ID}" >/dev/null 2>&1; then
      ok "sha256 verified"
    else
      bad "sha256 verification failed (re-run: ./download.sh --verify-only)"
    fi
  fi
fi

printf '\ndisk\n'
mkdir -p "${HF_HOME}" "${VLLM_CACHE_HOST}"
free_disk="$(df --output=avail -B1 "${HF_HOME}" | tail -1)"
if (( free_disk >= DISK_RESERVE_BYTES )); then
  ok "free on $(df --output=target "${HF_HOME}" | tail -1): $(human_bytes "${free_disk}")"
else
  bad "only $(human_bytes "${free_disk}") free, want $(human_bytes "${DISK_RESERVE_BYTES}")"
fi
# The packed PLE table is built on first launch and lives in vLLM's own cache.
ple_dir="${VLLM_CACHE_HOST}/ple_cache/${MODEL_ORG}--${MODEL_NAME}"
if ls "${ple_dir}"/*.packed_u8 >/dev/null 2>&1; then
  ok "packed PLE table already built ($(du -sh "${ple_dir}" | cut -f1))"
else
  warn "packed PLE table not built yet -- start.sh builds it once (~40 s, ~27 GiB in ${VLLM_CACHE_HOST}/ple_cache)"
fi

printf '\nmemory budget\n'
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

read -r b_weights b_kvneed b_cap b_budget b_gmu b_kvexp b_kvtok b_ctr b_maxctr b_capbinds b_kvfits b_ctrfits b_pinhigh <<<"$(
  printf '%s' "${budget_json}" | python3 -c '
import json, sys
d = json.load(sys.stdin)
print(d["weights_gpu_gib"], d["kv_need_gib"], d["budget_cap_gib"], d["budget_gib"],
      d["gmu"], d["kv_expect_gib"], d["kv_expect_tokens"], d["container_mem_gib"],
      d["max_container_gib"], int(d["cap_binds"]), int(d["kv_fits"]),
      int(d["container_fits"]), int(d["pinned_above_cap"]))')"

avail="$(mem_available_bytes)"
printf '  unified pool    %s total, %s available now\n' \
  "$(human_bytes "$(mem_total_bytes)")" "$(human_bytes "${avail}")"
printf '  swap used       %s\n' "$(human_bytes "$(host_swap_used_bytes)")"
printf '  weights on GPU  %s GiB  (checkpoint minus %s GiB PLE table)\n' "${b_weights}" "${PLE_GIB}"
printf '  KV for %s   %s GiB (%s)\n' "${MAX_MODEL_LEN}" "${b_kvneed}" "${KV_CACHE_DTYPE}"
printf '  host reserve    %s GiB => GPU budget cap %s GiB\n' "${HOST_RESERVE_GIB}" "${b_cap}"
printf '  GPU budget      %s GiB (gmu %s) => ~%s GiB KV (~%s tokens)\n' \
  "${b_budget}" "${b_gmu}" "${b_kvexp}" "${b_kvtok}"
printf '  cgroup cap      %s GiB (hard ceiling %s)\n' "${b_ctr}" "${b_maxctr}"

if (( b_capbinds )); then
  warn "KV target ${KV_TARGET_GIB} GiB reduced to ${b_kvexp} GiB by HOST_RESERVE_GIB=${HOST_RESERVE_GIB}"
fi
if (( b_pinhigh )); then
  bad "pinned GPU_MEMORY_UTILIZATION puts the budget above the host-side cap ${b_cap} GiB"
  printf '        this is the configuration that killed three servers upstream on 2026-09-04.\n'
fi
if (( b_kvfits )); then
  ok "KV pool holds a full ${MAX_MODEL_LEN}-token request"
else
  bad "budget leaves ${b_kvexp} GiB for KV but ${MAX_MODEL_LEN} tokens need ${b_kvneed} GiB"
  printf '        lower MAX_MODEL_LEN, or use the fp8 KV profiles (native/yarn512).\n'
fi
if (( b_ctrfits )); then
  ok "cgroup cap ${b_ctr} GiB is under the ${b_maxctr} GiB ceiling"
else
  bad "cgroup cap ${b_ctr} GiB exceeds the ${b_maxctr} GiB ceiling (pool minus OS_RESERVE_GIB=${OS_RESERVE_GIB})"
fi

# The container may grow to its cgroup cap; the pool has to have that free now.
needed_bytes="$(python3 -c "print(int((${b_ctr} + 4) * 2**30))")"
if (( avail >= needed_bytes )); then
  ok "available memory covers the cgroup cap plus 4 GiB"
else
  bad "only $(human_bytes "${avail}") available, the container may use ${b_ctr} GiB"
  others="$(docker ps --format '{{.Names}}' | grep -vx "${CONTAINER_NAME}" || true)"
  if [[ -n "${others}" ]]; then
    printf '\n  other containers are holding memory:\n'
    docker ps --format '{{.Names}}  {{.Image}}  {{.Status}}' \
      | grep -v "^${CONTAINER_NAME}  " | sed 's/^/    /' || true
    printf '  stop the one you do not need, then re-run preflight.\n'
  fi
fi

printf '\nGPU tenancy\n'
tenants="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
           --format=csv,noheader 2>/dev/null | sed '/^$/d' || true)"
if [[ -z "${tenants}" ]]; then
  ok "GPU idle"
elif container_running; then
  ok "GPU held by our own container ${CONTAINER_NAME}"
else
  if [[ "${REQUIRE_IDLE_GPU}" == "true" ]]; then
    bad "GPU is in use by something else (REQUIRE_IDLE_GPU=false to ignore)"
  else
    warn "GPU is in use, REQUIRE_IDLE_GPU=false"
  fi
  printf '%s\n' "${tenants}" | sed 's/^/    /'
fi

printf '\nkernel VM tunables\n'
vm_min_free_kb="$(cat /proc/sys/vm/min_free_kbytes 2>/dev/null || printf 0)"
vm_wsf="$(cat /proc/sys/vm/watermark_scale_factor 2>/dev/null || printf 0)"
if (( vm_min_free_kb < 1048576 || vm_wsf < 100 )); then
  # A warning, never a failure: applying these needs sudo and they change how
  # MemAvailable is accounted, which shifts every figure above.
  warn "at defaults (min_free_kbytes=${vm_min_free_kb}, watermark_scale_factor=${vm_wsf}): no free-page reserve for the NVIDIA driver"
  printf '        optional: sudo sysctl -p %s/files/sysctl-spark3.conf\n' "${EXPERIMENT_DIR}"
  printf '        read that file'"'"'s header first -- it shifts MemAvailable accounting.\n'
else
  ok "min_free_kbytes=${vm_min_free_kb}, watermark_scale_factor=${vm_wsf}"
fi

printf '\nport\n'
if ss -ltn "sport = :${PORT}" 2>/dev/null | tail -n +2 | grep -q .; then
  if container_running; then
    ok "port ${PORT} held by our own container ${CONTAINER_NAME}"
  else
    bad "port ${PORT} is already in use by something else"
  fi
else
  ok "port ${PORT} is free"
fi

printf '\n'
if (( fail )); then
  printf 'preflight FAILED\n' >&2
  exit 1
fi
printf 'preflight OK -- ./start.sh %s\n' "${PROFILE}"
