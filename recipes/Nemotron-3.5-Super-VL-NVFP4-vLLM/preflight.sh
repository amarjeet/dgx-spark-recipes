#!/usr/bin/env bash
#
# Assert everything start.sh depends on, before a 70 GiB load is attempted.
#
# Memory is the check that matters. vLLM budgets GPU_MEMORY_UTILIZATION of
# total memory and refuses to start unless that much is free; on unified memory
# "free" is MemAvailable, and a pool that is truly exhausted hangs the kernel
# rather than raising an OOM. This server therefore cannot run beside another
# large one -- the other containers are listed when the check fails.
#
# Usage: ./preflight.sh [spark|nodraft]
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
select_profile "${1:-${DEFAULT_PROFILE}}"

fail=0
ok()   { printf '  OK    %s\n' "$*"; }
warn() { printf '  WARN  %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; fail=1; }

printf 'preflight: %s (seqs %s, spec tokens %s, gmu %s)\n\n' \
  "${PROFILE}" "${MAX_NUM_SEQS}" "${SPEC_TOKENS}" "${GPU_MEMORY_UTILIZATION}"

printf 'platform\n'
if require_aarch64 2>/dev/null; then ok "aarch64"; else bad "not aarch64 (found $(uname -m))"; fi
for tool in docker curl python3 nvidia-smi; do
  if command -v "${tool}" >/dev/null 2>&1; then ok "${tool} on PATH"; else bad "${tool} is not on PATH"; fi
done
if nvidia-smi >/dev/null 2>&1; then ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"; else bad "nvidia-smi failed"; fi

printf '\nruntime image\n'
if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  ok "image present: ${IMAGE:0:60}"
  version="$(docker run --rm --entrypoint python3 "${IMAGE}" -c 'import vllm; print(vllm.__version__)' 2>/dev/null | tail -1 || true)"
  if [[ -n "${version}" ]]; then ok "vLLM ${version}"; else bad "could not read vllm.__version__ from the image"; fi
  # The card's flags, checked against the image rather than assumed: a tag that
  # predates them fails minutes into a load otherwise. Read from the source,
  # not `vllm serve --help`, which imports vllm._C and so needs the GPU runtime.
  missing_flags="$(docker run --rm --entrypoint sh "${IMAGE}" -c '
    cd /usr/local/lib/python3.12/dist-packages/vllm || exit 1
    for k in mamba_backend mamba_ssm_cache_dtype enable_mamba_cache_stochastic_rounding \
             mamba_cache_philox_rounds enable_mamba_shared_prefix_checkpoint \
             video_pruning_rate prefix_match_unit speculative_config; do
      grep -q "$k" engine/arg_utils.py || printf "%s " "$k"
    done
    grep -q dflash config/speculative.py || printf "dflash "
    grep -q nemotron_v3 reasoning/__init__.py || printf "nemotron_v3-parser "
    printf "END"' 2>/dev/null || printf 'UNREADABLE')"
  if [[ "${missing_flags}" == "END" ]]; then
    ok "image has every flag, the dflash method and the nemotron_v3 parser start.sh uses"
  else
    bad "image is missing: ${missing_flags% END}"
  fi
else
  bad "image not pulled (run: docker pull ${IMAGE})"
fi

printf '\ncheckpoints\n'
manifests=("${MODEL_MANIFEST}")
(( SPEC_TOKENS > 0 )) && manifests+=("${DRAFT_MANIFEST}")
for manifest in "${manifests[@]}"; do
  repo="$(manifest_field "${manifest}" repo)"
  revision="$(manifest_field "${manifest}" revision)"
  snap="$(snapshot_dir "${manifest}")"
  ref="${HF_HOME}/hub/models--${repo//\//--}/refs/main"
  if [[ ! -d "${snap}" ]]; then
    bad "${repo}: pinned snapshot not in the HF cache (run: ./download.sh)"; continue
  fi
  # vLLM resolves the repo id through refs/main, so it must be the pinned commit.
  if [[ "$(cat "${ref}" 2>/dev/null)" != "${revision}" ]]; then
    bad "${repo}: refs/main is not the pinned ${revision:0:12}"; continue
  fi
  if verify_manifest "${manifest}" >/dev/null 2>&1; then
    ok "${repo} @ ${revision:0:12}, sha256 verified"
  else
    bad "${repo}: verification failed (run: ./download.sh --verify-only)"
  fi
done

printf '\nmemory\n'
total="$(mem_total_bytes)"; avail="$(mem_available_bytes)"
need="$(python3 -c "print(int(${GPU_MEMORY_UTILIZATION} * ${total}) + ${MEM_HEADROOM_BYTES})")"
if (( avail >= need )); then
  ok "MemAvailable $(human_bytes "${avail}") >= budget + headroom $(human_bytes "${need}")"
else
  bad "MemAvailable $(human_bytes "${avail}") < budget + headroom $(human_bytes "${need}")"
  printf '        other containers:\n'
  docker ps --format '        {{.Names}}  {{.Image}}' | grep -v "^        ${CONTAINER_NAME} " || true
  printf '        stop the other model servers, or lower GPU_MEMORY_UTILIZATION\n'
fi
swap_used="$(awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print (t - f) * 1024}' /proc/meminfo)"
(( swap_used < 1073741824 )) && ok "swap in use $(human_bytes "${swap_used}")" || warn "swap in use $(human_bytes "${swap_used}")"

printf '\nport and disk\n'
if ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then bad "port ${PORT} is already in use"; else ok "port ${PORT} free"; fi
mkdir -p "${HF_HOME}" "${VLLM_CACHE_HOST}" "${FLASHINFER_CACHE_HOST}" "${TRITON_CACHE_HOST}"
free_disk="$(df --output=avail -B1 "${VLLM_CACHE_HOST}" | tail -1 | tr -d ' ')"
(( free_disk > DISK_RESERVE_BYTES )) && ok "disk free $(human_bytes "${free_disk}")" || bad "disk free $(human_bytes "${free_disk}") < reserve"

printf '\n'
if (( fail )); then printf 'preflight FAILED\n' >&2; exit 1; fi
printf 'preflight passed\n'
