#!/usr/bin/env bash
#
# Assert everything start.sh depends on, before a 52 GiB BF16 load is attempted.
#
# Memory is the check that matters. vLLM budgets GPU_MEMORY_UTILIZATION of
# total memory and refuses to start unless that much is free; on unified memory
# "free" is MemAvailable, and a pool that is truly exhausted hangs the kernel
# rather than raising an OOM. This server therefore cannot run beside another
# large one -- the other containers are listed when the check fails.
#
# The adapter is checked too: its rank, its targets and its tensor names must
# be what start.sh and vLLM's weight mapper expect, read from the safetensors
# header on disk, so a re-published adapter fails here and not after a load.
#
# Usage: ./preflight.sh [pi|pi-mtp|base]
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
select_profile "${1:-${DEFAULT_PROFILE}}"

fail=0
ok()   { printf '  OK    %s\n' "$*"; }
warn() { printf '  WARN  %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; fail=1; }

printf 'preflight: %s (lora %s, seqs %s, spec tokens %s, gmu %s, len %s)\n\n' \
  "${PROFILE}" "${ENABLE_LORA}" "${MAX_NUM_SEQS}" "${SPEC_TOKENS}" "${GPU_MEMORY_UTILIZATION}" "${MAX_MODEL_LEN}"

printf 'platform\n'
if require_aarch64 2>/dev/null; then ok "aarch64"; else bad "not aarch64 (found $(uname -m))"; fi
for tool in docker curl python3 nvidia-smi; do
  if command -v "${tool}" >/dev/null 2>&1; then ok "${tool} on PATH"; else bad "${tool} is not on PATH"; fi
done
if nvidia-smi >/dev/null 2>&1; then ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"; else bad "nvidia-smi failed"; fi

printf '\nruntime image\n'
if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  ok "image present: ${IMAGE:0:60}"
  # Read from the source, not `vllm serve --help`, which imports vllm._C and so
  # needs the GPU runtime. One container for every check.
  report="$(docker run --rm --entrypoint sh "${IMAGE}" -c '
    cd /usr/local/lib/python3.12/dist-packages/vllm || { echo UNREADABLE; exit 0; }
    printf "VERSION %s\n" "$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null)"
    m=""
    for k in mamba_cache_mode language_model_only enable_lora max_lora_rank \
             speculative_config enable_prefix_caching; do
      grep -q "$k" engine/arg_utils.py || m="$m $k"
    done
    grep -q default_chat_template_kwargs entrypoints/openai/cli_args.py || m="$m default_chat_template_kwargs"
    grep -q "\"qwen3_xml\"" tool_parsers/__init__.py 2>/dev/null || m="$m qwen3_xml-parser"
    grep -q "Qwen3_5ForConditionalGeneration" model_executor/models/registry.py || m="$m Qwen3_5ForConditionalGeneration"
    grep -q "qwen3_5_mtp" config/speculative.py || m="$m qwen3_5_mtp"
    grep -q "\"model.language_model.\": \"language_model.model.\"" model_executor/models/qwen3_vl.py || m="$m lora-weight-mapper"
    printf "MISSING%s\n" "$m"' 2>/dev/null || printf 'UNREADABLE')"
  version="$(sed -n 's/^VERSION //p' <<<"${report}")"
  missing="$(sed -n 's/^MISSING//p' <<<"${report}")"
  if [[ -n "${version}" ]]; then ok "vLLM ${version}"; else bad "could not read vllm.__version__ from the image"; fi
  if grep -q '^MISSING' <<<"${report}" && [[ -z "${missing// /}" ]]; then
    ok "image has the architecture, LoRA mapper, MTP drafter, parsers and every flag start.sh uses"
  else
    bad "image is missing:${missing:- (unreadable)}"
  fi
else
  bad "image not pulled (run: docker pull ${IMAGE})"
fi

printf '\ncheckpoints\n'
while IFS= read -r manifest; do
  repo="$(manifest_field "${manifest}" repo)"
  revision="$(manifest_field "${manifest}" revision)"
  snap="$(snapshot_dir "${manifest}")"
  ref="${HF_HOME}/hub/models--${repo//\//--}/refs/main"
  if [[ ! -d "${snap}" ]]; then
    bad "${repo}: pinned snapshot not in the HF cache (run: ./download.sh)"; continue
  fi
  # vLLM resolves the base repo id through refs/main, so it must be the pinned
  # commit. (The adapter is passed by snapshot path, but keep both honest.)
  if [[ "$(cat "${ref}" 2>/dev/null)" != "${revision}" ]]; then
    bad "${repo}: refs/main is not the pinned ${revision:0:12}"; continue
  fi
  if verify_manifest "${manifest}" >/dev/null 2>&1; then
    ok "${repo} @ ${revision:0:12}, sha256 verified"
  else
    bad "${repo}: verification failed (run: ./download.sh --verify-only)"
  fi
done < <(profile_manifests)

if (( ENABLE_LORA )) && [[ -d "$(snapshot_dir "${ADAPTER_MANIFEST}")" ]]; then
  printf '\nadapter\n'
  adapter_report="$(python3 -I - "$(snapshot_dir "${ADAPTER_MANIFEST}")" "${ADAPTER_RANK}" <<'PYEOF'
import json, os, re, struct, sys
root, want_rank = sys.argv[1], int(sys.argv[2])
cfg = json.load(open(os.path.join(root, "adapter_config.json")))
problems = []
if cfg.get("peft_type") != "LORA": problems.append(f"peft_type {cfg.get('peft_type')}")
if cfg.get("r") != want_rank: problems.append(f"r={cfg.get('r')}, start.sh passes --max-lora-rank {want_rank}")
if cfg.get("use_dora"): problems.append("DoRA, which vLLM does not serve")
if cfg.get("modules_to_save"): problems.append(f"modules_to_save {cfg['modules_to_save']}")
base = cfg.get("base_model_name_or_path")
with open(os.path.join(root, "adapter_model.safetensors"), "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
header.pop("__metadata__", None)
prefix = "base_model.model.model.language_model.layers."
stray = [k for k in header if not k.startswith(prefix)]
if stray: problems.append(f"{len(stray)} tensors outside {prefix}* (e.g. {stray[0]}); vLLM's mapper will not place them")
attn = sorted({int(m.group(1)) for k in header for m in [re.search(r"layers\.(\d+)\.self_attn\.", k)] if m})
if attn and any(i % 4 != 3 for i in attn): problems.append(f"attention LoRA on non-full-attention layers {attn}")
print(f"INFO base={base} r={cfg.get('r')} alpha={cfg.get('lora_alpha')} tensors={len(header)} attn_layers={len(attn)}")
for p in problems: print("PROBLEM " + p)
PYEOF
)" || adapter_report="PROBLEM could not read the adapter"
  info="$(sed -n 's/^INFO //p' <<<"${adapter_report}")"
  [[ -n "${info}" ]] && ok "${info}"
  if grep -q '^PROBLEM' <<<"${adapter_report}"; then
    while IFS= read -r p; do bad "${p#PROBLEM }"; done < <(grep '^PROBLEM' <<<"${adapter_report}")
  else
    ok "LoRA, rank ${ADAPTER_RANK}, every tensor under model.language_model.layers.*"
  fi
  [[ "${info}" == *"base=${MODEL_ID} "* ]] || warn "adapter's base_model_name_or_path is not ${MODEL_ID}"
fi

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
