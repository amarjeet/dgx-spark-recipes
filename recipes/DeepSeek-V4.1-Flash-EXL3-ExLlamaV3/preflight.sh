#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2026 Victor Cruz
# Copyright (C) 2026 amarjeet
#
# Derived from the guard block in one-spark-tp1/scripts/run_tp1.sh of
# vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe, which is
# Copyright (C) 2026 Victor Cruz and licensed AGPL-3.0-only.
# Modified 2026-09-17 by amarjeet: upstream checks MemAvailable, a duplicate
# process and the addressing mode inline at launch; this promotes them to a
# standalone check list and adds the checkpoint, re-laid-symlink, toolchain and
# port checks.
#
# Assert everything start.sh depends on, before a 40-second load.
#
# The expensive failure this prevents is not a crash. On unified memory an
# over-commit hangs the kernel with no OOM and no logs, and a pruned Hugging
# Face cache leaves a re-laid pack that looks complete and dies mid-load.
#
# Usage:
#   ./preflight.sh                # check the default profile
#   ./preflight.sh aliased        # check a specific profile
#   FORCE_VERIFY=1 ./preflight.sh # re-hash the checkpoint rather than trust the stamp
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

# Parse before select_profile, or `--help` is rejected as an unknown profile.
case "${1:-}" in
  -h|--help) sed -n '15,24p' "$0" | sed 's/^# \?//'; exit 0 ;;
esac
select_profile "${1:-${DEFAULT_PROFILE}}"

fail=0
ok()   { printf '  OK    %s\n' "$*"; }
warn() { printf '  WARN  %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; fail=1; }

printf 'preflight: %s (profile %s)\n' "${EXPERIMENT_NAME}" "${PROFILE}"

printf '\nplatform\n'
if require_aarch64 2>/dev/null; then
  ok "architecture $(uname -m)"
else
  bad "architecture $(uname -m); this recipe targets DGX Spark (GB10, aarch64)"
fi
for tool in python3 git curl nvidia-smi; do
  if command -v "${tool}" >/dev/null 2>&1; then
    ok "${tool} on PATH"
  else
    bad "${tool} not on PATH"
  fi
done

printf '\naddressing mode\n'
mode="$(addressing_mode)"
case "${mode}" in
  ATS) ok "addressing mode ATS -- zero-copy aliasing available" ;;
  unknown) warn "addressing mode unknown (nvidia-smi did not report it)" ;;
  *) bad "addressing mode ${mode}, expected ATS; every tensor would be copied and the pack will not fit" ;;
esac

printf '\ntoolchain\n'
if [[ -x "${VENV}/bin/python" ]]; then
  ok "venv ${VENV}"
  # A path check cannot answer this: after `pip install .` the package sits in
  # site-packages whether it came from the fork or from PyPI. Check contents.
  verdict="$(exllamav3_is_fork)"
  if [[ "${verdict}" == "ok" ]]; then
    ok "installed exllamav3 is the fork (V4.1 arch + MTP + ATS loader present)"
    ok "imported from $(exllamav3_home)"
  else
    bad "installed exllamav3 is not the fork build -- ${verdict:-import failed}"
    printf '        TabbyAPI may have overwritten it; re-run pip install . in %s\n' \
      "${EXL3_SRC}"
  fi
  cuda_ver="$("${VENV}/bin/python" -c 'import torch; print(torch.version.cuda or "none")' 2>/dev/null || printf 'no-torch')"
  case "${cuda_ver}" in
    13*) ok "torch built against CUDA ${cuda_ver}" ;;
    no-torch) bad "torch not installed in the venv -- see README.md -> Build" ;;
    *) warn "torch built against CUDA ${cuda_ver}, expected 13.x" ;;
  esac
else
  bad "venv not found at ${VENV} -- see README.md -> Build"
fi
if [[ -f "${EXL3_SRC}/util/align_safetensors.py" ]]; then
  head_sha="$(git -C "${EXL3_SRC}" rev-parse HEAD 2>/dev/null || printf 'unknown')"
  if [[ "${head_sha}" == "${EXL3_COMMIT}" ]]; then
    ok "fork checkout at pinned commit ${EXL3_COMMIT:0:12}"
  else
    warn "fork checkout at ${head_sha:0:12}, pinned is ${EXL3_COMMIT:0:12}"
  fi
else
  bad "fork checkout or util/align_safetensors.py missing at ${EXL3_SRC}"
fi
if [[ -f "${TABBY_DIR}/main.py" ]]; then
  ok "TabbyAPI at ${TABBY_DIR}"
else
  bad "TabbyAPI not found at ${TABBY_DIR} -- see README.md -> Serving"
fi

printf '\ncheckpoint\n'
if [[ -d "${SNAPSHOT_DIR}" ]]; then
  ok "pinned snapshot present (${MODEL_REVISION:0:12})"
  if verify_snapshot "${MODEL_ID}" >/dev/null 2>&1; then
    ok "all $(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["files"]))' "${MANIFEST}") files verified against the manifest"
  else
    bad "checkpoint does not match the manifest -- ./download.sh to repair"
  fi
else
  bad "pinned snapshot missing: ${SNAPSHOT_DIR} -- run ./download.sh"
fi

printf '\nre-laid pack\n'
dangling="$(relaid_dangling)"
if [[ "${dangling}" == "missing" ]]; then
  bad "re-laid pack not built: ${MODEL_ROOT} -- run ./relay.sh"
elif [[ -n "${dangling}" ]]; then
  # The whole reason this check exists: unchanged shards are symlinks into the
  # hub cache, so pruning it leaves a directory that looks complete.
  bad "dangling symlinks in ${MODEL_ROOT} (hub cache pruned?): ${dangling}"
else
  shards="$(find "${MODEL_ROOT}" -maxdepth 1 -name '*.safetensors' | wc -l)"
  ok "${shards} shards, every symlink resolves"
fi

printf '\nmemory budget\n'
printf '        resident %s + headroom %s = %s needed, pool is %s\n' \
  "$(human_bytes "${RESIDENT_BYTES}")" "$(human_bytes "${MEM_HEADROOM_BYTES}")" \
  "$(human_bytes "$((MIN_AVAIL_KB * 1024))")" "$(human_bytes "$(mem_total_bytes)")"
if (( PROFILE_FITS )); then
  ok "profile ${PROFILE} fits this pool"
else
  bad "profile ${PROFILE} cannot fit: it needs more than the whole pool"
  printf '        This pack is 111.16 GiB of text plus a 7.39 GiB drafter.\n'
  printf '        Upstream measured "whole model copied" on a 110 GiB build,\n'
  printf '        which this is not. Use the measured or aliased profile.\n'
fi
avail_kb="$(awk '/MemAvailable/{print $2}' /proc/meminfo)"
if (( avail_kb >= MIN_AVAIL_KB )); then
  ok "MemAvailable $(human_bytes "$((avail_kb * 1024))") >= $(human_bytes "$((MIN_AVAIL_KB * 1024))")"
else
  bad "MemAvailable $(human_bytes "$((avail_kb * 1024))") < $(human_bytes "$((MIN_AVAIL_KB * 1024))"); the load would take the host down"
fi
printf '        (load drives MemAvailable to roughly 5 GiB by design)\n'
# This recipe is native, so unlike the Docker recipes there is no cgroup cap
# standing between a bad budget and a hung kernel. Say so rather than imply one.
warn "native recipe: no cgroup cap and no watchdog -- the budget above is the only guard"

printf '\ntenancy\n'
# One GPU, one pool. A second model server does not fit alongside this one.
others="$(docker ps --format '{{.Names}}' 2>/dev/null || true)"
if [[ -n "${others}" ]]; then
  bad "containers hold GPU memory: $(printf '%s' "${others}" | tr '\n' ' ')"
else
  ok "no model containers running"
fi
if pgrep -f "[e]xllamav3" >/dev/null 2>&1; then
  bad "an exllamav3 process is already running"
else
  ok "no exllamav3 process running"
fi
if pid="$(server_pid)"; then
  bad "this recipe is already serving (pid ${pid}) -- ./stop.sh first"
else
  ok "no server from this recipe"
fi

printf '\nport\n'
if ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
  bad "port ${PORT} is already in use"
else
  ok "port ${PORT} free"
fi

printf '\nchat template\n'
# The pack ships none, and TabbyAPI carries only alpaca/chatml/lfm2. Without a
# template /v1/chat/completions fails while /v1/completions still works -- the
# "it started, then every request failed" outcome.
tok_cfg="${SNAPSHOT_DIR}/tokenizer_config.json"
if [[ -n "${PROMPT_TEMPLATE}" ]]; then
  if [[ -f "${PROMPT_TEMPLATE}" ]]; then
    ok "prompt template ${PROMPT_TEMPLATE}"
  else
    bad "PROMPT_TEMPLATE set but not found: ${PROMPT_TEMPLATE}"
  fi
elif [[ -f "${tok_cfg}" ]] && grep -q '"chat_template"' "${tok_cfg}" 2>/dev/null; then
  ok "pack carries its own chat_template"
else
  warn "no chat template: /v1/chat/completions will fail, /v1/completions will work"
  printf '        set PROMPT_TEMPLATE=/path/to/deepseek-v4.1.jinja to enable chat\n'
fi

printf '\nserving config\n'
printf '  profile       %s\n' "${PROFILE}"
printf '  copy regex    %s\n' "${EXL3_ATS_COPY:-<empty: alias everything>}"
printf '  mmap/draft    EXL3_ATS_MMAP=%s DRAFT=%s conf=%s\n' \
  "${EXL3_ATS_MMAP}" "${DRAFT}" "${EXL3_DSPARK_CONF}"
printf '  ctx / chunk   %s / %s\n' "${CTX}" "${CHUNK}"
printf '  endpoint      http://%s:%s/v1\n' "${HOST}" "${PORT}"

printf '\n'
if (( fail )); then
  printf 'preflight FAILED\n' >&2
  exit 1
fi
printf 'preflight OK -- ./start.sh %s\n' "${PROFILE}"
