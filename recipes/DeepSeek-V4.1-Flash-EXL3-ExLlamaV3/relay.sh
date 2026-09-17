#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2026 Victor Cruz
# Copyright (C) 2026 amarjeet
#
# Derived from the "Pack preparation: 64-byte re-lay" section of
# vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe (one-spark-tp1/README.md),
# which is Copyright (C) 2026 Victor Cruz and licensed AGPL-3.0-only.
# Modified 2026-09-17 by amarjeet: upstream documents the command in prose; this
# wraps it with the pinned source/destination paths, a disk check and a
# dangling-symlink check.
#
# Re-lay the EXL3 pack at 64-byte alignment.
#
# Zero-copy aliasing can only alias a tensor whose bytes already sit where the
# kernels expect them: EXL3 trellis kernels need their int16 data on a 16-byte
# boundary, and safetensors writers pack tensors back to back. Upstream measured
# that WITHOUT this step only 48.6 GiB aliased and 67.4 GiB were copied, which
# defeats the entire point of the ATS loader.
#
# Shards already on the grid are SYMLINKED, not copied, so this directory stays
# bound to the Hugging Face cache. Do not prune that cache afterwards -- see
# `preflight.sh`, which checks the chain before every launch.
#
# Usage:
#   ./relay.sh                    # re-lay into MODEL_ROOT
#   ./relay.sh --force            # rebuild even if MODEL_ROOT looks complete
#   RELAY_JOBS=8 ./relay.sh       # more parallelism
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
select_profile "${DEFAULT_PROFILE}"

FORCE=0
while (( $# )); do
  case "$1" in
    -h|--help) sed -n '14,29p' "$0" | sed 's/^# \?//'; exit 0 ;;
    --force) FORCE=1; shift ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

ALIGN_TOOL="${EXL3_SRC}/util/align_safetensors.py"

printf 'source    : %s\n' "${SNAPSHOT_DIR}"
printf 'target    : %s\n' "${MODEL_ROOT}"
printf 'align     : %s bytes, min %s, skip %s\n' \
  "${RELAY_ALIGN}" "$(human_bytes "${RELAY_MIN_BYTES}")" "${RELAY_SKIP}"
printf 'tool      : %s\n\n' "${ALIGN_TOOL}"

[[ -d "${SNAPSHOT_DIR}" ]] || {
  printf 'error: pinned snapshot not present: %s\n' "${SNAPSHOT_DIR}" >&2
  printf 'run ./download.sh first\n' >&2
  exit 1
}

[[ -f "${ALIGN_TOOL}" ]] || {
  printf 'error: align_safetensors.py not found at %s\n' "${ALIGN_TOOL}" >&2
  printf '\nClone the pinned fork first -- the branch is required, and the\n' >&2
  printf 'commit is what is pinned:\n\n' >&2
  printf '  git clone %s %s\n' "${EXL3_REPO}" "${EXL3_SRC}" >&2
  printf '  git -C %s checkout %s\n\n' "${EXL3_SRC}" "${EXL3_COMMIT}" >&2
  printf 'See README.md -> Build for the venv and aarch64 extension build.\n' >&2
  exit 1
}

if [[ -n "$(relaid_dangling)" && "$(relaid_dangling)" != "missing" ]]; then
  printf 'warning: existing target has dangling symlinks; rebuilding\n\n' >&2
  FORCE=1
fi

if [[ -d "${MODEL_ROOT}" && "${FORCE}" != "1" ]]; then
  existing="$(find "${MODEL_ROOT}" -maxdepth 1 -name '*.safetensors' | wc -l)"
  expected="$(python3 -c \
    'import json,sys; m=json.load(open(sys.argv[1])); print(sum(1 for f in m["files"] if f["path"].endswith(".safetensors")))' \
    "${MANIFEST}")"
  if (( existing == expected )); then
    printf 'already re-laid: %s shards present\n' "${existing}"
    printf '(--force to rebuild)\n'
    printf '\nnext: ./preflight.sh\n'
    exit 0
  fi
  printf 'incomplete target: %s of %s shards; rebuilding\n\n' "${existing}" "${expected}"
fi

# Only rewritten shards cost space; shards needing no change are symlinked. With
# RELAY_SKIP=.engram. the two Engram-only shards (189.13 GiB) are skipped and
# stay symlinks, so the cost is the text+drafter shards -- about 118.56 GiB,
# for a peak of roughly 426 GiB alongside the 307.72 GiB source.
#
# With upstream's .engram.embed. instead, `engram.wkv.weight` keeps both Engram
# shards off the grid and they are rewritten in full: 308 GiB written and a
# ~616 GiB peak. The skip pattern is load-bearing, hence this figure tracking it.
need="$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))
tb = m["tensor_bytes"]
skip = sys.argv[2]
# Engram shards are only skippable when the pattern actually covers every
# oversized tensor in them, which ".engram.embed." does not.
print(tb["text"] + tb["mtp"] if skip in (".engram.", ".engram") else m["total_bytes"])
' "${MANIFEST}" "${RELAY_SKIP}")"
free="$(df -B1 --output=avail "$(dirname "${MODEL_ROOT}")" 2>/dev/null | tail -1 \
        || df -B1 --output=avail "${HOME}" | tail -1)"
printf 'rewrite   : ~%s needed, %s free\n\n' \
  "$(human_bytes "${need}")" "$(human_bytes "${free}")"
if (( free < need + DISK_RESERVE_BYTES )); then
  printf 'error: need %s free (rewrite + reserve), have %s\n' \
    "$(human_bytes "$((need + DISK_RESERVE_BYTES))")" "$(human_bytes "${free}")" >&2
  exit 1
fi

mkdir -p "$(dirname "${MODEL_ROOT}")"

# Prefer the venv interpreter: align_safetensors.py imports safetensors, which
# is installed there rather than system-wide.
PY="${VENV}/bin/python"
[[ -x "${PY}" ]] || PY="python3"
printf 'python    : %s\n\n' "${PY}"

# Upstream's "~20 minutes" is for its own 18-shard, 110 GiB build. This pack
# rewrites ~118.56 GiB and byte-verifies every copy, so budget longer.
log_event "re-lay starting (writes ~$(human_bytes "${need}"), verified; expect tens of minutes on NVMe)"
"${PY}" "${ALIGN_TOOL}" \
  "${SNAPSHOT_DIR}" \
  "${MODEL_ROOT}" \
  --align "${RELAY_ALIGN}" \
  --min-bytes "${RELAY_MIN_BYTES}" \
  --skip "${RELAY_SKIP}" \
  --jobs "${RELAY_JOBS}"
log_event "re-lay complete"

dangling="$(relaid_dangling)"
if [[ -n "${dangling}" && "${dangling}" != "missing" ]]; then
  printf '\nerror: re-laid pack has dangling symlinks:\n%s\n' "${dangling}" >&2
  exit 1
fi

printf '\nre-laid pack: %s\n' "${MODEL_ROOT}"
printf 'Keep %s -- unchanged shards are symlinks into it.\n' "${HF_HOME}"
printf '\nnext: ./preflight.sh\n'
