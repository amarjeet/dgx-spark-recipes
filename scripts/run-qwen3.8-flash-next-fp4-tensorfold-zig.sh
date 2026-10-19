#!/usr/bin/env bash
#
# Thin launcher for recipes/Qwen3.8-Flash-Next-FP4-TensorFold-Zig.
#
# start.sh runs preflight itself before anything is stopped or loaded: a start
# needs ~108 GiB of MemAvailable, and on unified memory an over-commit hangs the
# kernel rather than raising an OOM. It does not run build.sh or download.sh: an
# image build and a 95 GiB download should be started on purpose.
#
# It sets no storage paths. The recipe's own defaults already point at each
# tool's standard location -- see CONVENTIONS.md.
#
# Usage: run-qwen3.8-flash-next-fp4-tensorfold-zig.sh [fp8x5|fp8x3|bf16x3|fp8x1]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE_DIR="${REPO_ROOT}/recipes/Qwen3.8-Flash-Next-FP4-TensorFold-Zig"
PROFILE="${1:-fp8x5}"

[[ -d "${RECIPE_DIR}" ]] || {
  printf 'error: recipe not found: %s\n' "${RECIPE_DIR}" >&2
  exit 1
}

printf '==> Qwen3.8-Flash-Next (NVIDIA FP4) via TensorFold Zig engine\n'
printf '==> profile: %s\n\n' "${PROFILE}"

cd "${RECIPE_DIR}"
exec bash ./start.sh "${PROFILE}"
