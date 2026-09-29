#!/usr/bin/env bash
#
# Thin launcher for recipes/Qwen3.8-Flash-Next-MLX4-TensorFold.
#
# start.sh runs preflight itself before anything is stopped or loaded: the
# default profile budgets ~97.8 GiB of a 121.7 GiB unified pool, and on
# unified memory an over-commit hangs the kernel rather than raising an OOM.
# It does not run build.sh or download.sh: an ~11 GB image pull and a 105 GiB
# download should be started on purpose.
#
# It sets no storage paths. The recipe's own defaults already point at each
# tool's standard location -- see CONVENTIONS.md.
#
# Usage: run-qwen3.8-flash-next-tensorfold.sh [int8x4|int8x5|int4x6|bf16x3]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE_DIR="${REPO_ROOT}/recipes/Qwen3.8-Flash-Next-MLX4-TensorFold"
PROFILE="${1:-int8x4}"

[[ -d "${RECIPE_DIR}" ]] || {
  printf 'error: recipe not found: %s\n' "${RECIPE_DIR}" >&2
  exit 1
}

printf '==> Qwen3.8-Flash-Next (MLX 4-bit + MTP) via TensorFold\n'
printf '==> profile: %s\n\n' "${PROFILE}"

cd "${RECIPE_DIR}"
exec bash ./start.sh "${PROFILE}"
