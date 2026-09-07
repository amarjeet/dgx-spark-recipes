#!/usr/bin/env bash
#
# Thin launcher for recipes/Qwen3.8-Flash-Next-NVFP4-vLLM.
#
# Runs preflight before start: this model needs ~72 GiB of weights resident
# plus a ~16 GiB KV pool inside a 121.7 GiB unified pool shared with the OS, and
# on unified memory an over-commit hangs the kernel rather than raising an OOM.
# Failing in preflight beats failing ten minutes into a cold load -- or worse.
#
# It sets no storage paths. The recipe's own defaults already point at each
# tool's standard location -- see CONVENTIONS.md.
#
# Usage: run-qwen3.8-flash-next-nvfp4.sh [native|yarn512|bf16kv]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE_DIR="${REPO_ROOT}/recipes/Qwen3.8-Flash-Next-NVFP4-vLLM"
PROFILE="${1:-native}"

[[ -d "${RECIPE_DIR}" ]] || {
  printf 'error: recipe not found: %s\n' "${RECIPE_DIR}" >&2
  exit 1
}

printf '==> Qwen3.8-Flash-Next (NVFP4, multimodal) via vLLM at TP=1\n'
printf '==> profile: %s\n\n' "${PROFILE}"

cd "${RECIPE_DIR}"
bash ./preflight.sh "${PROFILE}"
printf '\n'
exec bash ./start.sh "${PROFILE}"
