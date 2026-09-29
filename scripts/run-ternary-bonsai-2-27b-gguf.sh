#!/usr/bin/env bash
#
# Thin launcher for recipes/Ternary-Bonsai-2-27B-GGUF-llamacpp-prism.
#
# Runs preflight before start: the default profile commits 64 GiB of KV cache
# up front, and failing in preflight beats failing partway into a load.
#
# It does NOT run ./build.sh. That clones and compiles a CUDA tree for 15-25
# minutes, which is not something a launcher should do behind your back -- run
# it once by hand first.
#
# It sets no storage paths. The recipe's own defaults already point at the
# tool's standard locations -- see CONVENTIONS.md.
#
# Usage: run-ternary-bonsai-2-27b-gguf.sh [wide|deep|long|safe|ptq1]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE_DIR="${REPO_ROOT}/recipes/Ternary-Bonsai-2-27B-GGUF-llamacpp-prism"
PROFILE="${1:-wide}"

[[ -d "${RECIPE_DIR}" ]] || {
  printf 'error: recipe not found: %s\n' "${RECIPE_DIR}" >&2
  exit 1
}

printf '==> Ternary-Bonsai-2-27B (qwen35, 27.36B, ternary g128 at 1.72 bpw)\n'
printf '==> native PrismML llama.cpp fork -- stock llama.cpp cannot run these files\n'
printf '==> profile: %s\n\n' "${PROFILE}"

cd "${RECIPE_DIR}"
[[ -x "$(bash -c 'source ./profiles.sh && printf %s "${BIN_DIR}"')/llama-server" ]] || {
  printf 'error: the fork is not built yet. Run this once, it takes 15-25 minutes:\n' >&2
  printf '  cd %s && ./build.sh\n' "${RECIPE_DIR}" >&2
  exit 1
}
bash ./preflight.sh "${PROFILE}"
printf '\n'
exec bash ./start.sh "${PROFILE}"
