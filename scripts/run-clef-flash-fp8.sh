#!/usr/bin/env bash
#
# Thin launcher for recipes/Clef-Flash-FP8-transformers.
#
# start.sh runs preflight itself, which checks the load peak and the steady
# state against MemAvailable -- this recipe is meant to start beside a running
# server (Qwen3.8-Flash-Next TensorFold on int8x1), so that check is what keeps
# the co-tenant clear of its watchdog. It does not run setup.sh or
# download.sh: a several-GB venv and an 18 GiB download should be started on
# purpose.
#
# It sets no storage paths. The recipe's own defaults already point at each
# tool's standard location -- see CONVENTIONS.md.
#
# Usage: run-clef-flash-fp8.sh            (WEIGHTS=bf16 to serve the checkpoint as released)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE_DIR="${REPO_ROOT}/recipes/Clef-Flash-FP8-transformers"

[[ -d "${RECIPE_DIR}" ]] || {
  printf 'error: recipe not found: %s\n' "${RECIPE_DIR}" >&2
  exit 1
}

printf '==> Clef-Flash (Cloudflare) via transformers, SystemOne API\n'
printf '==> weights: %s\n\n' "${WEIGHTS:-fp8}"

cd "${RECIPE_DIR}"
exec bash ./start.sh
