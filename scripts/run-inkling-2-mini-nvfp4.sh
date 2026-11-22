#!/usr/bin/env bash
#
# Thin launcher for recipes/Inkling-2-Mini-NVFP4-vLLM.
#
# Runs preflight before start: every tier takes most of unified memory and
# will not run beside another large model, and failing in preflight beats
# failing minutes into a cold load. It sets no storage paths -- the recipe's
# own defaults already point at each tool's standard location (CONVENTIONS.md).
#
# Usage: run-inkling-2-mini-nvfp4.sh [standard|max|max-fp8]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE_DIR="${REPO_ROOT}/recipes/Inkling-2-Mini-NVFP4-vLLM"
PROFILE="${1:-standard}"

[[ -d "${RECIPE_DIR}" ]] || { printf 'error: recipe not found: %s\n' "${RECIPE_DIR}" >&2; exit 1; }

printf '==> Inkling 2 Mini (NVFP4, text + image, DSpark) via vLLM\n==> profile: %s\n\n' "${PROFILE}"
cd "${RECIPE_DIR}"
bash ./preflight.sh "${PROFILE}"
printf '\n'
exec bash ./start.sh "${PROFILE}"
