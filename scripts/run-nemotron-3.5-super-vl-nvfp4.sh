#!/usr/bin/env bash
#
# Thin launcher for recipes/Nemotron-3.5-Super-VL-NVFP4-vLLM.
#
# Runs preflight before start: the server budgets 75% of unified memory and
# will not run beside another large model, and failing in preflight beats
# failing minutes into a cold load. It sets no storage paths -- the recipe's
# own defaults already point at each tool's standard location (CONVENTIONS.md).
#
# Usage: run-nemotron-3.5-super-vl-nvfp4.sh [spark|nodraft]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE_DIR="${REPO_ROOT}/recipes/Nemotron-3.5-Super-VL-NVFP4-vLLM"
PROFILE="${1:-spark}"

[[ -d "${RECIPE_DIR}" ]] || { printf 'error: recipe not found: %s\n' "${RECIPE_DIR}" >&2; exit 1; }

printf '==> Nemotron-3.5-Super-VL (NVFP4, text + image) via vLLM\n==> profile: %s\n\n' "${PROFILE}"
cd "${RECIPE_DIR}"
bash ./preflight.sh "${PROFILE}"
printf '\n'
exec bash ./start.sh "${PROFILE}"
