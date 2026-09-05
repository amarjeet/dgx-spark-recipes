#!/usr/bin/env bash
#
# Thin launcher for recipes/Ling-3.0-flash-Fin-GGUF-llamacpp.
#
# Runs preflight before start: this model needs 60+ GiB resident, and failing
# in preflight beats failing four minutes into a cold load.
#
# It sets no storage paths. The recipe's own defaults already point at the
# tool's standard locations -- see CONVENTIONS.md.
#
# Usage: run-ling-3.0-flash-fin-gguf.sh [iq4xs|q3kxl|q4km]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE_DIR="${REPO_ROOT}/recipes/Ling-3.0-flash-Fin-GGUF-llamacpp"
PROFILE="${1:-iq4xs}"

[[ -d "${RECIPE_DIR}" ]] || {
  printf 'error: recipe not found: %s\n' "${RECIPE_DIR}" >&2
  exit 1
}

printf '==> Ling-3.0-flash-Fin (bailingmoe3, 124B total / 5.1B active) via llama.cpp\n'
printf '==> profile: %s\n\n' "${PROFILE}"

cd "${RECIPE_DIR}"
bash ./preflight.sh "${PROFILE}"
printf '\n'
exec bash ./start.sh "${PROFILE}"
