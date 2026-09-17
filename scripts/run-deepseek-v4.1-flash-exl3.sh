#!/usr/bin/env bash
#
# Thin launcher for recipes/DeepSeek-V4.1-Flash-EXL3-ExLlamaV3.
#
# Runs preflight before start: this pack is ~107 GiB resident in a 121.7 GiB
# unified pool shared with the OS, the load drives MemAvailable to roughly
# 5 GiB by design, and on unified memory an over-commit hangs the kernel rather
# than raising an OOM. Preflight also catches the two failures unique to this
# recipe -- a GPU not in ATS addressing mode, and a re-laid pack whose symlinks
# no longer resolve because the Hugging Face cache was pruned.
#
# It sets no storage paths. The recipe's own defaults already point at each
# tool's standard location -- see CONVENTIONS.md.
#
# Usage: run-deepseek-v4.1-flash-exl3.sh [measured|aliased|nodraft]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE_DIR="${REPO_ROOT}/recipes/DeepSeek-V4.1-Flash-EXL3-ExLlamaV3"
PROFILE="${1:-measured}"

[[ -d "${RECIPE_DIR}" ]] || {
  printf 'error: recipe not found: %s\n' "${RECIPE_DIR}" >&2
  exit 1
}

printf '==> DeepSeek-V4.1-Flash (EXL3 1.59 bpw) via native ExLlamaV3 at TP=1\n'
printf '==> profile: %s\n\n' "${PROFILE}"

cd "${RECIPE_DIR}"
bash ./preflight.sh "${PROFILE}"
printf '\n'
exec bash ./start.sh "${PROFILE}"
