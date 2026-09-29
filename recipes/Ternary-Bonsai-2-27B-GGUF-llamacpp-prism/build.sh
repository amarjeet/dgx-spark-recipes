#!/usr/bin/env bash
#
# Build the PrismML llama.cpp fork with CUDA for GB10 (sm_121).
#
# Why this step exists at all: these GGUFs cannot run on stock llama.cpp.
# PQ2_0 (ggml type id 142) and PTQ1_0 sit past upstream's GGML_TYPE_COUNT so it
# refuses them, and the weights are stored in a blockwise Hadamard-rotated
# basis whose matching activation transform lives only in the fork. The fork
# publishes prebuilt CUDA binaries for linux-x64 only -- there is no aarch64
# CUDA archive -- so on this box the binary has to be built.
#
# Do NOT use the fork's own scripts/build_cuda_linux.sh. It hardcodes
# CMAKE_CUDA_ARCHITECTURES="80;86;89;90;100;120a", and ggml only picks
# architectures for you when that variable is NOT defined. sm_120a is not
# forwards-compatible to GB10's sm_121 and there is no PTX entry to JIT from,
# so that script yields a binary which will not run on this machine.
#
# Usage:
#   ./build.sh                     # clone/fetch the pinned commit and build
#   ./build.sh --clean             # discard the build tree first
#   CUDA_ARCHS=native ./build.sh   # let ggml detect the local GPU instead
#   FORK_COMMIT=<sha> ./build.sh   # build a different commit
set -euo pipefail

case "${1:-}" in
  -h|--help) sed -n '3,22p' "$0" | sed 's/^# \?//'; exit 0 ;;
esac

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

CLEAN=0
[[ "${1:-}" == "--clean" ]] && CLEAN=1

require_aarch64

for tool in git cmake make; do
  command -v "${tool}" >/dev/null || {
    printf 'error: %s is not on PATH\n' "${tool}" >&2
    exit 1
  }
done

NVCC="${CUDA_PATH}/bin/nvcc"
[[ -x "${NVCC}" ]] || {
  printf 'error: nvcc not found at %s\n' "${NVCC}" >&2
  printf 'set CUDA_PATH to your CUDA toolkit root.\n' >&2
  exit 1
}
CUDA_VERSION="$("${NVCC}" --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -1)"

printf 'fork      : %s (%s)\n' "${FORK_REPO}" "${FORK_BRANCH}"
printf 'commit    : %s\n' "${FORK_COMMIT}"
printf 'checkout  : %s\n' "${FORK_DIR}"
printf 'build dir : %s\n' "${BUILD_DIR}"
printf 'cuda      : %s (v%s)\n' "${CUDA_PATH}" "${CUDA_VERSION}"
printf 'archs     : %s\n' "${CUDA_ARCHS}"
printf 'gpu       : %s\n\n' \
  "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || printf 'unknown')"

# --- checkout ----------------------------------------------------------------

if [[ ! -d "${FORK_DIR}/.git" ]]; then
  printf '=== cloning ===\n'
  mkdir -p "$(dirname "${FORK_DIR}")"
  # Blobless clone: the full history of llama.cpp is large and none of it is
  # needed to build one commit.
  git clone --filter=blob:none --branch "${FORK_BRANCH}" \
    "${FORK_REPO}" "${FORK_DIR}"
else
  printf '=== fetching ===\n'
  git -C "${FORK_DIR}" fetch --filter=blob:none origin "${FORK_BRANCH}"
fi

# Pinned commit, not the branch head. If the branch has been rewritten the
# commit may no longer be reachable from it, so ask for it directly.
if ! git -C "${FORK_DIR}" cat-file -e "${FORK_COMMIT}^{commit}" 2>/dev/null; then
  printf 'commit not present after fetch; requesting it directly\n'
  git -C "${FORK_DIR}" fetch --filter=blob:none origin "${FORK_COMMIT}"
fi
git -C "${FORK_DIR}" -c advice.detachedHead=false checkout --force "${FORK_COMMIT}"
git -C "${FORK_DIR}" submodule update --init --recursive >/dev/null 2>&1 || true
printf 'at %s\n\n' "$(git -C "${FORK_DIR}" log -1 --format='%h %s')"

# --- configure ---------------------------------------------------------------

if (( CLEAN )) && [[ -d "${BUILD_DIR}" ]]; then
  printf '=== removing %s ===\n' "${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

# No -G Ninja: ninja is not installed and this repo installs nothing (see the
# root README's requirements). The default Makefile generator is fine.
#
# LLAMA_BUILD_UI=OFF: the embedded web UI is fetched prebuilt from the
# ggml-org/llama-ui Hub bucket keyed by build number, and a fork's build number
# has no bucket entry -- so it falls back to building the assets with npm,
# which pulls a toolchain this recipe does not need. Everything here is driven
# through the OpenAI-compatible API. Set LLAMA_BUILD_UI=ON in EXTRA_CMAKE_ARGS
# if you want the browser chat page.
printf '=== configuring ===\n'
EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS:-}"
# shellcheck disable=SC2086
cmake -B "${BUILD_DIR}" -S "${FORK_DIR}" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="${NVCC}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_UI=OFF \
  ${EXTRA_CMAKE_ARGS}

# --- build -------------------------------------------------------------------

# nvcc is memory-hungry per translation unit; 16 is upstream's own cap and this
# box has the RAM for it.
JOBS="${JOBS:-$(nproc)}"
(( JOBS > 16 )) && JOBS=16

# Only what this recipe runs. Building the `all` target also compiles tools the
# recipe never invokes, for several more minutes. Override BUILD_TARGETS to get
# them.
BUILD_TARGETS="${BUILD_TARGETS:-llama-server llama-bench llama-cli}"

printf '\n=== building (-j %s): %s ===\n' "${JOBS}" "${BUILD_TARGETS}"
printf '(a single-arch CUDA build of this tree takes roughly 15-25 minutes)\n\n'
# shellcheck disable=SC2086
cmake --build "${BUILD_DIR}" -j "${JOBS}" --target ${BUILD_TARGETS}

# --- verify ------------------------------------------------------------------
#
# Two checks, both for failures that are otherwise silent rather than loud.

printf '\n=== verifying ===\n'
fail=0

if [[ -x "${BIN_DIR}/llama-server" ]]; then
  printf '  OK    %s\n' "${BIN_DIR}/llama-server"
else
  printf '  FAIL  llama-server was not produced in %s\n' "${BIN_DIR}"
  fail=1
fi

# Stock llama.cpp built into the same tree would pass every other check here
# and then emit fluent nonsense at run time, because it has no Hadamard
# activation runtime. The ternary type names are the cheapest proof.
if fork_has_ternary_kernels; then
  printf '  OK    ternary kernels present (PQ2_0 in libggml-base)\n'
else
  printf '  FAIL  no PQ2_0 type in libggml-base -- this looks like stock llama.cpp,\n'
  printf '        not the PrismML fork. Check FORK_REPO/FORK_COMMIT.\n'
  fail=1
fi

archs="$(cuda_backend_archs || true)"
if [[ -z "${archs}" ]]; then
  printf '  WARN  could not read architectures from libggml-cuda (cuobjdump missing?)\n'
elif [[ "${archs}" == *sm_121* ]]; then
  printf '  OK    CUDA backend contains sm_121 (%s)\n' "${archs}"
else
  printf '  FAIL  CUDA backend has no sm_121: %s\n' "${archs}"
  printf '        this binary cannot run on GB10. Rebuild with CUDA_ARCHS=121a-real.\n'
  fail=1
fi

build="$(bin_build_number || true)"
if [[ -z "${build}" ]]; then
  printf '  WARN  could not read a build number from llama-server --version\n'
elif (( build < MIN_LLAMA_BUILD )); then
  printf '  FAIL  build %s < %s (MIN_LLAMA_BUILD)\n' "${build}" "${MIN_LLAMA_BUILD}"
  fail=1
else
  printf '  OK    llama.cpp build %s >= %s\n' "${build}" "${MIN_LLAMA_BUILD}"
fi

printf '\n'
if (( fail )); then
  printf 'build FAILED verification\n' >&2
  exit 1
fi
printf 'build OK\n'
printf 'binaries : %s\n' "${BIN_DIR}"
printf 'next     : ./download.sh pq2 && ./preflight.sh long\n'
