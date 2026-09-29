#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 MiaAI-Lab (original)
# Copyright (c) 2026 amarjeet (port)
#
# Derived from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
# (scripts/prepare.sh, step 2), MIT. The Dockerfile is upstream's; the pull is
# pinned by digest and the result is verified rather than trusted.
#
# Make IMAGE: TensorFold v0.3.6.2 with patches/*.patch applied, on NVIDIA's
# PyTorch container.
#
#   ./build.sh            pull upstream's prebuilt image by digest (~11 GB) and
#                         tag it as IMAGE; build locally if the pull fails
#   PULL=0 ./build.sh     always build locally (a few minutes, needs BASE_IMAGE)
#   ./build.sh --rebuild  build locally from scratch, no layer cache
#
# Either way the image is then verified: its tf.patches label must equal the
# hash of patches/, TensorFold must report 0.3.6.2, and every patch must be
# present in the installed package (a reverse dry-run applies cleanly).

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

REBUILD=0
for arg in "$@"; do
  case "${arg}" in
    --rebuild) REBUILD=1 ;;
    -h|--help) sed -n '10,21p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) printf 'unknown argument: %s\n' "${arg}" >&2; exit 2 ;;
  esac
done

require_aarch64
command -v docker >/dev/null || { printf 'error: docker is not installed\n' >&2; exit 1; }

HASH="${PATCHES_HASH}"
if [[ "${HASH}" != "${PATCHES_HASH_PINNED}" ]]; then
  # The prebuilt image carries the pinned patches, not these.
  printf 'patches/ hash %s differs from the pinned %s: building %s locally\n' \
    "${HASH}" "${PATCHES_HASH_PINNED}" "${IMAGE}"
  PULL=0
fi

have_label="$(image_patches_label)"
if (( REBUILD == 0 )) && [[ "${have_label}" == "${HASH}" ]]; then
  printf 'image %s already present with patches %s\n' "${IMAGE}" "${HASH}"
else
  pulled=0
  if (( REBUILD == 0 )) && [[ "${PULL:-1}" == 1 ]]; then
    ref="${GHCR_IMAGE}@${PREBUILT_DIGEST}"
    printf 'pulling %s (~11 GB)\n' "${ref}"
    if docker pull "${ref}"; then
      docker tag "${ref}" "${IMAGE}"
      pulled=1
    else
      printf 'warn: pull failed; building locally\n' >&2
    fi
  fi
  if (( pulled == 0 )); then
    nocache=(); (( REBUILD )) && nocache=(--no-cache)
    docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1 || docker pull "${BASE_IMAGE}"
    printf 'building %s (TensorFold %s @ %s, patches %s)\n' "${IMAGE}" "${TF_VERSION}" "${TF_COMMIT:0:12}" "${HASH}"
    # Upstream's Dockerfile, with the install pinned to the release commit
    # rather than the tag.
    docker build "${nocache[@]}" -t "${IMAGE}" \
      --build-arg BASE_IMAGE="${BASE_IMAGE}" \
      --build-arg TF_SPEC="git+${TF_REPO}@${TF_COMMIT}" \
      --build-arg PATCHES_HASH="${HASH}" \
      -f - "${EXPERIMENT_DIR}/patches" <<'DOCKERFILE'
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}
ARG TF_SPEC
RUN pip install --no-cache-dir --upgrade "${TF_SPEC}" && tensorfold --version
COPY . /opt/tf-patches
RUN cd "$(python -c 'import os, tensorfold; print(os.path.dirname(os.path.dirname(tensorfold.__file__)))')" && \
    for p in /opt/tf-patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; patch -p0 --forward < "$p" || exit 1; done && \
    python -c "import tensorfold.cuda.reply_text"
ARG PATCHES_HASH
LABEL tf.patches=${PATCHES_HASH}
ENV HF_HOME=/root/.cache/huggingface \
    TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    TRITON_CACHE_DIR=/cache/triton
WORKDIR /workspace
DOCKERFILE
  fi
fi

# --- verify ------------------------------------------------------------------
printf '\nverifying %s\n' "${IMAGE}"
fail=0
label="$(image_patches_label)"
if [[ "${label}" == "${HASH}" ]]; then
  printf '  [ OK ] tf.patches label %s\n' "${label}"
else
  printf '  [FAIL] tf.patches label is %s, expected %s\n' "${label:-missing}" "${HASH}"; fail=1
fi
ver="$(image_tf_version)"
if [[ "v${ver}" == "${TF_VERSION}" ]]; then
  printf '  [ OK ] tensorfold %s\n' "${ver}"
else
  printf '  [FAIL] tensorfold reports %s, expected %s\n' "${ver:-nothing}" "${TF_VERSION}"; fail=1
fi
# The label only says what the builder claimed. Undoing the patches in reverse
# order on a scratch copy of the package proves every one of them is in it.
# Each must be undone on top of the later ones' undo, because later patches
# rewrite lines earlier ones added (0004 edits a line of 0003's).
if docker run --rm --entrypoint bash -v "${EXPERIMENT_DIR}/patches:/p:ro" "${IMAGE}" -c '
    site="$(python -c "import os, tensorfold; print(os.path.dirname(os.path.dirname(tensorfold.__file__)))")"
    mkdir -p /tmp/undo && cp -a "${site}/tensorfold" /tmp/undo/ && cd /tmp/undo
    rc=0
    for p in $(ls /p/*.patch | sort -r); do
      if patch -p0 -R -s -f --no-backup-if-mismatch < "$p" >/dev/null; then echo "  [ OK ] $(basename "$p")"
      else echo "  [FAIL] $(basename "$p") is not applied"; rc=1; fi
    done
    exit $rc'; then :; else fail=1; fi
(( fail == 0 )) || { printf '\nimage verification failed\n' >&2; exit 1; }
printf '\nimage ready: %s\n' "${IMAGE}"
