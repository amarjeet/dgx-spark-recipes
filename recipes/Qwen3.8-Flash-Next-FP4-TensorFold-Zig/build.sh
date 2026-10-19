#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 MiaAI-Lab (original)
# Copyright (c) 2026 amarjeet (port)
#
# Derived from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
# (scripts/prepare.sh, step 2), MIT. The Dockerfile is upstream's, less its
# recipe-private CUDA cache path; the result is verified rather than trusted.
#
# Make IMAGE: TensorFold's Zig engine (tensorfold-native), built from TF_REF
# plus patches/*.patch with Zig, on NVIDIA's PyTorch container, with the TP=1
# AOT kernel set. Always built locally: upstream's prebuilt images follow its
# own Dockerfile and patch set, not this recipe's.
#
#   ./build.sh            build IMAGE unless it is already there (20+ minutes cold)
#   ./build.sh --rebuild  build from scratch, no layer cache
#
# The image is then verified: tf.patches and tf.kernels labels, and an
# executable tensorfold-native. Needs ~40 GB of Docker disk while it builds.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

REBUILD=0
for arg in "$@"; do
  case "${arg}" in
    --rebuild) REBUILD=1 ;;
    -h|--help) sed -n '12,20p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) printf 'unknown argument: %s\n' "${arg}" >&2; exit 2 ;;
  esac
done

require_aarch64
command -v docker >/dev/null || { printf 'error: docker is not installed\n' >&2; exit 1; }

if (( REBUILD == 0 )) && [[ "$(image_patches_label)" == "${PATCHES_HASH}" ]]; then
  printf 'image %s already present with patches %s\n' "${IMAGE}" "${PATCHES_HASH}"
else
  docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1 || docker pull "${BASE_IMAGE}"
  printf 'building %s (TensorFold %s, Zig %s, patches %s, TP=1 kernel set)\n' \
    "${IMAGE}" "${TF_REF:0:12}" "${ZIG_VERSION}" "${PATCHES_HASH}"
  nocache=(); (( REBUILD )) && nocache=(--no-cache)
  mkdir -p "${OUT_DIR}"
  ctx="$(mktemp -d "${OUT_DIR}/build.XXXXXX")"
  trap 'rm -rf -- "${ctx}"' EXIT
  mkdir -p "${ctx}/patches"
  cp -- "${EXPERIMENT_DIR}"/patches/*.patch "${ctx}/patches/"
  # shellcheck disable=SC2086
  docker build ${DOCKER_BUILD_ARGS:-} "${nocache[@]}" -t "${IMAGE}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg TF_REPO="${TF_REPO}" --build-arg TF_REF="${TF_REF}" \
    --build-arg ZIG_VERSION="${ZIG_VERSION}" --build-arg ZIG_SHA256="${ZIG_SHA256}" \
    --build-arg PATCHES_HASH="${PATCHES_HASH}" --build-arg KERNEL_SOURCE_MTIME="${KERNEL_SOURCE_MTIME}" \
    -f - "${ctx}" <<'DOCKERFILE'
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE} AS build
ARG ZIG_VERSION
ARG ZIG_SHA256
RUN curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz" && \
    echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - && \
    mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz && \
    /opt/zig/zig version
ARG TF_REPO
ARG TF_REF
RUN git init -q /tensorfold && cd /tensorfold && git remote add origin "${TF_REPO}" && \
    git fetch -q --depth 1 origin "${TF_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${TF_REF}"
COPY patches /opt/tf-patches
RUN cd /tensorfold && \
    for p in /opt/tf-patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done
RUN cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast --prefix /opt/tensorfold \
      -j"$(nproc)" fatbins install native && \
    test -x /opt/tensorfold/native/bin/tensorfold-native
RUN mkdir -p /opt/tensorfold/share/doc/tensorfold && cd /tensorfold && \
    for f in LICENSE LICENSE.md LICENSES NOTICE THIRD_PARTY_NOTICES.md; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done
ARG KERNEL_SOURCE_MTIME
RUN cd /tensorfold && python -B -c 'import json, os, sys; t = int(sys.argv[1]); \
      [os.utime(os.path.join("src", f), (t, t)) for f in sorted({k["source"]["file"] for k in \
       json.load(open("zig/tests/cuda/flashnext/kernels.json"))["kernels"]})]' "${KERNEL_SOURCE_MTIME}" && \
    PYTHONPATH=/tensorfold/src python -B tools/zig/flashnext_aot.py build \
      --spec zig/tests/cuda/flashnext/kernels.json --jit zig/tests/cuda/flashnext/jit.json --tp 1 \
      --out /opt/tensorfold/share/tensorfold/cuda/sm121 > /tmp/aot.log || true && \
    cat /tmp/aot.log && \
    grep -q 'kernels ->' /tmp/aot.log && \
    bad=$(grep PROBLEM /tmp/aot.log | grep -v 'cubin sha256' || true) && \
    test -z "$bad" && \
    test -f /opt/tensorfold/share/tensorfold/cuda/sm121/aot.json

FROM ${BASE_IMAGE}
COPY --from=build /opt/tensorfold /opt/tensorfold
COPY --from=build /tensorfold/src/tensorfold /opt/tensorfold/python/tensorfold
RUN ln -s /opt/tensorfold/native/bin/tensorfold-native /usr/local/bin/tensorfold-native && \
    pip install --no-cache-dir "huggingface_hub>=1.0" && \
    pip install --no-cache-dir --no-deps "transformers==5.17.0" "av==19.0.1"
ARG PATCHES_HASH
LABEL tf.patches=${PATCHES_HASH} tf.kernels=present
ENV HF_HOME=/root/.cache/huggingface TENSORFOLD_CUDA_KERNELS=/opt/tensorfold/share/tensorfold/cuda/sm121 \
    TENSORFOLD_VISION_PYTHONPATH=/opt/tensorfold/python
WORKDIR /workspace
DOCKERFILE
  rm -rf -- "${ctx}"; trap - EXIT
fi

# --- verify ------------------------------------------------------------------
printf '\nverifying %s\n' "${IMAGE}"
fail=0
[[ "$(image_patches_label)" == "${PATCHES_HASH}" ]] \
  && printf '  [ OK ] tf.patches label %s\n' "${PATCHES_HASH}" \
  || { printf '  [FAIL] tf.patches label is %s, expected %s\n' "$(image_patches_label)" "${PATCHES_HASH}"; fail=1; }
[[ "$(image_label tf.kernels)" == present ]] \
  && printf '  [ OK ] tf.kernels label present\n' \
  || { printf '  [FAIL] no tf.kernels label: the AOT kernel set is missing\n'; fail=1; }
if docker run --rm --network none --entrypoint test "${IMAGE}" -x /opt/tensorfold/native/bin/tensorfold-native; then
  printf '  [ OK ] tensorfold-native\n'
else
  printf '  [FAIL] %s has no tensorfold-native\n' "${IMAGE}"; fail=1
fi
(( fail == 0 )) || { printf '\nimage verification failed\n' >&2; exit 1; }
printf '\nimage ready: %s\n' "${IMAGE}"
