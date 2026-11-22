#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 MiaAI-Lab (original)
# Copyright (c) 2026 amarjeet (port)
#
# Derived from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
# (scripts/prepare.sh, step 2) at b956dba, MIT. The Dockerfile is upstream's,
# less its recipe-private CUDA cache path; the result is verified rather than
# trusted.
#
# Make IMAGE: TensorFold's Zig engine (tensorfold-native, kernels embedded),
# built from TF_REF (v1.2.7) plus patches/*.patch with Zig, on NVIDIA's PyTorch
# container, with the libraries it loads: libjpeg-turbo, libwebp's decoder,
# PDFium, libtfvideo over a static FFmpeg, libtfgrammar. Every download is
# checked against a SHA-256 pinned in profiles.sh. Always built locally:
# upstream's prebuilt images follow its own Dockerfile, not this recipe's.
#
#   ./build.sh            build IMAGE unless it is already there (20+ minutes cold)
#   ./build.sh --rebuild  build from scratch, no layer cache
#
# The image is then verified: tf.patches and tf.kernels labels, an executable
# tensorfold-native whose serve parser runs, and every library where the
# engine looks for it. Needs ~40 GB of Docker disk while it builds.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"

REBUILD=0
for arg in "$@"; do
  case "${arg}" in
    --rebuild) REBUILD=1 ;;
    -h|--help) sed -n '/^# Make IMAGE/,/^[^#]/{/^#/p}' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) printf 'unknown argument: %s\n' "${arg}" >&2; exit 2 ;;
  esac
done

require_aarch64
command -v docker >/dev/null || { printf 'error: docker is not installed\n' >&2; exit 1; }

if (( REBUILD == 0 )) && [[ "$(image_patches_label)" == "${PATCHES_HASH}" ]]; then
  printf 'image %s already present with patches %s\n' "${IMAGE}" "${PATCHES_HASH}"
else
  docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1 || docker pull "${BASE_IMAGE}"
  printf 'building %s (TensorFold v%s %s, Zig %s, patches %s)\n' \
    "${IMAGE}" "${TF_RELEASE_VERSION}" "${TF_REF:0:12}" "${ZIG_VERSION}" "${PATCHES_HASH}"
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
    --build-arg LIBJPEG_TURBO_VERSION="${LIBJPEG_TURBO_VERSION}" --build-arg LIBJPEG_TURBO_SHA256="${LIBJPEG_TURBO_SHA256}" \
    --build-arg LIBWEBP_VERSION="${LIBWEBP_VERSION}" --build-arg LIBWEBP_SHA256="${LIBWEBP_SHA256}" \
    --build-arg PYPDFIUM2_WHEEL_URL="${PYPDFIUM2_WHEEL_URL}" --build-arg PYPDFIUM2_AARCH64_SHA256="${PYPDFIUM2_AARCH64_SHA256}" \
    --build-arg FFMPEG_VERSION="${FFMPEG_VERSION}" --build-arg FFMPEG_SHA256="${FFMPEG_SHA256}" \
    --build-arg TF_RELEASE_VERSION="${TF_RELEASE_VERSION}" --build-arg TF_RELEASE_AARCH64_SHA256="${TF_RELEASE_AARCH64_SHA256}" \
    --build-arg PATCHES_HASH="${PATCHES_HASH}" \
    -f - "${ctx}" <<'DOCKERFILE'
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE} AS codecs
ARG LIBJPEG_TURBO_VERSION
ARG LIBJPEG_TURBO_SHA256
ARG LIBWEBP_VERSION
ARG LIBWEBP_SHA256
ARG PYPDFIUM2_WHEEL_URL
ARG PYPDFIUM2_AARCH64_SHA256
RUN mkdir -p /src /opt/codecs && cd /src && \
    curl -fsSL -o jt.tar.gz "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/${LIBJPEG_TURBO_VERSION}/libjpeg-turbo-${LIBJPEG_TURBO_VERSION}.tar.gz" && \
    echo "${LIBJPEG_TURBO_SHA256}  jt.tar.gz" | sha256sum -c - && \
    tar -xzf jt.tar.gz && cmake -S libjpeg-turbo-${LIBJPEG_TURBO_VERSION} -B jt-build -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_SHARED=ON -DENABLE_STATIC=OFF -DWITH_TURBOJPEG=ON -DCMAKE_INSTALL_PREFIX=/opt/codecs/jpeg -DCMAKE_INSTALL_LIBDIR=lib && \
    cmake --build jt-build -j"$(nproc)" && cmake --install jt-build && \
    test -e /opt/codecs/jpeg/lib/libturbojpeg.so.0
RUN cd /src && \
    curl -fsSL -o webp.tar.gz "https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-${LIBWEBP_VERSION}.tar.gz" && \
    echo "${LIBWEBP_SHA256}  webp.tar.gz" | sha256sum -c - && \
    tar -xzf webp.tar.gz && cd libwebp-${LIBWEBP_VERSION} && \
    ./configure --prefix=/opt/codecs/webp --enable-libwebpdecoder --disable-static --disable-gl --disable-sdl --disable-png --disable-jpeg --disable-tiff --disable-gif --disable-wic && \
    make -j"$(nproc)" && make install && \
    test -e /opt/codecs/webp/lib/libwebpdecoder.so.3
# PDFium (PDF document parts, --vision-pdf) comes out of the pinned pypdfium2 aarch64 wheel: the library and the license
# files the wheel ships with it (BSD-3-Clause and the BUILD_LICENSES of what PDFium was built from) are what we take
RUN mkdir -p /opt/codecs/pdfium/lib /opt/codecs/pdfium/licenses && cd /src && \
    curl -fsSL -o pypdfium2.whl "${PYPDFIUM2_WHEEL_URL}" && \
    echo "${PYPDFIUM2_AARCH64_SHA256}  pypdfium2.whl" | sha256sum -c - && \
    python3 -c 'import os, zipfile; z = zipfile.ZipFile("pypdfium2.whl"); open("/opt/codecs/pdfium/lib/libpdfium.so", "wb").write(z.read("pypdfium2_raw/libpdfium.so")); [[os.makedirs(os.path.dirname(p) or ".", exist_ok=True), open(p, "wb").write(z.read(n))] for n in z.namelist() if "/licenses/" in n for p in [os.path.join("/opt/codecs/pdfium/licenses", n.split("/licenses/", 1)[1])]]' && \
    test -s /opt/codecs/pdfium/lib/libpdfium.so && \
    test -s /opt/codecs/pdfium/licenses/LICENSES/BSD-3-Clause.txt

FROM ${BASE_IMAGE} AS grammar
ARG TF_RELEASE_VERSION
ARG TF_RELEASE_AARCH64_SHA256
RUN mkdir -p /opt/tfrel && cd /tmp && \
    curl -fsSL -o tf-rel.tar.gz "https://github.com/ashhart/TensorFold/releases/download/v${TF_RELEASE_VERSION}/tensorfold-${TF_RELEASE_VERSION}-linux-aarch64.tar.gz" && \
    echo "${TF_RELEASE_AARCH64_SHA256}  tf-rel.tar.gz" | sha256sum -c - && \
    tar -xzf tf-rel.tar.gz -C /opt/tfrel --strip-components=1 \
      tensorfold-${TF_RELEASE_VERSION}-linux-aarch64/lib/libtfgrammar.so \
      tensorfold-${TF_RELEASE_VERSION}-linux-aarch64/LICENSES/xgrammar-Apache-2.0.txt \
      tensorfold-${TF_RELEASE_VERSION}-linux-aarch64/LICENSES/xgrammar-NOTICE.txt \
      tensorfold-${TF_RELEASE_VERSION}-linux-aarch64/LICENSES/dlpack-Apache-2.0.txt && \
    test -s /opt/tfrel/lib/libtfgrammar.so

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
# video input (VIDEO=1) is TensorFold's libtfvideo.so over a static FFmpeg. TensorFold's own tools/zig/build_ffmpeg.sh builds it: the five decoders and two
# demuxers Qwen video needs, --disable-network, --disable-programs, no shared objects. The tarball is fetched and checked here with sha256sum and its source
# dir handed to the script, which then skips its own download, so the build is gated on the SHA-256 pinned in scripts/config.sh (the same one that script carries).
ARG FFMPEG_VERSION
ARG FFMPEG_SHA256
RUN mkdir -p /tmp/ffmpeg && cd /tmp/ffmpeg && \
    curl -fsSL -o ffmpeg.tar.xz "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" && \
    echo "${FFMPEG_SHA256}  ffmpeg.tar.xz" | sha256sum -c - && tar -xJf ffmpeg.tar.xz && \
    /tensorfold/tools/zig/build_ffmpeg.sh /opt/ffmpeg /tmp/ffmpeg && \
    mkdir -p /opt/ffmpeg/share/doc/ffmpeg/objects && \
    cp "ffmpeg-${FFMPEG_VERSION}/COPYING.LGPLv2.1" "ffmpeg-${FFMPEG_VERSION}/LICENSE.md" /opt/ffmpeg/share/doc/ffmpeg/ && \
    cp /opt/ffmpeg/lib/*.a /opt/ffmpeg/share/doc/ffmpeg/objects/ && \
    rm -rf /tmp/ffmpeg
# fatbins + native + video only: the default install also builds tf-cuda-test, which does not compile on 1.1.4 (contract_controls.cu, w4a8_tests.zig).
# `video` installs libtfvideo.so to native/lib, where the server's dlopen would find it beside the binary; it moves to libexec/tfvideo, off every candidate
# path, so only VIDEO=1 (start.sh putting that directory on LD_LIBRARY_PATH) makes a server take video at all.
RUN cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast -Dffmpeg=/opt/ffmpeg --prefix /opt/tensorfold \
      -j"$(nproc)" fatbins native video && \
    test -x /opt/tensorfold/native/bin/tensorfold-native && test -s /opt/tensorfold/native/lib/libtfvideo.so && \
    mkdir -p /opt/tensorfold/libexec/tfvideo && \
    mv /opt/tensorfold/native/lib/libtfvideo.so /opt/tensorfold/libexec/tfvideo/libtfvideo.so && \
    test ! -e /opt/tensorfold/native/lib/libtfvideo.so
RUN mkdir -p /opt/tensorfold/share/doc/tensorfold && cd /tensorfold && \
    for f in LICENSE LICENSE.md LICENSES NOTICE THIRD_PARTY_NOTICES.md; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done && \
    cp -a /opt/ffmpeg/share/doc/ffmpeg /opt/tensorfold/share/doc/ffmpeg

FROM ${BASE_IMAGE}
COPY --from=build /opt/tensorfold /opt/tensorfold
COPY --from=codecs /opt/codecs/jpeg/lib/libturbojpeg.so* /opt/tensorfold/lib/
COPY --from=codecs /opt/codecs/webp/lib/libwebpdecoder.so* /opt/tensorfold/lib/
COPY --from=codecs /opt/codecs/pdfium/lib/libpdfium.so /opt/tensorfold/lib/libpdfium.so
COPY --from=codecs /opt/codecs/pdfium/licenses /opt/tensorfold/share/doc/tensorfold/LICENSES-pdfium
# the engine looks for libtfgrammar.so in <binary dir>/../lib, i.e. next to native/bin
COPY --from=grammar /opt/tfrel/lib/libtfgrammar.so /opt/tensorfold/native/lib/libtfgrammar.so
COPY --from=grammar /opt/tfrel/LICENSES /opt/tensorfold/share/doc/tensorfold/LICENSES-grammar
RUN ln -s /opt/tensorfold/native/bin/tensorfold-native /usr/local/bin/tensorfold-native
ARG PATCHES_HASH
LABEL tf.patches=${PATCHES_HASH} tf.kernels=embedded
ENV HF_HOME=/root/.cache/huggingface \
    TENSORFOLD_LIBTURBOJPEG=/opt/tensorfold/lib/libturbojpeg.so.0 TENSORFOLD_LIBWEBP=/opt/tensorfold/lib/libwebpdecoder.so.3 \
    TENSORFOLD_LIBPDFIUM=/opt/tensorfold/lib/libpdfium.so
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
[[ "$(image_label tf.kernels)" == embedded ]] \
  && printf '  [ OK ] tf.kernels embedded\n' \
  || { printf '  [FAIL] tf.kernels is %s, expected embedded\n' "$(image_label tf.kernels)"; fail=1; }
# What the engine runs and dlopens, each where it looks for it. libtfvideo
# must be in libexec/tfvideo and NOT beside the binary, or VIDEO=0 would not
# keep video off.
out="$(docker run --rm --network none --entrypoint sh "${IMAGE}" -c '
    t() { if [ "$1" "$2" ]; then echo "ok   $2"; else echo "FAIL $2"; fi; }
    t -x /opt/tensorfold/native/bin/tensorfold-native
    t -s /opt/tensorfold/native/lib/libtfgrammar.so
    t -e "$TENSORFOLD_LIBTURBOJPEG"
    t -e "$TENSORFOLD_LIBWEBP"
    t -s "$TENSORFOLD_LIBPDFIUM"
    t -s /opt/tensorfold/libexec/tfvideo/libtfvideo.so
    if [ -e /opt/tensorfold/native/lib/libtfvideo.so ]; then echo "FAIL libtfvideo.so is on the probe path"; fi
    tensorfold-native serve --help >/dev/null 2>&1 && echo "ok   serve --help" || echo "FAIL serve --help"' 2>&1 || true)"
while IFS= read -r line; do
  case "${line}" in
    ok*) printf '  [ OK ] %s\n' "${line#ok   }" ;;
    FAIL*) printf '  [FAIL] %s\n' "${line#FAIL }"; fail=1 ;;
  esac
done <<<"${out}"
[[ -n "${out}" ]] || { printf '  [FAIL] could not run %s\n' "${IMAGE}"; fail=1; }
(( fail == 0 )) || { printf '\nimage verification failed\n' >&2; exit 1; }
printf '\nimage ready: %s\n' "${IMAGE}"
