#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2026 amarjeet
#
# Written for this port. Part of the same AGPL-3.0-or-later combined work as
# the files it sits beside, which derive from MiaAI-Lab/
# Qwen3.8-Flash-Next-Single-DGX-Spark, Copyright (C) 2026 MiaAI Lab.
#
# Build the reduced MTP draft vocabulary, in the container, into vLLM's cache.
#
# upstream's files/build_draft_vocab.py needs transformers and the checkpoint's
# tokenizer, so this runs it inside IMAGE against the pinned snapshot rather
# than asking for a host Python environment. The output is a generated artifact
# reused across launches, so it lands in DRAFT_VOCAB_DIR (under VLLM_CACHE_HOST)
# and never in the recipe directory -- which is also what lets start.sh reach it
# through the cache mount it already has.
#
# The corpus that matters is the MODEL'S OWN OUTPUT distribution, not a generic
# text corpus and not your prompts: that is what the drafter has to predict.
# Coverage, not size, is the number to tune on.
#
# Usage:
#   ./scripts/build_draft_vocab.sh corpus.jsonl
#   ./scripts/build_draft_vocab.sh english.txt code.txt:3 model_out.jsonl:20
#   SIZE=32768 ./scripts/build_draft_vocab.sh corpus.jsonl
#   REPORT_ONLY=1 ./scripts/build_draft_vocab.sh corpus.jsonl   # coverage only
#
# Then serve with it:
#   MTP_DRAFT_VOCAB=<the path printed at the end> ./start.sh
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../profiles.sh"
select_profile "${PROFILE:-${DEFAULT_PROFILE}}"

# Rows to keep. Upstream ships 65536: on their corpus 32k covers 93.7% of the
# model's own English+code output against 97.3% at 65k, and 65k costs only
# 3 points of byte saving (18.0% vs 21.1%). Acceptance falls off a cliff below
# ~88-90% coverage.
SIZE="${SIZE:-65536}"
OUT_NAME="${OUT_NAME:-${MODEL_NAME}-$((SIZE / 1024))k.txt}"

(( $# > 0 )) || {
  sed -n '22,29p' "$0" | sed 's/^# \?//' >&2
  exit 2
}

command -v docker >/dev/null || { printf 'docker is not on PATH\n' >&2; exit 1; }
docker image inspect "${IMAGE}" >/dev/null 2>&1 || {
  printf 'image not pulled: %s (run: docker pull %s)\n' "${IMAGE}" "${IMAGE}" >&2
  exit 1
}
[[ -d "${SNAPSHOT_DIR}" ]] || {
  printf 'pinned snapshot is not in the HF cache -- run ./download.sh first\n' >&2
  exit 1
}

# Mount each corpus file read-only under /corpus, preserving the optional :N
# repeat weight that build_draft_vocab.py parses. Basenames are indexed so two
# corpora with the same name cannot shadow each other.
mounts=()
corpus_args=()
i=0
for spec in "$@"; do
  repeat=""
  path="${spec}"
  # Only a trailing all-digits field is a weight; a path may legitimately
  # contain a colon.
  if [[ "${spec}" == *:* && "${spec##*:}" =~ ^[0-9]+$ ]]; then
    path="${spec%:*}"
    repeat=":${spec##*:}"
  fi
  [[ -f "${path}" ]] || { printf 'no such corpus file: %s\n' "${path}" >&2; exit 1; }
  path="$(cd "$(dirname "${path}")" && pwd)/$(basename "${path}")"
  ctr="/corpus/${i}-$(basename "${path}")"
  mounts+=(-v "${path}:${ctr}:ro")
  corpus_args+=("${ctr}${repeat}")
  i=$((i + 1))
done

mkdir -p "${DRAFT_VOCAB_DIR}"
out_host="${DRAFT_VOCAB_DIR}/${OUT_NAME}"

printf 'building a %s-row draft vocabulary from %s corpus file(s)\n' "${SIZE}" "$#"
printf '  tokenizer  %s @ %s\n' "${MODEL_ID}" "${MODEL_REVISION:0:12}"
printf '  out        %s\n\n' "${out_host}"

docker run --rm --name "${CONTAINER_NAME}-vocabbuild" \
  --memory 8g --cpus 8 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HOME=/root/.cache/huggingface \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${DRAFT_VOCAB_DIR}:/out" \
  -v "${EXPERIMENT_DIR}/files/build_draft_vocab.py:/b.py:ro" \
  "${mounts[@]}" \
  --entrypoint python3 "${IMAGE}" -u /b.py \
    "${corpus_args[@]}" \
    --model "${SNAPSHOT_DIR/#${HF_HOME}//root/.cache/huggingface}" \
    --size "${SIZE}" \
    --out "/out/${OUT_NAME}" \
    ${REPORT_ONLY:+--report-only}

if [[ -z "${REPORT_ONLY:-}" ]]; then
  printf '\nserve with it:\n  MTP_DRAFT_VOCAB=%s ./start.sh %s\n' "${out_host}" "${PROFILE}"
fi
