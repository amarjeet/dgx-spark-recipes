#!/usr/bin/env bash
#
# Measurements. Results go to $OUT_DIR (see profiles.sh), never into the
# recipe directory.
#
# Three modes, answering three different questions:
#
#   depths  How do prefill and decode hold up as the context fills? Needs a
#           running server.
#   packs   Is PQ2_0 or PTQ1_0 faster on THIS hardware? The model card says
#           PQ2_0 wins on "the Blackwell cards", but it measured RTX 5090 and
#           RTX PRO 6000 -- roughly 6x GB10's memory bandwidth, where batch-1
#           decode is limited by instruction throughput rather than memory.
#           GB10 sits nearer upstream's L4 row, where PTQ1_0 won decode by
#           7.7% and lost 40% of prompt processing. Measured, not inherited.
#   needle  Does a given depth return a CORRECT answer? Delegates to
#           smoke.py --needle-sweep; see llama.cpp #27756.
#
# Usage:
#   ./bench.sh                 # depth ladder against the running server
#   ./bench.sh depths
#   ./bench.sh packs           # llama-bench, no server needed, stops nothing
#   ./bench.sh needle          # correctness across depths
set -euo pipefail

case "${1:-}" in
  -h|--help) sed -n '3,26p' "$0" | sed 's/^# \?//'; exit 0 ;;
esac

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
select_profile "$(cat "${ACTIVE_PROFILE_FILE}" 2>/dev/null || printf '%s' "${DEFAULT_PROFILE}")"

MODE="${1:-depths}"
mkdir -p "${OUT_DIR}"
stamp="$(date +%Y%m%dT%H%M%S)"

require_server() {
  curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || {
    printf 'server is not responding on port %s -- run ./start.sh first\n' "${PORT}" >&2
    exit 1
  }
}

case "${MODE}" in
  depths)
    require_server
    out="${OUT_DIR}/depths-${PROFILE}-${stamp}.json"
    printf 'profile %s (%s), %s per slot\n\n' "${PROFILE}" "${QUANT}" "${CTX_PER_SLOT}"
    python3 -u "${EXPERIMENT_DIR}/scripts/bench_depths.py" --port "${PORT}" --out "${out}"
    ;;

  packs)
    # llama-bench loads the weights itself, so this needs no server -- but it
    # does need the memory, so do not run it against a live wide profile.
    if server_running; then
      printf 'note: llama-server is running and holding %s.\n' \
        "$(human_bytes "$(budget_bytes)")" >&2
      printf 'llama-bench loads its own copy of the weights; stop the server first:\n' >&2
      printf '  ./stop.sh && ./bench.sh packs\n' >&2
      exit 1
    fi
    out="${OUT_DIR}/packs-${stamp}.txt"
    : >"${out}"
    for pack in pq2 ptq1; do
      ( select_profile "$( [[ "${pack}" == pq2 ]] && printf deep || printf ptq1 )"
        if [[ ! -f "${MODEL_FILE}" ]]; then
          printf '\n=== %s: not downloaded (./download.sh %s) ===\n' "${QUANT}" "${pack}" \
            | tee -a "${out}"
          exit 0
        fi
        printf '\n=== %s ===\n' "${QUANT}" | tee -a "${out}"
        # Two invocations deliberately: upstream notes the combined form emits
        # only pp512 for this model.
        for args in "-p 512 -n 0" "-p 0 -n 128"; do
          # shellcheck disable=SC2086
          LD_LIBRARY_PATH="${BIN_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
            "${BIN_DIR}/llama-bench" -m "${MODEL_FILE}" -ngl 99 -fa on ${args} -r 5 \
            2>&1 | grep -E '^\|' | tee -a "${out}"
        done )
    done
    printf '\nwrote %s\n' "${out}"
    printf 'decode is the column that should favour PTQ1_0; prefill should favour PQ2_0.\n'
    ;;

  needle)
    require_server
    out="${OUT_DIR}/needle-${PROFILE}-${stamp}.log"
    depths="${DEPTHS:-8192,32768,65536,98304,131072,196608,258048}"
    printf 'needle sweep at %s\n' "${depths}"
    printf '(a failure above a pass is expected: #27756 is non-monotonic)\n\n'
    python3 -u "${EXPERIMENT_DIR}/scripts/smoke.py" \
      --port "${PORT}" --quick --needle-sweep "${depths}" | tee "${out}"
    printf '\nwrote %s\n' "${out}"
    ;;

  *)
    printf 'usage: %s {depths|packs|needle}\n' "$0" >&2
    exit 2
    ;;
esac
