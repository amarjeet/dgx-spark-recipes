#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
# Copyright (C) 2026 amarjeet
#
# Derived from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark, which is
# Copyright (C) 2026 MiaAI Lab and licensed AGPL-3.0-or-later.
# Modified 2026-09-07 by amarjeet: extracted from upstream's inline shell/awk
# in start.sh into a helper, so start.sh, preflight.sh and the README quote one
# implementation
"""Every decode batch width the scheduler can actually build.

vLLM's default CUDA graph capture list is [1, 2, 4] plus multiples of 8, each
rounded up to a multiple of (1 + num_speculative_tokens) and then filtered to
<= (1 + K) * max_num_seqs before it becomes a decode key. At K=3,
max_num_seqs=4 that leaves keys {4, 8, 16}: a 3-sequence verify batch is 12
tokens and pads up to 16, and at max_num_seqs=5 a full 5-sequence batch is 20
tokens, matches nothing, and decodes eager.

An MTP verify batch of S sequences is exactly (1 + K(S)) * S tokens wide, so
the set of widths the scheduler can produce is small and enumerable. This
prints it, sorted and de-duplicated, for `--cudagraph-capture-sizes`.

K(S) follows MTP_K_SCHEDULE when one is set ("start:end:K,..." over inclusive
num_seqs ranges) and is the constant K otherwise. A scheduled K is clamped to
the constant: the schedule can only ever lower the depth, never raise it above
the number of draft tokens the speculator was configured for.

Usage:
  ./scripts/graph_widths.py --max-num-seqs 4 --mtp 3
  ./scripts/graph_widths.py --max-num-seqs 8 --mtp 3 --k-schedule 1:2:3,3:8:2
"""
import argparse


def parse_schedule(spec: str, max_seqs: int, k_default: int) -> dict[int, int]:
    """Expand "start:end:K,..." into {num_seqs: K} over inclusive ranges."""
    k_of: dict[int, int] = {}
    for part in filter(None, (p.strip() for p in spec.strip().split(","))):
        fields = part.split(":")
        if len(fields) != 3:
            raise ValueError(
                f"bad MTP_K_SCHEDULE range {part!r}: expected start:end:K")
        lo, hi, k = (int(x) for x in fields)
        if lo < 1 or hi < lo or k < 0:
            raise ValueError(
                f"bad MTP_K_SCHEDULE range {part!r}: need 1 <= start <= end, K >= 0")
        for seqs in range(lo, min(hi, max_seqs) + 1):
            # First range wins, so an overlapping schedule is read left to
            # right rather than silently taking the last match.
            k_of.setdefault(seqs, min(k, k_default))
    return k_of


def widths(max_seqs: int, k_default: int, schedule: str = "") -> list[int]:
    k_of = parse_schedule(schedule, max_seqs, k_default) if schedule else {}
    return sorted({(1 + k_of.get(s, k_default)) * s
                   for s in range(1, max_seqs + 1)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--mtp", type=int, default=0,
                        help="num_speculative_tokens; 0 means no drafting")
    parser.add_argument("--k-schedule", default="",
                        help='MTP_K_SCHEDULE, e.g. "1:2:3,3:8:2"')
    args = parser.parse_args()

    if args.max_num_seqs < 1:
        parser.error("--max-num-seqs must be >= 1")
    if args.mtp < 0:
        parser.error("--mtp must be >= 0")

    print(",".join(str(w) for w in
                   widths(args.max_num_seqs, args.mtp, args.k_schedule)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
