#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
# Copyright (C) 2026 amarjeet
#
# Derived from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark, which is
# Copyright (C) 2026 MiaAI Lab and licensed AGPL-3.0-or-later.
# Modified 2026-09-05 by amarjeet: implements the GPU/KV budget arithmetic
# from upstream's start.sh
"""Derive the GPU memory budget for a TP=1 launch on unified memory.

Both preflight.sh and start.sh need these numbers, and they must not drift, so
the arithmetic lives here once and each caller reads the JSON.

The thing this computes is not obvious, so it is worth stating plainly. On GB10
the CPU and GPU share one pool, and vLLM reads MemAvailable -- page cache
included -- as "free GPU memory", then fills the GPU side to exactly
`gpu_memory_utilization * MemTotal`. So the budget cannot be chosen from what
the model wants; it has to be capped from the host side:

    budget = min(weights + overhead + mtp + max(kv_need, kv_target),
                 MemTotal - host_reserve)

and the KV pool is whatever the capped budget leaves. Asking for KV above the
cap does not allocate more KV -- it takes the memory out of the PLE page cache
and the free pages the NVIDIA driver needs, which is how upstream lost three
servers in one day. The cap wins, and the caller is told when it binds.

`gmu` is floored to the three decimals vLLM is actually given, and the budget
recomputed from that, so these figures are what vLLM will do rather than what
it was asked for.

Usage:
  ./scripts/budget.py --mem-total-gib 121.69 --model-bytes 105935744618 \
      --max-model-len 262144 --kv-cache-dtype fp8 --mtp 3
"""
import argparse
import json
import math

GIB = 2 ** 30


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mem-total-gib", type=float, required=True)
    parser.add_argument("--model-bytes", type=int, required=True,
                        help="checkpoint size on disk, PLE table included")
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--mtp", type=int, default=0,
                        help="num_speculative_tokens; 0 disables MTP")
    parser.add_argument("--ple-gib", type=float, default=26.82)
    parser.add_argument("--overhead-gib", type=float, default=5.6)
    parser.add_argument("--kv-bytes-per-token", type=int, default=29482)
    parser.add_argument("--kv-target-gib", type=float, default=16.0)
    parser.add_argument("--host-reserve-gib", type=float, default=26.0)
    parser.add_argument("--host-slack-gib", type=float, default=5.0)
    parser.add_argument("--os-reserve-gib", type=float, default=16.0)
    parser.add_argument("--gmu", default="", help="pin gpu-memory-utilization")
    parser.add_argument("--container-mem-gib", default="",
                        help="pin the container cgroup cap, GiB")
    args = parser.parse_args()

    total = args.mem_total_gib
    # The PLE n-gram table is memory-mapped by the CPU offload worker, so it is
    # evictable page cache rather than resident GPU memory. Subtracting it is
    # what turns a checkpoint that cannot fit into one that can.
    weights_gpu = args.model_bytes / GIB - args.ple_gib
    mtp_gib = 1.49 if args.mtp > 0 else 0.0
    # FP8 halves the 12 full-attention layers (~84% of bytes/token) but the QSA
    # side and compressor caches stay BF16, so the real saving is ~1.7x, not 2x.
    kv_mult = 0.58 if args.kv_cache_dtype.startswith("fp8") else 1.0

    fixed = weights_gpu + args.overhead_gib + mtp_gib
    kv_need = args.max_model_len * args.kv_bytes_per_token * kv_mult / GIB
    wish = fixed + max(kv_need, args.kv_target_gib)
    cap = total - args.host_reserve_gib
    cap_binds = wish > cap

    if args.gmu:
        gmu = float(args.gmu)
        budget = gmu * total
        pinned_above_cap = budget > cap
    else:
        budget = min(wish, cap)
        gmu = math.floor(budget / total * 1000) / 1000
        budget = gmu * total
        pinned_above_cap = False

    kv_expect = budget - fixed
    kv_expect_tok = int(max(kv_expect, 0) * GIB / (args.kv_bytes_per_token * kv_mult))

    container_mem = (int(args.container_mem_gib) if args.container_mem_gib
                     else int(budget + args.host_slack_gib))
    max_container = int(total - args.os_reserve_gib)

    print(json.dumps({
        "mem_total_gib": round(total, 2),
        "weights_gpu_gib": round(weights_gpu, 2),
        "ple_gib": args.ple_gib,
        "overhead_gib": args.overhead_gib,
        "mtp_gib": mtp_gib,
        "kv_mult": kv_mult,
        "kv_need_gib": round(kv_need, 2),
        "kv_target_gib": args.kv_target_gib,
        "budget_cap_gib": round(cap, 2),
        "budget_gib": round(budget, 2),
        "gmu": gmu,
        "kv_expect_gib": round(kv_expect, 2),
        "kv_expect_tokens": kv_expect_tok,
        "container_mem_gib": container_mem,
        "max_container_gib": max_container,
        "cap_binds": cap_binds,
        "pinned_above_cap": pinned_above_cap,
        # Does the KV pool actually hold one full-length request?
        "kv_fits": kv_expect >= kv_need,
        "container_fits": container_mem <= max_container,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
