#!/usr/bin/env python3
"""Summarise harness results: per-model totals, per-task means, and a paired
sign-flip permutation test on per-task mean tool calls (base vs adapter)."""
import collections
import itertools
import json
import statistics
import sys

BASE, LORA = "qwen3.8-27b", "qwen3.8-pi"
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
by = collections.defaultdict(list)
for r in rows:
    by[(r["model"], r["task"])].append(r)
tasks = sorted({r["task"] for r in rows})


def agg(rs, k):
    return statistics.mean(r[k] for r in rs)


print(f"{'model':12s} {'runs':>4s} {'solved':>7s} {'calls':>6s} {'turns':>6s} {'tokens':>7s} {'secs':>6s} {'bad':>4s} {'toolerr':>7s} {'raw':>4s}")
for m in (BASE, LORA):
    rs = [r for r in rows if r["model"] == m]
    print(f"{m:12s} {len(rs):4d} {sum(r['solved'] for r in rs):3d}/{len(rs):<3d} {agg(rs,'tool_calls'):6.2f} {agg(rs,'turns'):6.2f} "
          f"{agg(rs,'completion_tokens'):7.0f} {agg(rs,'seconds'):6.1f} {sum(r['bad_calls'] for r in rs):4d} "
          f"{sum(r['tool_errors'] for r in rs):7d} {sum(r['raw_tool_text'] for r in rs):4d}")
b = [r for r in rows if r["model"] == BASE]; l = [r for r in rows if r["model"] == LORA]
print(f"\nadapter vs base: tool calls {100*(agg(l,'tool_calls')/agg(b,'tool_calls')-1):+.0f}%, "
      f"turns {100*(agg(l,'turns')/agg(b,'turns')-1):+.0f}%, tokens {100*(agg(l,'completion_tokens')/agg(b,'completion_tokens')-1):+.0f}%, "
      f"time {100*(agg(l,'seconds')/agg(b,'seconds')-1):+.0f}%")

print(f"\n{'task':20s} {'base solved':>11s} {'pi solved':>9s} {'base calls':>10s} {'pi calls':>8s} {'base tok':>8s} {'pi tok':>6s}")
diffs = []
for t in tasks:
    rb, rl = by[(BASE, t)], by[(LORA, t)]
    if not rb or not rl:
        continue
    diffs.append(agg(rl, "tool_calls") - agg(rb, "tool_calls"))
    print(f"{t:20s} {sum(r['solved'] for r in rb):5d}/{len(rb):<5d} {sum(r['solved'] for r in rl):3d}/{len(rl):<5d} "
          f"{agg(rb,'tool_calls'):10.1f} {agg(rl,'tool_calls'):8.1f} {agg(rb,'completion_tokens'):8.0f} {agg(rl,'completion_tokens'):6.0f}")

# Paired sign-flip permutation test over tasks (exact for <= 16 tasks).
obs = statistics.mean(diffs)
n = len(diffs)
count = sum(1 for signs in itertools.product((1, -1), repeat=n)
            if abs(statistics.mean(s * d for s, d in zip(signs, diffs))) >= abs(obs) - 1e-12)
print(f"\nper-task mean tool-call difference (pi - base): {obs:+.2f}; tasks where pi used fewer: "
      f"{sum(d < 0 for d in diffs)}/{n}; exact two-sided sign-flip p = {count / 2**n:.4f}")
