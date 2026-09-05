#!/usr/bin/env python3
"""Measure prefill and decode throughput against context depth.

Ling-3.0-flash-Fin's pitch is long-context financial work -- multi-document
reasoning over filings -- so the number that matters is not throughput at a
2K prompt, it is how prefill and decode hold up as the context fills. This
drives the running server at a ladder of depths and reports llama.cpp's own
`timings` for each.

Depths are reached with real tokens, not padding: a repeated block of
filing-style prose, trimmed to the requested length using the server's
/tokenize endpoint so the depth is exact rather than estimated.

Usage:
  ./scripts/bench_depths.py                              # 2K..64K ladder
  ./scripts/bench_depths.py --depths 4096 32768 131072
  ./scripts/bench_depths.py --out "$OUT_DIR/bench.json"
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

BLOCK = (
    "Item 7. Management's Discussion and Analysis of Financial Condition and "
    "Results of Operations. Revenue for the period increased primarily as a "
    "result of higher unit volumes in the commercial segment, partially offset "
    "by unfavorable foreign currency translation and continued pricing pressure "
    "in the legacy hardware line. Gross margin expanded on a favorable mix "
    "shift toward subscription revenue, and operating expenses grew more slowly "
    "than revenue, reflecting disciplined headcount growth. Cash provided by "
    "operating activities was driven by net income adjusted for non-cash "
    "charges, offset by an increase in accounts receivable consistent with the "
    "timing of fourth-quarter billings. "
)


def post(base: str, path: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def token_count(base: str, text: str, timeout: int) -> int:
    return len(post(base, "/tokenize", {"content": text}, timeout)["tokens"])


def prompt_of_depth(base: str, depth: int, timeout: int) -> str:
    """Build a prompt of almost exactly `depth` tokens.

    Grow by whole blocks until we overshoot, then binary-search the character
    cut. Tokenising is cheap next to a 64K-token prefill, so exactness is
    worth the extra round trips.
    """
    per_block = token_count(base, BLOCK, timeout)
    text = BLOCK * max(1, depth // max(per_block, 1) + 2)
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if token_count(base, text[:mid], timeout) < depth:
            lo = mid + 1
        else:
            hi = mid
    return text[:lo]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8008")))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--depths", type=int, nargs="+",
                        default=[2048, 8192, 16384, 32768, 65536])
    parser.add_argument("--predict", type=int, default=128,
                        help="tokens to decode at each depth")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--out", default=os.environ.get("BENCH_OUT", ""))
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=30):
            pass
    except OSError as exc:
        print(f"cannot reach {base}: {exc}", file=sys.stderr)
        return 1

    with urllib.request.urlopen(f"{base}/props", timeout=30) as response:
        props = json.load(response)
    n_ctx = props.get("default_generation_settings", {}).get("n_ctx") or props.get("n_ctx")
    print(f"server n_ctx : {n_ctx}")
    print(f"decoding     : {args.predict} tokens per sample, {args.repeats} repeat(s)\n")

    rows = []
    print(f"{'depth':>8}  {'prefill tok/s':>14}  {'decode tok/s':>13}  {'TTFT s':>8}")
    print("-" * 50)
    for depth in args.depths:
        if n_ctx and depth + args.predict > n_ctx:
            print(f"{depth:>8}  skipped (exceeds server n_ctx={n_ctx})")
            continue
        prompt = prompt_of_depth(base, depth, args.timeout)
        prefill, decode, ttft = [], [], []
        for _ in range(args.repeats):
            started = time.monotonic()
            body = post(base, "/completion", {
                "prompt": prompt,
                "n_predict": args.predict,
                # Deterministic so repeats measure the machine, not sampling.
                "temperature": 0.0,
                # Force a real prefill each time; otherwise repeat 2 hits the
                # prompt cache and reports a fictitious prefill rate.
                "cache_prompt": False,
            }, args.timeout)
            elapsed = time.monotonic() - started
            timings = body.get("timings") or {}
            prefill.append(timings.get("prompt_per_second", 0.0))
            decode.append(timings.get("predicted_per_second", 0.0))
            ttft.append(timings.get("prompt_ms", 0.0) / 1000.0 or elapsed)
        row = {
            "depth": depth,
            "prefill_tok_s": statistics.median(prefill),
            "decode_tok_s": statistics.median(decode),
            "ttft_s": statistics.median(ttft),
        }
        rows.append(row)
        print(f"{depth:>8}  {row['prefill_tok_s']:>14.1f}  "
              f"{row['decode_tok_s']:>13.2f}  {row['ttft_s']:>8.2f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump({
                "n_ctx": n_ctx,
                "predict": args.predict,
                "repeats": args.repeats,
                "rows": rows,
            }, handle, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
