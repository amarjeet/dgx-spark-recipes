#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2026 amarjeet
#
# Written for this port. Part of the same AGPL-3.0-or-later combined work
# as the files it sits beside, which derive from MiaAI-Lab/
# Qwen3.8-Flash-Next-Single-DGX-Spark, Copyright (C) 2026 MiaAI Lab.
"""Measure prefill and decode throughput against context depth.

The point of a ~1M-token KV pool is long context, so the number that matters is
not throughput at a 2K prompt -- it is how prefill and decode hold up as the
context fills. This drives the running server at a ladder of depths and reports
TTFT, prefill rate and decode rate for each.

Depths are reached with real tokens, not padding: a repeated block of prose,
trimmed to the requested length using the server's own /tokenize endpoint, so
the depth is exact rather than estimated.

vLLM's OpenAI endpoint returns no llama.cpp-style `timings` block, so timings
come from a streamed response: TTFT is the wall time to the first token, and
the prefill rate is depth/TTFT. That slightly understates prefill (TTFT
includes queueing and the first decode step) and is the honest number a client
actually sees.

Results go to $OUT_DIR, never into the recipe directory.

Usage:
  ./scripts/bench_depths.py                                  # 2K..128K ladder
  ./scripts/bench_depths.py --depths 4096 32768 262144
  ./scripts/bench_depths.py --out "$OUT_DIR/bench.json"
"""
import argparse
import json
import os
import secrets
import statistics
import sys
import time
import urllib.request

BLOCK = (
    "The quarterly review covered capacity planning across the three regional "
    "clusters, the migration of the ingest pipeline onto the new scheduler, and "
    "the backlog of reliability work carried over from the previous cycle. "
    "Throughput improved after the batching change, though tail latency "
    "regressed under bursty load and the team traced it to lock contention in "
    "the metadata service rather than to the storage layer. Follow-up items "
    "were assigned owners and due dates, and the risk register was updated to "
    "reflect the revised dependency on the upstream vendor's release schedule. "
)


def post(base: str, path: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        f"{base}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def token_count(base: str, model: str, text: str, timeout: int) -> int:
    body = post(base, "/tokenize", {"model": model, "prompt": text}, timeout)
    return body.get("count", len(body.get("tokens", [])))


def prompt_of_depth(base: str, model: str, depth: int, timeout: int) -> str:
    """Build a unique prompt of almost exactly `depth` tokens.

    Grow by whole blocks until we overshoot, then binary-search the character
    cut. Tokenising is cheap next to a 128K-token prefill, so exactness is
    worth the extra round trips.

    The leading nonce is what makes this a *prefill* benchmark. The server runs
    with prefix caching on, and every depth here is built from the same block
    of prose -- so without it the 8K prompt is a literal prefix of the 32K one,
    each repeat re-hits its own cached KV, and the reported rates are the
    prompt cache rather than the engine. Measured on this host, that inflated
    131K prefill to ~39,000 tok/s against a true ~1,500.
    """
    nonce = f"Reference {secrets.token_hex(16)}. "
    per_block = token_count(base, model, BLOCK, timeout)
    text = nonce + BLOCK * max(1, depth // max(per_block, 1) + 2)
    lo, hi = len(nonce), len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if token_count(base, model, text[:mid], timeout) < depth:
            lo = mid + 1
        else:
            hi = mid
    return text[:lo]


def timed_completion(base: str, model: str, prompt: str, predict: int, timeout: int):
    """Stream a completion; return (ttft_seconds, decode_tok_s, tokens)."""
    request = urllib.request.Request(
        f"{base}/v1/completions",
        data=json.dumps({
            "model": model,
            "prompt": prompt,
            "max_tokens": predict,
            # Deterministic so repeats measure the machine, not sampling.
            "temperature": 0.0,
            # Required, not cosmetic: on a long prompt of repeated filler the
            # model emits EOS on the first token, so without this every depth
            # past ~2K decodes exactly one token and reports nothing.
            "ignore_eos": True,
            "min_tokens": predict,
            "stream": True,
            # Ask for the usage chunk: with MTP speculative decoding a single
            # SSE chunk can carry several accepted tokens, so counting chunks
            # would understate decode. This gives the engine's own token count.
            "stream_options": {"include_usage": True},
        }).encode(),
        headers={"Content-Type": "application/json"})

    started = time.monotonic()
    ttft, chunks, completion_tokens = None, 0, 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                continue
            usage = payload.get("usage")
            if usage and usage.get("completion_tokens"):
                completion_tokens = usage["completion_tokens"]
            try:
                text = payload["choices"][0].get("text") or ""
            except (KeyError, IndexError):
                continue
            if text:
                if ttft is None:
                    ttft = time.monotonic() - started
                chunks += 1
    elapsed = time.monotonic() - started
    if ttft is None:
        return None, 0.0, 0
    # Fall back to chunks only if the server sent no usage block.
    produced = completion_tokens or chunks
    decode_rate = max(produced - 1, 0) / max(elapsed - ttft, 1e-9)
    return ttft, decode_rate, produced


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8888")))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--depths", type=int, nargs="+",
                        default=[2048, 8192, 32768, 131072])
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
        with urllib.request.urlopen(f"{base}/v1/models", timeout=30) as response:
            info = json.load(response)["data"][0]
    except OSError as exc:
        print(f"cannot reach {base}: {exc}", file=sys.stderr)
        return 1

    model = info["id"]
    max_len = info.get("max_model_len")
    print(f"model        : {model}")
    print(f"max_model_len: {max_len}")
    print(f"decoding     : {args.predict} tokens per sample, {args.repeats} repeat(s)\n")

    rows = []
    print(f"{'depth':>8}  {'TTFT s':>9}  {'prefill tok/s':>14}  {'decode tok/s':>13}")
    print("-" * 52)
    for depth in args.depths:
        if max_len and depth + args.predict > max_len:
            print(f"{depth:>8}  skipped (exceeds max_model_len={max_len})")
            continue
        ttfts, decodes = [], []
        for _ in range(args.repeats):
            # Rebuilt per repeat: a reused prompt is a prefix-cache hit and
            # would report a fictitious prefill rate.
            prompt = prompt_of_depth(base, model, depth, args.timeout)
            ttft, decode_rate, _ = timed_completion(
                base, model, prompt, args.predict, args.timeout)
            if ttft is None:
                continue
            ttfts.append(ttft)
            decodes.append(decode_rate)
        if not ttfts:
            print(f"{depth:>8}  no tokens returned")
            continue
        ttft_med = statistics.median(ttfts)
        row = {
            "depth": depth,
            "ttft_s": ttft_med,
            # TTFT includes queueing and the first decode step, so this is a
            # lower bound on the engine's raw prefill rate.
            "prefill_tok_s": depth / ttft_med,
            "decode_tok_s": statistics.median(decodes),
        }
        rows.append(row)
        print(f"{depth:>8}  {row['ttft_s']:>9.2f}  {row['prefill_tok_s']:>14.1f}  "
              f"{row['decode_tok_s']:>13.2f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump({
                "model": model,
                "max_model_len": max_len,
                "predict": args.predict,
                "repeats": args.repeats,
                "rows": rows,
            }, handle, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
