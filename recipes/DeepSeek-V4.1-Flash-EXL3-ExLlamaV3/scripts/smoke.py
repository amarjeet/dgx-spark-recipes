#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2026 amarjeet
#
# Written for this port. Part of the same AGPL-3.0-only combined work as the
# files it sits beside, which derive from
# vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe,
# Copyright (C) 2026 Victor Cruz.
"""Smoke-test the TabbyAPI OpenAI-compatible endpoint.

Stdlib only, like every helper in this repository: the recipes assume `python3`
and nothing else.

This is a liveness and sanity check, not a benchmark. It deliberately reports
decode tok/s anyway, because the one number worth eyeballing after a cold start
is whether speculation is working at all -- upstream measured 17.53 tok/s median
with the DSpark drafter and 15.13 without, so a result near 11 means the copy
regex did not take effect and every tensor was aliased.

Do not quote its numbers as benchmarks. BENCHMARKS.md records measurements with
their full runtime identity; this prints one sample from one prompt.

Usage:
  ./scripts/smoke.py
  ./scripts/smoke.py --port 8009 --max-tokens 128
  SMOKE_HOST=127.0.0.1 ./scripts/smoke.py
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def request(url: str, payload: dict | None, timeout: int) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("SMOKE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8009")))
    parser.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME",
                                                          "deepseek-v4.1-flash"))
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}/v1"

    try:
        models = request(f"{base}/models", None, 15)
    except (urllib.error.URLError, OSError) as exc:
        print(f"FAIL  {base}/models unreachable: {exc}", file=sys.stderr)
        print("      is the server up? ./start.sh", file=sys.stderr)
        return 1

    served = [m.get("id") for m in models.get("data", [])]
    if not served:
        print("FAIL  server lists no models", file=sys.stderr)
        return 1
    print(f"OK    endpoint up, serving: {', '.join(str(s) for s in served)}")

    # Use whatever the server actually reports rather than insisting on our own
    # name: TabbyAPI derives the model id from the pack directory, so a renamed
    # MODEL_ROOT would otherwise fail a test that has nothing to do with it.
    model = args.model if args.model in served else served[0]
    if model != args.model:
        print(f"      (using reported id {model!r}, not {args.model!r})")

    payload = {
        "model": model,
        "messages": [{"role": "user",
                      "content": "In one sentence, what is a trellis quantization code?"}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": False,
    }

    started = time.monotonic()
    try:
        result = request(f"{base}/chat/completions", payload, args.timeout)
    except (urllib.error.URLError, OSError) as exc:
        print(f"FAIL  chat/completions: {exc}", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - started

    choices = result.get("choices") or []
    if not choices:
        print(f"FAIL  no choices in response: {result}", file=sys.stderr)
        return 1
    text = (choices[0].get("message") or {}).get("content") or ""
    if not text.strip():
        print("FAIL  empty completion", file=sys.stderr)
        return 1

    usage = result.get("usage") or {}
    produced = usage.get("completion_tokens") or 0
    rate = produced / elapsed if produced and elapsed else 0.0

    print(f"OK    completion in {elapsed:.1f}s, {produced} tokens"
          + (f", {rate:.2f} tok/s" if rate else ""))
    print()
    print(text.strip()[:400])
    if rate and rate < 12.0:
        print()
        print(f"note: {rate:.2f} tok/s is below the ~15 tok/s upstream measured "
              "without a drafter.")
        print("      Check that EXL3_ATS_COPY took effect -- an all-aliased load "
              "lands near 11.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
