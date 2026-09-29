#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
"""Aggregate decode with N concurrent streaming requests. Standard library only.

Each request asks for prose without thinking, so every token is answer text,
and runs to max_tokens. Reports aggregate and per-request tok/s (completion
tokens over the time from the first token to the last) and the median time to
first token. Upstream's README reports this shape at 1, 2, 4 and 5 requests.

Usage: scripts/bench_concurrent.py [n ...]   (default 1 2 4 5; API_URL / PORT / SERVED_MODEL_NAME;
       MAX_TOKENS, default 512)
"""
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

API_URL = os.environ.get("API_URL", "http://127.0.0.1:" + os.environ.get("PORT", "8011")).rstrip("/")
MODEL = os.environ.get("SERVED_MODEL_NAME", "qwen3.8-flash-next")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
TOPICS = ["lighthouses", "glaciers", "printing presses", "honeybees", "suspension bridges", "tea ceremonies",
          "coral reefs", "steam engines"]


def one(i: int, out: list) -> None:
    body = {"model": MODEL, "stream": True, "max_tokens": MAX_TOKENS, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}, "min_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": f"Write a long, detailed essay about {TOPICS[i % len(TOPICS)]}."}]}
    req = urllib.request.Request(API_URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    start, first, last, tokens = time.time(), None, None, 0
    for line in urllib.request.urlopen(req, timeout=1800):
        line = line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        chunk = json.loads(line[5:])
        delta = (chunk.get("choices") or [{}])[0].get("delta", {})
        if delta.get("content") or delta.get("reasoning_content"):
            first = first or time.time()
            last = time.time()
        if chunk.get("usage"):
            tokens = chunk["usage"].get("completion_tokens", tokens)
    out.append({"ttft": (first or start) - start, "tokens": tokens, "start": first or start, "end": last or start})


def run(n: int) -> None:
    results: list = []
    threads = [threading.Thread(target=one, args=(i, results)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    per = [r["tokens"] / max(1e-9, r["end"] - r["start"]) for r in results]
    span = max(r["end"] for r in results) - min(r["start"] for r in results)
    total = sum(r["tokens"] for r in results)
    print(f"{n} concurrent: aggregate {total / span:6.1f} tok/s  per request {statistics.median(per):5.1f} tok/s  "
          f"TTFT {statistics.median(r['ttft'] for r in results) * 1000:5.0f} ms  ({total} tokens)", flush=True)


def main() -> None:
    for n in [int(a) for a in sys.argv[1:]] or [1, 2, 4, 5]:
        run(n)


if __name__ == "__main__":
    main()
