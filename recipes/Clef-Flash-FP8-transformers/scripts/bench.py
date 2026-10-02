#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
"""Request latency by input length, plus throughput under concurrent clients.

Standard library only. The question set is fixed (one choice, one score, one
noul) and only the state grows, so the curve is the backbone's prefill cost.
Input lengths are what the server reports in usage.input_tokens, not targets.

Latency is end to end (HTTP + encode + forward); the server's own forward
time, from the X-Clef-Forward-Ms header, is reported beside it.

Usage: scripts/bench.py [REPEATS]      (PORT, SMOKE_HOST)
"""
import json
import os
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HOST = os.environ.get("SMOKE_HOST", "127.0.0.1")
PORT = os.environ.get("PORT", "8012")
URL = f"http://{HOST}:{PORT}/v1/systemone"
REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 20

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages",
                                "sales": "Pricing or new contracts"}},
    "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
    "outage": {"type": "noul", "instructions": "Is a service down?"},
}
SENTENCE = "The checkout service returned intermittent 502 errors for EU customers after the deploy. "


def one(state: str) -> tuple[float, float, int]:
    body = json.dumps({"model": "clef-flash", "state": state, "questions": QUESTIONS}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        payload = json.load(resp)
        forward = float(resp.headers.get("X-Clef-Forward-Ms", "nan"))
    return (time.perf_counter() - started) * 1000, forward, payload["usage"]["input_tokens"]


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))]


print(f"{URL}, {REPEATS} sequential requests per row\n")
print(f"{'input tokens':>12}  {'p50 ms':>8}  {'p95 ms':>8}  {'forward p50':>11}  {'prefill tok/s':>13}")
for sentences in (1, 12, 50, 200, 800):
    state = SENTENCE * sentences
    one(state)                                            # first of a shape is not representative
    rows = [one(state) for _ in range(REPEATS)]
    total = [r[0] for r in rows]
    forward = [r[1] for r in rows]
    tokens = rows[0][2]
    print(f"{tokens:>12}  {statistics.median(total):>8.1f}  {pct(total, 95):>8.1f}  "
          f"{statistics.median(forward):>11.1f}  {tokens / (statistics.median(forward) / 1000):>13.0f}")

state = SENTENCE * 3
print(f"\nconcurrent clients, ~{one(state)[2]}-token requests, {REPEATS * 4} requests per row")
print(f"{'clients':>7}  {'req/s':>7}  {'p50 ms':>8}  {'p95 ms':>8}")
for clients in (1, 2, 4, 8):
    started = time.perf_counter()
    with ThreadPoolExecutor(clients) as pool:
        rows = list(pool.map(lambda _: one(state), range(REPEATS * 4)))
    elapsed = time.perf_counter() - started
    total = [r[0] for r in rows]
    print(f"{clients:>7}  {len(rows) / elapsed:>7.1f}  {statistics.median(total):>8.1f}  {pct(total, 95):>8.1f}")
