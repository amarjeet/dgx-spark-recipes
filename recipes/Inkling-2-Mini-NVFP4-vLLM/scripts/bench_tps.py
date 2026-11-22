#!/usr/bin/env python3
"""Decode throughput of a running Inkling 2 Mini server (stdlib only).

Streams natural-language and code prompts (random tokens would defeat the DSpark
drafter), MAX_TOKENS each, at each concurrency in turn. Every request is held to
exactly MAX_TOKENS (min_tokens) so the runs compare. Per level it prints one
JSON line: per-stream decode tok/s (completion tokens over the time after the
first token), aggregate tok/s over the whole span, mean TTFT, and the DSpark
mean acceptance length from vLLM's /metrics counters.

Usage: scripts/bench_tps.py [CONCURRENCIES] [MAX_TOKENS]    e.g. 1,2,4,8 1024
Env:   BASE (default http://127.0.0.1:$PORT), PORT (8015), MODEL (inkling-2-mini),
       EFFORT (reasoning_effort, default low)
"""
import json, os, sys, time, threading, urllib.request, re
BASE = os.environ.get("BASE", f"http://127.0.0.1:{os.environ.get('PORT', '8015')}")
MODEL = os.environ.get("MODEL", "inkling-2-mini")
EFFORT = os.environ.get("EFFORT", "low")
PROMPTS = [
  "Write a detailed explanation of how a hash map works, including collision handling, resizing and complexity.",
  "Write a Python implementation of an LRU cache with O(1) get and put, with docstrings and a short test.",
  "Explain the causes and consequences of the 1929 stock market crash in several paragraphs.",
  "Describe, step by step, how to plan a two-week trip through Japan on a moderate budget.",
  "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
  "Compare TCP and UDP in depth: reliability, ordering, congestion control and typical use cases.",
  "Explain how transformers use attention, from embeddings to the output layer, for a programmer audience.",
  "Write a Go HTTP server with graceful shutdown, structured logging and a health endpoint, with comments.",
]
CONCS = sys.argv[1] if len(sys.argv) > 1 else "1,2,4,8"
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 1024

def metrics():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    out = {}
    for name in ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_draft_tokens_total"):
        m = re.findall(r"^" + re.escape(name) + r"(?:\{[^}]*\})?\s+([0-9.e+]+)$", txt, re.M)
        out[name] = sum(float(x) for x in m) if m else None
    return out

def one(prompt, res):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAXTOK,
            "min_tokens": MAXTOK, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": EFFORT}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); tf = None; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            ev = json.loads(d)
            if ev.get("usage"): usage = ev["usage"]
            for ch in ev.get("choices", []):
                dl = ch.get("delta", {})
                if tf is None and (dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning")): tf = time.time()
    te = time.time()
    n = usage["completion_tokens"]
    res.append({"ttft": tf - t0, "tok": n, "decode": n / (te - tf), "wall": te - t0, "t0": t0, "te": te})

for conc in [int(c) for c in CONCS.split(",")]:
    m0 = metrics(); res = []
    ths = [threading.Thread(target=one, args=(PROMPTS[i % len(PROMPTS)], res)) for i in range(conc)]
    for t in ths: t.start()
    for t in ths: t.join()
    m1 = metrics()
    span = max(r["te"] for r in res) - min(r["t0"] for r in res)
    tot = sum(r["tok"] for r in res)
    dr = (m1["vllm:spec_decode_num_drafts_total"] or 0) - (m0["vllm:spec_decode_num_drafts_total"] or 0)
    acc = (m1["vllm:spec_decode_num_accepted_tokens_total"] or 0) - (m0["vllm:spec_decode_num_accepted_tokens_total"] or 0)
    print(json.dumps({"concurrency": conc, "max_tokens": MAXTOK, "effort": EFFORT, "tokens_each": [r["tok"] for r in res],
        "per_stream_decode_tok_s": {"min": round(min(r["decode"] for r in res),1), "mean": round(sum(r["decode"] for r in res)/conc,1), "max": round(max(r["decode"] for r in res),1)},
        "aggregate_tok_s": round(tot / span, 1), "ttft_s_mean": round(sum(r["ttft"] for r in res)/conc, 2),
        "accept_len": round(1 + acc / dr, 2) if dr else None}), flush=True)
