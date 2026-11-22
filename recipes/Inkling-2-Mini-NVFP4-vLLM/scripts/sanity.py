#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Tech2wild
#
# From tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark (sanity.py) at 300942d, MIT.
# Changed: the default BASE follows PORT (this recipe serves on 8015), and the
# usage lines name this path.
"""Light sanity checks for the Inkling 2 Mini server (stdlib only, no load test).

  ./scripts/sanity.py chat                      # 2 short chats: correctness + TTFT + decode tok/s
  ./scripts/sanity.py needle --tokens 200000    # one long-context needle check, reports TTFT

Env: BASE (default http://localhost:$PORT, PORT default 8015), MODEL (default inkling-2-mini).
Runs one request at a time. Never run this as a benchmark or with concurrency.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request

BASE = os.environ.get("BASE", f"http://localhost:{os.environ.get('PORT', '8015')}")
MODEL = os.environ.get("MODEL", "inkling-2-mini")


def post(path, body, timeout=60):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stream_chat(messages, effort="low", max_tokens=1024, timeout=1800):
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"reasoning_effort": effort},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    t_first = None
    content, reasoning, usage = [], [], None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece_c = d.get("content") or ""
                piece_r = d.get("reasoning_content") or d.get("reasoning") or ""
                if (piece_c or piece_r) and t_first is None:
                    t_first = time.time()
                content.append(piece_c)
                reasoning.append(piece_r)
    t_end = time.time()
    out_tok = (usage or {}).get("completion_tokens", 0)
    in_tok = (usage or {}).get("prompt_tokens", 0)
    ttft = (t_first or t_end) - t0
    decode_s = max(t_end - (t_first or t0), 1e-6)
    return {
        "content": "".join(content).strip(),
        "reasoning_chars": len("".join(reasoning)),
        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "ttft_s": round(ttft, 2),
        "total_s": round(t_end - t0, 2),
        "decode_tok_s": round(out_tok / decode_s, 1),
    }


def cmd_chat(_):
    tests = [
        ("What is 17 * 23? Answer with just the number.", ["391"]),
        (
            "List the eight planets of the solar system in order from the Sun, one per line, "
            "each with one short fact.",
            ["Mercury", "Neptune"],
        ),
    ]
    ok_all = True
    for q, must in tests:
        r = stream_chat([{"role": "user", "content": q}], effort="low", max_tokens=1500)
        ok = all(m.lower() in r["content"].lower() for m in must)
        ok_all &= ok
        print(json.dumps({"q": q[:60], "pass": ok, **{k: v for k, v in r.items() if k != "content"},
                          "answer_head": r["content"][:160]}))
    return 0 if ok_all else 1


WORDS = (
    "harbor lantern ledger orchard copper meadow signal river canvas engine garden window "
    "station market granite thunder velvet compass archive mirror timber falcon beacon quarry "
    "summit valley cobalt ember willow glacier circuit parcel anchor prairie violin saddle"
).split()
VERBS = "carried painted measured followed repaired counted traded mapped sorted guarded".split()


def filler_paragraph(rng):
    sents = []
    for _ in range(rng.randint(4, 7)):
        a, b, c = rng.sample(WORDS, 3)
        v = rng.choice(VERBS)
        n = rng.randint(2, 990)
        sents.append(f"The {a} keeper {v} {n} {b} crates past the {c} before noon.")
    return " ".join(sents)


def build_haystack(target_chars, needle, depth, seed=7):
    rng = random.Random(seed)
    paras, size = [], 0
    while size < target_chars:
        p = filler_paragraph(rng)
        paras.append(p)
        size += len(p) + 2
    paras.insert(int(len(paras) * depth), needle)
    return "\n\n".join(paras)


def cmd_needle(a):
    needle = "Important note: the access phrase for the north vault is VIOLET-HARBOR-7392."
    question = "What is the access phrase for the north vault? Reply with the phrase only."
    chars = int(a.tokens * 4.0)
    for _ in range(3):
        text = build_haystack(chars, needle, a.depth)
        msgs = [{"role": "user", "content": text + "\n\n" + question}]
        n = post("/tokenize", {"model": MODEL, "messages": msgs}, timeout=300).get("count", 0)
        print(json.dumps({"calibrate_chars": chars, "prompt_tokens": n}))
        if abs(n - a.tokens) / a.tokens < 0.03:
            break
        chars = int(chars * a.tokens / max(n, 1))
    r = stream_chat(msgs, effort=a.effort, max_tokens=2048, timeout=3600)
    ok = "VIOLET-HARBOR-7392" in r["content"].upper()
    print(json.dumps({"needle_pass": ok, "depth": a.depth,
                      "prefill_tok_s": round(r["prompt_tokens"] / max(r["ttft_s"], 1e-6), 0),
                      **{k: v for k, v in r.items() if k != "content"},
                      "answer_head": r["content"][:200]}))
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("chat")
    n = sp.add_parser("needle")
    n.add_argument("--tokens", type=int, default=200000)
    n.add_argument("--depth", type=float, default=0.5)
    n.add_argument("--effort", default="low")
    a = p.parse_args()
    sys.exit(cmd_chat(a) if a.cmd == "chat" else cmd_needle(a))


if __name__ == "__main__":
    main()
