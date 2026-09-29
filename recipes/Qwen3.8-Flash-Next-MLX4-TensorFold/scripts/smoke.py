#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
"""Smoke test against the running server. Standard library only.

Checks, in order:
  1. /v1/models lists the served id (and the upstream alias answers)
  2. a greedy non-thinking answer is correct -- a wrong activation or KV path
     still produces fluent text, so fluency proves nothing; arithmetic does
  3. thinking mode puts its trace in reasoning_content and the answer in content
  4. /health carries the live token counters (patch 0002)
  5. a tool call's array argument arrives as a JSON array (patch 0001)

Usage: scripts/smoke.py      (PORT, SMOKE_HOST, SERVED_MODEL_NAME, SERVED_ALIAS)
Exit code 1 on any failure.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOST = os.environ.get("SMOKE_HOST", "127.0.0.1")
PORT = os.environ.get("PORT", "8011")
BASE = f"http://{HOST}:{PORT}"
MODEL = os.environ.get("SERVED_MODEL_NAME", "qwen3.8-flash-next")
ALIAS = os.environ.get("SERVED_ALIAS", "Qwen3.8-Flash-Next")
failures = 0


def call(path: str, body: dict | None = None, timeout: float = 600) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def result(ok: bool, what: str) -> None:
    global failures
    failures += not ok
    print(f"  [{' OK ' if ok else 'FAIL'}] {what}", flush=True)


def chat(content: str, model: str = MODEL, **extra) -> tuple[dict, float]:
    t0 = time.time()
    r = call("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": content}], **extra})
    return r, time.time() - t0


def main() -> int:
    print(f"smoke test against {BASE} (model {MODEL})")
    try:
        ids = [m["id"] for m in call("/v1/models", timeout=10)["data"]]
    except (urllib.error.URLError, OSError) as exc:
        print(f"  [FAIL] cannot reach {BASE}: {exc} -- is it running? (./start.sh)")
        return 1
    result(MODEL in ids, f"/v1/models lists {MODEL} ({', '.join(ids)})")

    r, dt = chat("What is 17 * 23? Reply with the number only.", max_tokens=64, temperature=0,
                 chat_template_kwargs={"enable_thinking": False})
    answer = (r["choices"][0]["message"].get("content") or "").strip()
    result("391" in answer, f"greedy arithmetic: {answer[:40]!r} in {dt:.1f} s")

    if ALIAS and ALIAS != MODEL:
        try:
            r, _ = chat("Say ok.", model=ALIAS, max_tokens=16, chat_template_kwargs={"enable_thinking": False})
            result(bool(r["choices"][0]["message"].get("content")), f"alias {ALIAS} answers")
        except urllib.error.HTTPError as exc:
            result(False, f"alias {ALIAS}: HTTP {exc.code}")

    r, dt = chat("A train leaves at 09:40 and arrives at 13:05. How long is the trip? Answer briefly.",
                 max_tokens=2048, temperature=0, chat_template_kwargs={"reasoning_effort": "low"})
    msg = r["choices"][0]["message"]
    reasoning, content = msg.get("reasoning_content") or "", msg.get("content") or ""
    stats = r.get("tensorfold", {})
    result(bool(reasoning) and ("3 h" in content or "3 hours" in content or "3:25" in content or "205" in content),
           f"thinking: {len(reasoning)} chars of reasoning, answer {content.strip()[-60:]!r} "
           f"({r['usage']['completion_tokens']} tokens in {dt:.1f} s"
           + (f", {stats['decode_tps']:.1f} tok/s" if stats.get("decode_tps") else "") + ")")

    h = call("/health", timeout=10)
    result({"requests_running", "prompt_tokens_total", "completion_tokens_total"} <= h.keys(),
           f"/health live counters: completion_tokens_total {h.get('completion_tokens_total')}")

    tool = subprocess.run([sys.executable, str(Path(__file__).with_name("toolcheck.py"))],
                          env={**os.environ, "API_URL": BASE}, capture_output=True, text=True)
    result(tool.returncode == 0, (tool.stdout.strip().splitlines() or ["no output"])[-1])

    print("smoke: all checks passed" if failures == 0 else f"smoke: {failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
