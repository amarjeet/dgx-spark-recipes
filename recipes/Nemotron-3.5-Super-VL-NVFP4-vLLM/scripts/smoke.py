#!/usr/bin/env python3
"""Smoke-test a running Nemotron-3.5-Super-VL server (stdlib only).

Three checks specific to this model, not just "did it return 200":

  1. text: the reasoning parser splits the trace into `reasoning_content`, and
     a small arithmetic question comes out right;
  2. thinking off: `enable_thinking: false` yields an answer with no trace;
  3. image: a generated PNG (a solid blue square on white) is described with
     the right colour, which fails if the vision tower or processor is broken.

Usage:
  ./scripts/smoke.py              # port from PORT, default 8013
  ./scripts/smoke.py --max-tokens 4096
"""
import argparse
import base64
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib


def png(width: int, height: int, pixel) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))
    rows = b"".join(
        b"\x00" + b"".join(bytes(pixel(x, y)) for x in range(width)) for y in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def square(x: int, y: int):
    return (0, 0, 255) if 96 <= x < 160 and 96 <= y < 160 else (255, 255, 255)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8013")))
    parser.add_argument("--host", default=os.environ.get("SMOKE_HOST", "127.0.0.1"))
    # Reasoning tokens and the answer share this budget.
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    base = f"http://{args.host}:{args.port}"

    def post(payload: dict) -> tuple[dict, float]:
        request = urllib.request.Request(
            f"{base}/v1/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            return json.load(response), time.monotonic() - started

    try:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=30) as response:
            model = json.load(response)["data"][0]["id"]
    except (urllib.error.URLError, OSError) as exc:
        print(f"cannot reach {base}: {exc}\nis the server up? ./status.sh", file=sys.stderr)
        return 1
    print(f"server : {base}\nmodel  : {model}\n")

    failures = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
        if not ok:
            failures.append(f"{name} {detail}".strip())

    def message(body: dict) -> dict:
        return body["choices"][0]["message"]

    # 1. text, thinking on (the template default)
    body, secs = post({"model": model, "max_tokens": args.max_tokens, "messages": [
        {"role": "user", "content": "What is 17 * 23? Answer with just the number."}]})
    m = message(body)
    tokens = body["usage"]["completion_tokens"]
    print(f"text      {tokens} tokens in {secs:.1f}s ({tokens / secs:.1f} tok/s incl. prefill)")
    check("text answer is 391", "391" in (m.get("content") or ""), repr(m.get("content")))
    check("reasoning_content populated", bool((m.get("reasoning_content") or m.get("reasoning") or "").strip()))
    check("no <think> leak in content", "<think>" not in (m.get("content") or ""))

    # 2. thinking off
    body, _ = post({"model": model, "max_tokens": 256,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": "Name the capital of France in one word."}]})
    m = message(body)
    check("enable_thinking=false answers", "paris" in (m.get("content") or "").lower(), repr(m.get("content")))
    check("enable_thinking=false has no trace", not (m.get("reasoning_content") or m.get("reasoning") or "").strip())

    # 3. image
    url = "data:image/png;base64," + base64.b64encode(png(256, 256, square)).decode()
    body, secs = post({"model": model, "max_tokens": args.max_tokens, "temperature": 0.6, "top_p": 0.95,
                       "messages": [{"role": "user", "content": [
                           {"type": "image_url", "image_url": {"url": url}},
                           {"type": "text", "text": "What colour is the square in this image? One word."}]}]})
    m = message(body)
    print(f"image     {body['usage']['prompt_tokens']} prompt tokens, {secs:.1f}s")
    check("image: square reported as blue", "blue" in (m.get("content") or "").lower(), repr(m.get("content")))

    if failures:
        print("\nFAILED:\n" + "\n".join(f"  - {f}" for f in failures))
        return 1
    print("\nsmoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
