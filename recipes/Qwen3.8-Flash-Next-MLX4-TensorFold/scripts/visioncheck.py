#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 MiaAI-Lab (original)
# Copyright (c) 2026 amarjeet (port)
#
# Derived from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
# (tools/visioncheck.py), MIT. Standalone here: no import of bench.py.
"""Image input (VISION=1): draws a red circle and a blue square into a PNG,
sends it as a data URL and checks the model names both.

Usage: scripts/visioncheck.py      (API_URL, or PORT / SMOKE_HOST; SERVED_MODEL_NAME)
Exit code 1 on failure.
"""
import base64
import json
import os
import struct
import sys
import urllib.error
import urllib.request
import zlib

BASE = os.environ.get("API_URL") or f"http://{os.environ.get('SMOKE_HOST', '127.0.0.1')}:{os.environ.get('PORT', '8011')}"
MODEL = os.environ.get("SERVED_MODEL_NAME", "qwen3.8-flash-next")
W, H = 480, 240


def png() -> bytes:
    """A white 480 x 240 RGB image: a red circle on the left, a blue square on the right (stdlib only)."""

    rows = []
    for y in range(H):
        row = bytearray([0])                          # filter: none
        for x in range(W):
            if (x - 120) ** 2 + (y - 120) ** 2 <= 80 ** 2:
                row += bytes((220, 30, 30))
            elif 280 <= x < 440 and 40 <= y < 200:
                row += bytes((30, 60, 220))
            else:
                row += bytes((255, 255, 255))
        rows.append(bytes(row))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"".join(rows)))
            + chunk(b"IEND", b""))


def main() -> int:
    url = "data:image/png;base64," + base64.b64encode(png()).decode()
    body = {"model": MODEL, "max_tokens": 200, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "Which shapes are in this image, and what colour is each? One sentence."}]}]}
    req = urllib.request.Request(BASE.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        out = json.load(urllib.request.urlopen(req, timeout=600))
    except urllib.error.HTTPError as exc:     # a server started with VISION=0 answers 400
        print(f"visioncheck: FAILED: HTTP {exc.code} (is the server running with VISION=1?)")
        return 1
    text = (out["choices"][0]["message"].get("content") or "").strip()
    print(f"reply ({out.get('usage', {}).get('prompt_tokens')} prompt tokens): {text}")
    ok = all(word in text.lower() for word in ("red", "circle", "blue", "square"))
    print("visioncheck:", "OK: the model sees the red circle and the blue square" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
