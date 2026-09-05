#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2026 amarjeet
#
# Written for this port. Part of the same AGPL-3.0-or-later combined work
# as the files it sits beside, which derive from MiaAI-Lab/
# Qwen3.8-Flash-Next-Single-DGX-Spark, Copyright (C) 2026 MiaAI Lab.
"""Smoke-test a running Qwen3.8-Flash-Next server.

Checks the things that are specific to this deployment rather than just "did it
return 200":

  1. the reasoning trace arrives in its own field (`reasoning` on this vLLM
     build, `reasoning_content` on llama.cpp and older builds) and does not
     leak into `content` -- vLLM is started with `--reasoning-parser qwen3`,
     so a trace inline in `content` means the parser is not attached;
  2. arithmetic the trace has to actually carry out, so an empty or truncated
     reasoning budget shows up as a wrong number rather than a pass;
  3. the vision tower answers about an image, which is the half of this
     checkpoint a text-only test never touches. The image is generated here in
     pure stdlib -- no PIL, no fixture file in the recipe directory;
  4. decode throughput, measured from a streamed response (vLLM's OpenAI
     endpoint returns no llama.cpp-style `timings` block).

Usage:
  ./scripts/smoke.py                 # default port from profiles.sh
  PORT=8888 ./scripts/smoke.py
  ./scripts/smoke.py --skip-vision
  ./scripts/smoke.py --max-tokens 16384
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

PROMPT = (
    "A company reports FY2025 revenue of $4.20B (up from $3.55B in FY2024), "
    "gross margin of 61%, operating expenses of $1.98B, and a 22% effective "
    "tax rate. It carries $900M of debt at 5.4% and $1.30B of cash.\n\n"
    "Compute FY2025 operating income, net income, and net debt. Then state "
    "which single assumption in this set most limits the reliability of a "
    "forward FY2026 net income estimate, and why."
)


def png_solid(width: int, height: int, rgb: tuple) -> bytes:
    """Encode a solid-colour PNG with the standard library only."""
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def post(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def stream(url: str, payload: dict, timeout: int):
    """Yield (delta, is_reasoning) and report TTFT plus decode rate."""
    # include_usage: with MTP speculative decoding one SSE chunk can carry
    # several accepted tokens, so counting chunks understates decode by ~2.5x
    # on this model. Take the engine's own token count instead.
    payload = dict(payload, stream=True, stream_options={"include_usage": True})
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    started = time.monotonic()
    ttft, chunks, completion_tokens = None, 0, 0
    content, reasoning = [], []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                payload_chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            usage = payload_chunk.get("usage")
            if usage and usage.get("completion_tokens"):
                completion_tokens = usage["completion_tokens"]
            try:
                delta = payload_chunk["choices"][0]["delta"]
            except (KeyError, IndexError):
                continue
            piece = delta.get("content") or ""
            # This vLLM build emits `reasoning`; llama.cpp and older vLLM
            # builds emit `reasoning_content`. Accept either.
            trace = delta.get("reasoning") or delta.get("reasoning_content") or ""
            if piece or trace:
                if ttft is None:
                    ttft = time.monotonic() - started
                chunks += 1
            content.append(piece)
            reasoning.append(trace)
    elapsed = time.monotonic() - started
    produced = completion_tokens or chunks
    decode_rate = produced / max(elapsed - (ttft or 0), 1e-9)
    return "".join(content), "".join(reasoning), ttft, produced, decode_rate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8888")))
    parser.add_argument("--host", default=os.environ.get("SMOKE_HOST", "127.0.0.1"))
    # This checkpoint reasons at length; a small budget truncates mid-thought
    # and looks like an empty-response failure.
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--skip-vision", action="store_true")
    parser.add_argument("--vision-max-tokens", type=int, default=4096,
                        help="the reasoning trace is billed against this too")
    parser.add_argument("--prompt", default=PROMPT)
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    try:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=30) as response:
            models = json.load(response)["data"]
    except (urllib.error.URLError, OSError) as exc:
        print(f"cannot reach {base}: {exc}", file=sys.stderr)
        print("is the server up? ./status.sh", file=sys.stderr)
        return 1
    model_id = models[0]["id"]
    print(f"server : {base}")
    print(f"model  : {model_id}\n")

    failures = []

    print("=== 1. reasoning + arithmetic (streamed) ===")
    content, reasoning, ttft, tokens, rate = stream(
        f"{base}/v1/chat/completions",
        {"model": model_id,
         "messages": [{"role": "user", "content": args.prompt}],
         "max_tokens": args.max_tokens},
        args.timeout)

    print("--- reasoning ---")
    print(reasoning.strip()[:900] or "(empty)")
    print("\n--- content ---")
    print(content.strip()[:1500] or "(empty)")
    print(f"\n  TTFT {ttft:.2f}s   {tokens} tokens   ~{rate:.1f} tok/s decode"
          if ttft else "\n  (no streamed tokens)")

    print("\n--- checks ---")
    if content.strip():
        print("  OK   non-empty content")
    elif reasoning.strip():
        # The trace is billed against the same budget, so a long reasoner can
        # spend all of it before emitting a single content token.
        failures.append(
            f"empty content, but {len(reasoning)} chars of reasoning -- the trace "
            f"used the whole {args.max_tokens}-token budget. Retry with --max-tokens "
            f"{args.max_tokens * 2}.")
    else:
        failures.append("empty content and empty reasoning")

    if reasoning.strip():
        print("  OK   reasoning trace populated (qwen3 reasoning parser attached)")
    else:
        failures.append(
            "reasoning trace is empty -- --reasoning-parser qwen3 is not attached, "
            "or the chat template is wrong")

    if "<think>" in content:
        failures.append("<think> leaked into content (reasoning parser not applied)")
    else:
        print("  OK   no <think> leakage in content")

    # 4.20B * 0.61 = 2.562B gross profit; - 1.98B opex = 582M operating income.
    # Net debt = 900M - 1300M = -400M, i.e. a net cash position.
    if any(t in content for t in ("582", "0.58", "$582")):
        print("  OK   operating income ~$582M present")
    else:
        failures.append("expected operating income ~$582M in the answer")

    if any(t in content.lower() for t in ("net cash", "-400", "(400", "400m net")):
        print("  OK   net debt resolved to a net cash position")
    else:
        failures.append("expected the net debt figure to come out as ~$400M net cash")

    if not args.skip_vision:
        print("\n=== 2. vision tower ===")
        # A solid crimson square. Colour is the one property a 27-layer vision
        # tower cannot get right by guessing from the text prompt alone.
        data_url = "data:image/png;base64," + base64.b64encode(
            png_solid(336, 336, (220, 20, 60))).decode()
        try:
            body = post(f"{base}/v1/chat/completions", {
                "model": model_id,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text",
                     "text": "What single colour fills this image? Answer with the "
                             "colour name only."},
                ]}],
                # The reasoning trace is billed against this budget too, and
                # the model will happily spend hundreds of tokens deliberating
                # over a solid colour. 512 was not enough.
                "max_tokens": args.vision_max_tokens,
            }, args.timeout)
            message = body["choices"][0]["message"]
            answer = (message.get("content") or "").strip()
            trace = (message.get("reasoning") or message.get("reasoning_content") or "").strip()
            print(f"  answer: {answer[:200] or '(empty)'}")
            wanted = ("red", "crimson", "scarlet", "maroon")
            if any(w in answer.lower() for w in wanted):
                print("  OK   vision tower identified the colour")
            elif not answer and any(w in trace.lower() for w in wanted):
                # The tower saw it; only the token budget was short. Still a
                # failure, but say which one so it is not mistaken for blindness.
                failures.append(
                    f"vision tower saw the colour but spent all "
                    f"{args.vision_max_tokens} tokens reasoning -- raise "
                    f"--vision-max-tokens")
            else:
                failures.append(f"vision tower did not identify red/crimson (said: {answer[:120]!r})")
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            failures.append(f"vision request failed: {exc}")

    print()
    if failures:
        print("FAILED:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
