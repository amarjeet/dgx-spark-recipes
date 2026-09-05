#!/usr/bin/env python3
"""Smoke-test a running Ling-3.0-flash-Fin server.

Checks the three things that are actually specific to this model, rather than
just "did it return 200":

  1. the bailingmoe3 chat template loads and thinking mode is on by default
     (llama.cpp is started with --reasoning-format deepseek, so the trace must
     arrive in `reasoning_content`, not inline in `content`);
  2. the financial-reasoning behaviour the -Fin variant exists for;
  3. decode throughput, from llama.cpp's own `timings` block.

Usage:
  ./scripts/smoke.py                 # default port from profiles.sh
  PORT=8008 ./scripts/smoke.py
  ./scripts/smoke.py --max-tokens 1024
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

PROMPT = (
    "A company reports FY2025 revenue of $4.20B (up from $3.55B in FY2024), "
    "gross margin of 61%, operating expenses of $1.98B, and a 22% effective "
    "tax rate. It carries $900M of debt at 5.4% and $1.30B of cash.\n\n"
    "Compute FY2025 operating income, net income, and net debt. Then state "
    "which single assumption in this set most limits the reliability of a "
    "forward FY2026 net income estimate, and why."
)


def post(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8008")))
    parser.add_argument("--host", default=os.environ.get("SMOKE_HOST", "127.0.0.1"))
    # Thinking mode is on by default and this checkpoint reasons at length:
    # the finance prompt below routinely spends >2000 tokens in the trace
    # before emitting any content, so a smaller budget truncates mid-thought
    # and looks like an empty-response failure.
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=1800)
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

    body = post(
        f"{base}/v1/chat/completions",
        {
            "model": model_id,
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens,
            # Left unset deliberately: the server was started with the model
            # card's recommended temperature/top_p/top_k, and overriding them
            # here would test a configuration nobody runs.
        },
        args.timeout,
    )

    message = body["choices"][0]["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""

    print("--- reasoning_content ---")
    print(reasoning.strip()[:1200] or "(empty)")
    print("\n--- content ---")
    print(content.strip()[:2000] or "(empty)")

    timings = body.get("timings") or {}
    usage = body.get("usage") or {}
    print("\n--- timings ---")
    if timings:
        print(f"  prefill {timings.get('prompt_n', 0):>7} tok  "
              f"{timings.get('prompt_per_second', 0):.1f} tok/s")
        print(f"  decode  {timings.get('predicted_n', 0):>7} tok  "
              f"{timings.get('predicted_per_second', 0):.1f} tok/s")
    else:
        print(f"  (server returned no timings) usage={usage}")

    print("\n--- checks ---")
    failures = []

    if not content.strip():
        failures.append("empty content")
    else:
        print("  OK   non-empty content")

    if not reasoning.strip():
        # Thinking mode is on by default for this checkpoint, so an empty
        # reasoning_content means the template or --reasoning-format is wrong.
        failures.append(
            "reasoning_content is empty -- thinking mode is on by default for "
            "this model, so the chat template or --reasoning-format is wrong"
        )
    else:
        print("  OK   reasoning_content populated (thinking mode active)")

    if "<think>" in content:
        failures.append("<think> tag leaked into content (--reasoning-format not applied)")
    else:
        print("  OK   no <think> leakage in content")

    # 4.20B * 0.61 = 2.562B gross profit; - 1.98B opex = 582M operating income.
    # Net debt = 900M - 1300M = -400M (net cash). Accept either sign wording.
    if any(token in content for token in ("582", "0.58", "$582")):
        print("  OK   operating income ~$582M present")
    else:
        failures.append("expected operating income ~$582M in the answer")

    if any(token in content.lower() for token in ("net cash", "-400", "(400", "400m net")):
        print("  OK   net debt resolved to a net cash position")
    else:
        failures.append("expected the net debt figure to come out as ~$400M net cash")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nsmoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
