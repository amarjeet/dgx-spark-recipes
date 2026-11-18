#!/usr/bin/env python3
"""Smoke-test a running Qwen3.8-27B + pi LoRA server (stdlib only).

Checks specific to this recipe, not just "did it return 200":

  1. models: the base and the adapter are both listed (the adapter only when
     the server runs with LoRA);
  2. text, thinking off by default: a small arithmetic question comes out
     right with no reasoning trace, as the adapter was trained;
  3. tool call: a `tools` request comes back as a parsed `tool_calls` entry
     through the qwen3_xml parser, not as XML in `content` -- pi depends on it;
  4. adapter applied: on the same prompt, the adapter's next-token logprobs
     differ from the base's. No shift means the LoRA weights are not reaching
     the model.

Usage:
  ./scripts/smoke.py              # port from PORT, default 8014
  ./scripts/smoke.py --timeout 1800
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_shell",
        "description": "Run a shell command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run."}},
            "required": ["command"],
        },
    },
}]

# A raw-completion prompt in the model's chat format, thinking off.
PROBE_PROMPT = ("<|im_start|>user\nYou are in a repo. The test suite fails with ImportError. "
                "What is your first shell command?<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8014")))
    parser.add_argument("--host", default=os.environ.get("SMOKE_HOST", "127.0.0.1"))
    parser.add_argument("--base-model", default=os.environ.get("SERVED_MODEL_NAME", "qwen3.8-27b"))
    parser.add_argument("--lora-model", default=os.environ.get("LORA_NAME", "qwen3.8-pi"))
    # BF16 27B decodes at a few tokens per second here; be patient.
    parser.add_argument("--timeout", type=int, default=1200)
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
            models = [m["id"] for m in json.load(response)["data"]]
    except (urllib.error.URLError, OSError) as exc:
        print(f"cannot reach {base}: {exc}\nis the server up? ./status.sh", file=sys.stderr)
        return 1
    print(f"server : {base}\nmodels : {', '.join(models)}\n")

    failures = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
        if not ok:
            failures.append(f"{name} {detail}".strip())

    def message(body: dict) -> dict:
        return body["choices"][0]["message"]

    def trace(m: dict) -> str:
        return (m.get("reasoning_content") or m.get("reasoning") or "").strip()

    # 1. models
    check(f"base listed as {args.base_model}", args.base_model in models, repr(models))
    lora = args.lora_model in models
    print(f"  ----  adapter {args.lora_model}: {'listed' if lora else 'not served (base profile?)'}")
    model = args.lora_model if lora else args.base_model

    # 2. text, thinking off by default (--default-chat-template-kwargs)
    body, secs = post({"model": model, "max_tokens": 64, "temperature": 0, "messages": [
        {"role": "user", "content": "What is 17 * 23? Answer with just the number."}]})
    m = message(body)
    tokens = body["usage"]["completion_tokens"]
    print(f"\ntext      {tokens} tokens in {secs:.1f}s ({tokens / secs:.1f} tok/s incl. prefill)")
    check("text answer is 391", "391" in (m.get("content") or ""), repr(m.get("content")))
    check("thinking off by default: no trace", not trace(m), repr(trace(m)[:80]))
    check("no <think> leak in content", "<think>" not in (m.get("content") or ""))

    # 3. tool call through qwen3_xml
    body, secs = post({"model": model, "max_tokens": 256, "temperature": 0, "tools": TOOLS,
                       "messages": [{"role": "user", "content":
                                     "How many files are in the current directory? Use the tool."}]})
    m = message(body)
    calls = m.get("tool_calls") or []
    print(f"tool      {secs:.1f}s, {len(calls)} call(s)")
    ok = bool(calls) and calls[0]["function"]["name"] == "run_shell"
    check("tool call parsed into tool_calls", ok, repr(m.get("content"))[:200])
    if ok:
        try:
            cmd = json.loads(calls[0]["function"]["arguments"]).get("command", "")
        except json.JSONDecodeError:
            cmd = None
        check("tool arguments are JSON with a command", bool(cmd), repr(calls[0]["function"]["arguments"]))
    check("no raw <tool_call> in content", "<tool_call>" not in (m.get("content") or ""))

    # 4. the adapter changes the distribution. Compare logprobs, not text: on
    # an easy prompt the RL adapter often keeps the base's greedy choice (seen
    # here with a dedupe() prompt), while every token's logprob still moves.
    if lora:
        lps = {}
        for name in (args.base_model, args.lora_model):
            request = urllib.request.Request(
                f"{base}/v1/completions", headers={"Content-Type": "application/json"},
                data=json.dumps({"model": name, "prompt": PROBE_PROMPT, "max_tokens": 1,
                                 "temperature": 0, "logprobs": 5}).encode())
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                lps[name] = json.load(response)["choices"][0]["logprobs"]["top_logprobs"][0]
        shared = set(lps[args.base_model]) & set(lps[args.lora_model])
        delta = max((abs(lps[args.base_model][t] - lps[args.lora_model][t]) for t in shared), default=0.0)
        print(f"probe     max |dlogprob| over {len(shared)} shared top-5 first tokens: {delta:.3f}")
        check("adapter shifts the next-token distribution (LoRA is applied)", delta > 0.05,
              f"max |dlogprob| {delta:.4f}")

    if failures:
        print("\nFAILED:\n" + "\n".join(f"  - {f}" for f in failures))
        return 1
    print("\nsmoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
