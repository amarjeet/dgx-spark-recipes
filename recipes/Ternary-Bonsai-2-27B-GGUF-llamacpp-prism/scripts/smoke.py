#!/usr/bin/env python3
"""Smoke-test a running Ternary-Bonsai-2-27B server.

Checks the things that are specific to THIS model rather than "did it return
200". In rough order of how badly they fail silently:

  1. Needle retrieval at depth. llama.cpp issue #27756 is open against this
     exact architecture (48 gated-DeltaNet + 16 full-attention layers) and
     reports instant-EOS beyond roughly 130k tokens, non-monotonically: the
     prefill completes cleanly and then the model emits EOS as its first
     token. Nothing errors. A recipe that serves a 262144 context without
     testing it is advertising a number, not a capability -- so this test
     asks for a fact buried mid-prompt and distinguishes three outcomes:
     answered correctly, answered but lost the needle, and silent EOS.

  2. Arithmetic on a prompt the model card scores 95-98% on. These weights are
     stored in a Hadamard-rotated basis; a runtime that does not apply the
     matching activation transform produces FLUENT NONSENSE, not an error.
     Every other check here would pass on a silently wrong model.

  3. The server is serving the geometry it was asked for. llama.cpp can reduce
     a requested context to make it fit, so /props is compared against the
     profile rather than trusted.

  4. Thinking, reasoning_effort, XML tool calls and the vision projector.

Usage:
  ./scripts/smoke.py                          # full run, needle at the
                                              # server's own per-slot context
  ./scripts/smoke.py --needle-depth 65536     # needle at one explicit depth
  ./scripts/smoke.py --needle-depth -1        # skip the needle test
  ./scripts/smoke.py --needle-sweep 32768,65536,98304,131072,262144
                                              # walk depths to find the cliff
  ./scripts/smoke.py --quick                  # skip needle and vision
"""
import argparse
import base64
import json
import os
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib

# Answers: operating income 4.20*0.61 - 1.98 = 0.582B; net debt 0.9 - 1.3 =
# -0.4B, i.e. a net cash position. Chosen because both numbers are wrong in
# recognisable ways if the activation basis is wrong.
MATH_PROMPT = (
    "A company reports FY2025 revenue of $4.20B, gross margin of 61%, and "
    "operating expenses of $1.98B. It carries $900M of debt and $1.30B of "
    "cash.\n\n"
    "Compute FY2025 operating income and net debt. Give both as numbers."
)

# Filler prose for the needle test. Neutral, repetitive, and nothing like the
# needle, so recalling the needle cannot be confused with guessing.
FILLER = (
    "The regional logistics report notes that warehouse throughput remained "
    "within seasonal norms during the period under review, with inbound "
    "pallet volumes tracking slightly ahead of forecast and outbound dispatch "
    "times unchanged. Fleet utilisation was stable, maintenance windows were "
    "observed as scheduled, and no material deviations from the standing "
    "service agreement were recorded by the operations desk. "
)
NEEDLE = ("The maintenance access code for the Harbourgate cold-storage "
          "annex is 47-QUINCE-1983.")
NEEDLE_ANSWER = "47-QUINCE-1983"


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def active_geometry() -> tuple[str | None, str | None]:
    """The slot count and per-slot context the running profile asked for.

    Read from profiles.sh rather than taken from the environment, so a bare
    `./scripts/smoke.py` still checks the geometry. Without this the most
    important assertion in the file -- that llama.cpp did not silently shrink
    the context -- would quietly not run.
    """
    for name in ("PARALLEL", "CTX_PER_SLOT"):
        if os.environ.get(name):
            return os.environ.get("PARALLEL"), os.environ.get("CTX_PER_SLOT")
    script = (
        'source "$1/profiles.sh" >/dev/null 2>&1 || exit 1; '
        'select_profile "$(cat "$1/.profile.active" 2>/dev/null '
        '|| printf %s "${DEFAULT_PROFILE}")" >/dev/null 2>&1 || exit 1; '
        'printf "%s %s" "${PARALLEL}" "${CTX_PER_SLOT}"'
    )
    try:
        out = subprocess.run(["bash", "-c", script, "bash", ROOT],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None, None
    parts = out.stdout.split()
    return (parts[0], parts[1]) if len(parts) == 2 else (None, None)


def get(url: str, timeout: int) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def post_tolerant(url: str, payload: dict, timeout: int) -> tuple[dict | None, str]:
    """POST that survives an HTTP error body.

    llama.cpp's PEG tool-call parser THROWS on output it cannot parse, which
    surfaces as a 500 rather than as degraded content. That is a result worth
    reporting, not a reason to abort the run.
    """
    try:
        return post(url, payload, timeout), ""
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode())
            message = detail.get("error", {}).get("message") or str(detail)
        except Exception:
            message = f"HTTP {exc.code}"
        return None, message
    except (urllib.error.URLError, OSError) as exc:
        return None, str(exc)


def tiny_png(width: int = 96, height: int = 96) -> bytes:
    """A red field with a blue square, built with zlib and struct alone.

    Two flat colours because the assertion has to be robust: the point is that
    the projector ran at all, not that the model is good at vision.
    """
    rows = []
    for y in range(height):
        row = bytearray([0])  # PNG filter type 0
        for x in range(width):
            inside = width // 4 <= x < 3 * width // 4 and height // 4 <= y < 3 * height // 4
            row += bytes((0, 0, 255) if inside else (255, 0, 0))
        rows.append(bytes(row))
    raw = zlib.compress(b"".join(rows), 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    header = struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", raw) + chunk(b"IEND", b""))


def tokenize_len(base: str, text: str, timeout: int) -> int:
    return len(post(f"{base}/tokenize", {"content": text}, timeout)["tokens"])


def build_needle_prompt(base: str, depth: int, timeout: int) -> tuple[str, int]:
    """Filler of ~depth tokens with the needle at the midpoint.

    Depth is reached with real tokens and measured with the server's own
    tokenizer, then binary-searched to the character cut -- an approximate
    depth would make a non-monotonic cliff impossible to pin down.
    """
    per_block = max(1, tokenize_len(base, FILLER, timeout))
    blocks = max(2, depth // per_block + 2)
    text = FILLER * blocks
    # Grow until the text definitely exceeds the target before searching.
    # `depth // per_block + 2` is NOT enough: BPE merges across the seam
    # between copies, so N copies tokenize to fewer than N * per_block tokens
    # and the shortfall grows with N. Measured before this loop existed, a
    # request for 196608 produced a 193613-token prompt -- a 1.5% undershoot
    # that grows with depth, which is fatal when the whole point is to pin down
    # a non-monotonic cliff to an exact token count.
    while tokenize_len(base, text, timeout) < depth:
        blocks *= 2
        text = FILLER * blocks
    low, high = 0, len(text)
    while low < high:
        mid = (low + high) // 2
        if tokenize_len(base, text[:mid], timeout) < depth:
            low = mid + 1
        else:
            high = mid
    body = text[:low]
    middle = len(body) // 2
    prompt = body[:middle] + "\n\n" + NEEDLE + "\n\n" + body[middle:]
    return prompt, tokenize_len(base, prompt, timeout)


def needle_probe(base: str, depth: int, timeout: int, verbose: bool = True) -> dict:
    """One needle test. Returns a verdict dict; never raises for a model fault."""
    prompt, actual = build_needle_prompt(base, depth, timeout)
    if verbose:
        print(f"  building prompt: requested {depth}, actual {actual} tokens")
    started = time.monotonic()
    body, error = post_tolerant(
        f"{base}/v1/chat/completions",
        {
            "messages": [
                {"role": "user",
                 "content": prompt + "\n\nWhat is the maintenance access code "
                                     "for the Harbourgate cold-storage annex? "
                                     "Reply with the code only."},
            ],
            "max_tokens": 2048,
            "temperature": 0.0,
            # Keep the trace short: this test is about retrieval, not depth of
            # thought, and xhigh would spend thousands of tokens first.
            "chat_template_kwargs": {"reasoning_effort": "medium"},
        },
        timeout,
    )
    elapsed = time.monotonic() - started
    if body is None:
        return {"depth": actual, "verdict": "error", "detail": error, "seconds": elapsed}

    choice = body["choices"][0]
    message = choice.get("message") or {}
    content = (message.get("content") or "").strip()
    reasoning = (message.get("reasoning_content") or "").strip()
    predicted = (body.get("usage") or {}).get("completion_tokens", 0)
    timings = body.get("timings") or {}

    result = {
        "depth": actual,
        "seconds": elapsed,
        "completion_tokens": predicted,
        "finish_reason": choice.get("finish_reason"),
        "prefill_tok_s": timings.get("prompt_per_second"),
        "decode_tok_s": timings.get("predicted_per_second"),
    }
    # This is the #27756 signature: clean prefill, then nothing at all.
    if predicted <= 1 and not content and not reasoning:
        result["verdict"] = "silent_eos"
    elif NEEDLE_ANSWER.lower() in content.lower():
        result["verdict"] = "ok"
    elif not content:
        result["verdict"] = "empty"
    else:
        result["verdict"] = "wrong"
        result["detail"] = content[:200]
    return result


def describe(result: dict) -> str:
    depth = result["depth"]
    verdict = result["verdict"]
    rate = result.get("prefill_tok_s")
    rate_text = f", prefill {rate:.0f} tok/s" if rate else ""
    base = f"{depth:>8} tokens in {result['seconds']:.1f}s{rate_text}: "
    if verdict == "ok":
        return base + "needle recalled"
    if verdict == "silent_eos":
        return base + ("SILENT EOS -- prefill completed, then EOS as the first "
                       "token. This is llama.cpp #27756.")
    if verdict == "empty":
        return base + f"empty content ({result['completion_tokens']} tokens generated)"
    if verdict == "error":
        return base + f"server error: {result.get('detail')}"
    return base + f"needle LOST, answered instead: {result.get('detail', '')!r}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8010")))
    parser.add_argument("--host", default=os.environ.get("SMOKE_HOST", "127.0.0.1"))
    # This model thinks by default and at xhigh effort it reasons at length, so
    # a small budget truncates mid-thought and reads as an empty-response bug.
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--needle-depth", type=int, default=0,
                        help="0 = the server's own per-slot context, -1 = skip")
    parser.add_argument("--needle-sweep", default="",
                        help="comma-separated depths to walk instead of one test")
    parser.add_argument("--quick", action="store_true",
                        help="skip the needle and vision tests")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    failures: list[str] = []
    warnings: list[str] = []

    try:
        models = get(f"{base}/v1/models", 30)["data"]
    except (urllib.error.URLError, OSError) as exc:
        print(f"cannot reach {base}: {exc}", file=sys.stderr)
        print("is the server up? ./status.sh", file=sys.stderr)
        return 1
    model_id = models[0]["id"]

    props = get(f"{base}/props", 30)
    settings = props.get("default_generation_settings") or {}
    slot_ctx = settings.get("n_ctx") or props.get("n_ctx")
    total_slots = props.get("total_slots")
    print(f"server      : {base}")
    print(f"model       : {model_id}")
    print(f"slots       : {total_slots}")
    print(f"ctx per slot: {slot_ctx}")

    # The geometry the active profile asked for.
    want_slots, want_slot_ctx = active_geometry()
    print("\n--- geometry ---")
    if not want_slots or not want_slot_ctx:
        warnings.append(
            "could not read the active profile, so the server's geometry was "
            "not compared against it -- pass PARALLEL and CTX_PER_SLOT to check")
    if want_slots and str(total_slots) != want_slots:
        failures.append(f"server has {total_slots} slots, profile asked for {want_slots}")
    else:
        print(f"  OK   {total_slots} slot(s)")
    if want_slot_ctx and str(slot_ctx) != want_slot_ctx:
        # llama.cpp reduces context to make a config fit and says so only in
        # its log, so a silently shrunken server would otherwise pass.
        failures.append(
            f"server gives {slot_ctx} tokens per slot, profile asked for "
            f"{want_slot_ctx} -- llama.cpp reduced it to make it fit")
    else:
        print(f"  OK   {slot_ctx} tokens per slot")

    # --- arithmetic: the Hadamard guard ------------------------------------
    print("\n--- arithmetic (Hadamard basis guard) ---")
    body, error = post_tolerant(f"{base}/v1/chat/completions", {
        "messages": [{"role": "user", "content": MATH_PROMPT}],
        "max_tokens": args.max_tokens,
    }, args.timeout)
    if body is None:
        failures.append(f"math request failed: {error}")
    else:
        message = body["choices"][0].get("message") or {}
        content = (message.get("content") or "").strip()
        reasoning = (message.get("reasoning_content") or "").strip()
        timings = body.get("timings") or {}
        print(f"  prefill {timings.get('prompt_n', 0)} tok at "
              f"{timings.get('prompt_per_second', 0):.1f} tok/s")
        print(f"  decode  {timings.get('predicted_n', 0)} tok at "
              f"{timings.get('predicted_per_second', 0):.1f} tok/s")

        if content:
            print("  OK   non-empty content")
        else:
            failures.append("empty content on the math prompt")
        if reasoning:
            print("  OK   reasoning_content populated (thinking active)")
        else:
            # The template opens <think> in the assistant prefix, so an empty
            # reasoning_content means the parser never matched the block.
            failures.append(
                "reasoning_content is empty -- thinking is on by default for "
                "this model, so --jinja or --reasoning-format is wrong")
        if "<think>" in content:
            failures.append("<think> leaked into content (reasoning not extracted)")
        else:
            print("  OK   no <think> leakage")

        if any(token in content for token in ("582", "0.582", "$582")):
            print("  OK   operating income ~$582M")
        else:
            failures.append(
                "operating income ~$582M is absent -- on these weights a wrong "
                "activation basis gives fluent nonsense, so suspect the build "
                "before suspecting the model")
        if any(token in content.lower()
               for token in ("net cash", "-400", "(400", "-$400", "400m net", "400 million net")):
            print("  OK   net debt resolves to a ~$400M net cash position")
        else:
            warnings.append("net debt was not clearly stated as a net cash position")

    # --- reasoning_effort ---------------------------------------------------
    print("\n--- reasoning_effort ---")
    body, error = post_tolerant(f"{base}/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Name the capital of France."}],
        "max_tokens": 2048,
        "chat_template_kwargs": {"reasoning_effort": "medium"},
    }, args.timeout)
    if body is None:
        failures.append(f"reasoning_effort=medium was rejected: {error}")
    else:
        message = body["choices"][0].get("message") or {}
        if "paris" in (message.get("content") or "").lower():
            print("  OK   reasoning_effort=medium accepted and answered")
        else:
            warnings.append("reasoning_effort=medium accepted but the answer looked wrong")

    # --- tool calling -------------------------------------------------------
    print("\n--- tool calling (Qwen3-Coder XML dialect) ---")
    body, error = post_tolerant(f"{base}/v1/chat/completions", {
        "messages": [{"role": "user",
                      "content": "What is the weather in Reykjavik? Use the tool."}],
        "max_tokens": 4096,
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string",
                                            "description": "City name"}},
                    "required": ["city"],
                },
            },
        }],
    }, args.timeout)
    if body is None:
        # llama.cpp's PEG parser throws rather than degrading, so a malformed
        # call arrives as an HTTP error.
        failures.append(f"tool-call request failed: {error}")
    else:
        message = body["choices"][0].get("message") or {}
        calls = message.get("tool_calls") or []
        if calls and calls[0].get("function", {}).get("name") == "get_weather":
            arguments = calls[0]["function"].get("arguments") or "{}"
            print(f"  OK   parsed tool_call: get_weather {arguments}")
            if "reykjavik" not in arguments.lower():
                warnings.append("tool call parsed but the city argument looked wrong")
        else:
            failures.append(
                "no structured tool_calls -- the template's XML dialect should "
                "be picked up by llama.cpp's Qwen3-Coder parser under --jinja")

    # --- vision -------------------------------------------------------------
    if args.quick:
        print("\n--- vision --- skipped (--quick)")
    else:
        print("\n--- vision (mmproj) ---")
        data_uri = "data:image/png;base64," + base64.b64encode(tiny_png()).decode()
        body, error = post_tolerant(f"{base}/v1/chat/completions", {
            "messages": [{"role": "user", "content": [
                {"type": "text",
                 "text": "Name the two colours in this image. Two words."},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]}],
            "max_tokens": 2048,
            "chat_template_kwargs": {"reasoning_effort": "medium"},
        }, args.timeout)
        if body is None:
            failures.append(f"vision request failed (is --mmproj loaded?): {error}")
        else:
            answer = (body["choices"][0].get("message", {}).get("content") or "").lower()
            if "red" in answer and "blue" in answer:
                print("  OK   projector ran and both colours were named")
            elif "red" in answer or "blue" in answer:
                warnings.append(f"projector ran but only one colour was named: {answer[:80]!r}")
            else:
                failures.append(f"image was not described: {answer[:120]!r}")

    # --- needle at depth: the #27756 test -----------------------------------
    if args.quick and not args.needle_sweep:
        print("\n--- needle at depth --- skipped (--quick)")
    elif args.needle_sweep:
        depths = [int(d) for d in args.needle_sweep.split(",") if d.strip()]
        print(f"\n--- needle sweep ({len(depths)} depths) ---")
        print("  a failure ABOVE a pass is expected: #27756 is non-monotonic\n")
        results = []
        for depth in depths:
            if slot_ctx and depth > slot_ctx:
                print(f"{depth:>8} tokens: skipped (server per-slot ctx is {slot_ctx})")
                continue
            result = needle_probe(base, depth, args.timeout, verbose=False)
            results.append(result)
            print("  " + describe(result))
        bad = [r for r in results if r["verdict"] != "ok"]
        if bad:
            worst = min(r["depth"] for r in bad)
            failures.append(
                f"needle sweep: {len(bad)} of {len(results)} depths did not answer "
                f"correctly, shallowest at {worst} tokens")
    elif args.needle_depth == -1:
        print("\n--- needle at depth --- skipped (--needle-depth -1)")
    else:
        depth = args.needle_depth
        if depth == 0:
            # Test what the server actually advertises, with a small margin for
            # the question and the answer.
            depth = max(4096, (slot_ctx or 8192) - 4096)
        print(f"\n--- needle at depth {depth} ---")
        if depth > 65536:
            print("  (a deep prefill takes minutes; --needle-depth -1 skips it)")
        result = needle_probe(base, depth, args.timeout)
        print("  " + describe(result))
        if result["verdict"] != "ok":
            failures.append(
                f"needle test failed at {result['depth']} tokens "
                f"({result['verdict']}) -- this context depth is not usable")

    # --- verdict ------------------------------------------------------------
    print("\n--- result ---")
    for warning in warnings:
        print(f"  WARN  {warning}")
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("  smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
