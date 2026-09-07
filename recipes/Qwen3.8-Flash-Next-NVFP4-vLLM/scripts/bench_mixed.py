#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
# Copyright (C) 2026 amarjeet
#
# Derived from bench/mixed.py in MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark,
# which is Copyright (C) 2026 MiaAI Lab and licensed AGPL-3.0-or-later.
# Modified 2026-09-07 by amarjeet: port and port only -- the measurement design
# is upstream's. Parameterised on --port / --model / --out instead of hardcoded
# localhost:8888; the prompt and the big context are salted with a nonce so
# repeat runs are not served out of the prefix cache; and per-chunk figures are
# reported as ENGINE STEPS with token counts taken from the engine's own usage
# block, because with MTP one streamed chunk carries several accepted tokens.
"""Mixed traffic: what a long prefill does to streams that are already decoding.

Decode-only and prefill-only ladders both miss the case an agent harness
actually produces -- a 64k-token prompt arriving while earlier turns are still
streaming. Chunked prefill puts a MAX_NUM_BATCHED_TOKENS-wide chunk in the same
engine step as every co-scheduled decode, so a decoder's inter-token latency
during a prefill is set by the chunk width, not by the decode batch.

    N prose streams start decoding (ignore_eos, temperature 0, thinking off).
    At --inject-after seconds one large prompt is submitted.
    The streams' inter-chunk gap is reported for the prefill window (injection
    to the big prompt's first token) and, as the control, for the quiet window
    before it.

WHAT THE NUMBERS MEAN. vLLM emits one streamed chunk per engine step, and with
MTP that chunk carries every token the step accepted (~2.8 of 4 on prose here).
So the gap between chunks is a per-STEP latency, and per-token latency is
roughly that divided by the accepted-tokens-per-step this prints alongside it.
Upstream's first published version of this table labelled the gaps per token;
that is the one thing the port deliberately does not reproduce.

Requires an idle server: any other traffic lands in the same engine steps.

Usage:
  ./scripts/bench_mixed.py --port 8888 --out mixed.json
  ./scripts/bench_mixed.py --decoders 2 --context 64000 --chunk-note 2048
"""
import argparse
import json
import os
import secrets
import statistics
import sys
import threading
import time
import urllib.request

# ~25 tokens per entry, so a context target converts to a line count.
FILLER = ("Entry {i:06d}: the quarterly logistics audit recorded a routine "
          "variance in the northbound depot inventory.\n")
PROSE = ("Write a flowing, continuous essay about the history of maritime "
         "navigation. Use ordinary narrative prose, no lists, no headings.")
BIG_TASK = "In one short sentence, what kind of document is the log above?"


def build_ctx(target_tokens: int, nonce: str) -> str:
    """A ~target_tokens context, salted so it is not a prefix-cache hit.

    The nonce leads, which is what makes the big request a real prefill: the
    server runs with prefix caching on, and without it a repeat run reuses the
    first run's KV and reports a TTFT that has nothing to do with prefill.
    """
    body = "".join(FILLER.format(i=i)
                   for i in range(max(1, int(target_tokens / 25))))
    return f"Reference {nonce}.\n" + body


class Decoder(threading.Thread):
    """One prose stream. Records when each chunk arrived, and the engine's own
    token count for the whole completion."""

    def __init__(self, idx: int, max_tokens: int, t_origin: float,
                 base: str, model: str, nonce: str):
        super().__init__(daemon=True)
        self.idx, self.max_tokens, self.t_origin = idx, max_tokens, t_origin
        self.base, self.model, self.nonce = base, model, nonce
        self.marks: list[float] = []
        self.completion_tokens = 0
        self.error: str | None = None

    def run(self) -> None:
        payload = {
            "model": self.model,
            "messages": [{"role": "user",
                          "content": f"({self.nonce}-{self.idx}) " + PROSE}],
            "max_tokens": self.max_tokens, "min_tokens": self.max_tokens,
            "ignore_eos": True, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream_options": {"include_usage": True},
        }
        try:
            for kind, value in sse(self.base, payload):
                if kind == "chunk":
                    self.marks.append(time.time() - self.t_origin)
                elif kind == "usage":
                    self.completion_tokens = value.get("completion_tokens", 0)
        except Exception as exc:                       # noqa: BLE001
            self.error = repr(exc)


class BigPrompt(threading.Thread):
    """The long prompt injected into the middle of those streams."""

    def __init__(self, ctx: str, t_origin: float, base: str, model: str):
        super().__init__(daemon=True)
        self.ctx, self.t_origin = ctx, t_origin
        self.base, self.model = base, model
        self.submitted: float | None = None
        self.ttft: float | None = None
        self.prompt_tokens = 0
        self.error: str | None = None

    def run(self) -> None:
        payload = {
            "model": self.model,
            "messages": [{"role": "user",
                          "content": self.ctx + "\n\n" + BIG_TASK}],
            "max_tokens": 16, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream_options": {"include_usage": True},
        }
        self.submitted = time.time() - self.t_origin
        try:
            for kind, value in sse(self.base, payload):
                if kind == "usage":
                    self.prompt_tokens = value.get("prompt_tokens", 0)
                elif kind == "chunk" and self.ttft is None:
                    self.ttft = time.time() - self.t_origin - self.submitted
        except Exception as exc:                       # noqa: BLE001
            self.error = repr(exc)


def sse(base: str, payload: dict, timeout: int = 1800):
    """Yield ("chunk", None) per streamed chunk carrying text, ("usage", dict)
    for the final usage block."""
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                return
            obj = json.loads(body)
            if obj.get("usage"):
                yield "usage", obj["usage"]
            for choice in obj.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning"):
                    yield "chunk", None


def gap_stats(marks: list[float], lo: float, hi: float) -> dict | None:
    """Gaps between consecutive chunk arrivals whose start falls in [lo, hi)."""
    gaps = [(b - a) * 1000 for a, b in zip(marks, marks[1:]) if lo <= a < hi]
    if not gaps:
        return None
    ordered = sorted(gaps)

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))
        return ordered[idx]

    return {"n": len(gaps), "mean_ms": round(statistics.fmean(gaps), 1),
            "p50_ms": round(pct(50), 1), "p95_ms": round(pct(95), 1),
            "p99_ms": round(pct(99), 1), "max_ms": round(max(gaps), 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8888)))
    ap.add_argument("--host", default=os.environ.get("SMOKE_HOST", "127.0.0.1"))
    ap.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME",
                                                      "qwen3.8-flash-next"))
    ap.add_argument("--decoders", type=int, default=2)
    ap.add_argument("--decode-tokens", type=int, default=800)
    ap.add_argument("--context", type=int, default=64000,
                    help="approximate prompt length of the injected request")
    ap.add_argument("--inject-after", type=float, default=5.0)
    ap.add_argument("--out", default=os.environ.get("BENCH_OUT", ""))
    ap.add_argument("--tag", default="mixed")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    nonce = secrets.token_hex(8)
    ctx = build_ctx(args.context, nonce)

    t0 = time.time()
    decoders = [Decoder(i, args.decode_tokens, t0, base, args.model, nonce)
                for i in range(args.decoders)]
    for d in decoders:
        d.start()
    time.sleep(args.inject_after)
    big = BigPrompt(ctx, t0, base, args.model)
    big.start()
    big.join(timeout=1800)
    for d in decoders:
        d.join(timeout=1800)

    if big.ttft is None:
        print(f"big prompt produced no token (error={big.error})",
              file=sys.stderr)
    win_lo = big.submitted or 0.0
    win_hi = win_lo + (big.ttft or 0.0)
    window_s = max(win_hi - win_lo, 1e-9)

    quiet = [gap_stats(d.marks, 0.0, win_lo) for d in decoders]
    during = [gap_stats(d.marks, win_lo, win_hi) for d in decoders]
    after = [gap_stats(d.marks, win_hi, 1e9) for d in decoders]

    steps_in_window = sum(sum(1 for m in d.marks if win_lo <= m < win_hi)
                          for d in decoders)
    # Accepted tokens per streamed chunk, from the engine's own counts. This is
    # what converts every per-step figure above into a per-token one.
    total_chunks = sum(len(d.marks) for d in decoders)
    total_tokens = sum(d.completion_tokens for d in decoders)
    tok_per_step = (total_tokens / total_chunks) if total_chunks else None

    row = {
        "kind": "mixed", "tag": args.tag, "note": args.note,
        "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model, "decoders": args.decoders,
        "decode_tokens": args.decode_tokens,
        "big_prompt_tokens": big.prompt_tokens,
        "big_ttft_s": round(big.ttft, 2) if big.ttft else None,
        "inject_at_s": round(win_lo, 2),
        "prefill_window_s": round(window_s, 2),
        "steps_in_window": steps_in_window,
        "steps_per_s_in_window": round(steps_in_window / window_s, 2),
        "tokens_per_step": round(tok_per_step, 2) if tok_per_step else None,
        # Steps actually delivered in the window x the run's measured
        # acceptance. An estimate, and labelled as one: the engine reports
        # usage per request, not per window.
        "est_decode_tps_in_window": (
            round(steps_in_window * tok_per_step / window_s, 2)
            if tok_per_step else None),
        "gap_quiet_per_step": quiet,
        "gap_during_prefill_per_step": during,
        "gap_after_per_step": after,
        "errors": [e for e in ([d.error for d in decoders] + [big.error]) if e],
        "wall_s": round(time.time() - t0, 1),
    }

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as handle:
            handle.write(json.dumps(row) + "\n")

    print(f"[{args.tag}] big prompt {big.prompt_tokens:,} tok, "
          f"TTFT {row['big_ttft_s']} s (injected at {row['inject_at_s']} s)")
    print(f"        acceptance: {row['tokens_per_step']} tokens per engine step "
          f"({total_tokens:,} tokens in {total_chunks:,} chunks)")
    print(f"        during that window: {row['steps_per_s_in_window']} steps/s "
          f"across {args.decoders} streams "
          f"(~{row['est_decode_tps_in_window']} tok/s)")
    for label, stats in (("quiet  ", quiet), ("prefill", during),
                         ("after  ", after)):
        for i, st in enumerate(stats):
            print(f"        gap/step {label} stream {i}: {st}")
    if row["errors"]:
        print(f"        errors: {row['errors']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
