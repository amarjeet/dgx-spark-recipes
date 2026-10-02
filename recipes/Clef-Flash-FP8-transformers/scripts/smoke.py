#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
"""Smoke test against the running server. Standard library only.

Every check has an answer known in advance, because a broken head or a
wrong-dtype backbone still returns well-formed probabilities: shape proves
nothing, so the checks assert which option wins.

  1. /health and /v1/models answer with the served id
  2. the model card's invoice example: status "overdue", total above 1000
  3. the model card's SystemOne example: "technical", an outage, urgent
  4. an image: a solid red PNG, built here, is judged red
  5. a malformed request is a 400, not a 500
  6. eight concurrent requests each get their own answer back

Usage: scripts/smoke.py      (PORT, SMOKE_HOST)
Exit code 1 on any failure.
"""
import base64
import json
from concurrent.futures import ThreadPoolExecutor
import os
import struct
import sys
import urllib.error
import urllib.request
import zlib
from typing import Any

HOST = os.environ.get("SMOKE_HOST", "127.0.0.1")
PORT = os.environ.get("PORT", "8012")
BASE = f"http://{HOST}:{PORT}"
failures = 0


def call(path: str, body: dict | None = None) -> tuple[dict, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp), resp.headers  # case-insensitive; the server sends lowercase names


def check(name: str, ok: bool, detail: str) -> None:
    global failures
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {detail}")
    failures += not ok


def solid_png(rgb: tuple[int, int, int], size: int = 64) -> str:
    """A size x size PNG of one colour, base64-encoded, with no imaging library."""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    row = b"\x00" + bytes(rgb) * size
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(row * size))
           + chunk(b"IEND", b""))
    return base64.b64encode(png).decode()


print(f"smoke test against {BASE}")

health, _ = call("/health")
models, _ = call("/v1/models")
model_id = models["data"][0]["id"]
check("health", health.get("status") == "ok", f"{health['model']} max_length {health['max_length']}")
check("models", model_id == health["model"], model_id)

# 2. The model card's first example, through the HTTP API rather than Python.
r, headers = call("/v1/systemone", {
    "model": model_id,
    "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
    "questions": {
        "status": {"type": "choice", "instructions": "What is the invoice status?",
                   "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."}},
        "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
    },
})
a = r["answers"]
check("invoice status", a["status"]["choice"] == "overdue",
      f"{a['status']['choice']} ({a['status']['confidence']}) in {headers.get('X-Clef-Forward-Ms', '?')} ms")
check("invoice > 1000", a["large"]["noul"] > 0.5, f"p(true) = {a['large']['noul']}")
check("usage", r["usage"]["output_tokens"] == 0 and r["usage"]["input_tokens"] > 0, json.dumps(r["usage"]))

# 3. The model card's SystemOne example.
r, headers = call("/v1/systemone", {
    "model": model_id,
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle the message?",
                       "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
})
a = r["answers"]
check("department", a["department"]["choice"] == "technical", f"{a['department']['choice']} ({a['department']['confidence']})")
check("outage", a["outage"]["noul"] > 0.5, f"p(true) = {a['outage']['noul']}")
check("urgency", a["urgency"]["score"] > 1.0, f"expected score {a['urgency']['score']} of 0..2")

# 4. Vision: the image path runs the encoder, and the answer depends on it.
r, headers = call("/v1/systemone", {
    "model": model_id,
    "state": "Look at the attached image.",
    "images": [solid_png((220, 20, 20))],
    "questions": {"colour": {"type": "choice", "instructions": "What colour fills the image?",
                             "criteria": {"red": "Red", "green": "Green", "blue": "Blue"}}},
})
c = r["answers"]["colour"]
check("image", c["choice"] == "red", f"{c['choice']} ({c['confidence']}) in {headers.get('X-Clef-Forward-Ms', '?')} ms")

# 5. Validation is the model code's own; it must surface as a client error.
try:
    call("/v1/systemone", {"model": model_id, "state": "x", "questions": {"q": {"type": "maybe"}}})
    check("bad request", False, "accepted a question of type 'maybe'")
except urllib.error.HTTPError as exc:
    check("bad request", exc.code == 400, f"HTTP {exc.code}")

# 6. Concurrent clients. Distinct states, so a response crossed between
# requests would show up as a different answer than the same request alone.
tickets = [
    "Refund my duplicate charge from last month.", "The API returns 500 on every call since 9am.",
    "Can I get a quote for 200 seats?", "Invoice INV-2231 shows the wrong VAT number.",
    "Login page is down for all users in Europe.", "What discount do you give nonprofits?",
    "My card was charged twice for the same order.", "Webhooks stopped firing after your deploy.",
]
def ask(text: str) -> tuple[dict, Any]:
    return call("/v1/systemone", {"model": model_id, "state": text, "questions": {
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages", "sales": "Pricing or new contracts"}},
        "outage": {"type": "noul", "instructions": "Is a service down?"}}})
alone = [ask(t)[0]["answers"] for t in tickets]
with ThreadPoolExecutor(len(tickets)) as pool:
    together = list(pool.map(ask, tickets))
drift = max(max(abs(a["team"]["probabilities"][k] - b[0]["answers"]["team"]["probabilities"][k]) for k in a["team"]["probabilities"])
            for a, b in zip(alone, together))
same = all(a["team"]["choice"] == b[0]["answers"]["team"]["choice"] for a, b in zip(alone, together))
check("concurrent", same and drift < 0.02, f"same winners as alone: {same}, max prob drift {drift:.4f}")
print("  teams: " + ", ".join(a["team"]["choice"] for a in alone))

print("PASS" if not failures else f"{failures} FAILED")
sys.exit(1 if failures else 0)
