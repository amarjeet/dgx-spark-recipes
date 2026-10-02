# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
"""How far FP8 weights move Clef-Flash's answers from BF16. The gate for WEIGHTS=fp8.

Loads the model in BF16 exactly as the server does, answers a fixed set of
records, converts the decoder to FP8 in place (server/fp8.py), answers them
again, and compares question by question:

  * winners -- the chosen option (choice), the most likely level (score),
    or p(true) > 0.5 (noul): the decision a client acts on
  * probability drift -- the largest absolute change of any option's
    probability, and of a score's expected value

The set is built here rather than downloaded, so the gate runs offline and
needs nothing beyond the venv: ticket routing, graded sentiment, NLI-style
true/false, invoice JSON with numeric thresholds, multi-question schemas, a
long state and three images.

Needs the server stopped -- it loads its own copy of the model. drift.sh
checks that, and runs it under the watchdog.
Usage: ./drift.sh
"""

from __future__ import annotations

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("WARMUP", "0")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

import app  # noqa: E402
import fp8  # noqa: E402

jsm = app.jsm

TEAMS = {"billing": "Payments, refunds or invoices", "technical": "Bugs, errors or outages",
         "sales": "Pricing, quotes or new contracts", "account": "Login, profile or access management"}
TICKETS = [
    "I was charged twice for my subscription this month.", "Our API calls return HTTP 500 since this morning.",
    "Can we get volume pricing for 500 seats?", "I can't reset my password, the email never arrives.",
    "Please send a corrected invoice with our new VAT number.", "The dashboard is blank in Safari.",
    "We'd like to upgrade to the enterprise plan next quarter.", "How do I add a second admin to our workspace?",
    "Refund the annual plan, we cancelled within 14 days.", "Webhooks stopped firing after your release.",
    "Is there a discount for nonprofits?", "My account got locked after too many attempts.",
    "The export to CSV truncates rows over 10,000.", "Our card expired, how do we update payment details?",
    "Need a quote for on-prem deployment.", "SSO login loops back to the sign-in page.",
]
REVIEWS = [
    "Absolutely terrible, broke after a day and support ignored me.", "It's fine. Does the job, nothing special.",
    "Best purchase I've made this year, works flawlessly.", "Mostly good, but the battery life is disappointing.",
    "Arrived damaged and the replacement was also broken.", "Exceeded my expectations in every way.",
    "Not great, not awful. Wouldn't buy again though.", "Love it! Setup took two minutes.",
    "The app crashes constantly, unusable.", "Solid build quality, a bit overpriced.",
    "Meh. Returned it.", "Five stars, would recommend to anyone.",
]
NLI = [
    ("The meeting was moved from Monday to Wednesday.", "The meeting happens on Monday."),
    ("All three servers in the EU region failed health checks.", "At least one EU server is unhealthy."),
    ("She has lived in Lisbon since 2015.", "She lived in Lisbon in 2018."),
    ("The contract renews automatically unless cancelled 30 days before expiry.", "The contract never renews."),
    ("Revenue grew 12% year over year.", "Revenue declined compared to last year."),
    ("The package was delivered to the front desk at 3pm.", "The package was delivered."),
    ("No employees were injured in the incident.", "Several employees were hurt."),
    ("The library opens at 9 and closes at 17 on weekdays.", "The library is open at noon on Tuesday."),
    ("The patch fixes the memory leak but not the crash on startup.", "The startup crash is fixed."),
    ("Tickets cost 20 euros for adults and are free for children.", "Children need to pay for tickets."),
    ("The vote passed with 61 in favour and 39 against.", "Most voters supported the measure."),
    ("The train was cancelled due to a strike.", "The train ran on time."),
]
INVOICES = [
    {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue", "days_past_due": 12},
    {"vendor": "Globex", "total": 80.5, "currency": "USD", "status": "paid", "paid_on": "2026-09-01"},
    {"vendor": "Initech", "total": 4300.0, "currency": "EUR", "status": "draft"},
    {"vendor": "Umbrella", "total": 999.99, "currency": "USD", "status": "sent", "due": "2026-10-30"},
    {"vendor": "Hooli", "total": 15000.0, "currency": "USD", "status": "overdue", "days_past_due": 45},
    {"vendor": "Stark", "total": 300.0, "currency": "GBP", "status": "paid"},
    {"vendor": "Wayne", "total": 1001.0, "currency": "USD", "status": "sent"},
    {"vendor": "Tyrell", "total": 50.0, "currency": "USD", "status": "draft"},
]
LONG = ("Incident log. " + " ".join(
    f"At 10:{i:02d} the checkout service reported latency of {200 + i * 7} ms and {i % 5} failed payments."
    for i in range(60)))


def records() -> list[dict]:
    out = []
    for t in TICKETS:
        out.append({"state": t, "questions": {
            "team": {"type": "choice", "instructions": "Which team should handle this message?", "criteria": TEAMS},
            "urgent": {"type": "noul", "instructions": "Does this need attention today?"}}})
    for r in REVIEWS:
        out.append({"state": {"review": r}, "questions": {
            "sentiment": {"type": "score", "instructions": "How positive is the review?",
                          "criteria": ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]}}})
    for premise, hypothesis in NLI:
        out.append({"state": {"premise": premise, "hypothesis": hypothesis}, "questions": {
            "entailed": {"type": "noul", "instructions": "Does the premise entail the hypothesis?"}}})
    for inv in INVOICES:
        out.append({"state": {"invoice": inv}, "questions": {
            "status": {"type": "choice", "instructions": "What is the invoice status?",
                       "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.",
                                    "draft": "Not sent.", "sent": "Sent, not yet due."}},
            "large": {"type": "noul", "instructions": "Is the total above 1000 in its currency?"},
            "priority": {"type": "score", "criteria": ["Ignore", "Review this week", "Act today"]}}})
    out.append({"state": LONG, "questions": {
        "severity": {"type": "score", "criteria": ["Minor", "Moderate", "Severe"]},
        "payments_affected": {"type": "noul", "instructions": "Were any payments affected?"},
        "owner": {"type": "choice", "criteria": TEAMS}}})
    for colour, rgb in (("red", (220, 20, 20)), ("green", (20, 200, 40)), ("blue", (30, 40, 220))):
        out.append({"state": "Look at the attached image.", "images": [Image.new("RGB", (64, 64), rgb)],
                    "questions": {"colour": {"type": "choice", "instructions": "What colour fills the image?",
                                             "criteria": {"red": "Red", "green": "Green", "blue": "Blue"}}}})
    return out


def answer(model, processor, recs: list[dict]) -> list[dict]:
    results = []
    for rec in recs:
        encoded = jsm.encode_record(processor.tokenizer, rec, processor=processor)
        batch = jsm.collate_records([encoded], processor.tokenizer.pad_token_id, torch.device("cuda"))
        with torch.inference_mode():
            logits = model(batch)[0]
        results.append({q.question_id: (rec["questions"][q.question_id]["type"],
                                         dict(zip(q.option_ids, l.float().softmax(-1).tolist())))
                        for q, l in zip(encoded.questions, logits)})
    return results


def winner(kind: str, probs: dict) -> str:
    if kind == "noul":
        return "true" if probs["true"] > 0.5 else "false"
    return max(probs, key=probs.get)


def expected(probs: dict) -> float:
    return sum(int(k) * v for k, v in probs.items())


def main() -> int:
    recs = records()
    snapshot = os.environ["CLEF_SNAPSHOT"]
    model, processor = app.load_clef(snapshot)
    started = time.perf_counter()
    bf16 = answer(model, processor, recs)
    bf16_s = time.perf_counter() - started
    bf16_bytes = torch.cuda.memory_allocated()

    converted, saved = fp8.quantize_decoder(model.language_model)
    fp8_bytes = torch.cuda.memory_allocated()
    answer(model, processor, recs[:2])                    # first FP8 calls JIT nothing, but be fair
    started = time.perf_counter()
    quant = answer(model, processor, recs)
    fp8_s = time.perf_counter() - started

    flips, drifts, score_drifts, n = [], [], [], 0
    for i, (a, b) in enumerate(zip(bf16, quant)):
        for qid, (kind, pa) in a.items():
            pb = b[qid][1]
            n += 1
            drifts.append(max(abs(pa[k] - pb[k]) for k in pa))
            if kind == "score":
                score_drifts.append(abs(expected(pa) - expected(pb)))
            if winner(kind, pa) != winner(kind, pb):
                flips.append((i, qid, kind, winner(kind, pa), round(max(pa.values()), 3),
                               winner(kind, pb), round(max(pb.values()), 3)))

    print(f"records {len(recs)}, questions {n}")
    print(f"decoder linears converted  {converted}; weight bytes given back {saved / 2**30:.2f} GiB")
    print(f"GPU allocated              BF16 {bf16_bytes / 2**30:.2f} GiB -> FP8 {fp8_bytes / 2**30:.2f} GiB")
    print(f"time for the set           BF16 {bf16_s:.2f} s -> FP8 {fp8_s:.2f} s")
    print(f"same winner                {n - len(flips)}/{n}")
    print(f"max |dp| per question      median {statistics.median(drifts):.4f}, p95 "
          f"{sorted(drifts)[int(0.95 * (len(drifts) - 1))]:.4f}, max {max(drifts):.4f}")
    if score_drifts:
        print(f"score expectation drift    max {max(score_drifts):.4f} (levels are 1 apart)")
    for f in flips:
        print(f"  FLIP record {f[0]} {f[1]} ({f[2]}): BF16 {f[3]} @ {f[4]} -> FP8 {f[5]} @ {f[6]}")
    # Gate: no decision changes except where BF16 itself was a coin toss.
    hard = [f for f in flips if f[4] >= 0.6]
    print("GATE", "PASS" if not hard else f"FAIL: {len(hard)} confident decision(s) changed")
    return 0 if not hard else 1


if __name__ == "__main__":
    sys.exit(main())
