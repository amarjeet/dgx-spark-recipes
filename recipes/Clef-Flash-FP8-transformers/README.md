# Clef-Flash on transformers (DGX Spark / GB10), beside a running server

Serves [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash), a 9B multimodal **decision model**,
on a single GB10, natively, behind its own SystemOne API on port 8012. It's built to run **next to** the
[`Qwen3.8-Flash-Next-MLX4-TensorFold`](../Qwen3.8-Flash-Next-MLX4-TensorFold/) server on the same pool, not instead
of it. That's the opposite of every other recipe here.

| | |
|---|---|
| Model | Qwen3.5-9B backbone (with its vision encoder) + a joint schema head. Text, JSON, images or video in; one probability per allowed option of every question out, in one forward pass. **It generates no text.** |
| Checkpoint | [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash) @ `17f0b0ad64ef`, BF16, 17 files, 17.77 GiB, every one checksummed in `manifests/bf16.json` |
| Served as | **FP8** (e4m3) decoder linear layers, converted at load; everything else BF16. `WEIGHTS=bf16` serves the checkpoint as released |
| Runtime | transformers 5.10.2 + torch 2.11 (cu130) in a uv project, venv at `~/venvs/clef-flash`; FastAPI/uvicorn; flash-linear-attention for the GDN layers |
| API | `POST /v1/systemone`, the body that the model's own `systemone()` defines. No OpenAI chat endpoint, because there is nothing to chat with |
| Footprint | 18.7 GiB of `MemAvailable` measured (11.32 GiB weights on the GPU) |
| Latency | 100 ms at ~330 input tokens, 9.3 req/s, 5.1 s at 13.9k tokens (GB10, the co-tenant idle) |
| Licence | Model: Apache-2.0. This recipe: MIT. The model's `joint_schema_model.py` is imported from the verified snapshot, not vendored |

## Quick start

```bash
cd recipes/Clef-Flash-FP8-transformers

./setup.sh            # uv sync --locked into ~/venvs/clef-flash, then proves torch drives this GPU
./download.sh         # 17.77 GiB into the HF cache, pinned revision, checksum-verified, resumable
./preflight.sh        # venv, snapshot, load-peak and steady-state memory, co-tenants, port
./start.sh            # serves on :8012; load + FP8 conversion + a 16k-token warmup, ~30 s
./scripts/smoke.py    # answers with known winners, an image, a 400, concurrent clients
./status.sh
./stop.sh
```

**Beside Qwen, start Qwen first, on its one-stream profile:**

```bash
../Qwen3.8-Flash-Next-MLX4-TensorFold/start.sh restart int8x1   # 1 x 262,144; leaves ~33.7 GiB
./start.sh                                                      # leaves ~14.9 GiB
```

The order matters. TensorFold reads `MemAvailable` once when it starts and holds back a tenth of RAM from that, so
with Clef already loaded, even one Qwen stream would barely be admitted.

## Why not `vllm serve`

The model card lists `vllm serve` and SGLang. Those snippets are Hugging Face's generic widget for a
`Qwen3_5ForConditionalGeneration` repo. They would load the backbone as a chat model and **drop the joint schema
head**, which is the part that produces the decisions. The card itself was tested only with transformers on an H200.
So this recipe runs the model's own code (`joint_schema_model.py` from the snapshot) and exposes the request and
response body that code defines.

## API

```bash
curl -s localhost:8012/v1/systemone -H 'content-type: application/json' -d '{
  "model": "clef-flash",
  "state": "Our checkout started returning errors and orders are blocked.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle the message?",
                   "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
    "urgency":    {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
    "outage":     {"type": "noul", "instructions": "Is a service down?"}
  }
}'
```

```json
{"model": "clef-flash",
 "answers": {"department": {"type": "choice", "choice": "technical", "confidence": 0.9552,
                            "probabilities": {"billing": 0.0448, "technical": 0.9552}},
             "urgency": {"type": "score", "score": 1.7968, "confidence": 0.8663,
                         "legend": {"0": "Can wait", "1": "This week", "2": "Today"},
                         "probabilities": {"0": 0.0695, "1": 0.0642, "2": 0.8663}},
             "outage": {"type": "noul", "noul": 0.8347}},
 "usage": {"input_tokens": 300, "output_tokens": 0}}
```

- `state` is any string or JSON value. `type` is `noul` (true/false), `choice` (named options) or `score` (ordered
  options). See the [model card](https://huggingface.co/Cloudflare/clef-flash#input-format).
- **Images:** `"images": ["<base64>", "data:image/png;base64,..."]`. **Video:** `"videos": [["<frame>", ...]]`,
  a list of base64 frames. Nothing is fetched by URL, so a request can't make the server reach out anywhere.
- Limits: `MAX_LENGTH` tokens of input (longer states are truncated by the model's own encoder), `MAX_IMAGES`,
  `MAX_VIDEOS`, `MAX_BODY_MIB`. A malformed request is a 400.
- `GET /health` reports readiness, the weight mode and the memory the process holds. `GET /v1/models` lists the id.
- The time of the forward pass is in the `X-Clef-Forward-Ms` response header, so the body stays exactly the
  SystemOne shape.

Requests run one at a time. Token-budgeted batching was built and measured here, and it was removed. Every request
carries a ~250-token system prompt and schema, and from there GB10 is compute-bound, so a batch costs the sum of its
requests: 7.1–7.8 req/s against 7.4 run one at a time, with a worse p95.

## Weights: why FP8, and what it changed

**BF16 doesn't fit beside Qwen on `int8x1` reliably.** Measured: BF16 Clef costs 25 GiB of the pool and leaves
8.3 GiB. Both recipes' watchdogs treat `MemFree` under 2 GiB as danger once `MemAvailable` is under 10 GiB (the
NVIDIA driver refuses allocations when free pages run out). At 8.3 GiB that condition is armed permanently, and the
first 50k-token Qwen prefill (whose SSD reads churn the page cache) had Clef's watchdog kill Clef. Qwen was never
touched, which is the design, but a model that dies whenever its neighbour works isn't a service.

So the decoder's 248 linear layers are converted to FP8 right after the BF16 load (`server/fp8.py`):

- **Scheme.** e4m3 weights with one scale per output row; activations quantized per token on every call;
  `torch._scaled_mm` with a BF16 result. No new dependency.
- **Not quantized:** token embeddings and `lm_head` (the head scores options by reading `lm_head`'s weight
  directly), the vision tower, and the joint schema head.
- **Chunked at 4,096 rows per call.** Row-wise `_scaled_mm` on GB10 falls off a cliff above ~4k rows (35.9 ms at
  16k tokens against BF16's 17.4). Each row has its own scale, so splitting by rows gives exactly the same numbers
  and keeps every length at least as fast as BF16. Per-tensor scaling would be faster still, but it lets one outlier
  token set the scale for all, so it wasn't adopted.

**The gate:** `./drift.sh` (`server/fp8_drift.py`) answers 52 records (86 questions: ticket routing, graded sentiment,
NLI-style true/false, invoice JSON, multi-question schemas, a long incident log, three images) in BF16, converts,
and answers them again:

| | |
|---|---|
| Same winner | **85 / 86** |
| The one that changed | an "is this urgent?" `noul` on a refund ticket: BF16 0.541 false, FP8 0.506 true. A coin toss either way |
| Largest probability change per question | median 0.0023, p95 0.0395, max 0.0568 |
| `score` expected-value change | max 0.063 (levels are 1 apart) |
| GPU weights | 17.76 -> 11.32 GiB |
| Time for the set | BF16 7.9–9.9 s -> FP8 5.0 s |

The gate fails if any decision BF16 made with at least 0.6 confidence changes. Re-run it after changing anything in
`server/fp8.py`. It needs the server stopped.

## Memory

**Three things this recipe found the hard way, all measured on this host:**

1. **transformers' own loader isn't usable here.** `load_release_model` calls `from_pretrained(device_map=cuda)`.
   On GB10 with transformers 5.10.2, that stages every tensor through a CPU copy on four threads. Host anonymous
   memory climbed ~140 MB/s past a 12 GiB cgroup cap while the weights loaded at a few tensors a second, and the
   load took 119 s. `load_clef()` in `server/app.py` builds the model on the GPU with initialisation skipped and
   copies each tensor in from `safetensors`' CUDA reader: **9.5 s**, no host-side copy. The logits are identical on
   text and agree to four decimal places on an image.
2. **The caching allocator keeps the load's staging blocks**, 1.7 GiB, mostly the two 1.9 GiB vocabulary matrices.
   `torch.cuda.empty_cache()` after the load returns them before the warmup sets the real high-water mark.
3. **A SIGTERM watchdog doesn't stop a load.** uvicorn is still inside its startup hook and ignores SIGTERM until the
   app is up. An early version gave 20 s of grace, the load kept allocating, and **TensorFold's watchdog fired first
   and stopped Qwen**. This one SIGKILLs at once: the server holds no state worth a clean shutdown.

**The budget** (`profiles.sh`). Preflight checks two moments:

| | Bytes | Must leave |
|---|---|---|
| Load peak | BF16 weights 17.77 GiB + host side 5 GiB = 22.8 GiB (no activations yet) | `LOAD_FLOOR_GIB` 8 |
| Steady (FP8) | weights 11.3 + host side 5 + activations 16,384 x 144 KiB = 18.6 GiB | `HOST_FLOOR_GIB` 12 |

Measured at start: **18.7 GiB against 18.6 predicted**, 14.9 GiB left with Qwen on `int8x1`. The load dipped to
9.9 GiB, above both `MemAvailable` floors. There's no KV cache (one forward pass, `use_cache=False`), so the
footprint doesn't grow with traffic once the allocator has seen its peak. `start.sh` makes it see that peak by
warming up at exactly `MAX_LENGTH` tokens before it reports ready, while the watchdog is running.

**Two watchdogs, one pool.** Clef's is set so that it always fires first: every condition contains TensorFold's,
and it reacts ~7 s sooner.

| | `MemAvailable` floor | `MemFree` floor (gated) | Debounce | Then |
|---|---|---|---|---|
| Clef (`scripts/memwatch.sh`) | 7 GiB | 2 GiB while `MemAvailable` < 10.5 GiB | 3 x 1 s | SIGKILL |
| TensorFold | 6 GiB | 2 GiB while `MemAvailable` < 10 GiB | 5 x 2 s | `docker stop` |

The `MemFree` condition must be gated. Loading Clef allocates 18 GiB on the GPU in one burst, faster than the kernel
reclaims Qwen's SSD page cache, so `MemFree` touches ~1 GiB for a few seconds while `MemAvailable` is still 17–20
GiB. A 12 GiB gate killed every load for nothing. Since the switch to SIGKILL, the ordering has been exercised three
times here (a load, a 50k-token Qwen prefill beside BF16 Clef, a diagnostic probe). Clef went every time, and Qwen's
watchdog never got past one low sample of five.

**The cgroup cap** (`CGROUP_MEM_GIB`, 8 GiB, a systemd user scope) bounds the host side only. Verified here: CUDA
allocations aren't charged to the cgroup on GB10. Measured steady: 4.5 GiB.

## Measured on this host

GB10, 121.7 GiB unified memory, driver 580.173.02, torch 2.11.0+cu130, transformers 5.10.2, 2026-10-02.
Qwen3.8-Flash-Next TensorFold `int8x1` loaded beside it in every row. `./bench.sh`, 10 requests per row, the same
three questions, a growing state:

| Input tokens | FP8 p50 | FP8 prefill | BF16 p50 |
|---:|---:|---:|---:|
| 327 | 100 ms | 3,296 tok/s | 141 ms |
| 514 | 143 ms | 3,631 tok/s | 170 ms |
| 1,160 | 327 ms | 3,565 tok/s | 327 ms |
| 3,710 | 1.23 s | 3,026 tok/s | 1.11 s |
| 13,910 | 5.07 s | 2,746 tok/s | 4.15 s |

Throughput at ~360 tokens: **9.3 req/s** at 1–8 concurrent clients (BF16: 7.4). FP8 wins wherever decision-sized
inputs live and loses ~20% at the far end, where quantizing activations on every call costs more than the cheaper
matmuls save. FP8 rows were run while Qwen's token counters didn't move. BF16 rows come from earlier the same day,
before that check existed.

**Sharing the GPU.** Memory is partitioned, compute isn't. With both servers saturated at once, each ran at about half
speed: Clef 3.7k tokens in 2.8 s (from 1.2 s); Qwen prefill 2,455 -> 1,107 tok/s and sampled decode 59 -> 24 tok/s.
Neither watchdog recorded a low sample during that run (lowest `MemAvailable` 14.8 GiB, `MemFree` 7.6 GiB).

## Configuration

Every variable is optional. Storage follows [CONVENTIONS.md](../../CONVENTIONS.md): this is a native recipe and
exports no cache variables. Weights go to the HF cache, Triton's kernels to `~/.triton`, uv's packages to
`~/.cache/uv`, and the venv outside the recipe.

| Variable | Default | Means |
|---|---|---|
| `WEIGHTS` | `fp8` | `fp8` converts the decoder at load; `bf16` serves the checkpoint as released (preflight refuses it beside `int8x1`) |
| `MAX_LENGTH` | `16384` | Longest input, in tokens; the warmup runs at exactly this. The main lever on activations (144 KiB/token) |
| `PORT` / `HOST` | `8012` / `0.0.0.0` | **No API key.** `HOST=127.0.0.1` keeps it off the LAN |
| `SERVED_MODEL_NAME` | `clef-flash` | The id in `/v1/models`, and the default `model` in a response |
| `MAX_IMAGES` / `MAX_VIDEOS` / `MAX_BODY_MIB` | `8` / `1` / `64` | Per-request limits |
| `VENV` | `~/venvs/clef-flash` | The uv project's environment (`UV_PROJECT_ENVIRONMENT`), outside the recipe |
| `HOST_FLOOR_GIB` / `LOAD_FLOOR_GIB` | `12` / `8` | What preflight requires to remain after the steady state / the load peak |
| `CGROUP_MEM_GIB` | `8` | Host-side cap (systemd user scope `clef-flash-<port>`) |
| `MEMWATCH_MIN_GIB` / `MEMWATCH_MIN_FREE_GIB` / `MEMWATCH_FREE_GATE_GIB` | `7` / `2` / `10.5` | Watchdog floors. Keep each at or above TensorFold's (6 / 2 / 10) or the ordering inverts. Preflight warns |
| `MEMWATCH_SAMPLES` / `MEMWATCH_INTERVAL` | `3` / `1` | Debounce |
| `RUNTIME_RESERVE_BYTES` / `ACTIVATION_BYTES_PER_TOKEN` / `FP8_SAVED_BYTES` | 5 GiB / 144 KiB / 6.44 GiB | Budget terms, all measured |
| `PREFLIGHT` | `1` | `0` skips preflight in `start.sh` |
| `OUT_DIR` | `~/.local/state/dgx-spark-recipes/Clef-Flash-FP8-transformers` | Pidfile, logs, bench and drift results, verification stamp |

## Files

```
pyproject.toml, uv.lock   the serving environment (uv project; the venv lives at $VENV)
profiles.sh               paths, budget, watchdog thresholds, helpers -- the only place they are defined
setup.sh                  uv sync --locked, then a real bf16 matmul on the GPU and the fla check
download.sh               pinned, checksum-verified, resumable snapshot download (stdlib)
preflight.sh              venv, snapshot, both memory moments, co-tenants, watchdog ordering, port
start.sh                  systemd scope + cgroup cap, watchdog, readiness, measured vs predicted
stop.sh / status.sh       pidfile-driven; status shows torch and cgroup memory and the shared pool
bench.sh                  latency by input length, throughput by client count -> $OUT_DIR/bench
drift.sh                  the FP8 gate -> $OUT_DIR/drift
server/app.py             the API, the direct-to-GPU loader, the warmup
server/fp8.py             FP8Linear and the decoder conversion
server/fp8_drift.py       the BF16 vs FP8 comparison
scripts/                  smoke.py, bench.py (stdlib), memwatch.sh, download_snapshot.py
manifests/bf16.json       revision, per-file sizes and SHA-256 / git blob ids
```

## Known gaps

- **Video is wired but untested.** Frames arrive as base64 images and are stacked into the array the processor
  takes; no video request was run here.
- **The GDN conv kernel is the torch fallback.** transformers logs "fast path not available" because
  `causal-conv1d` isn't installed (it needs a CUDA build). flash-linear-attention, which carries the expensive
  part, is installed, and the model uses it.
- **The drift set is built here, not a benchmark.** It covers the question types and input kinds the API takes, and
  says nothing about task accuracy on Cloudflare's Decision Index. FP8 was admitted on agreement with BF16, not on
  scores.
- **One request at a time.** See [API](#api) for why batching was measured and removed.
