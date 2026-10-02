# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
"""Clef-Flash behind a SystemOne-compatible HTTP API.

Clef-Flash is a Qwen3.5-9B backbone plus a joint schema head that scores every
allowed option of every question in one forward pass. It generates no text, so
an OpenAI-compatible chat server cannot serve it: `vllm serve` would load the
backbone as a chat model and drop the head. This server instead runs the
model's own code, `joint_schema_model.py` from the pinned snapshot, and exposes
the request/response body that code already defines:

    POST /v1/systemone   body and response exactly as `systemone()` defines them
    GET  /v1/models      OpenAI-shaped model list, so generic clients can probe
    GET  /health         readiness plus the memory this process holds

Images travel as base64 (optionally a `data:` URL) in `images`; a video is a
list of base64 frames in `videos`. Nothing is fetched by URL, so a request
cannot make this server reach out anywhere.

Requests run one at a time behind a lock, which does two jobs. Memory: one
forward pass of at most MAX_LENGTH tokens is exactly the high-water mark the
warmup already set, so traffic cannot raise this server's footprint -- which
is what lets it run beside another server on one unified-memory pool. And it
costs no throughput: every request carries a ~250-token system prompt and
schema, and from there GB10 is compute-bound at ~3,400 tokens/s, so a batch
costs the sum of its requests. Token-budgeted batching was built and measured
here: 7.1-7.8 req/s against 7.4 serial, with a worse p95. It was removed.

Weights: WEIGHTS=fp8 (the default) converts the decoder's linear layers to
FP8 right after the BF16 load -- see server/fp8.py for why and what, and
server/fp8_drift.py for the comparison that admitted it. WEIGHTS=bf16 serves
the checkpoint as released.

Configuration is environment variables set by start.sh (see profiles.sh).
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

SNAPSHOT = os.environ["CLEF_SNAPSHOT"]
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "clef-flash")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "16384"))
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "8"))
MAX_VIDEOS = int(os.environ.get("MAX_VIDEOS", "1"))
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_MIB", "64")) * 1024 * 1024
WARMUP = os.environ.get("WARMUP", "1") == "1"
WEIGHTS = os.environ.get("WEIGHTS", "fp8")
if WEIGHTS not in ("fp8", "bf16"):
    raise SystemExit(f"WEIGHTS={WEIGHTS}: expected fp8 or bf16")

log = logging.getLogger("clef")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# The model's own code, from the snapshot preflight.sh verified by SHA-256.
sys.path.insert(0, SNAPSHOT)
import joint_schema_model as jsm  # noqa: E402

STATE: dict[str, Any] = {}
LOCK = threading.Lock()


def _gib(n: int | float) -> float:
    return round(n / 2**30, 2)


def _meminfo(key: str) -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024
    return 0


def _memory() -> dict[str, float]:
    return {
        "torch_allocated_gib": _gib(torch.cuda.memory_allocated()),
        "torch_reserved_gib": _gib(torch.cuda.memory_reserved()),
        "torch_peak_reserved_gib": _gib(torch.cuda.max_memory_reserved()),
        "host_mem_available_gib": _gib(_meminfo("MemAvailable")),
    }


def load_clef(path: str) -> tuple[Any, Any]:
    """What jsm.load_release_model does, with the weights read straight onto the GPU.

    load_release_model calls from_pretrained(device_map=cuda). Measured on
    GB10 with transformers 5.10.2, that path stages every tensor through a
    CPU copy on four threads: host anonymous memory climbed ~140 MB/s past a
    12 GiB cap while the weights loaded at a few tensors a second. Next to a
    co-tenant that is fatal, and it is the opposite of what unified memory
    needs. safetensors' own CUDA reader moves a 4.6 GiB shard in 1.6 s with
    no host-side copy at all.

    So: build the model directly on the GPU with initialisation skipped (one
    allocation, and the rotary buffers are computed as usual), then copy each
    tensor in from safe_open(device="cuda"). The transient is one tensor. The
    head, processor and ClefModel wrapper are the release's own, exactly as
    load_release_model builds them. Keys are required to match 1:1 -- the
    checkpoint was saved by the same transformers version -- and any mismatch
    is a hard error, not a warning.
    """
    import json
    from pathlib import Path

    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration
    from transformers.initialization import no_init_weights

    root = Path(path)
    config = AutoConfig.from_pretrained(root)
    with torch.device("cuda"), no_init_weights():
        backbone = Qwen3_5ForConditionalGeneration._from_config(config, dtype=torch.bfloat16)
    params = dict(backbone.named_parameters())
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    if set(weight_map) != set(params):
        missing, extra = set(params) - set(weight_map), set(weight_map) - set(params)
        raise RuntimeError(f"checkpoint/model key mismatch: missing {sorted(missing)[:5]}, unexpected {sorted(extra)[:5]}")
    with torch.no_grad():
        for shard in sorted(set(weight_map.values())):
            with safe_open(root / shard, framework="pt", device="cuda") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    if tensor.shape != params[key].shape:
                        raise RuntimeError(f"{key}: checkpoint {tuple(tensor.shape)}, model {tuple(params[key].shape)}")
                    params[key].copy_(tensor)
                    del tensor
            # The shard is on the GPU now; its page cache is a second copy of
            # 4-5 GiB that the co-tenant's own SSD reads would rather have.
            fd = os.open(root / shard, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
    backbone.config.use_cache = False
    head = jsm.JointSchemaHead(**json.loads((root / "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(root / "joint_head.safetensors"), strict=True)
    head = head.to(device="cuda", dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(root)
    # Each get_tensor allocated a staging tensor on the GPU, and the caching
    # allocator keeps the freed blocks -- measured 1.7 GiB reserved over what
    # the weights use, mostly the two 1.9 GiB vocabulary matrices. On unified
    # memory that is host memory a co-tenant cannot have, so give it back
    # before the warmup sets the real high-water mark.
    torch.cuda.empty_cache()
    return jsm.ClefModel(backbone, head).eval(), processor


def _warmup(model: Any, processor: Any) -> None:
    """One MAX_LENGTH forward pass, so the allocator reaches its high-water mark now.

    encode_record truncates the state to fit MAX_LENGTH, so an over-long state
    lands at exactly MAX_LENGTH tokens: the largest text request this server
    will ever run.
    """
    request = {
        "model": SERVED_MODEL_NAME,
        "state": "warmup " * (MAX_LENGTH * 2),
        "questions": {"ok": {"type": "noul", "instructions": "Is this a warmup?"}},
    }
    started = time.perf_counter()
    response = jsm.systemone(model, processor, request, max_length=MAX_LENGTH)
    torch.cuda.synchronize()
    log.info(
        "warmup: %d tokens in %.2f s; %s",
        response["usage"]["input_tokens"], time.perf_counter() - started, _memory(),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    started = time.perf_counter()
    avail_before = _meminfo("MemAvailable")
    model, processor = load_clef(SNAPSHOT)
    log.info("loaded in %.1f s; %s", time.perf_counter() - started, _memory())
    if WEIGHTS == "fp8":
        import fp8

        converted, saved = fp8.quantize_decoder(model.language_model)
        log.info("fp8: %d decoder linears converted, %.2f GiB given back; %s", converted, saved / 2**30, _memory())
    if WARMUP:
        _warmup(model, processor)
    STATE.update(
        model=model,
        processor=processor,
        load_seconds=round(time.perf_counter() - started, 1),
        host_cost_gib=_gib(avail_before - _meminfo("MemAvailable")),
    )
    log.info("ready: host MemAvailable fell %.2f GiB during load and warmup", STATE["host_cost_gib"])
    yield
    STATE.clear()


app = FastAPI(title="clef-flash", lifespan=lifespan)


def _decode_image(value: Any, where: str):
    from PIL import Image

    if not isinstance(value, str):
        raise ValueError(f"{where}: expected a base64 string or data: URL")
    if value.startswith("data:"):
        _, _, value = value.partition(",")
    try:
        raw = base64.b64decode(value, validate=True)
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (binascii.Error, OSError) as exc:
        raise ValueError(f"{where}: not a decodable image ({exc})") from exc
    return image.convert("RGB")


def _decode_media(request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    images = request.get("images") or []
    videos = request.get("videos") or []
    if not isinstance(images, list) or not isinstance(videos, list):
        raise ValueError("images and videos must be lists")
    if len(images) > MAX_IMAGES:
        raise ValueError(f"at most {MAX_IMAGES} images per request (MAX_IMAGES)")
    if len(videos) > MAX_VIDEOS:
        raise ValueError(f"at most {MAX_VIDEOS} videos per request (MAX_VIDEOS)")
    out = dict(request)
    if images:
        out["images"] = [_decode_image(v, f"images[{i}]") for i, v in enumerate(images)]
    if videos:
        decoded = []
        for i, frames in enumerate(videos):
            if not isinstance(frames, list) or not frames:
                raise ValueError(f"videos[{i}]: expected a non-empty list of base64 frames")
            decoded.append(np.stack([np.asarray(_decode_image(f, f"videos[{i}][{j}]")) for j, f in enumerate(frames)]))
        out["videos"] = decoded
    return out


@app.get("/health")
def health() -> dict[str, Any]:
    if "model" not in STATE:
        raise HTTPException(503, "loading")
    return {
        "status": "ok",
        "model": SERVED_MODEL_NAME,
        "revision": os.path.basename(SNAPSHOT.rstrip("/")),
        "weights": WEIGHTS,
        "max_length": MAX_LENGTH,
        "load_seconds": STATE["load_seconds"],
        "host_cost_gib": STATE["host_cost_gib"],
        **_memory(),
    }


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": SERVED_MODEL_NAME, "object": "model", "owned_by": "cloudflare"}]}


@app.post("/v1/systemone")
async def systemone(request: Request) -> JSONResponse:
    if int(request.headers.get("content-length") or 0) > MAX_BODY_BYTES:
        raise HTTPException(413, f"body larger than {MAX_BODY_BYTES // 2**20} MiB (MAX_BODY_MIB)")
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        raise HTTPException(413, f"body larger than {MAX_BODY_BYTES // 2**20} MiB (MAX_BODY_MIB)")
    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(400, f"body is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(400, "body must be a JSON object")
    # Answer whatever model id a client sends, as SystemOne does, but default it
    # so a request without one is not rejected by systemone()'s own check.
    payload.setdefault("model", SERVED_MODEL_NAME)

    def run() -> tuple[dict[str, Any], float]:
        # Off the event loop: image decoding and the processor are CPU work,
        # and only the model call itself needs the lock.
        decoded = _decode_media(payload)
        with LOCK:
            started = time.perf_counter()
            result = jsm.systemone(STATE["model"], STATE["processor"], decoded, max_length=MAX_LENGTH)
            return result, (time.perf_counter() - started) * 1000

    from starlette.concurrency import run_in_threadpool

    try:
        result, elapsed_ms = await run_in_threadpool(run)
    except (ValueError, KeyError, TypeError) as exc:
        # Malformed requests: systemone()'s own checks raise ValueError, and a
        # question missing a field the encoder indexes surfaces as Key/TypeError.
        raise HTTPException(400, f"{type(exc).__name__}: {exc}") from exc
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise HTTPException(507, f"out of GPU memory for this request: {exc}") from exc
    # Timing rides in a header so the body stays exactly the SystemOne shape.
    return JSONResponse(result, headers={"X-Clef-Forward-Ms": f"{elapsed_ms:.1f}"})

if __name__ == "__main__":
    import uvicorn

    # One worker: a second would load a second 18 GiB copy of the model.
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8012")),
        workers=1,
        log_level="info",
        timeout_graceful_shutdown=10,
    )
