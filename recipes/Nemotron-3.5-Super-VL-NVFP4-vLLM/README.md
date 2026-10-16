# Nemotron-3.5-Super-VL-120B-A12B-NVFP4 on vLLM

[nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4):
a Mamba-2/Transformer hybrid MoE (512 routed experts, 120B total / 12B active)
with a C-RADIO vision encoder and a 1M-token window, in NVFP4. This recipe runs
the model card's own **DGX Spark command**: one stream, text and image,
dflash speculative decoding with the `-LiLiCorr` draft model at 7 tokens.

**Status: runs on this host; smoke test passes.** Numbers are in Measurements.

## Run

```bash
./download.sh        # 69.9 GiB model + 1.2 GiB draft, sha256-verified, resumable
./preflight.sh       # flags, checkpoints, memory, port, disk
./start.sh           # serves on :8013
./scripts/smoke.py   # reasoning split, enable_thinking=false, an image
./stop.sh
```

Or `scripts/run-nemotron-3.5-super-vl-nvfp4.sh [spark|nodraft]` from the repo
root, which runs preflight and then start. `DRY_RUN=1 ./start.sh` prints the
`docker run` command.

| Profile | Meaning |
|---|---|
| `spark` (default) | The card's Spark recipe: `--max-num-seqs 1`, `--gpu-memory-utilization 0.75`, dflash + LiLiCorr, 7 speculative tokens |
| `nodraft` | Same without speculative decoding: a baseline, and a way to isolate a failure to the draft model. Does not need the draft downloaded |

## What is pinned

| | |
|---|---|
| Model | `579ffa0b659b…` (114 files, 69.9 GiB) |
| Draft | `7c7221aeb4cb…` (5 files, 1.2 GiB) |
| Image | `vllm/vllm-openai@sha256:cb9bc64f…`, the `nightly-3f1bcfd8…` build the card names (arm64 index) |

**The card also names `vllm/vllm-openai:v0.32.0`, which was not on Docker Hub on
2026-10-16** (newest release tag: v0.31.0, 2026-10-04). The recipe therefore
pins the nightly. `preflight.sh` checks the image for every Mamba, video and
speculative flag `start.sh` passes, so a different tag fails there rather than
after a long load. Regenerate manifests with `scripts/make_manifest.py`.

## Memory

vLLM budgets `GPU_MEMORY_UTILIZATION` (0.75, about 91 GiB) of total memory and
refuses to start unless that much is free. On a unified-memory GB10 that is
`MemAvailable`, so **this server cannot run beside the Qwen or Clef recipes**;
`preflight.sh` demands the budget plus 6 GiB and lists the containers in the way.
There is no watchdog: the budget is fixed by vLLM rather than grown under traffic.

## Storage

Nothing heavy is in this directory. Both checkpoints live in the shared Hugging
Face cache; the vLLM, flashinfer and triton caches are the shared defaults,
mounted onto the container's own default paths (`/root/.cache/huggingface`,
`/root/.cache/vllm`, `/root/.cache/flashinfer`, `/root/.triton`) so no cache env
var is set inside. The container runs with `HF_HUB_OFFLINE=1`. Logs go to
`OUT_DIR` (`~/.local/state/dgx-spark-recipes/Nemotron-3.5-Super-VL-NVFP4-vLLM/`).

## Knobs

`PORT` (8013), `HOST`, `IMAGE`, `CONTAINER_NAME`, `SERVED_MODEL_NAME`
(`nemotron-3.5-super-vl`), `GPU_MEMORY_UTILIZATION`, `MAX_NUM_SEQS`,
`MAX_NUM_BATCHED_TOKENS`, `SPEC_TOKENS`, `MAX_MODEL_LEN` (unset: the model's
own window, as in the card), `MEM_HEADROOM_BYTES`, and the storage variables in
[CONVENTIONS.md](../../CONVENTIONS.md). The server has no API key; set
`HOST=127.0.0.1` to keep it off the LAN.

## Using it

OpenAI-compatible at `http://<host>:8013/v1`. Reasoning is on by default and
comes back in `reasoning_content`; `chat_template_kwargs: {"enable_thinking":
false}` turns it off, and `{"enable_thinking": true, "medium_effort": true}`
shortens it. Reasoning tokens share the `max_tokens` budget. The card recommends
temperature 1.0, top_p 0.95 for text and 0.6 / 0.95 for vision. Tool calls use
the `qwen3_coder` parser. Video flags are in the command as the card has them,
but **the card lists video for larger hardware only**; on one Spark it is
unqualified here.

## Licenses

The model is under OpenMDW-1.1. The recipe's own scripts are MIT, except
`scripts/download_snapshot.py`, copied from
`Qwen3.8-Flash-Next-NVFP4-vLLM` with its AGPL-3.0-or-later header, plus one fix:
this repo contains an empty `__init__.py`, which the original could not fetch.

## Measurements

DGX Spark (GB10, 121.7 GiB), `spark` profile, image `nightly-3f1bcfd8` (vLLM
0.32.0rc1.dev190), 2026-10-16, nothing else running.

- **Load:** about 12 min cold from the first start, including a 26 s compile and
  a 265 s engine init. vLLM reported 65.3 GiB used against its 91.3 GiB budget.
- **KV pool:** 2,232,889 tokens (FP8), 2.13x concurrency at the full 1,048,576-token window.
- **Memory:** MemAvailable fell to 17 GiB while the weights loaded and settled at
  about 17.7 GiB with the server idle.
- **Decode, one stream, thinking off, 1,024 tokens:** 26.7 tok/s greedy, 24.3 tok/s
  at temperature 1.0 (wall time, includes prefill). Dflash acceptance length
  was 2.3 to 2.7 of 7 drafted tokens, from the server's own metric.
- **Smoke test:** passes (reasoning split, `enable_thinking=false`, image).

Not measured: long-context prefill, video, the `nodraft` baseline, or the
effect of the draft model on throughput.
