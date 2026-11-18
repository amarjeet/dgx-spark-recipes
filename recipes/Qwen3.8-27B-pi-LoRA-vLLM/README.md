# Qwen3.8-27B + pi LoRA on vLLM

[harness-hub/Qwen3.8-27B-pi-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted):
a PEFT LoRA (rank 32, alpha 64) on
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), trained with
reinforcement learning (AsyncGRPO) to drive the
[pi](https://github.com/earendil-works/pi) coding agent. The base is a dense
27B hybrid: 48 linear-attention and 16 full-attention layers out of 64. This
recipe runs the adapter card's own **vLLM evaluation command** on the BF16 base:
LoRA served beside the base model, thinking off, `qwen3_xml` tool calls. The one
change is the window: the base's native 262,144 tokens, not the card's 131,072.

**Status: runs on this host with the `pi` profile, and the smoke test passes.**
Numbers are in Measurements. The `pi-mtp` and `base` profiles have not been
started.

## Run

```bash
./download.sh        # 0.6 GiB adapter + 51.8 GiB BF16 base, sha256-verified, resumable
./preflight.sh       # image contents, checkpoints, adapter shape, memory, port, disk
./start.sh           # serves on :8014
./scripts/smoke.py   # thinking off, tool call parsed, adapter changes output
./stop.sh
```

Or `scripts/run-qwen3.8-27b-pi-lora.sh [pi|pi-mtp|base]` from the repo root,
which runs preflight and then start. `DRY_RUN=1 ./start.sh` prints the
`docker run` command.

| Profile | Meaning |
|---|---|
| `pi` (default) | The card's command. Serves `qwen3.8-27b` (base) and `qwen3.8-pi` (base + LoRA) from one server, so A/B comparison needs no restart |
| `pi-mtp` | `pi` plus MTP speculative decoding (2 tokens) from the base's own MTP head. Lossless, but the head never saw the adapter, so acceptance may be low. **Not in the card; unqualified.** It may also fail to start if this vLLM build does not combine LoRA, MTP and Mamba prefix caching |
| `base` | The base model alone, no LoRA: a baseline for LoRA overhead, and a way to isolate a failure to the adapter. Does not need the adapter downloaded |

## What is pinned

| | |
|---|---|
| Base | `Qwen/Qwen3.8-27B` @ `37561816e66b…` (32 files, 51.8 GiB, BF16, includes a vision tower and an MTP head) |
| Adapter | `harness-hub/Qwen3.8-27B-pi-adapted` @ `fedf3083b1bb…` (12 files, 0.6 GiB; the weights are F32) |
| Image | `vllm/vllm-openai@sha256:3016f449…`, the `v0.26.0-aarch64-ubuntu2404` tag |

Regenerate the manifests with `scripts/make_manifest.py`.

## Departures from the model card

- **The adapter id.** The card's vLLM and merge commands name
  `harness-hub/Qwen3.8-27B-pi`. On 2026-11-18 that repo returned HTTP 401,
  meaning it does not exist or is private. The adapter is at
  `-pi-adapted`, the name the card's own PEFT snippet uses.
- **The adapter is passed by snapshot path**
  (`/root/.cache/huggingface/hub/models--harness-hub--Qwen3.8-27B-pi-adapted/snapshots/<rev>`),
  not by repo id. The served adapter is then the pinned revision even if
  `refs/main` moves. `start.sh` derives that path from the `HF_HOME` mount, so
  the two cannot disagree.
- **The window.** `--max-model-len 262144`, the base model's native window.
  The card evaluated at 131,072, so **the adapter is untested past 131,072
  tokens**. `MAX_MODEL_LEN=131072 ./start.sh` restores the card's setting.
- **Concurrency.** `--max-num-seqs 4`. The card leaves this at vLLM's default.
- **The chat template** is the base's. The adapter repo ships a
  `chat_template.jinja`, but it is byte-identical (sha256 `af16f654…`), so no
  `--chat-template` override is needed.

## Why it should load (checked without loading)

These are read-only checks from 2026-11-18. `preflight.sh` repeats each one
against the image and the files on disk:

- vLLM 0.26.0 registers `Qwen3_5ForConditionalGeneration` (with `IsHybrid`)
  and inherits `SupportsLoRA` from its Qwen3-VL base.
- The adapter's 512 tensors are all named
  `base_model.model.model.language_model.layers.*`. vLLM's Qwen3-VL weight
  mapper rewrites exactly `model.language_model.` → `language_model.model.`.
- The MLP LoRA covers all 64 layers. The q/k/v/o LoRA covers only the 16
  full-attention layers (3, 7, … 63). The linear-attention layers have no
  adapter.
- The `q_proj` LoRA output is 12,288 wide, twice 24 heads × 256, because the
  attention is gated. Only a real load could prove this fits vLLM's merged
  `qkv_proj`, and it does: the adapter loads and is applied (see
  Measurements). If a re-published adapter breaks that, run
  `./start.sh base` to confirm the base on its own works.

## Memory

vLLM budgets `GPU_MEMORY_UTILIZATION` (0.75, about 91 GiB) of total memory and
refuses to start unless that much is free. The expected split:

- About 51 GiB of BF16 weights; the vision tower is skipped with
  `--language-model-only`.
- About 1.2 GiB for the LoRA weights, upcast from F32.
- The rest goes to KV cache, Mamba state and activations. Only 16 layers keep
  KV (4 KV heads × 256, about 64 KiB per token in BF16), so a full
  262,144-token sequence costs about 16 GiB. `align`-mode prefix caching keeps
  additional Mamba state per cached block.

**This server cannot run beside the Qwen3.8-Flash-Next, Nemotron or DeepSeek
recipes.** `preflight.sh` demands the budget plus 6 GiB and names the containers
in the way. If MemAvailable sits well above the budget plus headroom, drop the
page cache (`sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`) before a cold
load. A 52 GiB read on top of a full page cache is where unified memory
gets tight.

## Speed

Decode is bound by memory bandwidth. Each token reads all ~51 GiB of BF16
weights, and the GB10 has about 273 GB/s, so one stream tops out around
5 tok/s. Measured: 4.4 tok/s for the adapter, and about 17 tok/s across four
streams (see Measurements). Prefill is compute-bound and
much faster, and prefix caching keeps agent loops from re-prefilling their
shared history. With pi, a SWE or terminal task takes minutes. That is fine for
evaluating the adapter and slow for daily use. The `pi-mtp` profile exists to
measure how much MTP recovers.

## Storage

Nothing heavy is in this directory. Both checkpoints live in the shared Hugging
Face cache. The vLLM, flashinfer and triton caches are the shared defaults,
mounted onto the container's own default paths (`/root/.cache/huggingface`,
`/root/.cache/vllm`, `/root/.cache/flashinfer`, `/root/.triton`), so no cache
env var is set inside. The container runs with `HF_HUB_OFFLINE=1`. Logs go to
`OUT_DIR` (`~/.local/state/dgx-spark-recipes/Qwen3.8-27B-pi-LoRA-vLLM/`).

A merged checkpoint (the card's alternative to serving the LoRA) would be
about 54 GB of run output. It belongs under `${DGX_SPARK_ROOT}/outputs/`, never
in this directory. This recipe does not merge.

## Knobs

`PORT` (8014), `HOST` (also the address the port is published on), `IMAGE`,
`CONTAINER_NAME`, `SERVED_MODEL_NAME` (`qwen3.8-27b`), `LORA_NAME`
(`qwen3.8-pi`), `GPU_MEMORY_UTILIZATION`, `MAX_MODEL_LEN` (262144, the model's own
window; 131072 is the card's), `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, `SPEC_TOKENS`,
`MEM_HEADROOM_BYTES`, and the storage variables in
[CONVENTIONS.md](../../CONVENTIONS.md). The server has no API key. Set
`HOST=127.0.0.1` to keep it off the LAN.

## Using it with pi

OpenAI-compatible at `http://<host>:8014/v1`. Request model `qwen3.8-pi` for
the adapter and `qwen3.8-27b` for the base. Thinking is off by default,
**as the adapter was trained**. Do not turn it on for the adapter. Tool calls
come back parsed in `tool_calls` (parser `qwen3_xml`).

pi must use the chat-completions API. Without it, pi uses the Responses API,
which the adapter was neither trained nor evaluated with. In
`~/.pi/agent/models.json`, add a provider with `"baseUrl":
"http://localhost:8014/v1"`, `"api": "openai-completions"`, `"apiKey":
"dummy"` and model id `qwen3.8-pi`. pi needs Node.js 22.19 or newer. The card
evaluated with temperature 0.8.

## Claims to keep in perspective

The card reports Terminal-Bench 2.1 rising from 39.5% to 55.8% and SWE-bench
Pro (hard 51) from 29.4% to 62.7%, both under pi against the base. These
are the publisher's own numbers. The checkpoint (step 160 of a run stopped at
step 191) was chosen using scores on SWE-bench Pro and a Terminal-Bench subset,
so expect some selection bias.

## Licenses

Base and adapter are Apache-2.0. The recipe's own scripts are MIT, except
`scripts/download_snapshot.py`, copied unchanged from
`Nemotron-3.5-Super-VL-NVFP4-vLLM` with its AGPL-3.0-or-later header.

## Measurements

DGX Spark (GB10, 121.7 GiB), `pi` profile, image `v0.26.0-aarch64-ubuntu2404`
(by digest), 2026-11-18, nothing else running.

- **Load:** about 9 min cold on the first start. Weights took 356 s
  (50.73 GiB), torch.compile 32 s, and engine init 160 s in total.
- **Memory:** vLLM used 50.73 GiB for weights, 2.55 GiB peak activation,
  2.99 GiB non-torch and 0.35 GiB CUDA graphs, against a 91.3 GiB budget.
  MemAvailable was about 28 GiB with the server up.
- **KV pool:** 34.43 GiB, which is 541,613 tokens at a 131,072-token window
  (4.13x concurrency). With the 262,144 default it is 552,731 tokens, 2.11x
  concurrency. That second start took 379 s for weights and 142 s for engine
  init. The compile ran again, because the window is part of vLLM's cache key. vLLM set the attention block size
  to 784 tokens so its pages match the Mamba state pages (`align` mode).
- **Decode, 512 tokens, temperature 0.8, wall time including prefill:**
  4.58 tok/s for the base alone and **4.41 tok/s for the adapter**, so LoRA
  costs about 4%. Four adapter streams together reached 17.25 tok/s.
- **Adapter applied:** greedy output on an easy coding prompt was identical to
  the base's. On an agent-style prompt the adapter switched the top first
  token (`` ` `` instead of `` ``` ``), and logprobs over 24 tokens differed by
  up to 1.02. The smoke test checks the logprob shift for this reason, not
  the text.
- **Smoke test:** passes. Thinking is off by default with no trace, the tool
  call is parsed into `tool_calls`, and the adapter shifts the distribution.
- vLLM warns that `align`-mode prefix caching for Mamba layers is
  experimental, and that it found no LoRA kernel tuning configs for this GPU
  ("Using default LoRA kernel configs").

Not measured: the card's benchmarks, an end-to-end pi task, long-context
prefill, and the `pi-mtp` and `base` profiles.
