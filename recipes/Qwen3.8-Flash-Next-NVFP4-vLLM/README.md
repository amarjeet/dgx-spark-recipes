# Qwen3.8-Flash-Next — NVFP4 on vLLM, one DGX Spark

Serves [`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4)
— a multimodal (text + image + video) checkpoint with MXFP8 attention and a
4-bit NVFP4 PLE table — on a single **NVIDIA DGX Spark** (GB10, `aarch64`,
121.69 GiB unified memory) at `TP=1`, behind an OpenAI-compatible API.

## Credit

**This recipe is a port of [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark),
Copyright © 2026 [MiaAI Lab](https://x.com/MiaAI_lab), AGPL-3.0-or-later.**

MiaAI Lab did the hard part. Everything that makes this checkpoint fit on one
Spark is their work, not ours:

- the insight that the 26.82 GiB PLE table can be served memory-mapped from a
  CPU offload worker, so it is evictable page cache instead of resident GPU
  memory — the single reason a 98.66 GiB checkpoint fits in a 121.69 GiB pool;
- the two GB10-specific vLLM bug fixes (CUDA stream memory ops deadlocking the
  GPU worker after graph capture; offload rows needing codes *and* scales);
- the MXFP8 fallback and the FP8-KV patch;
- the host-side budget model — `HOST_RESERVE_GIB`, the cgroup cap and the
  watchdog floors — together with the incident analysis behind those numbers,
  which they paid for with three lost servers in a day;
- every measured constant this recipe's arithmetic depends on: 26.82 GiB PLE,
  5.6 GiB overhead, 1.49 GiB MTP, 28.8 KiB/token KV, the 0.58 FP8 multiplier;
- the four-part optimisation pass of 2026-09-05/06, all of it theirs and all of
  it measured over ten launches and a 45-minute soak: batching the PLE row
  gather's page faults, reduced-vocabulary MTP drafting, the BF16 GDN recurrent
  state, capturing every decode graph width, and the discovery that the MTP
  draft config copy silently downgrades those graphs unless the V2 model runner
  is pinned. Together they roughly doubled single-stream decode.

They also publish the [NVFP4 checkpoint](https://huggingface.co/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4)
itself. All seven files under `files/` are theirs, copied verbatim.

Our contribution is packaging: pinning and checksumming the revision, moving
config into `profiles.sh`, extracting the budget arithmetic so preflight can
check it, adding a preflight, and applying this repo's storage and networking
conventions. See [Differences from upstream](#differences-from-upstream) for the
full list, and [License](#license) — which matters, because this recipe is AGPL
while the rest of this repo is MIT.

Upstream in turn credits [lancelind/qwen3.8-Flash-DGX](https://github.com/lancelind/qwen3.8-Flash-DGX)
(Apache-2.0) for the FP8-KV approach reimplemented in
`files/patch_qsa_fp8_kv.py`. That credit applies to that one patch and travels
with it here.

## The problem this solves

The checkpoint is 98.66 GiB on disk and the pool is 121.69 GiB. Loading all of
it as resident GPU memory leaves nothing for KV, the runtime, or the operating
system — and on unified memory, exhausting the pool does not raise an OOM. It
hangs the kernel: no error, no logs, a power cycle.

Two things make it fit:

1. **The PLE table is not on the GPU.** The 26.82 GiB n-gram table is served by
   vLLM's CPU-offload worker from a memory-mapped pre-packed file, built once on
   first launch. File-backed pages are *evictable page cache*, so the
   deployment's non-evictable footprint is ~78 GiB + KV rather than ~104 GiB +
   KV. That margin is the whole recipe.
2. **The GPU budget is capped from the host side.** vLLM reads `MemAvailable`
   — page cache included — as "free GPU memory" on this integrated GPU, then
   fills the GPU side to exactly `gpu_memory_utilization × MemTotal`. So the
   budget cannot be chosen from what the model wants; it is capped at
   `MemTotal − HOST_RESERVE_GIB`, and the KV pool is whatever that leaves.

## How it fits

| | GiB | |
|---|---:|---|
| unified pool | 121.69 | LPDDR5X, CPU and GPU share it |
| checkpoint on disk | 98.66 | |
| — of which PLE table | 26.82 | memory-mapped, **not** on the GPU |
| weights on GPU | 71.84 | |
| runtime overhead | 5.60 | non-torch 3.37 + activation 1.92 + graphs 0.12 |
| MTP draft model | 1.49 | when `MTP_NUM_SPECULATIVE_TOKENS > 0` |
| **GPU budget** | **95.65** | `gmu 0.786`, capped by `HOST_RESERVE_GIB=26` |
| **KV cache** | **16.72** | ~1.11M FP8 tokens — 4.2× a full 262K request |
| container cgroup cap | 100 | bounds host-side memory only (ceiling 105) |

`./preflight.sh` prints this table for your actual `MemTotal` and profile before
anything long-running starts. The arithmetic lives in one place,
[`scripts/budget.py`](scripts/budget.py), so preflight and start cannot drift.

`KV_TARGET_GIB` is a **wish, not a grant**: it ships at 20, and on a 121.69 GiB
host `HOST_RESERVE_GIB` clips it to 16.72 on every launch — `start.sh` says so
(`KV target 20 GiB reduced to 16.72 GiB`). The cap, not this knob, is what
bounds the budget. On a host with more memory the cap is looser and 20 may be
granted in full. The KV figures above are `budget.py`'s arithmetic, not a
measurement; what vLLM's own profiling leaves is printed at launch, and
upstream measured 15.98 GiB = 1,132,586 tokens surviving it on their final
2026-09-06 launch.

## Quick start

```bash
./download.sh          # 98.66 GiB, pinned + checksummed, resumable
./preflight.sh         # image, checkpoint, budget, GPU tenancy, port
./start.sh             # ~12.5 min to /health from cold; much faster warm
./scripts/smoke.py     # reasoning channel, arithmetic, vision tower
./status.sh
./stop.sh
```

`start.sh` never downloads and `download.sh` never launches. Each fails with the
name of the other.

## Profiles

One checkpoint, one quantization — so unlike the GGUF recipes here, the profiles
are serving trade-offs rather than different weight files. All three load the
same 98.66 GiB and differ only in what the KV pool costs.

| Profile | Context | KV dtype | SSM state | KV pool | Notes |
|---|---:|---|---|---|---|
| `native` *(default)* | 262,144 | FP8 | BF16 | ~1.11M tokens | the model's full native context |
| `yarn512` | 524,288 | FP8 | BF16 | ~1.11M tokens | YaRN rope scaling, 2× native |
| `bf16kv` | 262,144 | BF16 | FP32 | ~610K tokens | quantizes nothing |

`bf16kv` keeps the checkpoint's float32 recurrent state as well as unquantized
keys: it exists to be the profile that quantizes nothing, and a recurrent state
is a worse place to lose precision than a KV block.

```bash
./start.sh yarn512
MAX_MODEL_LEN=65536 ./start.sh          # any profile, shorter context
MTP_NUM_SPECULATIVE_TOKENS=0 ./start.sh # give back 1.49 GiB
REASONING_EFFORT=low ./start.sh         # shorter thinking traces, server-wide
MAMBA_SSM_CACHE_DTYPE= ./start.sh       # float32 recurrent state
```

### FP8 KV is a capacity trade, not a free win

It roughly doubles the KV pool — ~1.85× here, because the 12 full-attention
layers halve but the QSA side and compressor caches stay BF16 — and that is what
makes a 512K context reachable at all.

It also perturbs which blocks the sparse-attention indexer selects. Upstream
scored 11/11 on their reasoning suite with FP8 KV, same as BF16; the reference
implementation they cite measured a long-reasoning benchmark falling from 6/6 to
2/6. **Neither number was measured on this host.** If long-reasoning quality
matters more to you than context length, use `./start.sh bf16kv` and measure it
yourself.

## Storage

Follows the repo's [storage convention](../../CONVENTIONS.md): the recipe
directory holds code, config and manifests, and nothing heavy. Every cache is
bind-mounted onto the path the tool already defaults to *inside* the container,
so no cache environment variable is set in the container at all.

| Data | Host path | Mounted at |
|---|---|---|
| checkpoint (98.66 GiB) | `$HF_HOME` (`~/.cache/huggingface`) | `/root/.cache/huggingface` |
| packed PLE table (~27 GiB) | `$VLLM_CACHE_HOST/ple_cache/` (`~/.cache/vllm`) | `/root/.cache/vllm` |
| torch.compile cache | `$VLLM_CACHE_HOST` | `/root/.cache/vllm` |
| flashinfer JIT cache | `$FLASHINFER_CACHE_HOST` (`~/.cache/flashinfer`) | `/root/.cache/flashinfer` |
| triton kernel cache | `$TRITON_CACHE_HOST` (`~/.triton`) | `/root/.triton` |
| reduced draft vocabulary | `$DRAFT_VOCAB_DIR` (`~/.cache/vllm/draft_vocab`) | *(inside the vLLM cache mount)* |
| bench results, verify stamps, archived logs | `$OUT_DIR` | — |

`$OUT_DIR` defaults to
`${XDG_STATE_HOME:-~/.local/state}/dgx-spark-recipes/Qwen3.8-Flash-Next-NVFP4-vLLM`.
The only files written into the recipe directory are `.vllm.log`, `.vllm.pid`,
`.profile.active` and the regenerated patch outputs under `files/` — all
gitignored.

`start.sh` asserts that `MODEL_PATH` sits under `HF_HOME`, because the
container-side path is derived by swapping that prefix. An `HF_HOME` that does
not contain the snapshot would otherwise produce a container path that does not
exist, and you would find out minutes into a load.

`MTP_DRAFT_VOCAB` gets the same assert against `VLLM_CACHE_HOST`. It needs no
mount of its own precisely *because* it lives in the vLLM cache tree, which is
already mounted at vLLM's own in-container default — upstream bind-mounts the
file separately at `/root/draft_vocab.txt`, and that extra mount is the one
thing this port deliberately does not copy.

## Safety

On unified memory the failure mode is a hung kernel, not a killed process. Three
independent mechanisms, none needing `sudo`:

- **`HOST_RESERVE_GIB=26`** caps the GPU budget at `MemTotal − 26`. It covers,
  in order: other containers and sessions (~7 GiB is normal here), vLLM's own
  host-side processes (~6), PLE page cache (≥6), the NVIDIA driver's free-page
  reserve (≥3), and 2–3 GiB of per-request growth that is never returned. Raise
  it in 2 GiB steps if the watchdog log shows `MemAvailable` idling under ~9 GiB.
  **Do not lower it to buy KV** — that is exactly the change that cost upstream
  three servers in one day.
- **A hard cgroup cap** on the container. GPU parameter allocations are not
  charged to it on GB10, so it bounds the host-side footprint (Python processes,
  pinned buffers, page cache) while the GPU budget bounds the GPU side. It does
  not protect the host from the GPU side; `HOST_RESERVE_GIB` does.
- **A watchdog** ([`files/memwatch.sh`](files/memwatch.sh)) polling every second
  with a 5-sample debounce. It stops the container if `MemAvailable` stays below
  6 GiB, or if `MemFree` stays below 2 GiB *while* `MemAvailable` is under 10 GiB
  — the driver starts refusing allocations (`NV_ERR_NO_MEMORY`) at `MemFree`
  ~3 GiB while `MemAvailable` still reads 6+. Logs are archived to `$OUT_DIR`
  before it acts.

`./stop.sh` is graceful by default: vLLM gets `SIGTERM` and 30 s to unlink the
POSIX shared-memory segments the PLE handshake allocates. The container runs
`--ipc host`, so a `SIGKILL` leaks them onto `/dev/shm` until reboot.

Two more things worth knowing:

- **`PLE_OFFLOAD=false` is refused at `TP=1`.** Pushing 98.66 GiB through UVM
  hung the host upstream. The script will not do it.
- **Kernel VM tunables.** Stock values give the driver essentially no free-page
  reserve (`min_free_kbytes` ~44 MB on a 121 GiB box). `preflight.sh` warns;
  [`files/sysctl-spark3.conf`](files/sysctl-spark3.conf) holds values that ran
  crash-free upstream. Nothing here applies them — read its header first, they
  shift `MemAvailable` accounting and therefore every number above.

## Runtime patches

Five generators rewrite vLLM sources from the image's own copies on every
launch. `start.sh` extracts each `*.orig` from the container once, regenerates
the patched file, and bind-mounts it read-only over the original. Nothing is
baked into an image and nothing is vendored — if the image moves, the extraction
fails loudly, and `preflight.sh` checks all nine target paths by name before
you get there.

| Generator | Fixes |
|---|---|
| [`patch_ple_layer.py`](files/patch_ple_layer.py) | NVFP4/FP8 dispatch; offload rows must carry codes **and** scales; batches the row gather's page faults through `posix_fadvise(WILLNEED)` |
| [`patch_ple_offload.py`](files/patch_ple_offload.py) | CUDA stream memory ops are unsupported on GB10 and deadlocked the GPU worker after graph capture; host-side handshake and `MADV_RANDOM` mmap instead; opens the second fd the prefetch pass advises on |
| [`patch_modelopt_mxfp8.py`](files/patch_modelopt_mxfp8.py) | BF16 fallback for MXFP8 shapes the kernel rejects |
| [`patch_qsa_fp8_kv.py`](files/patch_qsa_fp8_kv.py) | quantized K/V tiles with per-tensor scales; compiles out at BF16 KV, so it is applied unconditionally |
| [`patch_mtp_draft_vocab.py`](files/patch_mtp_draft_vocab.py) | adds `get_top_tokens()` to the MTP drafter so it can read a reduced lm_head; inert unless `MTP_DRAFT_VOCAB` is set, so it is applied unconditionally |

## Decode optimisations

Four changes from upstream's 2026-09-05/06 pass, all on by default. Every
number here is **upstream's, measured on their host at 512K YaRN** — the
figures in [Measurements](#measurements) are ours and predate all four, so
treat this section as what to expect, not as what this host has recorded.

| Knob | Default | What it buys upstream |
|---|---|---|
| PLE gather prefetch | *(always on)* | **+7–10% prefill.** A 2,048-token chunk gathers ~32,768 unrelated 90-byte rows out of the 27 GiB table, one 4 KiB fault at a time at queue depth 1; naming the pages up front lets the NVMe see them together. Worth ~3× more on prefill than decode, which gathers ~256 rows per step. |
| `MAMBA_SSM_CACHE_DTYPE` | `bfloat16` | **+8.5% decode at 8 streams** (151.6 → 164.5 tok/s), +6.8% at 1. The GDN recurrent state is ~0.23 GB per sequence read *and* written every step. Needles 15/15 at 32K, same as float32. |
| `CUDAGRAPH_CAPTURE_SIZES` | `auto` | Removes an eager-decode cliff rather than adding speed. vLLM's own list leaves widths the scheduler can build but never captured — at MTP 3 and `MAX_NUM_SEQS=5`, a full 5-sequence verify batch is 20 tokens and had no graph at all. |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | **A safety pin.** The MTP draft config copy is not in the V2 default set, falls back to V1, and mutates the `compilation_config` it *shares* with the target — turning FULL decode graphs into PIECEWISE. That cost +24% on the single-stream step and drove the driver to 99.4 GiB against a 94.87 GiB budget until the watchdog stopped the server. |

**Verified on this host on 2026-09-07**, first launch of the ported code
(`yarn512`, 524,288 context, `REASONING_EFFORT=low`):

- `Overriding cudagraph_mode` appeared **0 times**, so the V2 runner pin held
  and the decode graphs stayed FULL. This is the check worth repeating on any
  image change — the failure is silent.
- `Capturing CUDA graphs (FULL)` captured **4/4** widths in 2 s, exactly the
  `4,8,12,16` that `scripts/graph_widths.py` derives for `MAX_NUM_SEQS=4` at
  MTP 3.
- **4 `NV_ERR_NO_MEMORY` during KV allocation and graph capture, then none.**
  That startup burst is expected. `MemFree` dipped to 1,394 MiB inside it while
  `MemAvailable` was 17 GiB, and the watchdog's `MemFree` floor correctly did
  not fire, because it is gated on `MemAvailable < 10 GiB` — the gate exists
  for exactly this window.

Two knobs ship as documented non-defaults because upstream measured them and
they did not earn a change:

- **`MTP_K_SCHEDULE`** (dynamic speculative depth). The static sweep found
  K=3 optimal at *every* concurrency, K=2 tied, K=1 losing 8–14%. There is no
  crossover, so there is nothing to schedule. Setting it also makes vLLM
  override `cudagraph_mode` to PIECEWISE.
- **`COMPILATION_MODE=3`** (Inductor fusion). +0.3% at one stream, +1.0% at
  four — inside noise. Decode here is bandwidth-bound, and fusion only helps
  the part of the step that is not.

### Reduced-vocabulary drafting

Off by default, and the largest single win upstream measured: **−16.9%
single-stream step time.**

The MTP drafter carries its own BF16 `lm_head` over the whole 248,320-token
vocabulary — 1.18 GiB, read once per draft step, which at MTP 3 is three of the
four `lm_head` reads in an engine step — to produce one argmax. A 65,536-row
slice is 0.31 GiB, saving 2.61 GiB of traffic per step.

```bash
./scripts/build_draft_vocab.sh model_output.jsonl        # writes into ~/.cache/vllm/draft_vocab/
MTP_DRAFT_VOCAB=~/.cache/vllm/draft_vocab/<name>.txt ./start.sh
```

**Accuracy is safe structurally, not by luck.** The rejection sampler keeps a
draft only when it equals the target model's own argmax, and emits the target's
token otherwise, so a reduced-vocabulary drafter is indistinguishable from a
merely less accurate one — which is the case rejection sampling exists to
handle. Upstream confirmed on MGSM: English 94.8% against 93.6% full-vocabulary
(+13.4% throughput), Chinese identical at 86.4% either way despite the shipped
vocabulary covering only 50.6% of the tokens the model emits in Chinese.
Coverage buys speed, never correctness, and out-of-vocabulary traffic comes out
break-even rather than slower.

Build the vocabulary from **the model's own output**, not from a general corpus
and not from your prompts: that is the distribution the drafter has to predict.
Tune on coverage, not size — upstream found acceptance falling off a cliff
below ~88–90%.

## Measurements

Measured on this host on 2026-09-05: DGX Spark GB10, `aarch64`, 121.69 GiB
unified memory, kernel 6.17.0-1031-nvidia, vLLM `0.1.dev20073+g8e685d198`,
profile `native` (262,144 context, FP8 KV, MTP 3), idle server.

> **The throughput numbers below predate
> [the decode optimisations](#decode-optimisations)**, which landed here on
> 2026-09-07. Expect prefill ~7–10% better and single-stream decode
> substantially better; re-run `./bench.sh` before quoting them. The *memory*
> figures have been re-measured on the new configuration — see
> [What the engine actually allocated](#what-the-engine-actually-allocated).

### Startup, from cold page cache

| | |
|---|---|
| packed PLE table build (first launch only) | 35 s |
| weight load | 489.6 s |
| KV sizing, autotune, graph capture, API up | ~150 s |
| **cold start to `/health`** | **~12.5 min** |

Restarts are much faster: the PLE table, the 138 autotuned flashinfer configs
and the triton kernels all persist in the shared caches.

### Steady state, serving and idle

`MemAvailable` settles at **~16.3 GiB** of the 121.69 GiB pool, with the NVIDIA
driver holding ~95.7 GiB and zero `NV_ERR_NO_MEMORY` in the kernel log. That is
comfortably above the watchdog's 6 GiB floor, but it is lower than the 26 GiB
`HOST_RESERVE_GIB` nominally sets aside — the reserve bounds the *GPU budget*,
not the total footprint, and vLLM's host-side processes spend part of it. 16 GiB
idle is healthy here; the number to act on is ~9 GiB, at which point raise
`HOST_RESERVE_GIB` by 2.

### What the engine actually allocated

Two launches on this host, one per configuration.

**`native` at `KV_TARGET_GIB=16`, float32 SSM** — what this recipe shipped
until 2026-09-07 (measured 2026-09-05):

| | Predicted by `budget.py` | Reported by vLLM |
|---|---|---|
| KV cache | 15.99 GiB | **16.08 GiB** |
| KV tokens (FP8) | 1,003,933 | **969,678** |
| concurrency at 262,144 | 3.83× | **3.70×** |

**`yarn512` at `KV_TARGET_GIB=20`, BF16 SSM** — the shipped configuration
(measured 2026-09-07, `REASONING_EFFORT=low`):

| | Predicted by `budget.py` | Reported by vLLM |
|---|---|---|
| KV cache | 16.72 GiB | **16.62 GiB** |
| KV tokens (FP8) | 1,108,533 | **1,179,259** |
| concurrency at 524,288 | 2.11× | **2.25×** |

**Memory is predicted to within 0.6% in both**, which is the figure that
matters: it is what `HOST_RESERVE_GIB` and the cgroup cap are sized against, so
`preflight.sh` can be trusted before a 12-minute load rather than after it.

The BF16 SSM multiplier is confirmed. This host's 15,132 bytes/token matches
upstream's independently measured 15,131.5 to three thousandths of a percent,
so `KV_SSM_BF16_MULT` is a real number rather than a borrowed one.

**Known limitation: `KV_BYTES_PER_TOKEN` is global, and bytes/token is not.**
The two launches above imply 17,805 bytes/token at 262K and 15,132 at 524K —
vLLM sizes the attention and mamba pages differently as `max_model_len` moves,
so one constant cannot fit both, and the token counts come out 3.5% high at
262K and 6.0% low at 524K. The error direction is safe (capacity is
under-promised, and the memory cap is unaffected), which is why it is recorded
here rather than patched over with a constant retuned for one context length.
Making it per-profile with both measured values is the fix.

### Throughput

| depth | TTFT | prefill tok/s | decode tok/s |
|---:|---:|---:|---:|
| 2,048 | 1.14 s | 1,792 | 35.5 |
| 8,192 | 3.95 s | 2,076 | 36.7 |
| 32,768 | 15.49 s | 2,116 | 37.9 |
| 131,072 | 65.51 s | 2,001 | 35.2 |

Prefill is essentially flat at ~2,000 tok/s from 8K to 131K, which is the
sparse-attention design doing its job. Upstream measured 1,769 tok/s at 32K on
their host at the time; after their PLE prefetch change they measure 2,314 at
32K and 1,944 at 256K.

**On real prose, decode is 27.1 tok/s single-stream** (from `scripts/smoke.py`,
14,364 tokens) — which matched what upstream reported at the time. The 35–38
tok/s in the table is higher because the ladder decodes with `ignore_eos` over
repeated filler, which is unusually easy for MTP to predict. Treat ~27 tok/s as
the number you will see *on this recipe as measured* and the ladder's decode
column as an upper bound.

Upstream now measures 48.7 tok/s on prose single-stream, and 162.9 aggregate at
8 streams, after the four changes in
[Decode optimisations](#decode-optimisations). Those changes are now in this
recipe but the 27.1 above was taken before them.

MTP acceptance measured ~2.46 tokens per streamed chunk (128 tokens in 52 SSE
chunks). Upstream measures 2.80 of a possible 4 post-optimisation, with
per-position acceptance 0.80 / 0.59 / 0.41.

### Two things the benchmark has to do, and why

Both cost real accuracy if skipped, and both are easy to get wrong:

- **Bust the prefix cache.** The server runs with prefix caching on and every
  depth is built from the same block of prose, so without a unique nonce per
  request the 8K prompt is a literal prefix of the 32K one and each repeat
  re-hits its own KV. That reported 39,206 tok/s at 131K against a true ~2,000.
- **Count tokens, not SSE chunks.** With MTP a single chunk carries ~2.5
  accepted tokens, so chunk-counting understated decode as 11.5 tok/s against a
  true 27–38. `bench_depths.py` and `smoke.py` both request
  `stream_options.include_usage` and use the engine's own count.

A third, smaller one: the model emits EOS on the first token when handed a long
prompt of repeated filler, so the ladder passes `ignore_eos` and `min_tokens` to
force a fixed decode length.

```bash
./bench.sh              # depth ladder against the running server
./bench.sh mixed        # a long prompt injected into live decode streams
./bench.sh mtp          # A/B MTP self-speculation (restarts twice)
./bench.sh kv           # A/B fp8 vs bf16 KV (restarts twice)
./bench.sh chunk        # A/B the prefill chunk width (restarts twice)
```

`mixed` is the case an agent harness actually produces and the other modes
miss: a 64K prompt arriving while earlier turns are still streaming. Chunked
prefill co-schedules that prompt's chunk with every live decode, so it is where
`MAX_NUM_BATCHED_TOKENS` shows up as latency. Its gap figures are **per engine
step, not per token** — with MTP one streamed chunk carries every token the
step accepted, and the harness prints the measured tokens-per-step next to them
so the two cannot be confused. Upstream's first published version of that table
labelled the gaps per token; `chunk` exists because lowering the chunk width to
1,024 was a near miss on their bar (p95 1.67×, p99 2.08×, for 5.5% of prefill
at 64K) and is very likely right for prefill-heavy traffic.

## Using it

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next",
       "messages":[{"role":"user","content":"Why is the sky blue?"}],
       "max_tokens":1024}'
```

Reasoning arrives in its own **`reasoning`** field, not in `content` —
`--reasoning-parser qwen3` is attached. (Note the name: this vLLM build uses
`reasoning`, where llama.cpp and older vLLM builds use `reasoning_content`.) Budget at least ~400
completion tokens or the trace consumes the whole allowance and `content` comes
back empty. Tool calling is enabled (`--tool-call-parser qwen3_coder`), and the
vision tower takes images and video with no extra GPU budget beyond the weights
already loaded.

### Thinking effort

The chat template takes a `reasoning_effort` of `low`, `medium` or `xhigh`, and
**defaults to `xhigh` when nothing sets it** — the shipped `start.sh` does not,
so every request arrives carrying *"Reasoning effort is set to xhigh. Please
think carefully through the task, validate key assumptions, consider plausible
alternatives..."* in its system message. That is the right default for hard
work and the wrong one for a classifier: the trace is billed to `max_tokens`
and to KV, and at `--max-num-seqs 4` a long trace on one sequence is a
noticeable share of the pool.

Pin it for the whole server at startup:

```bash
REASONING_EFFORT=low ./start.sh          # or medium, or xhigh
```

`start.sh` validates the value before launch and passes it as
`--default-chat-template-kwargs '{"reasoning_effort":"..."}'` — note the
`default-` prefix; `chat_template_kwargs` is the *request* field name and is not
a CLI flag, and vLLM rejects it at startup. The startup summary
prints the level in effect, or `xhigh (chat template default, not pinned)` when
the variable is unset. Validating early matters because the template raises on
an unknown effort *per request* — an unchecked typo would give you a server
that comes up healthy and fails every completion.

Or per request. Request-level `chat_template_kwargs` is merged over the server
default, so a client can still override a pinned level per call:

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next",
       "messages":[{"role":"user","content":"Classify: positive or negative?"}],
       "chat_template_kwargs":{"reasoning_effort":"low"},
       "max_tokens":1024}'
```

`{"enable_thinking":false}` in the same field turns thinking off outright — the
template then emits an empty `<think></think>` block and the model answers
directly. Note that `medium` is not a middle instruction but the *absence* of
one: it sets no reasoning text at all, leaving the model to its untuned
behaviour.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `preflight` fails on missing modules | The tag is mutable and you have a stock `vllm-openai`. `docker pull vllm/vllm-openai:qwen38-flash-next` |
| `preflight` fails on available memory, names another container | Another model server holds the pool. `docker stop <name>`, then re-run |
| `GPU is in use by something else` | Same, from the GPU side. `REQUIRE_IDLE_GPU=false` to override |
| Container OOM-killed by its cgroup cap | The host survived as designed. `MAX_MODEL_LEN=131072 ./start.sh` |
| Host hangs with no logs | The pool was exhausted. Raise `HOST_RESERVE_GIB` by 2 and apply the sysctl tunables |
| `content` empty, `reasoning` long | The trace used the whole budget. Raise `max_tokens`, or drop the effort: `REASONING_EFFORT=low ./start.sh` |
| `<think>` inline in `content` | The reasoning parser is not attached — check the image |
| Slow every launch | Check the torch.compile and flashinfer caches are mounted; `status.sh` shows the paths |
| `Overriding cudagraph_mode` in the log | The draft config copy downgraded the decode graphs to PIECEWISE. `VLLM_USE_V2_MODEL_RUNNER=1` is the default that prevents it — check you have not set `0`, and that `MTP_K_SCHEDULE` is empty |
| Decode slower than expected at one specific concurrency | A verify width with no CUDA graph, decoding eager. `CUDAGRAPH_CAPTURE_SIZES=auto` (the default) captures every width the scheduler can build; `preflight.sh` prints the list |
| `MTP_DRAFT_VOCAB is outside VLLM_CACHE_HOST` | It has to live in the vLLM cache tree, which is what the container can see. Put it under `~/.cache/vllm/draft_vocab/`, or build it with `./scripts/build_draft_vocab.sh` |
| Drafting seems not to use the reduced head | The patch only engages when `use_local_argmax_reduction` is set, which `start.sh` adds only alongside `MTP_DRAFT_VOCAB`. It is also TP=1 only |
| Fails in the first forward pass after setting the SSM dtype | The fused GDN kernel takes `float32` or `bfloat16` and nothing else. `MAMBA_SSM_CACHE_DTYPE=` for the checkpoint's own float32 |

Full log: `.vllm.log`. Watchdog and archived logs: `$OUT_DIR/logs/`.

## Differences from upstream

| | Upstream | Here |
|---|---|---|
| Config | `.env` file, required, with an environment/`.env` precedence dance | `profiles.sh`, sourced; three named profiles; no file to copy |
| Revision | `ls snapshots \| head -1` — whichever snapshot is on disk | pinned to `925d7be6c14c`, with a 51-file size + SHA-256 manifest |
| Download | `huggingface_hub` on the host, or inside the 20 GB image | `scripts/download_snapshot.py`, pure stdlib, resumable, verified before linking |
| Budget math | inline in `start.sh` | `scripts/budget.py`, so `preflight.sh` checks what `start.sh` launches |
| Preflight | none — `start.sh` does its own checks | `preflight.sh`, including the nine patch targets inside the image, the decode graph widths and the draft-vocabulary path |
| Caches | HF and vLLM only | HF, vLLM, flashinfer and triton, each at its own default |
| Logs | `logs/` inside the recipe | `.vllm.log` in the recipe; archives and watchdog logs in `$OUT_DIR` |
| Co-tenant guard | hardcoded for upstream's `comfy-h3.service` | generic: GPU compute apps, and other containers named on failure |
| Restart policy | — | `no` by default: a server that exhausts the pool should not restart into the same wall |
| Networking | `--network host` | bridge + `-p $HOST:$PORT:$PORT`, so `HOST=127.0.0.1` genuinely keeps it off the LAN. The PLE handshake uses POSIX shm via `--ipc host`, not the network, so this should be immaterial — but it is the one deviation to watch on first launch. |
| Draft vocabulary | its own `-v $MTP_DRAFT_VOCAB:/root/draft_vocab.txt:ro` mount | no mount: it lives under `VLLM_CACHE_HOST`, which is already mounted at vLLM's in-container default, and `start.sh` asserts the containment |
| V2 runner pin | a string inside `.env`'s `EXTRA_DOCKER_ARGS` | its own `VLLM_USE_V2_MODEL_RUNNER` variable, so setting the escape hatch cannot silently drop a safety default |
| Graph widths | inline shell + `awk` + `python3 -c` in `start.sh` | `scripts/graph_widths.py`, so `start.sh`, `preflight.sh` and this README quote one implementation |
| Mixed-traffic bench | `bench/mixed.py`, hardcoded to `localhost:8888`, unsalted prompts, gaps labelled per token | `scripts/bench_mixed.py` + `./bench.sh mixed`, parameterised, nonce-salted so repeats are not prefix-cache hits, and gaps labelled per **engine step** with the measured tokens-per-step beside them |
| Concurrency sweep | `bench/sweep.py`, drives sparkDash on `localhost:5555` | not ported — it depends on a tool that is not part of this recipe |
| `--no-launch` output | prints the resolved `HF_TOKEN` | token redacted; the output is meant to be pasted into bug reports |

`files/` is upstream's work, copied **verbatim** — the five patch generators,
the PLE table builder, the draft-vocabulary builder, the watchdog and the sysctl
file. `memwatch.sh` reads its log paths from the environment, which is how its
output lands in `$OUT_DIR` without editing it.

Upstream also ships a `CHANGELOG.md` and three `docs/` write-ups carrying the
measurement detail behind the numbers quoted here. They are not duplicated into
this repo; the [upstream repository](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
is the place to read them.

## License

**AGPL-3.0-or-later** — the same licence as upstream, and the only one this
port can carry.

```
Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)   — original work
Copyright (C) 2026 amarjeet                              — this port
```

The full text is in [`LICENSE`](LICENSE), byte-identical to upstream's. Every
file carries an `SPDX-License-Identifier`, and the files derived from upstream
also carry both copyright lines plus a note saying what was changed and when,
as AGPL §5(a) requires of a modified work.

### Is the port compatible with upstream's licence?

Yes, and not by choice — by necessity. The two licences are *the same licence*,
so there is no compatibility question to resolve:

| | Licence | Why |
|---|---|---|
| Upstream | AGPL-3.0-or-later | MiaAI Lab's choice |
| **This port** | **AGPL-3.0-or-later** | AGPL §5(c): a work derived from AGPL code must be released under the same licence |

`files/` is upstream's code verbatim, and `start.sh`, `stop.sh`, `download.sh`,
`profiles.sh`, `preflight.sh`, `scripts/budget.py`, `scripts/graph_widths.py`
and `scripts/bench_mixed.py` are modified or derived from it. That makes the directory a derivative work, and AGPL §5(c) leaves one
option: AGPL-3.0-or-later. Relicensing any of it as MIT would be a licence
violation, so this recipe ships its own `LICENSE` rather than inheriting the
repository's.

### Then why is the rest of the repo MIT?

Because that direction is fine. The relationship is one-way:

- **MIT → AGPL works.** MIT is permissive and GPL-compatible, so MIT code can
  be incorporated into an AGPL work.
- **AGPL → MIT does not.** Upstream's code cannot be relicensed MIT by us or
  by anyone else without MiaAI Lab's permission.

A repository may hold differently-licensed subdirectories: §5 treats a covered
work stored alongside separate, independent works as an "aggregate", and being
in an aggregate does not spread the AGPL to the sibling recipes. What it does require is that the boundary be unambiguous, so:
this directory has its own `LICENSE`, every file here is marked, and the
[repository README](../../README.md) names this recipe as the exception to its
MIT default.

**If you fork or redistribute:** take this directory as AGPL. Copying parts of
it into an MIT project is the one thing that is not permitted.

### The network clause

AGPL §13 is the reason upstream chose this licence, and it is worth being
precise about what it covers, because this recipe exists to run a server.

It applies to **these scripts**, not to the inference you serve. If you modify
the launcher, the patch generators or the watchdog and then offer that modified
version to users over a network, §13 obliges you to offer those users the
corresponding source of your modifications. Answering chat completions with the
model does not trigger it — the model server is vLLM, which is Apache-2.0.

### What this licence does not cover

Three things travel under their own terms, and nothing here relicenses them:

- **vLLM** — Apache-2.0, and *not redistributed here*. `start.sh` extracts the
  pristine `*.orig` sources from the container image at runtime and the patch
  generators emit modified copies onto your machine only; both are gitignored.
  Those generated files keep vLLM's own Apache-2.0 headers and stay Apache-2.0
  works. (Apache-2.0 → AGPL-3.0 is compatible in that direction anyway, so even
  if they were shipped there would be no conflict.)
- **The checkpoint** — `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` is governed by its
  own terms on the Hub, not by this repository's.
- **`files/patch_qsa_fp8_kv.py`** — implements an approach credited upstream to
  [lancelind/qwen3.8-Flash-DGX](https://github.com/lancelind/qwen3.8-Flash-DGX)
  (Apache-2.0), reimplemented by MiaAI Lab against this image's own sources.
