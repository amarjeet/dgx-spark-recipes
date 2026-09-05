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
  5.6 GiB overhead, 1.49 GiB MTP, 28.8 KiB/token KV, the 0.58 FP8 multiplier.

They also publish the [NVFP4 checkpoint](https://huggingface.co/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4)
itself. All five files under `files/` are theirs, copied verbatim.

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
| **GPU budget** | **94.92** | `gmu 0.780`, capped by `HOST_RESERVE_GIB=26` |
| **KV cache** | **16.08** | 969,678 FP8 tokens — 3.70× a full 262K request (measured) |
| container cgroup cap | 99 | bounds host-side memory only (ceiling 105) |

`./preflight.sh` prints this table for your actual `MemTotal` and profile before
anything long-running starts. The arithmetic lives in one place,
[`scripts/budget.py`](scripts/budget.py), so preflight and start cannot drift.

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

| Profile | Context | KV dtype | KV pool | Notes |
|---|---:|---|---|---|
| `native` *(default)* | 262,144 | FP8 | 969,678 tokens *(measured)* | the model's full native context |
| `yarn512` | 524,288 | FP8 | ~970K tokens | YaRN rope scaling, 2× native |
| `bf16kv` | 262,144 | BF16 | ~560K tokens | no quantized keys |

```bash
./start.sh yarn512
MAX_MODEL_LEN=65536 ./start.sh          # any profile, shorter context
MTP_NUM_SPECULATIVE_TOKENS=0 ./start.sh # give back 1.49 GiB
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

Four generators rewrite vLLM sources from the image's own copies on every
launch. `start.sh` extracts each `*.orig` from the container once, regenerates
the patched file, and bind-mounts it read-only over the original. Nothing is
baked into an image and nothing is vendored — if the image moves, the extraction
fails loudly, and `preflight.sh` checks all eight target paths by name before
you get there.

| Generator | Fixes |
|---|---|
| [`patch_ple_layer.py`](files/patch_ple_layer.py) | NVFP4/FP8 dispatch; offload rows must carry codes **and** scales |
| [`patch_ple_offload.py`](files/patch_ple_offload.py) | CUDA stream memory ops are unsupported on GB10 and deadlocked the GPU worker after graph capture; host-side handshake and `MADV_RANDOM` mmap instead |
| [`patch_modelopt_mxfp8.py`](files/patch_modelopt_mxfp8.py) | BF16 fallback for MXFP8 shapes the kernel rejects |
| [`patch_qsa_fp8_kv.py`](files/patch_qsa_fp8_kv.py) | quantized K/V tiles with per-tensor scales; compiles out at BF16 KV, so it is applied unconditionally |

## Measurements

Measured on this host on 2026-09-05: DGX Spark GB10, `aarch64`, 121.69 GiB
unified memory, kernel 6.17.0-1031-nvidia, vLLM `0.1.dev20073+g8e685d198`,
profile `native` (262,144 context, FP8 KV, MTP 3), idle server.

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

| | Predicted by `budget.py` | Reported by vLLM |
|---|---|---|
| KV cache | 15.99 GiB | **16.08 GiB** |
| KV tokens (FP8) | 1,003,933 | **969,678** |
| concurrency at 262,144 | 3.83× | **3.70×** |

Within 0.6% on memory and 3.5% on tokens, so `preflight.sh` can be trusted
before a 12-minute load rather than after it.

### Throughput

| depth | TTFT | prefill tok/s | decode tok/s |
|---:|---:|---:|---:|
| 2,048 | 1.14 s | 1,792 | 35.5 |
| 8,192 | 3.95 s | 2,076 | 36.7 |
| 32,768 | 15.49 s | 2,116 | 37.9 |
| 131,072 | 65.51 s | 2,001 | 35.2 |

Prefill is essentially flat at ~2,000 tok/s from 8K to 131K, which is the
sparse-attention design doing its job. Upstream measured 1,769 tok/s at 32K and
1,495 at 400K on their host.

**On real prose, decode is 27.1 tok/s single-stream** (from `scripts/smoke.py`,
14,364 tokens) — the same figure upstream reports. The 35–38 tok/s in the table
is higher because the ladder decodes with `ignore_eos` over repeated filler,
which is unusually easy for MTP to predict. Treat ~27 tok/s as the number you
will see and the ladder's decode column as an upper bound.

MTP acceptance measured ~2.46 tokens per streamed chunk (128 tokens in 52 SSE
chunks).

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
./bench.sh mtp          # A/B MTP self-speculation (restarts twice)
./bench.sh kv           # A/B fp8 vs bf16 KV (restarts twice)
```

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

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `preflight` fails on missing modules | The tag is mutable and you have a stock `vllm-openai`. `docker pull vllm/vllm-openai:qwen38-flash-next` |
| `preflight` fails on available memory, names another container | Another model server holds the pool. `docker stop <name>`, then re-run |
| `GPU is in use by something else` | Same, from the GPU side. `REQUIRE_IDLE_GPU=false` to override |
| Container OOM-killed by its cgroup cap | The host survived as designed. `MAX_MODEL_LEN=131072 ./start.sh` |
| Host hangs with no logs | The pool was exhausted. Raise `HOST_RESERVE_GIB` by 2 and apply the sysctl tunables |
| `content` empty, `reasoning_content` long | Raise `max_tokens`; the trace used the whole budget |
| `<think>` inline in `content` | The reasoning parser is not attached — check the image |
| Slow every launch | Check the torch.compile and flashinfer caches are mounted; `status.sh` shows the paths |

Full log: `.vllm.log`. Watchdog and archived logs: `$OUT_DIR/logs/`.

## Differences from upstream

| | Upstream | Here |
|---|---|---|
| Config | `.env` file, required, with an environment/`.env` precedence dance | `profiles.sh`, sourced; three named profiles; no file to copy |
| Revision | `ls snapshots \| head -1` — whichever snapshot is on disk | pinned to `925d7be6c14c`, with a 51-file size + SHA-256 manifest |
| Download | `huggingface_hub` on the host, or inside the 20 GB image | `scripts/download_snapshot.py`, pure stdlib, resumable, verified before linking |
| Budget math | inline in `start.sh` | `scripts/budget.py`, so `preflight.sh` checks what `start.sh` launches |
| Preflight | none — `start.sh` does its own checks | `preflight.sh`, including the eight patch targets inside the image |
| Caches | HF and vLLM only | HF, vLLM, flashinfer and triton, each at its own default |
| Logs | `logs/` inside the recipe | `.vllm.log` in the recipe; archives and watchdog logs in `$OUT_DIR` |
| Co-tenant guard | hardcoded for upstream's `comfy-h3.service` | generic: GPU compute apps, and other containers named on failure |
| Restart policy | — | `no` by default: a server that exhausts the pool should not restart into the same wall |
| Networking | `--network host` | bridge + `-p $HOST:$PORT:$PORT`, so `HOST=127.0.0.1` genuinely keeps it off the LAN. The PLE handshake uses POSIX shm via `--ipc host`, not the network, so this should be immaterial — but it is the one deviation to watch on first launch. |

`files/` is upstream's work, copied **verbatim** — the four patch generators,
the PLE table builder, the watchdog and the sysctl file. `memwatch.sh` reads its
log paths from the environment, which is how its output lands in `$OUT_DIR`
without editing it.

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
`profiles.sh`, `preflight.sh` and `scripts/budget.py` are modified or derived
from it. That makes the directory a derivative work, and AGPL §5(c) leaves one
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
