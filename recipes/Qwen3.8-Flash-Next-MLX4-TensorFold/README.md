# Qwen3.8-Flash-Next · MLX 4-bit · TensorFold

Serves [Qwen3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) from one DGX Spark through an
OpenAI-compatible API, with **four concurrent requests at the full 262,144-token window**, or upstream's five. It runs
[TensorFold](https://github.com/ashhart/TensorFold) v0.3.6.2 in NVIDIA's PyTorch container, with eight patches that
change speed and never an output token.

Port of [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)
at `4cd9956`. What changed in the port is under [Port notes](#port-notes).

| | |
|---|---|
| Checkpoint | [`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP) @ `dadefa8066e3`, MLX 4-bit, group size 32, with the MTP draft head. 35 files, 105.46 GiB, every one checksummed in `manifests/mlx4.json` |
| Runtime | TensorFold v0.3.6.2 (`71377a5`) + `patches/*.patch` (hash `82e893ed2bcc`) on `nvcr.io/nvidia/pytorch:26.07-py3` |
| Image | upstream's prebuilt image pinned by digest, or built locally; verified either way |
| Endpoint | `:8011`, model id `qwen3.8-flash-next`, alias `Qwen3.8-Flash-Next` |
| KV pool | 4 streams x 262,144 = 1,048,576 tokens, int8 KV (default here); upstream's 5 streams = 1,310,720 is `int8x5` |

**This recipe and [`Qwen3.8-Flash-Next-NVFP4-vLLM`](../Qwen3.8-Flash-Next-NVFP4-vLLM/) serve the same model** from
different checkpoints and engines. They cannot run at the same time: each wants most of the unified pool. Both answer
to `qwen3.8-flash-next`, so a client moves between them by changing the port.

## Quick start

```bash
cd dgx-spark-recipes/recipes/Qwen3.8-Flash-Next-MLX4-TensorFold

./build.sh              # the image: pinned prebuilt pull (~11 GB), or PULL=0 to build; then verified
./download.sh           # 105.46 GiB into the HF cache, pinned revision, checksum-verified, resumable
./preflight.sh          # image + patches, snapshot, port, GPU tenancy, memory budget, serve arguments
./start.sh              # serves on :8011; runs preflight first
./scripts/smoke.py      # arithmetic, thinking, typed tool calls, live counters
./status.sh
./stop.sh
```

`build.sh` and `download.sh` are separate on purpose: `start.sh` never starts an 11 GB pull or a 105 GiB download
by itself.

```bash
curl -s http://<spark>:8011/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 1000
}'
```

The model thinks before it answers, into `reasoning_content`, so give replies room in `max_tokens`. Per request,
`"chat_template_kwargs": {"enable_thinking": false}` answers directly and `{"reasoning_effort": "low"}` (or `xhigh`)
sets the effort. A top-level OpenAI `reasoning_effort` field is ignored.

## Profiles

One checkpoint, so the profiles are KV-pool trade-offs. TensorFold gives every stream a full window, so the pool is
streams x window. The estimate column is TensorFold's own startup estimate as upstream reported it; the server
prints the real one at every start.

| Profile | Streams x window | KV | Pool | Estimate | Note |
|---|---|---|---:|---:|---|
| `int8x4` | 4 x 262,144 | int8 | 1,048,576 | 97.8 GiB | **default here**; see [Memory](#memory) |
| `int8x5` | 5 x 262,144 | int8 | 1,310,720 | 102.6 GiB | upstream's default; tight on this host |
| `int4x6` | 6 x 262,144 | int4 | 1,572,864 | 97.7 GiB | int4 changes output; quality not measured |
| `bf16x3` | 3 x 262,144 | bf16 | 786,432 | 102.1 GiB | full-precision KV |

`PARALLEL`, `CONTEXT` and `KV_DTYPE` override any profile, for example `PARALLEL=6 CONTEXT=220000 ./start.sh`.
TensorFold refuses a setting that does not fit before any weights load, and names a window that would.

## Memory

TensorFold sizes itself. Its budget is `MemAvailable` minus the larger of 4 GiB and a tenth of `MemTotal`, read at
start, and there is no knob for that reserve. On this host a tenth is 12.2 GiB, so an idle Spark gives a budget of
about 104 to 106 GiB. TensorFold refuses a profile whose estimate does not fit, before any weights load.

That leaves the host roughly the reserve and no more, and on unified memory an exhausted pool hangs the kernel
instead of raising an OOM. So the recipe adds two host-side guards around TensorFold's admission:

- **A cgroup cap** of `CONTAINER_MEM_GIB`, 24 GiB. GPU allocations are not charged to it on GB10, so it bounds the
  host-side footprint only. The measured peak was 12.0 GiB across the `int8x5` load and
  benchmarks, and 6.6 GiB on `int8x4`.
- **A watchdog**, `scripts/memwatch.sh`. It stops the container when `MemAvailable` stays under 6 GiB, or `MemFree`
  stays under 2 GiB while `MemAvailable` is under 10 GiB, for five samples two seconds apart. Its log is in
  `$OUT_DIR/logs/`.

**Why the default is `int8x4`, not upstream's `int8x5`.** Measured on this host, from a fresh start through the
benchmarks below:

| Profile | Startup estimate | Lowest MemAvailable | Watchdog low samples |
|---|---:|---:|---|
| `int8x5` | 102.60 of 105.71 GiB | 7.7 GiB | `MemFree` under 2 GiB for 3 of 5 consecutive samples |
| `int8x4` | 97.78 of 104.20 GiB | 17.9 GiB | none |

Both dips came during the KV allocation at load, not under traffic. `int8x5` ran every benchmark here, but it
operates within about a gigabyte of the point where the NVIDIA driver starts refusing allocations. If the watchdog
trips during an `int8x5` load, the start fails and the host is fine. Use `./start.sh int8x5` when the fifth stream
matters more than the margin.

## Measured on this host

GB10, 121.7 GiB unified memory, kernel driver 580.173.02 (the container runs CUDA 13.3 in forward-compatibility
mode). Image `tensorfold-qwen38:v0.3.6.2-82e893ed2bcc`, pulled by digest and verified. Nothing else on the GPU.
Measured through the OpenAI API with `./bench.sh`, on 2026-09-29.

**Startup.** The first start compiles CUDA kernels: 239 s to ready, 80 to 90 s of it kernel warm-up. A warm start
reuses `~/.triton` and `~/.cache/torch_extensions` and takes 149 to 151 s, with kernels warmed in 3.9 s.

**Prefill**, `int8x5`, fresh random prose per run so the prompt cache never hits:

| Prompt | Prefill | Rate | TTFT |
|---:|---:|---:|---:|
| 855 | 0.50 s | 1,724 tok/s | 0.50 s |
| 3,202 | 1.47 s | 2,176 tok/s | 1.48 s |
| 12,642 | 5.09 s | 2,483 tok/s | 5.12 s |
| 50,355 | 20.77 s | 2,425 tok/s | 20.86 s |

This matches upstream's ~2,340 to 2,480 tok/s over 3k to 50k tokens.

**Needle.** A passphrase at 60% depth of a 194,893-token prompt, asked greedily, was recalled on both profiles.
Prefill took 96.0 s on `int8x5` and 93.9 s on `int8x4`. Upstream reports ~97 s at ~195k.

**Decode, single request:** 71.7 tok/s on greedy code and 59.0 tok/s on sampled chat, medians of five, against
upstream's 62.4 tok/s on prose.

**Decode, concurrent** (`scripts/bench_concurrent.py`: 512-token essays, thinking off, aggregate over first to last
token):

| Requests | `int8x5` aggregate | `int8x5` per request | `int8x4` aggregate | `int8x4` per request | Upstream, `int8x5` |
|---:|---:|---:|---:|---:|---:|
| 1 | 54.4 | 54.4 | 54.4 | 54.4 | 62.4 |
| 2 | 75.0 | 38.2 | 74.5 | 38.0 | 90.5 |
| 4 | 89.4 | 22.9 | 98.2 | 25.2 | 106.7 |
| 5 | 100.6 | 21.5 | 85.9 | 25.1 | 119.3 |

These run 12 to 18% under upstream's. The workloads differ: upstream's prompts are not published, and MTP draft
acceptance depends on the text, so this gap is not evidence of a slower host. On `int8x4` a fifth request waits for a
stream, which is why its 5-request aggregate falls.

## Configuration

Every knob lives in `profiles.sh` and can be set from the environment.

| Variable | Default | Means |
|---|---|---|
| `PARALLEL` / `CONTEXT` / `KV_DTYPE` | per profile | streams, window per stream, `bf16` / `int8` / `int4` |
| `PLE_ON_SSD` | `1` | read the 29.8 GiB n-gram tables from SSD at each lookup instead of holding them in memory |
| `MTP_DRAFTS` / `MTP_CONFIDENCE` | `6` / `0.60` | at most 6 MTP drafts a round; a chain stops before a draft under 60% |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | `1.0` / `0.95` / `20` | Qwen's thinking-mode sampling; a request's own values win |
| `THINKING` | `1` | `0` answers directly unless a request asks to think |
| `REASONING_EFFORT` | `medium` | TensorFold's server default, which adds no system-prompt text |
| `TENSORFOLD_PREFILL_ROWS` | `4096` | rows per prompt chunk (patch 0007); TensorFold's own default is 2048 |
| `TENSORFOLD_MTP_COPY` | `1` | prompt-lookup drafts for replies that repeat the prompt (patch 0008; needs 2+ streams) |
| `EXTRA_SERVE_ARGS` | *(empty)* | appended to `tensorfold serve`, so they win; checked by TensorFold's own parser in preflight |
| `PORT` / `HOST` | `8011` / `0.0.0.0` | published port and bind address; there is no API key |
| `CONTAINER_MEM_GIB` | `24` | host-side cgroup cap |
| `MEMWATCH_MIN_GIB` / `MEMWATCH_MIN_FREE_GIB` | `6` / `2` | watchdog floors |
| `PULL` | `1` | `0` makes `build.sh` build locally instead of pulling |
| `TRITON_CACHE_HOST` / `TORCH_EXTENSIONS_HOST` / `TENSORFOLD_CACHE_HOST` | `~/.triton` / `~/.cache/torch_extensions` / `~/.cache/tensorfold` | host caches, mounted at each tool's in-container default |

Any other `TENSORFOLD_*` or `TF_*` variable in the environment is passed into the container, as upstream does.

## What the patches change

Upstream's eight patches, carried byte for byte (unified diffs against TensorFold's site-packages, applied with
`patch -p0`). `build.sh` proves each one is present in the image by reverse-applying it in dry-run mode, rather than
trusting the image label.

| Patch | Change | Effect upstream measured |
|---|---|---|
| `0001-cuda-typed-tool-parameters` | tool-call arguments decoded by the tool's JSON schema | arrays, numbers and objects arrive typed ([TensorFold #75](https://github.com/ashhart/TensorFold/pull/75)) |
| `0002-cuda-live-token-counters` | `/health` reports live token totals | monitoring ([#79](https://github.com/ashhart/TensorFold/pull/79)) |
| `0003-flash-next-ssd-read-ahead` | a chunk's n-gram rows read from SSD while the GPU runs the previous chunk | multi-chunk prefill +50% |
| `0004-flash-next-ssd-native-reader` | those reads on a C++ thread pool outside the GIL | short-prompt TTFT -35%, decode +4% |
| `0005-flash-next-qsa-tiled-select` | sparse-attention block selection no longer spills registers past 128k | 149k prompts 25% faster |
| `0006-cuda-stream-draft-stats` | `drafted` / `accepted` counts in concurrent requests' stats | observability |
| `0007-flash-next-prefill-rows` | configurable prompt chunk size (port of [#40](https://github.com/ashhart/TensorFold/pull/40)) | +2-5% at 4,096 rows |
| `0008-flash-next-copy-drafts` | drafts copied from earlier text when the reply repeats the prompt | +6% on quoting replies |

Upstream checked that outputs are unchanged by comparing reply hashes against unpatched TensorFold, sampled and
greedy, up to 149k tokens. Any request can send `"draft": false` for the serial one-token-a-round reference.

## Port notes

- **Pinned revision.** Upstream serves the repo id, which resolves to the newest snapshot on disk. `start.sh` passes
  the pinned snapshot directory, and `download.sh` verifies all 35 files against the manifest: SHA-256 for the
  weights, the git blob id for the small files.
- **Pinned image.** The prebuilt image is pulled by digest, since a tag can be re-pushed. A local build installs
  TensorFold at the release commit, not the tag. Either result must carry the right label, report 0.3.6.2, and
  contain every patch.
- **Caches at the tools' defaults.** Upstream keeps compiled kernels in a recipe-private `~/.cache/tensorfold-qwen38`.
  Here triton's cache is `~/.triton` and torch's extension cache is `~/.cache/torch_extensions`, mounted at their
  in-container defaults and shared with the other recipes. The HF cache is mounted read-only.
- **Host-side guards.** Upstream trusts TensorFold's admission alone. This adds a cgroup cap and a memory watchdog, as
  the repo's other self-budgeting recipe does.
- **Port 8011 and bridge networking.** Upstream uses `--network host` on 8888, which the vLLM recipe for this model
  already claims.
- **Setup split out.** Upstream's `start.sh` runs its setup on demand. Here that is `build.sh` and `download.sh`, and
  `start.sh` runs `preflight.sh` before it stops or loads anything.
- **Dropped:** the terminal banner, `publish-image.sh` (pushing to upstream's registry is upstream's job), and the
  `.github` templates.

## Known gaps

- **Concurrent throughput is under upstream's**, by 12 to 18%, on a different workload. Not reproduced with
  upstream's own prompts, which are not published.
- **`int8x5` is tight on this host.** See [Memory](#memory). Five concurrent long requests were not run here;
  upstream reports the host keeping at least 9.7 GiB free through them.
- **`int4x6` and `bf16x3` were not run here.** Their estimates are upstream's, and int4 KV quality is unmeasured
  anywhere.
- **The image is a third party's.** It is pinned by digest and verified: the right label, TensorFold 0.3.6.2, and all
  eight patches present. Once the patches are undone, the package matches the v0.3.6.2 source exactly. That was
  checked once by hand, not on every build. `PULL=0 ./build.sh` builds the same image locally instead.
- **No API key.** TensorFold serves without auth. Set `HOST=127.0.0.1` off a trusted network.

## License

MIT, like upstream. [`LICENSE`](LICENSE) carries upstream's notice and TensorFold's, which covers the patches. The
weights are not part of this repository and are under the Qwen Community License 1.0.

The image is based on NVIDIA's PyTorch container, whose NVIDIA software is governed by the
[NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and the [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/).
The container prints them at every start; pulling or running it accepts them.

## Credits

The recipe is MiaAI Lab's ([@MiaAI_lab](https://x.com/MiaAI_lab)): the patches, the settings and every upstream
measurement quoted here. It runs on [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart and its
contributors, [Qwen3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by Qwen, and
[Vontra's MLX 4-bit checkpoint](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP). Patch 0007 ports
TensorFold #40 by MovieMaker93. Upstream's
[CREDITS.md](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/main/CREDITS.md) has the
full list.
