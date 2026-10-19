# Qwen3.8-Flash-Next · NVIDIA FP4 · TensorFold (Zig engine)

Serves [Qwen3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) from one DGX Spark through an
OpenAI-compatible API, with **five concurrent requests at the full 262,144-token window** (more with YaRN, see
[Context](#context)), image and video input, and MTP speculative decoding. It runs TensorFold's Zig engine
(`tensorfold-native`) in NVIDIA's PyTorch container, with MiaAI-Lab's patch series.

Port of [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)
at `656879f`. It **supersedes** [`Qwen3.8-Flash-Next-MLX4-TensorFold`](../Qwen3.8-Flash-Next-MLX4-TensorFold/), which
ported upstream's earlier Python TensorFold recipe; upstream moved to this engine and checkpoint at its 1.0.0.

| | |
|---|---|
| Checkpoint | [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) @ `b9ce18a6887e`: NVIDIA's quantization-aware-distilled FP4 export (NVFP4 routed experts, MXFP8 attention and linears, FP8 n-gram table, MTP head, vision tower). 33 files, 95.24 GiB, every one checksummed in `manifests/fp4.json` |
| Runtime | TensorFold `db281878ddb8` (branch `zig-flashnext`) + `patches/0001`-`0015`, Zig 0.17.0, on `nvcr.io/nvidia/pytorch:26.07-py3`, TP=1 AOT kernel set |
| Image | built locally by `build.sh` (about 25 GB), tagged `tensorfold-qwen38:zig-db28187-<hash>` |
| Endpoint | `:8011`, model id `qwen3.8-flash-next`, alias `Qwen3.8-Flash-Next`, bound to `0.0.0.0` |
| Modalities | text, images and video in; text out (and typed tool calls). No audio |

**This recipe, [`Qwen3.8-Flash-Next-NVFP4-vLLM`](../Qwen3.8-Flash-Next-NVFP4-vLLM/) and the superseded MLX4 recipe serve
the same model.** They cannot run together: each wants most of the unified pool. All answer to `qwen3.8-flash-next`,
so a client moves between them by changing the port.

## Quick start

```bash
cd dgx-spark-recipes/recipes/Qwen3.8-Flash-Next-FP4-TensorFold-Zig

./build.sh              # the image, from source (20+ minutes cold), then verified
./download.sh           # 95.24 GiB into the HF cache, pinned revision, checksum-verified, resumable
./preflight.sh          # image, snapshot, port, GPU tenancy, memory budget, serve arguments
./start.sh              # serves on :8011; runs preflight first
./bench.sh checks       # tool calls, image input, the 262,144-token window boundary
./status.sh
./stop.sh
```

`build.sh` and `download.sh` are separate on purpose: `start.sh` never starts a build or a 95 GiB download by itself.

```bash
curl -s http://<spark>:8011/v1/models
http GET http://<spark>:8011/v1/models          # httpie, from another machine

curl -s http://<spark>:8011/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 1000
}'
```

Images and video go in OpenAI's content parts, as data URLs (`VISION_URLS=1` also fetches `https://` ones). The model
thinks before it answers, into `reasoning_content`; a request without `max_tokens` gets 32,768 (`MAX_TOKENS`).
Per request, `"chat_template_kwargs": {"enable_thinking": false}` answers directly.

## Profiles

One checkpoint, so the profiles are KV-pool trade-offs.

| Profile | Streams x window | KV | Note |
|---|---|---|---|
| `fp8x5` | 5 x 262,144 | fp8 | **default**, as upstream; image and video input |
| `fp8x3` | 3 x 262,144 | fp8 | a smaller pool |
| `bf16x3` | 3 x 262,144 | bf16 | exact cache; one window is 7.5 GiB |
| `fp8x1` | 1 x 262,144 | fp8 | one stream |

`PARALLEL`, `CONTEXT` and `KV_DTYPE` override any profile, for example `PARALLEL=8 CONTEXT=131072 ./start.sh`. fp8 KV is
about 1.84x the pool of bf16 and lossy (upstream measured about 98.8% top-1 agreement with a bf16 cache); bf16 is exact.
Only `fp8x5` has been run on this host. The others are upstream's settings, not measured here.

## Context

The native window is 262,144 tokens. `CONTEXT` up to 1,048,576 turns on Qwen's YaRN (factor 4). One sequence's cache is
17,888 bytes a token in fp8 (4.37 GiB at 262,144; **17.46 GiB at 1M**) and 30,784 in bf16 (30.06 GiB at 1M). With vision
on, the pool this host measured after loading is 27.71 GiB, so **one 1M-token stream fits and a second does not**:
two need 34.9 GiB. `VISION=0` returns about 2 GiB to the pool, still one stream. A 1M window was **not run here**.
Slots grow on demand, so `PARALLEL` is a cap, not a reservation.

## Memory

The engine sizes its pool from `MemAvailable` after the weights load, less `TENSORFOLD_MEMORY_RESERVE_GIB` (10), and
refuses a request that would not fit. A start needs about 108 GiB of `MemAvailable`: 81 weights + one full window +
8 scratch + 2 vision workspace + 10 reserve + 2. `preflight.sh` itemises it and refuses to start below it.

On unified memory an exhausted pool hangs the kernel instead of raising an OOM, so two host-side guards sit around the
engine's own, as in the other TensorFold recipe: a **cgroup cap** of `CONTAINER_MEM_GIB` (24; GPU allocations are not
charged to it on GB10) and a **watchdog**, `scripts/memwatch.sh`, which stops the container when `MemAvailable` stays under
6 GiB, or `MemFree` under 2 GiB while `MemAvailable` is under 10, for five samples two seconds apart. The reserve sits
above the watchdog's floor, so the engine stops growing before the watchdog would fire.

## Measured on this host

GB10, 121.7 GiB unified memory. Image `tensorfold-qwen38:zig-db28187-69448ec006ba`. Profile `fp8x5`, 2026-10-18 and 19.

- **Start:** the server answered after 39 s, with the weights warm in the page cache. The first start also had the
  checkpoint cold on disk.
- **Pool:** the engine reported 27.71 GiB of sequence memory, 10.0 GiB kept free, fp8 KV at 1,056 bytes a position a layer.
- **Memory:** `MemAvailable` 39.1 GiB right after start; it later read 30.8 GiB with the watchdog's low mark at 29.3 GiB
  (host page cache; it is reclaimable). Container cgroup 3.1 GiB of 24.
- **Checks (`./bench.sh checks`):** `toolcheck.py` 3/3 (array parameter whole and streamed, and a tool-result turn);
  `visioncheck.py` passed (a red circle and a blue square recognised); `context_boundary.py` 16/16 at 262,144.

**Not measured here yet:** decode and prefill speed, concurrent decode, the 195k-token needle, exactness against
`"draft": false`, five concurrent long prompts, video input, 1M context, and running beside Clef-Flash. `./bench.sh all`
runs the first five. Upstream's own numbers on this checkpoint (its README, measured with sparkDash on a Spark):
prose decode 46 tok/s with one request to 132 with five, code 67 to 150, prefill 2,325 tok/s at 4k tokens and 1,947
at 128k. Those are upstream's, not this host's.

## Configuration

Every knob lives in `profiles.sh` and can be set from the environment.

| Variable | Default | Means |
|---|---|---|
| `PARALLEL` / `CONTEXT` / `KV_DTYPE` | per profile | streams, window per stream, `fp8` / `bf16` |
| `TF_FLASHNEXT_DEPTH` | `6` | MTP drafts a round (the engine's 15 costs 5 to 20% at 3 to 5 streams) |
| `TF_FLASHNEXT_PROMPT_EXPERT_TILE` | `2` | prompt-expert tile kernel (patch 0014): identical output, cold prefill +14% |
| `TF_FLASHNEXT_FILL_FAIR` | unset | `1`: fair sliced prefill (patch 0015), short requests do not wait behind long prompts; costs ~20% on a mixed load |
| `DRAFTS` | `1` | `0` turns MTP drafting off |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | `1.0` / `0.95` / `20` | Qwen's thinking-mode sampling; a request's own values win |
| `THINKING` / `MAX_TOKENS` | `1` / `32768` | thinking default; reply length for a request without `max_tokens` |
| `VISION` / `VISION_URLS` | `1` / `0` | image and video input; `1` also fetches `https://` URLs |
| `MAX_VIDEOS` / `TENSORFOLD_MAX_IMAGES` | `4` / `50` | per request; images share `TENSORFOLD_IMAGE_TOKENS` (16,384) |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | `10` | what the engine keeps out of `MemAvailable` |
| `PORT` / `HOST` | `8011` / `0.0.0.0` | published port and bind address; **there is no API key**, set `HOST=127.0.0.1` off a trusted network |
| `CONTAINER_MEM_GIB` | `24` | host-side cgroup cap |
| `MEMWATCH_MIN_GIB` / `MEMWATCH_MIN_FREE_GIB` | `6` / `2` | watchdog floors |
| `CUDA_CACHE_HOST` / `TRITON_CACHE_HOST` / `TENSORFOLD_CACHE_HOST` | `~/.nv` / `~/.triton` / `~/.cache/tensorfold` | host caches, mounted at each tool's in-container default |
| `EXTRA_SERVE_ARGS` | *(empty)* | appended to `tensorfold-native serve`; checked by the engine's own parser in preflight |

Any other `TENSORFOLD_*` or `TF_FLASHNEXT_*` variable in the environment is passed into the container.

## Port notes

- **Pinned revision and image inputs.** Upstream serves the repo id; `start.sh` passes the pinned snapshot directory, and
  `download.sh` verifies all 33 files against the manifest. The image tag hashes the patches, this recipe's Dockerfile
  and the build inputs, and `build.sh` verifies the labels and the binary.
- **Built locally, not pulled.** Upstream's prebuilt images follow its own Dockerfile and patch set, which differ from
  this recipe's (no recipe-private CUDA cache path, and patches 0014 and 0015 are unreleased upstream).
- **Caches at the tools' defaults.** Upstream keeps compiled kernels in `~/.cache/tensorfold-qwen38`. Here the CUDA JIT
  cache is `~/.nv`, triton's is `~/.triton`, and the kernel set is built into the image.
- **Host-side guards.** Upstream trusts the engine's admission alone; this adds a cgroup cap and a watchdog.
- **Port 8011 and bridge networking.** Upstream uses `--network host` on 8888, which the vLLM recipe already claims.
- **Setup split out.** Upstream's `start.sh` runs `prepare.sh` on demand. Here that is `build.sh` and `download.sh`, and
  `start.sh` runs `preflight.sh` before it stops or loads anything.
- **One checkpoint.** Upstream's optional INT4-AutoRound (`CHECKPOINT=int4`) is not carried.
- **Dropped:** the terminal banner, `publish-image.sh`, the `.github` templates, `./.env` parsing, upstream's tests of its
  own `start.sh`, and the draft-language image (gone upstream too).

## Known gaps

- **Little measured here.** See [Measured on this host](#measured-on-this-host).
- **Clef-Flash co-hosting is untested.** The weights take about 81 GiB, so the old `int8x1` co-tenant arrangement
  does not carry over, and the other recipes' co-tenant notes still describe the MLX4 recipe.
- **The image is a third party's source.** It is built from TensorFold at a pinned commit plus MiaAI-Lab's patches; the
  build verifies labels and the binary, not the patched source.
- **No API key.** TensorFold serves without auth.

## License

MIT, like upstream; [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) carry upstream's notices and TensorFold's. TensorFold is
Apache-2.0 from 0.6.0, so [`LICENSE.Apache-2.0`](LICENSE.Apache-2.0) ships too. The weights are not part of this
repository: NVIDIA Open Model Agreement, plus the base model's Qwen Community License 1.0. The image is based on NVIDIA's
PyTorch container, whose terms the container prints at every start; building or running it accepts them.
See [`CREDITS.md`](CREDITS.md).
