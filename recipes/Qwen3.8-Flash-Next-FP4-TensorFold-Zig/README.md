# Qwen3.8-Flash-Next · NVIDIA FP4 · TensorFold (Zig engine)

Serves [Qwen3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) from one DGX Spark through an
OpenAI-compatible API, with **five concurrent requests at the full 262,144-token window**, image input (video and PDF
opt-in), structured output, and MTP speculative decoding. It runs TensorFold's Zig engine (`tensorfold-native`,
release v1.2.7) in NVIDIA's PyTorch container, with MiaAI-Lab's patches. `ABLIT=1` serves the same checkpoint with an
abliteration overlay grafted in (see [ABLIT](#ablit-the-abliterated-variant)).

Port of [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)
at `b956dba` (2026-11-18). The previous version of this recipe ported `656879f` (TensorFold `db281878` with patches
0001 to 0015); see [Since the previous version](#since-the-previous-version). It **supersedes**
[`Qwen3.8-Flash-Next-MLX4-TensorFold`](../Qwen3.8-Flash-Next-MLX4-TensorFold/), which ported upstream's earlier Python
TensorFold recipe.

| | |
|---|---|
| Checkpoint | [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) @ `b9ce18a6887e`: NVIDIA's quantization-aware-distilled FP4 export (NVFP4 routed experts, MXFP8 attention and linears, FP8 n-gram table, MTP head, vision tower). 33 files, 95.24 GiB, every one checksummed in `manifests/fp4.json` |
| Runtime | TensorFold v1.2.7 (`93a2e93ecebf`) + `patches/0001`-`0005`, Zig 0.17.0, on `nvcr.io/nvidia/pytorch:26.07-py3`, kernels embedded in the binary |
| Image | built locally by `build.sh`, tagged `tensorfold-qwen38:v1.2.7-93a2e93-<hash>`; carries libjpeg-turbo, libwebp's decoder, PDFium, libtfvideo (FFmpeg 9.0.2) and libtfgrammar, each from a pinned SHA-256 |
| Endpoint | `:8011`, model id `qwen3.8-flash-next`, alias `Qwen3.8-Flash-Next`, bound to `0.0.0.0` |
| Modalities | text and images in (video with `VIDEO=1`, PDF with `PDF=1`); text out, typed tool calls, JSON-schema output. No audio |

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

The model thinks before it answers, into `reasoning_content`; a request without `max_tokens` gets 32,768 (`MAX_TOKENS`).
Per request, `"chat_template_kwargs": {"enable_thinking": false}` answers directly. `response_format` with a JSON schema
and `tool_choice: "required"` (or a named function) are enforced by the engine's grammar library, libtfgrammar.

## Images, video and PDF

- **Images** (`VISION=1`, default): OpenAI's `image_url` content parts, as data URLs (`VISION_URLS=1` also fetches public
  `https://` ones). The engine decodes JPEG and WebP itself and runs the Qwen-VL tower on the GPU; up to 50 images
  (`TENSORFOLD_MAX_IMAGES`) sharing 16,384 image tokens a request. `VISION=0` is text only and returns 2 GiB to the pool.
- **Video** (`VIDEO=1`, off by default): `video_url` parts, MP4/MOV and WebM with H.264, HEVC, VP8, VP9 or MPEG-4. Sampled at
  2 frames a second, two frames a group; each group counts as one image, so the defaults hold about 50 seconds of footage a
  request. The decoder (`libtfvideo.so` over FFmpeg) is in the image but off the loader's path unless `VIDEO=1`.
  Upstream added it in patch 0005 and has not run a live video request; **not run here either**.
- **PDF** (`PDF=1`, off by default): Anthropic's `document`, OpenAI's `file` / `input_file`, or a
  `data:application/pdf;base64,` `image_url`, rendered to page images by the engine's PDFium. 32 MiB and 16 pages a document
  (`TENSORFOLD_MAX_PDF_PAGES` lowers it); the pages share the image budget. **Not run here.**

Both need `VISION=1`; `profiles.sh` refuses `VIDEO=1` or `PDF=1` with `VISION=0`.

## ABLIT: the abliterated variant

`ABLIT=1` serves NVIDIA's checkpoint with drowzeys's Keys "house-qsa" overlay
([`drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-FP4-QAD-house-qsa-L3-47`](https://huggingface.co/drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-FP4-QAD-house-qsa-L3-47)
@ `c44f5b92`) grafted in. **The model has had its safety refusals removed.** The overlay replaces the `self_attn.o_proj` weight and scale of
the 12 QSA layers (3, 7, ... 47; MXFP8 g32, requantized after a rank-1 refusal-direction projection) with its
~186 MiB sidecar. GDN `out_proj`, every expert, the MTP head, the n-gram table, vision and the chat template stay NVIDIA's.

The overlay is **gated** (free, approved automatically): accept its Responsible Use Agreement on the Hub while logged in,
with the account whose token is in `HF_TOKEN` or `~/.cache/huggingface/token`. Then:

```bash
scripts/make_manifest.py drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-FP4-QAD-house-qsa-L3-47 \
  c44f5b92ee3d307b82aab1c55f4cc4ae1d440344 manifests/ablit.json \
  --include fp4-local/house-qsa-oproj-mxfp8-L3-47.safetensors   # once; pins the sidecar's SHA-256
./download.sh --ablit       # the sidecar into the HF cache, verified, then the grafted folder
ABLIT=1 ./start.sh restart
./status.sh                 # "weights: ABLIT=1, ..." while it serves the overlay
```

The manifest has to be made after the gate is accepted: until then the Hub withholds the sidecar's SHA-256, and
`make_manifest.py` refuses a listing without it. `scripts/ablit_view.py` (upstream's, unchanged) builds the folder the
engine reads, `~/.cache/huggingface/tensorfold-qwen38/ablit-b9ce18a6-c44f5b92`: relative symlinks to every file of the stock
snapshot, the sidecar as `tensors/house-qsa-...` (which sorts before the stock shards), and an index that points the 24
tensor names at it. Nothing in either snapshot is copied or changed, and `ABLIT=0` (the default) serves the stock
weights from the same cache. It fits only checkpoint revision `b9ce18a6`; `profiles.sh` refuses any other.
**Run here** (2026-11-20, `fp8x5`): the server answered in 42 s with the same 30.9 GiB pool as stock. `./bench.sh checks` passed
(tool calls 3/3, vision, window boundary 16/16), and `./bench.sh exact` passed (drafted, non-drafted and concurrent replies
identical). Five of its six greedy replies differ from stock's, so the grafted weights are the ones served. Refusal
behaviour and quality were not evaluated here; the overlay author's own checks ran on another stack.

## Profiles

One checkpoint, so the profiles are KV-pool trade-offs.

| Profile | Streams x window | KV | Note |
|---|---|---|---|
| `fp8x5` | 5 x 262,144 | fp8 | **default**, as upstream |
| `fp8x3` | 3 x 262,144 | fp8 | a smaller pool |
| `bf16x3` | 3 x 262,144 | bf16 | exact cache; one window is 7.5 GiB |
| `fp8x1` | 1 x 262,144 | fp8 | one stream |

`PARALLEL`, `CONTEXT` and `KV_DTYPE` override any profile, for example `PARALLEL=8 CONTEXT=131072 ./start.sh`. fp8 KV is
about 1.84x the pool of bf16 and lossy (upstream measured about 98.8% top-1 agreement with a bf16 cache); bf16 is exact.
`ABLIT`, `VIDEO` and `PDF` combine with any profile.

## Context

**262,144 tokens is the maximum.** The Flash Next CUDA engine of TensorFold 1.2.7 refuses a scaled rope, so the YaRN
extension to 1,048,576 the previous build offered is gone; `profiles.sh` refuses a larger `CONTEXT`. One sequence's cache
is 17,888 bytes a token in fp8 (4.37 GiB at 262,144) and 30,784 in bf16 (7.52 GiB). Slots grow on demand, so `PARALLEL` is
a cap, not a reservation.

## Memory

The engine sizes its pool from `MemAvailable` after the weights load, less `TENSORFOLD_MEMORY_RESERVE_GIB` (9, down from 10:
upstream measured that five full windows need 23.9 GiB of pool), and refuses a request that would not fit. A start needs
about 107 GiB of `MemAvailable`: 81 weights + one full window + 8 scratch + 2 vision workspace + 9 reserve + 2.
`preflight.sh` itemises it and refuses to start below it.

On unified memory an exhausted pool hangs the kernel instead of raising an OOM, so two host-side guards sit around the
engine's own, as in the other TensorFold recipe: a **cgroup cap** of `CONTAINER_MEM_GIB` (24; GPU allocations are not
charged to it on GB10) and a **watchdog**, `scripts/memwatch.sh`, which stops the container when `MemAvailable` stays under
6 GiB, or `MemFree` under 2 GiB while `MemAvailable` is under 10, for five samples two seconds apart. The reserve sits
above the watchdog's floor, so the engine stops growing before the watchdog would fire.

## Performance

### Measured on this host

GB10, 121.7 GiB unified memory. Image `tensorfold-qwen38:v1.2.7-93a2e93-4e8f162f81db`, profile `fp8x5` (defaults), 2026-11-20.
`./bench.sh streams` (`tools/bench_streams.py`: greedy, 256 tokens, median of 3, every stream the same prompt):

| | 1 stream | 3 streams | 5 streams |
| --- | ---: | ---: | ---: |
| Decode prose, total tok/s (TTFT) | 57.9 (60 ms) | 153.9 (124 ms) | 225.9 (186 ms) |
| Decode code, total tok/s (TTFT) | 50.2 (60 ms) | 132.6 (121 ms) | 196.7 (183 ms) |

| Prompt tokens | 3,985 | 7,912 | 15,775 | 31,471 | 62,892 | 125,728 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill, tok/s | 2,085 | 2,133 | 2,165 | 2,180 | 2,146 | 2,072 |

A second run at 4k / 31k / 126k read 2,266 / 2,226 / 2,056 tok/s, and the same with `TF_FLASHNEXT_PROMPT_MX128=0` (patch 0004 off)
2,133 / 2,058 / 1,924: **the prompt tile is worth +6 to +8% here**, as upstream found. Run to run, prefill moves by about 5%.

- **Start:** the server answered after 42 s (weights warm in the page cache); smoke request OK.
- **Pool:** the engine reported 74.81 GiB of model, 30.8 GiB left after the 9.0 GiB reserve, 4.83 GiB a stream at a full
  262,144-token fp8 window, five streams, a 6.7 GiB prompt cache. `MemAvailable` 39.4 GiB right after start; cgroup 2.5 GiB of 24.
- **Checks (`./bench.sh checks`):** `toolcheck.py` 3/3, `visioncheck.py` passed, `context_boundary.py` 16/16 at 262,144.
- **Exactness (`./bench.sh exact`):** six greedy replies identical with drafts on (557 drafts accepted), with `"draft": false`,
  and six at once against one at a time.

Against upstream's A/B below, decode is within 1% (prose) to 5% (code) of its v1.2.7 figures, and prefill is 5 to 10% under
its v1.2.7 + 0004 figures. **Not measured here:** video, PDF, ABLIT speed, the 195k-token needle, and five concurrent long prompts.

### Upstream

Upstream's A/B of its v1.1.4 and v1.2.7 builds (`tools/bench_streams.py`, same Spark, defaults, greedy, 256 tokens,
median of 3, 2026-11-12/13):

| | v1.1.4 build | v1.2.7 build | Change |
| --- | ---: | ---: | ---: |
| Decode prose, 1 / 3 / 5 streams (total tok/s) | 57.8 / 133.4 / 167.9 | 58.5 / 156.1 / 230.1 | +1% / +17% / +37% |
| Decode code, 1 / 3 / 5 streams (total tok/s) | 48.2 / 104.0 / 141.3 | 51.9 / 138.8 / 206.8 | +8% / +33% / +46% |
| TTFT at 5 streams, prose / code (ms) | 440 / 372 | 177 / 173 | -60% / -53% |
| Prefill 4k / 31k / 126k tokens (tok/s), with patch 0004 | 2,329 / 2,566 / 2,397 | 2,303 / 2,443 / 2,293 | -1% / -5% / -4% |

Upstream also measured, with sparkDash, its v1.1.4 build ahead of the `db281878` build this recipe used to run: +18% to
+45% aggregate decode at five requests. Speed on the previous build was never measured on this host, so there is no
local before/after; the previous build started in 39 s here with a 27.71 GiB pool (10 GiB reserve) and passed the same checks.

## Configuration

Every knob lives in `profiles.sh` and can be set from the environment.

| Variable | Default | Means |
|---|---|---|
| `PARALLEL` / `CONTEXT` / `KV_DTYPE` | per profile | streams (1 to 16), window per stream (up to 262,144), `fp8` / `bf16` |
| `ABLIT` | `0` | `1`: the Keys house-qsa abliteration overlay ([ABLIT](#ablit-the-abliterated-variant)) |
| `VISION` / `VISION_URLS` | `1` / `0` | image input; `1` also fetches public `https://` URLs |
| `VIDEO` / `PDF` | `0` / `0` | video parts; PDF document parts. Both need `VISION=1` |
| `MAX_VIDEOS` / `TENSORFOLD_MAX_IMAGES` | `4` / `50` | per request; images share `TENSORFOLD_IMAGE_TOKENS` (16,384), video `TENSORFOLD_VIDEO_TOKENS` (16,384) |
| `DRAFTS` / `TF_FLASHNEXT_DRAFTS` | `1` / `6` | MTP drafting on; at most 6 drafts a round |
| `TF_FLASHNEXT_CONFIDENCE` / `TF_FLASHNEXT_SHARED_DRAFTS` | `-0.4` / `0` | a draft chain stops when the product of its probabilities falls under 0.4, beside other streams too |
| `TF_FLASHNEXT_DRAFT_HEAD` | `affine4` | the MTP draft head in 4 bits; `bf16` is the exact head (drafts only, replies unchanged) |
| `TF_FLASHNEXT_CHUNK` | `3072` | prompt rows a pass (the engine's 1,536 prefills ~4% slower) |
| `TF_FLASHNEXT_PROMPT_MX128` | `1024` | patch 0004's MXFP8 prompt tile for projections this wide; `0` keeps the exact kernels |
| `TF_FLASHNEXT_PROMPT_EXPERT_TILE` / `TF_FLASHNEXT_WARM_AHEAD` | `2` / `1` | NVFP4 prompt experts on K32 tiles; n-gram rows read ahead (opt-in in the engine, on here) |
| `TF_FLASHNEXT_SAMPLED_HEADS` / `_KEEP_LATE` / `_SHARED_SIDE` / `_GDN_LANES` | `1` | already the engine's defaults, listed so they can be turned off |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | `1.0` / `0.95` / `20` | Qwen's thinking-mode sampling; a request's own values win |
| `THINKING` / `MAX_TOKENS` | `1` / `32768` | thinking default; reply length for a request without `max_tokens` |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | `9` | what the engine keeps out of `MemAvailable` |
| `PORT` / `HOST` | `8011` / `0.0.0.0` | published port and bind address; **there is no API key by default** (`TENSORFOLD_API_KEY` sets one), set `HOST=127.0.0.1` off a trusted network |
| `CONTAINER_MEM_GIB` | `24` | host-side cgroup cap |
| `MEMWATCH_MIN_GIB` / `MEMWATCH_MIN_FREE_GIB` | `6` / `2` | watchdog floors |
| `CUDA_CACHE_HOST` / `TENSORFOLD_CACHE_HOST` | `~/.nv` / `~/.cache/tensorfold` | host caches, mounted at each tool's in-container default |
| `EXTRA_SERVE_ARGS` | *(empty)* | appended to `tensorfold-native serve`; checked by the engine's own parser in preflight |

Any other `TENSORFOLD_*` or `TF_FLASHNEXT_*` variable in the environment is passed into the container.

## Since the previous version

Upstream moved from TensorFold's `zig-flashnext` branch (`db281878`) through v1.1.4 to the v1.2.7 release. v1.2.7 carries in
its own code most of what the old patch series did, so the series went from 0001-0015 (about 190k lines, including the
registry server, kernels and config) to 0001-0005 (about 3.3k lines): the FP4 vision tower, a 16,384 image-token budget, the
fp8 KV cache, the MXFP8 prompt tile, and chai-md's video patch. What changes for this recipe:

- **Removed:** YaRN past 262,144; the AOT kernel set and its Triton cache (`~/.triton` is no longer mounted); the Python
  vision helper (transformers, PyAV, Pillow); `TF_FLASHNEXT_DEPTH`, `_PRODUCT_STREAMS`, `_PREFILL_TAIL` and `_FILL_FAIR`,
  which v1.2.7 no longer reads.
- **Renamed:** the serve flag `--kv-dtype` is `--kv-cache` (v1.2.7's `--kv-dtype` no longer takes `fp8`).
- **New:** `ABLIT`, `VIDEO`, `PDF`, structured output and required tool calls, upstream's v1.2.7 tuning settings, a
  9 GiB reserve, and the `bench.py` fix that no longer counts the chat template's shared prefix as a prompt-cache hit.

## Port notes

- **Pinned revision and image inputs.** Upstream serves the repo id; `start.sh` passes the pinned snapshot directory (or the
  ABLIT view), and `download.sh` verifies every file against its manifest. The image tag hashes the patches, this recipe's
  Dockerfile and every pinned build input, and `build.sh` checks the labels, the binary and each library the engine loads.
- **Built locally, not pulled.** Upstream's prebuilt image follows its own Dockerfile, which differs from this recipe's
  (no recipe-private CUDA cache path).
- **Caches at the tools' defaults.** Upstream keeps the CUDA JIT cache under `~/.cache/tensorfold-qwen38`; here it is
  `~/.nv`. The ABLIT view does sit under `~/.cache/huggingface/tensorfold-qwen38/`, as upstream's, because its symlinks
  must resolve inside the container's read-only mount of the HF cache.
- **Host-side guards.** Upstream trusts the engine's admission alone; this adds a cgroup cap and a watchdog.
- **Port 8011 and bridge networking.** Upstream uses `--network host` on 8888, which the vLLM recipe already claims.
- **Setup split out.** Upstream's `start.sh` runs `prepare.sh` on demand. Here that is `build.sh` and `download.sh`, and
  `start.sh` runs `preflight.sh` before it stops or loads anything.
- **One checkpoint.** Upstream's optional INT4-AutoRound (`CHECKPOINT=int4`) is not carried (upstream's v1.2.7 branch no longer serves it either).
- **Dropped:** the terminal banner, `publish-image.sh`, `harden.sh` (an opt-in CVE-reduction tag over the image), the
  `.github` templates, `./.env` parsing, and upstream's tests of its own `start.sh`, `stop.sh` and README. Carried:
  `tools/test_ablit_view.py` and `tools/test_bench_stats.py`, both offline.

## Known gaps

- **Video and PDF have not been run on this host**, and upstream has not run a live video request. ABLIT ran the
  functional checks only; its refusal behaviour and quality are not evaluated.
- **Clef-Flash co-hosting is untested.** The weights take about 81 GiB, and the other recipes' co-tenant notes still
  describe the MLX4 recipe.
- **The image is a third party's source.** It is built from TensorFold v1.2.7 plus MiaAI-Lab's patches; the build verifies
  checksums, labels, the binary and the libraries, not the patched source.
- **No API key by default.** Set `TENSORFOLD_API_KEY` or keep `HOST` on a trusted network.

## License

MIT, like upstream; [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) carry upstream's notices, TensorFold's and those of the
libraries in the image. TensorFold is Apache-2.0 from 0.6.0, so [`LICENSE.Apache-2.0`](LICENSE.Apache-2.0) ships too. The
weights are not part of this repository: NVIDIA Open Model Agreement, plus the base model's Qwen Community License 1.0;
the ABLIT overlay is gated under its own Responsible Use Agreement and inherits both. The image is based on NVIDIA's
PyTorch container, whose terms the container prints at every start; building or running it accepts them.
See [`CREDITS.md`](CREDITS.md).
