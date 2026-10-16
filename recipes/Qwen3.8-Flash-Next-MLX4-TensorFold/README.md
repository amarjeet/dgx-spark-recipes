# Qwen3.8-Flash-Next · MLX 4-bit · TensorFold

Serves [Qwen3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) from one DGX Spark through an
OpenAI-compatible API, with **five concurrent requests at the full 262,144-token window**, and image and video
input. It runs [TensorFold](https://github.com/ashhart/TensorFold) v0.6.1 in NVIDIA's PyTorch container, with one
patch that adds video, many images and speed, and changes no output token.

Port of [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)
at `4c0dea8` (upstream's recipe 0.6.0). What changed in the port is under [Port notes](#port-notes).

| | |
|---|---|
| Checkpoint | [`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP) @ `dadefa8066e3`, MLX 4-bit, group size 32, with the MTP draft head and vision tower. 35 files, 105.46 GiB, every one checksummed in `manifests/mlx4.json` |
| Runtime | TensorFold v0.6.1 (`17c73e1`) + `patches/*.patch` (hash `797af1d9df4d`) on `nvcr.io/nvidia/pytorch:26.07-py3` |
| Image | upstream's prebuilt image pinned by digest, or built locally; verified either way |
| Endpoint | `:8011`, model id `qwen3.8-flash-next`, alias `Qwen3.8-Flash-Next` |
| KV pool | 5 streams x 262,144 = 1,310,720 tokens, int8 KV (default, as upstream) |

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
./scripts/smoke.py      # arithmetic, thinking, typed tool calls, live counters, image input
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

Images and video go in OpenAI's content parts, as data URLs (`VISION_URLS=1` also fetches `https://` ones):

```bash
curl -s http://<spark>:8011/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    {"type": "text", "text": "What is in this picture?"}]}]
}'
```

The model thinks before it answers, into `reasoning_content`. A request without `max_tokens` gets 32,768 tokens
(`MAX_TOKENS`), enough for thinking to finish; TensorFold's own default, 4,096, could end a reply before it answered. Per request,
`"chat_template_kwargs": {"enable_thinking": false}` answers directly and `{"reasoning_effort": "low"}` (or `xhigh`)
sets the effort. A top-level OpenAI `reasoning_effort` field is ignored.

## Profiles

One checkpoint, so the profiles are KV-pool trade-offs. TensorFold admits every stream at a full window, so the
pool is streams x window. The estimate column is TensorFold's own startup estimate, measured here on v0.6.1; the
server prints the real one at every start.

| Profile | Streams x window | KV | Pool | Estimate | Note |
|---|---|---|---:|---:|---|
| `int8x5` | 5 x 262,144 | int8 | 1,310,720 | 85.39 GiB | **default**, as upstream; image and video input |
| `int8x4` | 4 x 262,144 | int8 | 1,048,576 | 85.18 GiB | the v0.3.6.2 default here |
| `int4x6` | 6 x 262,144 | int4 | 1,572,864 | not run | int4 changes output; quality not measured |
| `bf16x3` | 3 x 262,144 | bf16 | 786,432 | not run | full-precision KV |
| `int8x1` | 1 x 262,144 | int8 | 262,144 | 87.57 GiB | one stream, text only, to share the box; see below |

The estimates are close because v0.6.1 counts weights and starting caches, not full windows: the caches grow
later, under traffic (see [Memory](#memory)). A lone stream is the exception and is allocated whole, which is why
`int8x1` estimates highest.

**`int8x1` is the co-tenant profile.** One full-window stream, measured here with 33.0 GiB of `MemAvailable` left,
both idle and after a 226k-token prompt. That is room for
[`Clef-Flash-FP8-transformers`](../Clef-Flash-FP8-transformers/) (~19 GiB) with its 12 GiB floor intact, about
0.7 GiB less than on v0.3.6.2. It is **text only**: TensorFold's Flash Next image input needs two or more streams,
so the profile sets `VISION=0`, and preflight refuses `VISION=1` with one stream. Start this one first, since
TensorFold reads `MemAvailable` once at start. Measured on v0.3.6.2 with both up: each server runs at roughly half
speed while the other is busy (prefill 2,455 -> 1,107 tok/s, sampled decode 59 -> 24 tok/s with Clef saturated).
The pair has not been re-run on v0.6.1.

`PARALLEL`, `CONTEXT` and `KV_DTYPE` override any profile, for example `PARALLEL=6 CONTEXT=220000 ./start.sh`.
TensorFold refuses a setting that does not fit before any weights load, and names a window that would.

### Switching streams with Clef-Flash alongside

Keep `int8x1` while Clef is up. With two or more streams, v0.6.1 grows the caches *after* Clef has started, so Clef's
preflight check at its own start no longer bounds what Qwen will take. TensorFold's gate stops growth near
`MemAvailable` 14.2 GiB (its 12.2 GiB reserve plus 2 GiB), just above Clef's 12 GiB floor. But a stream decoding
alone grows past the gate to finish its reply, up to one more window (about 4.5 GiB at int8), which can put Clef
under its floor. Not measured with Clef running.

Restarting Qwen *while Clef is up* no longer protects Clef either. With Clef holding ~19 GiB, TensorFold's budget
is about 87 GiB, and v0.6.1's estimates (85.2 to 85.4 GiB) fit it, so more streams are admitted and then grow into
Clef's room. So to change streams, stop Clef first, then restart in this order:

```bash
../Clef-Flash-FP8-transformers/stop.sh

./start.sh restart                                   # back to int8x5, Qwen alone
./start.sh restart int8x1                            # back to the co-tenant profile

../Clef-Flash-FP8-transformers/start.sh              # its preflight decides whether it fits
```

Clef's preflight is the authority: it reads the real `MemAvailable` and refuses to start if either floor would be
broken. Overriding the floors (`HOST_FLOOR_GIB`, `LOAD_FLOOR_GIB`) puts Clef where heavy Qwen traffic gets it killed.
See [Clef's README](../Clef-Flash-FP8-transformers/README.md#memory).

## Memory

TensorFold sizes itself. Its budget is `MemAvailable`, read at start, minus a reserve: `TENSORFOLD_MEMORY_RESERVE_GIB`,
or the larger of 4 GiB and a tenth of `MemTotal` (12.2 GiB here) when unset. It refuses a profile whose full windows
do not fit, before any weights load.

**What changed in v0.6.** The KV caches are no longer allocated whole at load. Each stream's slot starts small and
grows with its context. Before each growth, a gate checks the live `MemAvailable` for the growth plus 2 GiB on top
of the same reserve. So the reserve is also the floor the server grows down to under traffic. When the gate refuses,
a request waits, or with several streams decoding the newest is ended with an error naming the cause. A stream
decoding alone grows anyway.

Upstream sets the reserve to 2 GiB, the knob's minimum, and measured `MemAvailable` down to 2.9 GiB through five
~236k prompts. That would trip this recipe's watchdog. Here the reserve stays TensorFold's own 12.2 GiB, and five
full windows are still admitted on an idle Spark: 85.39 within 105.87 GiB. Set `TENSORFOLD_MEMORY_RESERVE_GIB` to
override it; preflight warns when the floor it implies is under the watchdog's.

On unified memory an exhausted pool hangs the kernel instead of raising an OOM. So the recipe keeps two host-side
guards around TensorFold's own:

- **A cgroup cap** of `CONTAINER_MEM_GIB`, 24 GiB. GPU allocations are not charged to it on GB10, so it bounds the
  host-side footprint only.
- **A watchdog**, `scripts/memwatch.sh`. It stops the container when `MemAvailable` stays under 6 GiB, or `MemFree`
  stays under 2 GiB while `MemAvailable` is under 10 GiB, for five samples two seconds apart. Its log is in
  `$OUT_DIR/logs/`.

**Worst case, measured.** Every stream filled to ~226k tokens at once, with the default reserve:

| Profile | Requests | Completed | Lowest `MemAvailable` | Watchdog |
|---|---:|---:|---:|---|
| `int8x5` | 5 x ~226k | 5 | 14.4 GiB | no trip |
| `int8x4` | 4 x ~226k | 4 | 18.5 GiB | no trip |
| `int8x1` | 1 x ~226k | 1 | 33.0 GiB | no trip |

`int8x5` held at the gate's floor, as designed. `MemFree` dipped to 1.8 GiB there, which is reclaimable page cache:
the watchdog counts `MemFree` only under 10 GiB of `MemAvailable`. On v0.3.6.2, `int8x5` reached 7.7 GiB during its
*load*, which is why this recipe defaulted to `int8x4` until now.

## Measured on this host

GB10, 121.7 GiB unified memory, kernel driver 580.178.04 (the container runs CUDA 13.3 in forward-compatibility
mode). Image `tensorfold-qwen38:v0.6.1-797af1d9df4d`, pulled by digest and verified. Nothing else on the GPU.
Measured through the OpenAI API with `./bench.sh`, on 2026-10-02. The v0.3.6.2 column is this recipe's earlier
port, same host, 2026-09-29.

**Startup.** The first start on v0.6.1 builds its CUDA extensions and was ready in 105 s. Later starts, with the
weights still in the page cache, answered in 24 to 30 s.

**Prefill**, fresh random prose per run so the prompt cache never hits:

| Prompt | `int8x5` v0.6.1 | `int8x4` v0.6.1 | `int8x5` v0.3.6.2 |
|---:|---:|---:|---:|
| ~810 | 0.65 s | 0.57 s | 0.50 s |
| ~3,170 | 2.32 s | 2.15 s | 1.47 s |
| ~12,600 | 5.96 s | 5.29 s | 5.09 s |
| ~50,300 | 21.18 s (2,374 tok/s) | 20.52 s (2,451 tok/s) | 20.77 s (2,425 tok/s) |

Short prompts are slower than on v0.3.6.2. The recipe now prefills in 2,048-row pieces, where v0.3.6.2 used 4,096.
Upstream measured 2,048 faster from about 5k tokens with the n-gram tables on SSD, and slower below that, as here.
The first prefill run after a fresh extension build was up to twice as slow; the table is a warm run.

**Needle.** A passphrase at 60% depth of a 194,851-token prompt, asked greedily, was recalled on `int8x4`; prefill
took 98.6 s.

**Decode, single request:** 68.1 tok/s on greedy code and 63.8 tok/s on sampled chat (`int8x5`, medians of five),
against 71.7 and 59.0 on v0.3.6.2.

**Decode, concurrent** (`scripts/bench_concurrent.py`: 512-token essays, thinking off, aggregate over first to last
token, tok/s):

| Requests | `int8x5` aggregate | per request | `int8x4` aggregate | per request | `int8x5` v0.3.6.2 | Upstream, `int8x5` v0.3.6.2 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 52.9 | 52.9 | 50.1 | 50.1 | 54.4 | 62.4 |
| 2 | 76.8 | 40.0 | 65.2 | 33.7 | 75.0 | 90.5 |
| 4 | 100.8 | 25.7 | 93.0 | 23.7 | 89.4 | 106.7 |
| 5 | 107.7 | 22.3 | 85.2 | 26.0 | 100.6 | 119.3 |

Upstream's prompts are not published, and MTP draft acceptance depends on the text, so the gap to upstream's
column is not evidence of a slower host. On `int8x4` a fifth request waits for a stream.

**Checks.** `scripts/smoke.py` passed on `int8x5` and `int8x1`: arithmetic, alias, thinking, live counters, typed
tool arguments, and on `int8x5` image input (`scripts/visioncheck.py`). Video was not tested here.

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
| `MAX_TOKENS` | `32768` | reply length for a request without `max_tokens`; TensorFold's own 4,096 can end a thinking reply before it answers |
| `VISION` | `1` | image and video input (`--vision`); `0` is text only |
| `VISION_URLS` | `0` | `1` also fetches public `https://` image and video URLs; the default takes data URLs only |
| `VISION_MAX_IMAGES` | `50` | images a request may carry, a chat's turns all counted; they share `TENSORFOLD_IMAGE_TOKENS` (16,384) |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | per profile | what TensorFold leaves out of `MemAvailable` at start, and the floor its caches grow down to; see [Memory](#memory) |
| `TENSORFOLD_PREFILL_ROWS` | `2048` | prompt piece rows while `PLE_ON_SSD=1`; empty leaves TensorFold's choice |
| `TENSORFOLD_MTP_COPY` | `1` | prompt-lookup drafts for replies that repeat the prompt (needs 2+ streams) |
| `EXTRA_SERVE_ARGS` | *(empty)* | appended to `tensorfold serve`, so they win; checked by TensorFold's own parser in preflight |
| `PORT` / `HOST` | `8011` / `0.0.0.0` | published port and bind address; there is no API key |
| `CONTAINER_MEM_GIB` | `24` | host-side cgroup cap |
| `MEMWATCH_MIN_GIB` / `MEMWATCH_MIN_FREE_GIB` | `6` / `2` | watchdog floors |
| `PULL` | `1` | `0` makes `build.sh` build locally instead of pulling |
| `TRITON_CACHE_HOST` / `TORCH_EXTENSIONS_HOST` / `TENSORFOLD_CACHE_HOST` | `~/.triton` / `~/.cache/torch_extensions` / `~/.cache/tensorfold` | host caches, mounted at each tool's in-container default |

Any other `TENSORFOLD_*` or `TF_*` variable in the environment is passed into the container, as upstream does.

## What the patch changes

Upstream's one patch, `0002-flash-next-v061`, carried byte for byte (a unified diff against TensorFold's
site-packages, applied with `patch -p0`). It replaces the eight v0.3.6.2 patches: typed tool parameters and the live
`/health` counters are in TensorFold itself now, and so is Flash Next image input (TensorFold #146, adapted from
upstream's own patch). What the patch still adds:

- **video input**, and up to 50 images a request sharing 16,384 tokens (`--vision-max-images`,
  `TENSORFOLD_IMAGE_TOKENS`; TensorFold alone takes 4)
- **copy drafts**: prompt-lookup drafts ahead of MTP when the reply repeats the prompt (`TENSORFOLD_MTP_COPY`)
- **SSD read-ahead** for the n-gram tables, and `TENSORFOLD_PREFILL_ROWS` as an override of TensorFold's own choice
- the first token before the next draft round, no MTP logits while prompts absorb, and 96 MiB request bodies

`build.sh` proves the patch is in the image by reverse-applying it, rather than trusting the label. Upstream checked
that output is unchanged: greedy token ids equal across 34 replies between v0.6.0 and v0.6.1, and drafted replies
equal serial ones. Any request can send `"draft": false` for the serial one-token-a-round reference.

## Port notes

- **Pinned revision.** Upstream serves the repo id, which resolves to the newest snapshot on disk. `start.sh` passes
  the pinned snapshot directory, and `download.sh` verifies all 35 files against the manifest: SHA-256 for the
  weights, the git blob id for the small files.
- **Pinned image.** The prebuilt image is pulled by digest, since a tag can be re-pushed. A local build installs
  TensorFold at the release commit, not the tag. Either result must carry the right label, report 0.6.1, and
  contain the patch.
- **Caches at the tools' defaults.** Upstream keeps compiled kernels in a recipe-private `~/.cache/tensorfold-qwen38`.
  Here triton's cache is `~/.triton` and torch's extension cache is `~/.cache/torch_extensions`, mounted at their
  in-container defaults and shared with the other recipes. The HF cache is mounted read-only.
- **Host-side guards.** Upstream trusts TensorFold's admission alone. This adds a cgroup cap and a memory watchdog, as
  the repo's other self-budgeting recipe does.
- **TensorFold's own memory reserve.** Upstream sets `TENSORFOLD_MEMORY_RESERVE_GIB=2` for its five streams. Here the
  reserve stays TensorFold's default (12.2 GiB) unless a profile needs less, because the reserve is also the floor
  the caches grow down to, and 2 GiB is under the watchdog's. See [Memory](#memory).
- **Port 8011 and bridge networking.** Upstream uses `--network host` on 8888, which the vLLM recipe for this model
  already claims.
- **Setup split out.** Upstream's `start.sh` runs its setup on demand. Here that is `build.sh` and `download.sh`, and
  `start.sh` runs `preflight.sh` before it stops or loads anything.
- **Dropped:** the terminal banner, `publish-image.sh` (pushing to upstream's registry is upstream's job), the
  `.github` templates, and `./.env` parsing (every knob is an environment variable here, as in the other recipes).
- **Not ported yet: the language image.** Upstream's opt-in second image (`DRAFT_LANGUAGE`, patch
  `languages/0010`) adds Chinese or Japanese tokens to what MTP may draft: +48% decode on Chinese, same output.
  It is a 103k-line patch and a second image to pin; English and code get slightly slower with it.

## Known gaps

- **Concurrent throughput is under upstream's**, by 12 to 18%, on a different workload. Not reproduced with
  upstream's own prompts, which are not published.
- **`int8x5` is tight on this host.** See [Memory](#memory). Five concurrent long requests were not run here;
  upstream reports the host keeping at least 9.7 GiB free through them.
- **`int4x6` and `bf16x3` were not run here.** Their estimates are upstream's, and int4 KV quality is unmeasured
  anywhere.
- **The image is a third party's.** It is pinned by digest and verified: the right label, TensorFold 0.6.1, and the
  patch present. Once the patch is undone, the package matches the v0.6.1 source (`17c73e1`) exactly, apart from
  three READMEs the wheel does not ship. That was checked once by hand, on 2026-10-02, not on every build. `PULL=0 ./build.sh` builds the same image locally instead.
- **No API key.** TensorFold serves without auth. Set `HOST=127.0.0.1` off a trusted network.

## License

MIT, like upstream. [`LICENSE`](LICENSE) carries upstream's notice and TensorFold's NOTICE. TensorFold is
Apache-2.0 from 0.6.0, and the patch carries lines of v0.6.1, so a copy of that license is in
[`LICENSE.Apache-2.0`](LICENSE.Apache-2.0). The
weights are not part of this repository and are under the Qwen Community License 1.0.

The image is based on NVIDIA's PyTorch container, whose NVIDIA software is governed by the
[NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and the [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/).
The container prints them at every start; pulling or running it accepts them.

## Credits

The recipe is MiaAI Lab's ([@MiaAI_lab](https://x.com/MiaAI_lab)): the patches, the settings and every upstream
measurement quoted here. It runs on [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart and its
contributors, [Qwen3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by Qwen, and
[Vontra's MLX 4-bit checkpoint](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP). The prefill-rows override
ports TensorFold #40, and the `MAX_TOKENS` default upstream #11, both by [MovieMaker93](https://github.com/MovieMaker93). Upstream's
[CREDITS.md](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/main/CREDITS.md) has the
full list.
