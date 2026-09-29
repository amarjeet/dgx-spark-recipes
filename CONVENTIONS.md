# Conventions

Every recipe in this repo follows the same rules. They exist so that recipes can
share a machine without fighting each other over disk, memory or ports.

## Storage: heavy data lives at each tool's own default location

A recipe directory holds code, config and manifests. Nothing heavy. Model
weights and compiler caches live exactly once, at the location the tool would
use with no configuration at all:

| Data | Path | Env var the tool actually reads |
|---|---|---|
| HF hub cache, datasets | `~/.cache/huggingface/` | `HF_HOME` |
| llama.cpp GGUF downloads | `~/.cache/llama.cpp/` | `LLAMA_CACHE` |
| vLLM torch.compile cache | `~/.cache/vllm/` | `VLLM_CACHE_ROOT` |
| flashinfer JIT cache | `~/.cache/flashinfer/` | `FLASHINFER_WORKSPACE_BASE` (base is `~`) |
| triton kernel cache | `~/.triton/cache/` | `TRITON_CACHE_DIR` |
| Bench results, verify stamps | `${XDG_STATE_HOME:-~/.local/state}/dgx-spark-recipes/<recipe>/` | `OUT_DIR` |

Two rules follow:

1. **Never introduce a recipe-local cache default.** No `./.cache` inside the
   recipe directory. That pattern silently re-downloads the same 60–80 GB
   checkpoint once per recipe until the disk fills.
2. **Every path is an env-overridable variable whose default is the standard
   location.** The override makes the recipe portable; the default makes it
   correct with no setup.

For Docker recipes, bind-mount the host path onto the *same path the tool
defaults to inside the container*. When the mount target is already the tool's
default, nothing needs a cache environment variable at all.

**For native recipes there is no mount to redirect anything**, so each variable
must be the one the tool itself reads — `VLLM_CACHE_ROOT`, not
`VLLM_CACHE_HOST`; `FLASHINFER_WORKSPACE_BASE`, not `FLASHINFER_CACHE_HOST`.
Best case, export nothing: the defaults already point at shared storage.

A native recipe also loses the cgroup cap a container gets for free. On unified
memory that matters more than anywhere else, so such a recipe must derive its
memory budget explicitly and refuse to launch a configuration that cannot fit,
rather than discovering it during a load. See
`DeepSeek-V4.1-Flash-EXL3-ExLlamaV3` and
`Ternary-Bonsai-2-27B-GGUF-llamacpp-prism` for worked examples (in the latter,
`budget_bytes` in its `profiles.sh`, itemised by `preflight.sh`).

Where a recipe has to *build* its runtime rather than pull it, the checkout is
third-party source shared across recipes, not recipe data: it belongs under
`SRC_ROOT` (this host's existing `~/src` convention), and the binaries are run
out of the build tree so nothing is copied into the recipe directory.

The full version of this convention, including the native (non-Docker) case, is
packaged as an agent skill:
[`amarjeet/agent-skills` → `dgx-spark-layout`](https://github.com/amarjeet/agent-skills/tree/main/skills/dgx-spark-layout).

## Environment variable reference

Every variable below is optional — the default is what a recipe uses with no
configuration. They fall into three groups, and the distinction matters:

**Storage.** These decide where bytes land. Defaults are each tool's own
standard location, so overriding one relocates data for every recipe that reads
it, not just this one.

| Variable | Default | Means |
|---|---|---|
| `HF_HOME` | `~/.cache/huggingface` | Root of the Hugging Face cache — hub downloads, datasets, xet. Read by the HF libraries themselves, not just by these recipes. |
| `HF_TOKEN` | *(unset)* | Hugging Face access token, needed for gated repos. Falls back to the token file below. The Hub works anonymously at a lower rate limit. |
| `HF_TOKEN_PATH` | `$HF_HOME/token` | Where to read the token from when `HF_TOKEN` is unset — the file `hf auth login` writes. |
| `LLAMA_CACHE` | `~/.cache/llama.cpp` | llama.cpp's own GGUF store, and the tree bind-mounted into the container. The variable llama.cpp actually reads. |
| `MODEL_STORE` | `$LLAMA_CACHE/<repo>` | *(llama.cpp recipes)* This model's subtree of the store. Must stay under `LLAMA_CACHE` or the bind mount cannot reach it; `start.sh` asserts this. |
| `MODEL_ROOT` | `$MODEL_STORE/<QUANT>-<rev12>` | *(llama.cpp recipes)* The exact revision+quantization directory holding the shards. |
| `MODEL_REVISION` | recipe-specific | Pinned Hub commit. Recipes resolve exactly this snapshot rather than whichever one is on disk. |
| `VLLM_CACHE_HOST` | `~/.cache/vllm` | *(vLLM recipes)* Host side of vLLM's cache root — torch.compile cache, and any packed table a model builds. Mounted onto `/root/.cache/vllm`, the in-container default. Natively, vLLM reads `VLLM_CACHE_ROOT`. |
| `FLASHINFER_CACHE_HOST` | `~/.cache/flashinfer` | *(vLLM recipes)* Mounted onto `/root/.cache/flashinfer`. Natively, flashinfer reads `FLASHINFER_WORKSPACE_BASE` (base is `~`). |
| `TRITON_CACHE_HOST` | `~/.triton` | *(vLLM recipes)* The whole `.triton` tree, not just `cache/`, so triton's own sub-layout applies inside. Natively, triton reads `TRITON_CACHE_DIR`. |
| `DRAFT_VOCAB_DIR` | `$VLLM_CACHE_HOST/draft_vocab` | *(vLLM recipes, speculative decoding)* Generated reduced draft vocabularies. Under `VLLM_CACHE_HOST` deliberately: that tree is already mounted at vLLM's in-container default, so a vocabulary there needs no mount of its own. Recipes assert the containment. |
| `OUT_DIR` | `${XDG_STATE_HOME:-~/.local/state}/dgx-spark-recipes/<recipe>` | Bench results and verification stamps. Never inside the recipe directory. |
| `XDG_STATE_HOME` | `~/.local/state` | Standard base for `OUT_DIR`; honored rather than assumed. |
| `SRC_ROOT` | `~/src` | *(native recipes)* Third-party source checkouts, pinned by commit and shared across recipes. This host's existing convention; the layout skill has no row for source trees, and that silence is a cue to follow the host rather than invent a path under the workspace root. |
| `VENV` | `~/venvs/<name>` | *(native recipes)* The virtualenv holding a source build. Several GB of torch, so it is shared rather than rebuilt per recipe, and it never lives in the recipe directory. |
| `MODEL_ROOT` | recipe-specific, under `${DGX_SPARK_ROOT}/base-models/` | *(native recipes)* A derived, non-HF weight artifact — for example a re-laid pack. It does not go in the hub cache, which `huggingface_hub` owns and may prune. |
| `TORCH_EXTENSIONS_DIR` | `~/.cache/torch_extensions` | PyTorch's JIT extension cache. Read only on the `cpp_extension.load()` path, so it is **inert** for an ahead-of-time `setup.py` build — set to the shared default anyway, because a recipe-private cache default is the pattern these rules exist to prevent. |
| `TORCH_EXTENSIONS_HOST` | `~/.cache/torch_extensions` | *(Docker recipes that JIT-build extensions)* Host side of torch's extension cache, mounted onto `/root/.cache/torch_extensions`. Torch keys it by Python and CUDA version (`py312_cu130/`) **only when `TORCH_EXTENSIONS_DIR` is unset**; set, even to the default path, it builds straight into that directory, and a second torch version sharing the cache can load the first one's `.so`. So when an image sets `TORCH_EXTENSIONS_DIR` or `TRITON_CACHE_DIR`, as TensorFold's does, `start.sh` *unsets* them (`env -u`) rather than overriding them. |
| `TENSORFOLD_CACHE_HOST` | `~/.cache/tensorfold` | *(TensorFold recipes)* TensorFold's own cache, mounted at its in-container default. |

**Runtime and serving.** These change how the server runs. Safe to set per
invocation.

| Variable | Default | Means |
|---|---|---|
| `PORT` | `8008` | Port the server listens on and publishes. `preflight.sh` verifies it is free. |
| `HOST` | `0.0.0.0` | Bind address. **Set to `127.0.0.1` to keep the server off the LAN** — there is no API key. |
| `IMAGE` | `ghcr.io/ggml-org/llama.cpp:server-cuda` | Container image. The tag is mutable, hence `MIN_LLAMA_BUILD`. |
| `CONTAINER_NAME` | `<model>-gguf` | Docker container name. Change it to run two recipes of the same model side by side. |
| `DEFAULT_PROFILE` | recipe-specific | Which quantization profile is used when none is named. |
| `SERVED_MODEL_NAME` | `<model>-<quant>` | The `model` id clients send. Without it llama.cpp reports the full GGUF path. |
| `CTX_SIZE` | per profile | Context window. The largest single lever on memory use. |
| `PARALLEL` | `1` | Server slots. Total context is `CTX_SIZE`, divided across slots. |
| `BATCH_SIZE` / `UBATCH_SIZE` | `4096` / `2048` | Prefill batch sizes. Lower them if a load fails on compute buffers. |
| `SPEC_TYPE` | `draft-mtp` | Speculative decoding mode. `none` disables it — the first thing to try if the server exits at startup. |
| `MLOCK` | `1` | Lock weights in RAM so they never reach swap. `0` opts out, allowing oversubscription. |
| `FORK_REPO` / `FORK_BRANCH` / `FORK_COMMIT` | recipe-specific | *(native recipes that build their runtime)* The upstream to compile and the exact commit to pin. A branch head is not a pin; `build.sh` checks out the commit. |
| `FORK_DIR` | `$SRC_ROOT/<tool>-<slug>` | *(native recipes that build their runtime)* The source checkout. Shared, never inside the recipe directory. |
| `BUILD_DIR` / `BIN_DIR` | `$FORK_DIR/build-<backend>[/bin]` | *(native recipes that build their runtime)* Where the build lands and where the binaries are run from. Binaries run in place, with `LD_LIBRARY_PATH` pointed at `BIN_DIR`, so the shared libraries beside them are found without `patchelf` and nothing is copied out. |
| `CUDA_PATH` | `/usr/local/cuda` | *(native CUDA builds)* Toolkit root. `build.sh` uses `$CUDA_PATH/bin/nvcc`. |
| `CUDA_ARCHS` | `121a-real` | *(native CUDA builds)* `CMAKE_CUDA_ARCHITECTURES`. GB10 is compute capability 12.1 and the arch-specific `a` suffix is load-bearing: `sm_120a` is **not** forwards-compatible to `sm_121` and carries no PTX to JIT from, so a binary built for `120a` will not run on this box at all. `preflight.sh` reads the architectures back out of the built library rather than trusting the flag. |
| `CTX_CHECKPOINTS` / `CACHE_RAM_MIB` | `8` / `8192` | *(llama.cpp recipes, hybrid/GDN models)* `-ctx-checkpoints` and `-cache-ram`. A hybrid model cannot partially evict its recurrent state, so an edited conversation re-prefills from a checkpoint or from zero; checkpoints bound that cost, and each one is roughly a slot's worth of recurrent state. Set explicitly because the default is per-slot and large enough to matter to a memory budget. |
| `MMPROJ_CPU` | `0` | *(multimodal recipes)* `1` keeps the vision projector in system RAM (`--no-mmproj-offload`), trading a slower image prefill for the projector's memory. |
| `IMAGE_MAX_TOKENS` | *(unset)* | *(multimodal recipes)* Cap on vision tokens per image, which are prefill. Unset leaves the backend's own default; `0` disables capping. |
| `RESTART_POLICY` | `unless-stopped` | Docker restart policy; survives reboot. `no` for a one-off run. |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | `1.0` / `0.95` / `20` | Server-side sampling defaults, so clients that send nothing still get the model card's recommendation. |
| `REASONING_EFFORT` | *(unset)* | *(thinking models)* Thinking depth pinned server-wide through the chat template. Unset leaves the template's own default, which is not necessarily the cheap one — Qwen3.8-Flash-Next defaults to `xhigh`. Clients override it per request with `chat_template_kwargs`. |
| `MAMBA_SSM_CACHE_DTYPE` | per profile | *(vLLM recipes, hybrid/GDN models)* dtype of the recurrent (SSM) state. **Empty is a meaningful value** — it selects the checkpoint's own dtype — so recipes read it with `${VAR-default}`, not `${VAR:-default}`, and tell "unset" from "set to empty". |
| `CUDAGRAPH_CAPTURE_SIZES` | `auto` | *(vLLM recipes)* Which decode batch widths get a CUDA graph. `auto` enumerates every width the scheduler can actually build, so none falls back to eager; empty keeps vLLM's own list. Also read with `${VAR-default}`. |
| `COMPILATION_MODE` | `0` | *(vLLM recipes)* `torch.compile` level passed through `--compilation-config`. `0` is no compilation; `3` is Inductor fusion. |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | *(vLLM recipes with speculative decoding)* Pins the V2 model runner on **every** config copy. A draft config that falls back to V1 can mutate the `compilation_config` it shares with the target and silently downgrade its CUDA graphs. Its own variable rather than a default for `EXTRA_DOCKER_ARGS`, so setting that escape hatch cannot drop a safety default. |
| `MTP_DRAFT_VOCAB` | *(unset)* | *(vLLM recipes with MTP)* Path to a reduced draft vocabulary. Must sit under `VLLM_CACHE_HOST`; see `DRAFT_VOCAB_DIR`. |
| `MTP_K_SCHEDULE` | *(unset)* | *(vLLM recipes with MTP)* Speculative depth per batch-size range, `start:end:K,...`. Empty keeps a constant depth. |

**Preflight and tooling.** Thresholds and helper knobs.

| Variable | Default | Means |
|---|---|---|
| `MIN_LLAMA_BUILD` | recipe-specific | Minimum llama.cpp build number. `preflight.sh` reads the real number out of the image — or, for a native recipe, out of the built binary — and refuses to run below it, rather than trusting a mutable tag. |
| `COMPUTE_RESERVE_BYTES` | recipe-specific | *(native recipes)* Compute and graph buffers plus slack for the OS and driver: the one term in a memory budget that is a guess rather than arithmetic. Generous on purpose, and corrected against a measured start. |
| `MEM_HEADROOM_BYTES` | `6 GiB` | Free memory demanded *on top of* the weights, for KV, compute buffers and the OS. A floor, not a budget. |
| `DISK_RESERVE_BYTES` | `10 GiB` | Free disk demanded beyond the remaining download. |
| `FORCE_VERIFY` | `0` | `1` re-runs the full SHA-256 pass. Normally verification is stamped by `(path, size, mtime)` so a restart does not re-hash tens of gigabytes. |
| `RETRIES` | `20` | Download attempts before giving up. |
| `SMOKE_HOST` | `127.0.0.1` | Host `scripts/smoke.py` targets. |
| `BENCH_OUT` | *(unset)* | Default output path for `scripts/bench_depths.py`. |

## Recipe shape

Each recipe is self-contained and driven by a single sourced config:

```
profiles.sh     the only place paths, the profile table and helpers are defined
build.sh        (only if the runtime has to be compiled) pinned checkout + build
download.sh     checksum-verified, resumable weight download
preflight.sh    assert everything start.sh depends on, before a long load
start.sh        launch the server
stop.sh         stop it -- remove the container, or signal the pidfile
status.sh       is it up, what is it serving, is the host healthy
bench.sh        measurements
manifests/      pinned revision, per-shard sizes and SHA-256
scripts/        stdlib-only Python helpers
```

Every script resolves its own directory, so a recipe runs from any working
directory and can be cloned anywhere.

A `build.sh` stays out of the repo-root launcher: a launcher that silently
starts a twenty-minute compile is worse than one that tells you to run the
build yourself.

## Reproducibility

- **Pin the model revision.** Manifests record the Hub revision, every shard's
  byte size and its SHA-256.
- **Pin the runtime.** Container tags are mutable. Where a recipe needs a
  minimum build, `preflight.sh` reads the build number out of the image and
  refuses to proceed below it rather than trusting the tag. A recipe that
  builds its runtime pins a commit, not a branch, and verifies the built
  artifact — that it contains the kernels the model needs, and that it was
  compiled for this GPU — because both failures are otherwise silent.
- **Measurements name the machine they came from.** Numbers in a recipe README
  were measured on the host described there, not predicted.

## Sharing a host

A DGX Spark has one GPU and one pool of unified memory. Two model servers
generally do not fit. Recipes therefore:

- check free memory against weights plus headroom in `preflight.sh`, and list
  the other running containers when the check fails;
- claim a port and verify it is free before starting;
- `mlock` weights so a load that does not fit fails outright rather than
  degrading into swap.

On this hardware the CPU and GPU share one pool, so a runtime that sizes itself
from "free GPU memory" is really reading `MemAvailable` — page cache included —
and will happily take memory the OS and the NVIDIA driver still need. Exhausting
the pool hangs the kernel: no OOM, no logs. A recipe whose runtime budgets
itself that way must therefore cap its budget *from the host side* and leave an
explicit reserve, rather than trusting a utilization fraction. See
`Qwen3.8-Flash-Next-NVFP4-vLLM` for a worked example (`HOST_RESERVE_GIB`, a
cgroup cap, and a memory watchdog). `Qwen3.8-Flash-Next-MLX4-TensorFold` is the other case:
TensorFold's budget is `MemAvailable` minus a fixed tenth of RAM with no knob,
so that recipe adds the cgroup cap and the watchdog around it instead.
