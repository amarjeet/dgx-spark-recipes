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
| `OUT_DIR` | `${XDG_STATE_HOME:-~/.local/state}/dgx-spark-recipes/<recipe>` | Bench results and verification stamps. Never inside the recipe directory. |
| `XDG_STATE_HOME` | `~/.local/state` | Standard base for `OUT_DIR`; honored rather than assumed. |

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
| `RESTART_POLICY` | `unless-stopped` | Docker restart policy; survives reboot. `no` for a one-off run. |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | `1.0` / `0.95` / `20` | Server-side sampling defaults, so clients that send nothing still get the model card's recommendation. |

**Preflight and tooling.** Thresholds and helper knobs.

| Variable | Default | Means |
|---|---|---|
| `MIN_LLAMA_BUILD` | recipe-specific | Minimum llama.cpp build number. `preflight.sh` reads the real number out of the image and refuses to run below it, rather than trusting a mutable tag. |
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
download.sh     checksum-verified, resumable weight download
preflight.sh    assert everything start.sh depends on, before a long load
start.sh        launch the server
stop.sh         remove the container
status.sh       is it up, what is it serving, is the host healthy
bench.sh        measurements
manifests/      pinned revision, per-shard sizes and SHA-256
scripts/        stdlib-only Python helpers
```

Every script resolves its own directory, so a recipe runs from any working
directory and can be cloned anywhere.

## Reproducibility

- **Pin the model revision.** Manifests record the Hub revision, every shard's
  byte size and its SHA-256.
- **Pin the runtime.** Container tags are mutable. Where a recipe needs a
  minimum build, `preflight.sh` reads the build number out of the image and
  refuses to proceed below it rather than trusting the tag.
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
cgroup cap, and a memory watchdog).
