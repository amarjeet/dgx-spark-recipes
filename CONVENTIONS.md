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
