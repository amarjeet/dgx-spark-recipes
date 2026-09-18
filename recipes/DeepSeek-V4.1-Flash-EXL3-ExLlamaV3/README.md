<!--
SPDX-License-Identifier: AGPL-3.0-only

Copyright (C) 2026 Victor Cruz
Copyright (C) 2026 amarjeet

Derived from one-spark-tp1/README.md of
vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe, which is
Copyright (C) 2026 Victor Cruz and licensed AGPL-3.0-only.
Modified 2026-09-17 by amarjeet: storage paths moved to each tool's own default,
the unpublished ENTRY driver replaced with a TabbyAPI serving path, the recipe
given this repository's script shape, and the pinned revision + manifest added.
-->

# DeepSeek-V4.1-Flash, EXL3 1.59 bpw, native ExLlamaV3 on one DGX Spark

Single DGX Spark (GB10), TP1. **No Docker, no vLLM** — this is the only recipe
in this repository that runs natively, in a virtualenv, against a source build.

A 1.59 bpw DeepSeek-V4.1-Flash pack is roughly 107 GiB resident. One Spark has
128 GB of unified LPDDR5X shared between CPU and GPU. That fits, but only if the
loader does not keep a second copy of the weights, and only if the drafter is not
made resident alongside the main model.

Native ExLlamaV3 on a GB10 can do this because the GPU runs in **ATS addressing
mode**: it shares the process page tables and can read host virtual addresses
directly, so weight tensors are aliased straight out of an `mmap` of the
safetensors files instead of being copied into CUDA allocations.

Confirm the mode before anything else — `preflight.sh` checks it, but it is the
one precondition nothing can work around:

```bash
nvidia-smi -q | grep -i "addressing mode"
    Addressing Mode                   : ATS
```

## Credit

**This recipe is a port of
[vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe)
(`one-spark-tp1`), Copyright © 2026 Victor Cruz, AGPL-3.0-only.**

Victor Cruz did the hard part. Everything that makes this checkpoint fit on one
Spark is his work, not ours:

- the GB10 **ATS zero-copy loader** and the `DeepseekV41ForCausalLM` architecture
  class, which exist only on the `feat/gb10-ats-load` branch of his ExLlamaV3
  fork — upstream ExLlamaV3 registers `DeepseekV4ForCausalLM` and cannot load
  this model at all;
- the discovery that a **64-byte re-lay** is what makes aliasing work, and the
  `align_safetensors.py` tool that does it;
- the `EXL3_ATS_COPY='^(?!mtp\.)'` placement split — copy everything except the
  drafter — which is the configuration that fits and is fastest;
- **every measured number** in [`BENCHMARKS.md`](BENCHMARKS.md), including the
  negative results, which are the expensive kind to produce;
- the TabbyAPI configuration this recipe serves through.

Our contribution is packaging: pinning the revision and adding a checksum
manifest, moving the storage paths to each tool's own default location, giving
it this repository's script shape, and closing the gap where upstream's launcher
`exec`s an `ENTRY` driver script that was never published.

## Storage

Nothing heavy lives in this directory. This is a **native** recipe, so there is
no bind mount to redirect anything — each variable below is the one the owning
tool itself reads. See [`CONVENTIONS.md`](../../CONVENTIONS.md).

| Data | Path | Variable |
|---|---|---|
| EXL3 pack, 307.72 GiB | `~/.cache/huggingface` | `HF_HOME` |
| 64-byte re-laid pack | `~/dgx-spark/base-models/DSV4.1-Flash-SAGE-EXL3-1.59bpw-a64` | `MODEL_ROOT` |
| torch JIT extensions | `~/.cache/torch_extensions` | `TORCH_EXTENSIONS_DIR` |
| ExLlamaV3 fork checkout | `~/src/exllamav3` | `EXL3_SRC` |
| `vllm-exl3` (aarch64 build patch) | `~/src/vllm-exl3` | `VLLM_EXL3_SRC` |
| TabbyAPI checkout | `~/src/tabbyAPI` | `TABBY_DIR` |
| virtualenv | `~/venvs/exl3-v41` | `VENV` |
| verify stamps, archived logs | `~/.local/state/dgx-spark-recipes/DeepSeek-V4.1-Flash-EXL3-ExLlamaV3` | `OUT_DIR` |

Upstream hardcoded `/models/...`, `/home/markus/models` and a recipe-private
`~/.cache/torch_extensions_v41`. All three are gone.

Two notes on that table, because the reasoning matters more than the values:

- **`TORCH_EXTENSIONS_DIR` is inert on the documented build path.** ExLlamaV3 is
  built ahead-of-time by `setup.py` (`CUDAExtension` + `BuildExtension`), not
  through `torch.utils.cpp_extension.load()`, so neither the upstream value nor
  this one is read unless you take the JIT path. It is set to the shared default
  rather than a recipe-private one because a recipe-private cache default is the
  thing this repository's conventions exist to prevent — not because the `_v41`
  suffix was causing rebuilds.
- **`~/src`, not `~/dgx-spark/src`.** This host already keeps third-party
  checkouts in `~/src`. The layout skill has no row for source checkouts, and its
  silence is a cue to follow the host rather than licence to invent a second
  convention.
- Upstream's venv is `exl3_v41` (underscore); this uses `exl3-v41` (hyphen). If
  you follow upstream's build block verbatim you will end up with two venvs.

### The re-lay binds this to the HF cache

`relay.sh` **symlinks** shards that are already on the 64-byte grid rather than
copying them. `align_safetensors.py` resolves with `os.path.realpath`, so
`MODEL_ROOT` points **directly at the blob** in
`~/.cache/huggingface/hub/models--…/blobs/<sha256>` — one hop, not through the
snapshot directory.

Three consequences:

- Deleting the *snapshot* directory does not break the pack; deleting *blobs*
  does. Hugging Face's own GC counts references from snapshots and knows nothing
  about your external symlinks, so a prune that collects an apparently
  unreferenced blob is exactly the failure mode. **Do not prune the cache.**
- The realpath is frozen at re-lay time, so moving `HF_HOME` dangles every
  symlink.
- A truncated blob still resolves, so `preflight.sh` checks the symlink target's
  **size** against the manifest, not just its existence.

### Why the skip pattern is `.engram.` and not upstream's `.engram.embed.`

`align_safetensors.py` decides per **shard**: if any tensor is over
`--min-bytes`, unmatched by `--skip` and off the grid, the whole shard is
rewritten. Shards 16 and 17 hold only Engram tensors — but each also holds
`engram.wkv.weight`, 157 MB, which `.engram.embed.` does not match and which is
off the 64-byte grid in both (offset `%64` = 24 and 48).

With upstream's pattern both 94.6 GiB shards are rewritten in full: **616 GiB
peak instead of 426 GiB**, and 189 GiB of pointless copying. Widening to
`.engram.` skips them.

Peak disk with the fix is about **426 GiB**: 307.72 GiB of source pack plus
118.56 GiB of rewritten text and drafter shards.

### Engram is read from disk, not made resident

Shards 16 and 17 are 189.13 GiB — 61% of the download — and they are **not**
part of the resident footprint. The model card is explicit: *"~101 GiB of routed
experts on device; the Engram tables are read from disk"*, and the fork's
`doc/gb10_ats_loading.md` describes `EXL3_ENGRAM_ATS` and `EXL3_ENGRAM_PREFETCH`
reading rows on demand.

So they must be downloaded, and they are read by row off NVMe during generation,
competing for the same page cache the aliased drafter lives in. That is a
throughput property of this recipe, not a detail.

## Quick start

```bash
cd recipes/DeepSeek-V4.1-Flash-EXL3-ExLlamaV3

# one-time: build the fork and install TabbyAPI into the same venv (see Build)
./download.sh          # 307.72 GiB, pinned revision, checksum-verified, resumable
./relay.sh             # 64-byte re-lay, ~20 min on NVMe
./preflight.sh         # ATS mode, memory, symlink chain, toolchain, port
./start.sh             # serves on :8009
./scripts/smoke.py
./stop.sh
```

Or from the repo root, which runs preflight then start:

```bash
./scripts/run-deepseek-v4.1-flash-exl3.sh
```

## Profiles

The main model and the drafter cannot both be resident. Each profile is a
different answer to that; all three numbers are upstream's.

| Profile | Copies | Resident | Decode tok/s | Fits? |
|---|---|---:|---:|---|
| `measured` *(default)* | everything except `mtp.*` | 111.16 GiB | 17.53 median | yes, 4.5 GiB slack |
| `aliased` | nothing | ~0 (page cache) | 11.46 median | yes |
| `nodraft` | everything | 118.56 GiB | 15.13 – 15.22 | **no — see below** |

Resident figures are measured from this pack's shard headers and recorded under
`tensor_bytes` in the manifest: 111.16 GiB of text, 7.39 GiB of drafter,
189.13 GiB of Engram. The tok/s figures are upstream's, from a different build.

**`nodraft` does not fit this pack on this host.** 118.56 GiB resident plus 6 GiB
headroom exceeds a 121.69 GiB pool. Upstream measured "whole model copied into
CUDA" at ~107 GiB resident on an 18-shard, 110 GiB build; the published pack is
17 shards and 118.56 GiB of text+drafter. `profiles.sh` computes this and
`preflight.sh` refuses the profile rather than letting it hang the kernel. It is
kept because it is upstream's measured configuration and becomes viable on a
larger pool or a smaller pack.

`EXL3_ATS_COPY` is read with `${VAR-default}`, not `${VAR:-default}`: **empty is
a meaningful value** meaning "alias every tensor", so
`EXL3_ATS_COPY= ./start.sh` is the documented way to ask for it. Upstream's
`run_tp1.sh` uses `${EXL3_ATS_COPY:-^(?!mtp\.)}`, which cannot express "alias
everything" at all — the `aliased` profile is a fix over upstream, not a copy.

## Build

The branch is required and the **commit** is what is pinned — the branch tip
moves.

```bash
git clone https://github.com/vcruz305/exllamav3.git ~/src/exllamav3
git -C ~/src/exllamav3 checkout 954a8ca6e59d48c3e3462068ecf083fe9990f4dc

# The aarch64 patch is NOT in the fork at this commit. It lives here.
git clone https://github.com/vcruz305/vllm-exl3.git ~/src/vllm-exl3

python3 -m venv ~/venvs/exl3-v41
source ~/venvs/exl3-v41/bin/activate

# torch first: setup.py degrades to a metadata-only install without it, and
# aarch64 + CUDA 13 needs a matching wheel rather than the default index.
pip install torch --index-url https://download.pytorch.org/whl/cu130

cd ~/src/exllamav3
python3 ~/src/vllm-exl3/tools/patch_exllamav3_aarch64.py exllamav3/exllamav3_ext

CUDA_HOME=/usr/local/cuda-13.0 \
TORCH_CUDA_ARCH_LIST=12.1a \
MAX_JOBS=8 \
  pip install --no-build-isolation --no-deps .

# --no-deps installs the package and the extension but none of its requirements
pip install -r requirements.txt

python3 -c "import exllamav3, exllamav3_ext; print('ok')"
python3 -c "from exllamav3.architecture import deepseek_v41, deepseek_v41_mtp; print('v41 ok')"
```

Upstream's README says to run `util/patch_exllamav3_aarch64.py` from the fork.
**That file does not exist at the pinned commit** — `util/` contains
`align_safetensors.py` and thirteen other tools, but no aarch64 patch. The copy
in [`vcruz305/vllm-exl3`](https://github.com/vcruz305/vllm-exl3) at
`tools/patch_exllamav3_aarch64.py` is the real one. It replaces
`__builtin_ia32_pause` / `_mm_pause` with `std::this_thread::yield()` and stubs
the AVX2/AVX-512 target functions so the extension compiles on aarch64. Without
it the build fails; upstream's `2>/dev/null || true` would hide that until the
compile errors.

`preflight.sh` verifies the *installed* package is the fork by checking for the
V4.1 architecture class, the MTP drafter and the `EXL3_ATS_COPY` loader — not by
comparing `exllamav3.__file__`, which after `pip install` points at
site-packages whether the source was the fork or a stock wheel.

## Serving

TabbyAPI is ExLlamaV3's own API server and the only one exposing `draft_mode:
mtp`, which is the drafting mode this model's DSpark drafter needs.

```bash
git clone https://github.com/theroyallab/tabbyAPI ~/src/tabbyAPI
source ~/venvs/exl3-v41/bin/activate
cd ~/src/tabbyAPI && pip install .          # plain install, no extras
```

**Install into the same venv with no extras, and never run TabbyAPI's own
`start.sh` / `start.py` / `update_scripts/`.** `start.py` auto-detects
`nvidia-smi`, picks a CUDA feature set and runs pip, which can bump
`tokenizers`, `numpy` or `huggingface_hub` underneath your built extension.
This recipe's `start.sh` `exec`s `main.py` directly and never invokes theirs.

A correction to upstream's warning, which this port repeats more accurately:
upstream says TabbyAPI "enforces the latest Exllamav3 version". On **aarch64**
that is not the immediate risk — every exllamav3 and torch wheel in TabbyAPI's
`pyproject.toml` is gated `platform_machine == 'x86_64'` or
`platform_system == 'Windows'`, and exllamav3 is not in its base dependencies at
all, so `pip install .[cu13]` matches nothing here. The real risk is the
transitive dependency churn above. Either way, re-verify after any TabbyAPI
update; `preflight.sh` checks before every launch.

`start.sh` renders `tabbyapi/config.yml` — a template with `@PLACEHOLDER@`
tokens, not absolute paths — into `$OUT_DIR` and exports the ATS variables
before launching, because TabbyAPI has no knowledge of them. Without
`EXL3_ATS_COPY` the model is copied wholesale and will not fit.

## Status: not a qualified path

**TabbyAPI has not been run end-to-end against this pack on a DGX Spark**, by
upstream or by this port. Upstream marks its TabbyAPI directory "configuration
guidance, not a qualified path" and publishes no TabbyAPI throughput numbers;
neither do we. The measured numbers in [`BENCHMARKS.md`](BENCHMARKS.md) come
from driving ExLlamaV3 directly through a driver script upstream did not publish.

This is a starting point, not a supported configuration.

## Known unknowns

Recorded rather than guessed. Each is a real test, not a formality.

Upstream's four, carried over unchanged:

- Whether `draft_mode: mtp` needs `draft_model_name` set when the MTP head lives
  inside the main pack as `mtp.*` tensors.
- Whether TabbyAPI's loader path preserves the `EXL3_ATS_COPY` placement split,
  or forces its own device placement and defeats the aliasing.
- Whether TabbyAPI tolerates the `__align_pad__.*` tensors a re-laid pack
  contains. The fork's loader skips that prefix and TabbyAPI calls the same
  loader, so it should — untested.
- Quantized KV (`cache_mode: "8,8"`) behaviour for this pack on GB10.

Two this port adds:

- **Whether `RELAY_SKIP=.engram.` is safe.** The justification is that the alias
  grid is the dtype's item size, and every tensor in those shards is 1-byte
  (`F8_E4M3` / `F8_E8M0`) or 2-byte (`BF16`), never the `int16` the 64-byte
  trellis rule exists for. That comes from the fork's prose documentation, not
  from reading the loader. **Verify after the first load** by printing
  `config.stc.ats_bytes` and checking `copied` is ~0.3 GiB rather than 0; the
  worst case if the reasoning is wrong is that the loader copies 300 MiB.
- **Chat template is ours, and partial.** The pack's `tokenizer_config.json` is
  801 bytes with no `chat_template`, DeepSeek publish no Jinja template for
  V4.1 at all (the model card says so and ships a Python reference encoder
  instead), and TabbyAPI bundles only alpaca, chatml and lfm2.
  `tabbyapi/deepseek-v4.1-chat.jinja` is derived from that encoder and is the
  `PROMPT_TEMPLATE` default. It covers `thinking_mode="chat"` with string
  content — system/user/assistant, multi-turn, generation prompt — and matched
  the encoder on 18/18 conversation shapes plus upstream's published vectors.
  It does **not** cover thinking mode, tool calls, images, `latest_reminder` or
  the `task`/`mask` fields; those raise rather than mis-encode. Set
  `PROMPT_TEMPLATE=` (empty) to serve `/v1/completions` only.

## Security

`HOST` defaults to `127.0.0.1` and `TABBY_DISABLE_AUTH` to `1`, so the server is
loopback-only with no API key. TabbyAPI's own default is auth *enabled*, which
writes an `api_tokens.yml` into its working directory and 401s every
unauthenticated request including the smoke test.

`start.sh` refuses to start with auth disabled and a non-loopback `HOST`. If you
bind wider, set `TABBY_DISABLE_AUTH=0` and pass the generated key yourself. If
you genuinely want a keyless server on a network you control,
`TABBY_ALLOW_INSECURE_BIND=1` turns that refusal into a warning — it exists so
the choice is stated, not stumbled into. Never set it on an untrusted network:
anyone who can route to the port can use the model, with `allowed_origins` open.

## Deviations from `CONVENTIONS.md`

Deliberate, so they read as choices rather than oversights:

- **No `status.sh`, no `bench.sh`.** The repository's recipe shape lists both.
  Benchmarking a path that has never completed a qualified run would produce
  numbers worth less than upstream's, which [`BENCHMARKS.md`](BENCHMARKS.md)
  already records with full provenance.
- **An extra `relay.sh`.** No other recipe has a build step between download and
  launch. `start.sh` cannot work without it.
- **Native, not Docker.** Every other recipe bind-mounts host caches onto the
  tool's in-container default. Here there is no container, so the variables must
  be the ones the tools actually read.

## Files

```
profiles.sh     paths, pins, the profile table and shared helpers (sourced)
download.sh     checksum-verified, resumable fetch of the pinned revision
relay.sh        64-byte re-lay; writes MODEL_ROOT
preflight.sh    ATS mode, toolchain, checkpoint, symlinks, memory, port
start.sh        render config, export ATS vars, launch TabbyAPI
stop.sh         terminate by pidfile
manifests/      pinned revision, per-file size and SHA-256
scripts/        stdlib-only helpers
tabbyapi/       config template and serving notes
```

## License

**AGPL-3.0-only** — the same licence as upstream, and the only one this port can
carry.

```
Copyright (C) 2026 Victor Cruz    — original work
Copyright (C) 2026 amarjeet       — this port
```

The full text is in [`LICENSE`](LICENSE). Every file carries an
`SPDX-License-Identifier`, and files derived from upstream also carry both
copyright lines plus a dated note saying what was changed, as AGPL §5(a)
requires of a modified work.

**Why our `LICENSE` differs from upstream's.** Upstream's `LICENSE` is a
490-byte pointer — an SPDX line, a copyright line and a URL — which is why
GitHub's detector reports `NOASSERTION` for that repository rather than AGPL.
The authority for the licence is that file's `SPDX-License-Identifier:
AGPL-3.0-only` plus `one-spark-tp1/README.md` ("Recipe content in this folder is
AGPL-3.0-only"). AGPL §4 and §5 require conveying a *copy* of the License with
the work, so this directory ships the full 661-line FSF text. That is more
compliant than upstream, not merely consistent with the sibling recipe.

### Why `-only` and not `-or-later`

This matters, and it differs from the other AGPL recipe in this repository.

| | Licence |
|---|---|
| `Qwen3.8-Flash-Next-NVFP4-vLLM` | AGPL-3.0-**or-later** (MiaAI Lab's choice) |
| **This recipe** | AGPL-3.0-**only** (Victor Cruz's choice) |

Upstream's `LICENSE` says `AGPL-3.0-only`. Marking this port `-or-later` would
grant permission to use it under a future licence version that upstream did not
give, so the tag is carried across exactly. The two AGPL recipes in this
repository legitimately differ.

`scripts/download_snapshot.py` is reused from the `-or-later` recipe. That needs
no licence theory at all: both copies are `Copyright (C) 2026 amarjeet`, written
for this repository, so it is the author relicensing their own code. Its header
here was rewritten rather than having its SPDX tag swapped — the original ends
"part of the same AGPL-3.0-or-later combined work … which derive from
MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark", which would be a false
provenance statement in this directory.

The accurate copy-back rule is narrower than "nothing may move": **nothing
derived from Victor Cruz's work may be copied into the `-or-later` recipe or the
MIT root**, and a third party's future contribution to this directory could not
move either without their permission. Our own originally-authored files are not
trapped here.

One boundary to police: `scripts/run-deepseek-v4.1-flash-exl3.sh` at the repo
root carries no SPDX header and is MIT, which is correct **only while it stays a
thin launcher**. The moment it copies option parsing or environment logic out of
`start.sh` it becomes a derivative work and must go AGPL. Thin launchers grow;
this is the rule.

### Then why is the rest of the repo MIT?

Because that direction is fine. MIT is permissive and GPL-compatible, so MIT code
can be incorporated into an AGPL work; AGPL code cannot be relicensed MIT.

A repository may hold differently-licensed subdirectories: AGPL §5 treats a
covered work stored alongside separate, independent works as an "aggregate", and
being in an aggregate does not spread the AGPL to sibling recipes. What it does
require is that the boundary be unambiguous, so this directory has its own
`LICENSE`, every file here is marked, and the
[repository README](../../README.md) names it as an exception to the MIT default.

**If you fork or redistribute:** take this directory as AGPL-3.0-only. Copying
parts of it into an MIT project is the one thing that is not permitted.

### The network clause

AGPL §13 applies to **these scripts**, not to the inference you serve. If you
modify the launcher or the re-lay wrapper and offer that modified version to
users over a network, §13 obliges you to offer those users the corresponding
source. Answering chat completions with the model does not trigger it.

Note that TabbyAPI is itself AGPL-3.0, so serving through it carries its own §13
obligation independently of this recipe.

### What this licence does not cover

Nothing here relicenses any of the following, and none of it is redistributed:

| Component | Licence | Role |
|---|---|---|
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | MIT, © 2025 Turboderp | EXL3 trellis format, codebooks, packed execution |
| [vcruz305/exllamav3](https://github.com/vcruz305/exllamav3) @ `954a8ca6e59d48c3e3462068ecf083fe9990f4dc` | MIT (inherits Turboderp's notice verbatim) | the ATS zero-copy loader, `DeepseekV41ForCausalLM`, `util/align_safetensors.py` |
| [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3) | **AGPL-3.0-only** | `tools/patch_exllamav3_aarch64.py`, a required build step on aarch64; fetched, not vendored |
| [TabbyAPI](https://github.com/theroyallab/tabbyAPI) | **AGPL-3.0** | the API server; installed by you, not vendored |
| [`vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw) @ `0c29707b4fbc` | MIT | the EXL3 pack; governed by its own terms on the Hub |
| [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | MIT | model architecture and checkpoint origin |
| NVIDIA CUDA, drivers, DGX Spark | proprietary | NVIDIA products with their own licences |

Upstream keeps a `THIRD_PARTY_NOTICES.md`, and its `AGENTS.md` rule 10 asks that
third-party attribution be preserved with exact URLs and commits — which the
table above does. That file itself is not carried over because it describes the
**vLLM TP2/TP4 path** this port does not use: it pins ExLlamaV3 at
`be57335b…` (turboderp-org) and never mentions the `vcruz305/exllamav3` fork
that this recipe is built on, so it does not describe this runtime at all.
