<!--
SPDX-License-Identifier: AGPL-3.0-only

Copyright (C) 2026 Victor Cruz
Copyright (C) 2026 amarjeet

Derived from one-spark-tp1/tabbyapi/README.md of
vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe, which is
Copyright (C) 2026 Victor Cruz and licensed AGPL-3.0-only.
Modified 2026-09-17 by amarjeet: the hand-rolled launcher is replaced by
../start.sh, and config.yml became a rendered template rather than a file with
an absolute home directory committed into it.
-->

# TabbyAPI on one Spark (TP1)

**Status: configuration guidance, not a qualified path.**

TabbyAPI is the official API server for ExLlamaV3 and exposes the drafting mode
this model needs (`draft_mode: mtp`). It has **not** been run end-to-end against
this pack on a DGX Spark — not by upstream, and not by this port — so no
TabbyAPI throughput numbers are published and nothing here should be read as a
pass.

The measured numbers in [`../BENCHMARKS.md`](../BENCHMARKS.md) come from driving
ExLlamaV3 directly, not through TabbyAPI.

## The one thing that will break this

TabbyAPI installs ExLlamaV3 itself and, per its own documentation, **"enforces
the latest Exllamav3 version for compatibility purposes"**. Any upgrade through
its GPU-library extra will overwrite a custom install.

This recipe needs a **fork commit**, not the released wheel:

```
https://github.com/vcruz305/exllamav3.git @ 954a8ca6e59d (branch feat/gb10-ats-load)
```

Upstream ExLlamaV3 registers `DeepseekV4ForCausalLM`. `DeepseekV41ForCausalLM`
and the GB10 ATS zero-copy loader only exist on that branch. If TabbyAPI
replaces it, the model stops loading.

So:

1. Build the fork into a venv first — [`../README.md`](../README.md) → Build.
2. Install TabbyAPI into **that same venv**, without its GPU-library extra.
3. Re-install the fork with `pip install .` after any TabbyAPI update.

`../preflight.sh` checks this before every launch and names it explicitly when
it has happened, so you find out in a check list rather than mid-load:

```bash
python3 -c "from exllamav3.architecture import deepseek_v41, deepseek_v41_mtp; print('v41 ok')"
python3 -c "import exllamav3, exllamav3_ext; print('ext ok')"
```

Note that `exllamav3.__file__` alone proves nothing: after `pip install .` it is
site-packages either way. `../preflight.sh` checks for the V4.1 architecture,
the MTP drafter and the `EXL3_ATS_COPY` loader instead.

## ATS environment variables

TabbyAPI has no knowledge of the zero-copy loader. These are read from the
process environment by the fork's loader and **must be exported before TabbyAPI
starts** — [`../start.sh`](../start.sh) does it for you:

```bash
export EXL3_ATS_MMAP=1
export EXL3_ATS_COPY='^(?!mtp\.)'   # copy everything except the drafter into CUDA
export EXL3_DSPARK_CONF=0.7
```

Without `EXL3_ATS_COPY`, either everything is aliased (slower) or everything is
copied (does not fit: 111.16 GiB of text plus a 7.39 GiB drafter against a
121.69 GiB pool on this host).

## Config is a template, not a file to edit

[`config.yml`](config.yml) carries `@PLACEHOLDER@` tokens. `../start.sh` renders
them from `profiles.sh` into `$OUT_DIR/tabbyapi-config.<profile>.yml` and points
TabbyAPI at that.

This is deliberate. Upstream committed `model_dir: /home/markus/models` — a
specific person's home directory — into the tracked config, which is the kind of
thing this repository's storage conventions exist to prevent. Rendering keeps
`MODEL_ROOT` overridable and keeps machine-specific paths out of git.

To see exactly what would run, without launching:

```bash
../start.sh --no-launch
```

Two rendered settings carry real consequences:

- `chunk_size` — 2048 once the model is resident in CUDA memory. 4096 measured
  faster while weights were still aliased but does not fit with the model copied
  in, so the `aliased` profile renders 4096 and the others 2048.
- `tensor_parallel: false` — ExLlamaV3 tensor parallelism is single-host only
  (one `multiprocessing.Process` per *local* CUDA index, shared-memory payloads,
  `EXLLAMA_MASTER_ADDR` defaulting to `127.0.0.1`). It does not span two Sparks.

## Known unknowns

Recorded rather than guessed:

- Whether `draft_mode: mtp` needs `draft_model_name` set when the MTP head lives
  inside the main pack as `mtp.*` tensors. If TabbyAPI errors asking for a draft
  model, set `draft_model_dir` and `draft_model_name` to the same values as the
  `model` section.
- Whether TabbyAPI's loader path preserves the `EXL3_ATS_COPY` placement split,
  or forces its own device placement and defeats the aliasing.
- Whether TabbyAPI tolerates the `__align_pad__.*` tensors in a re-laid pack.
  The fork's loader skips that prefix, and TabbyAPI calls the same loader, so it
  should — untested.
- Quantized KV (`cache_mode: "8,8"`) behaviour for this pack on GB10.

Until these are run, treat this folder as a starting point rather than a
supported configuration.
