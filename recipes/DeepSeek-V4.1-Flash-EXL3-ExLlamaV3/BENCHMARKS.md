<!--
SPDX-License-Identifier: AGPL-3.0-only

Copyright (C) 2026 Victor Cruz
Copyright (C) 2026 amarjeet

Derived from one-spark-tp1/BENCHMARKS.md of
vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe, which is
Copyright (C) 2026 Victor Cruz and licensed AGPL-3.0-only.
Modified 2026-09-17 by amarjeet: measurements carried over unchanged; added the
provenance banner making clear these were measured on upstream's host and have
not been reproduced here, and the note on what the port changed.
-->

# TP1 measured results, with provenance

> **These numbers were measured by upstream, not on this host.**
>
> Every figure below is Victor Cruz's, measured on his DGX Spark with a local
> 64-byte re-laid build and reproduced here unchanged. This repository's rule is
> that measurements name the machine they came from, so they are attributed
> rather than re-badged. Nothing here has been re-measured on this Spark.
>
> They were also produced by driving ExLlamaV3 **directly**, through an `ENTRY`
> driver script that upstream did not publish. This port serves through TabbyAPI
> instead, which is a different code path and is **not** the path these numbers
> came from. See [README.md](README.md) → Known unknowns.

Nothing in this file is projected, scaled or carried over from another topology.

## Runtime identity

Recorded per upstream's `AGENTS.md` rule 8. This identity applies to every
number in this file.

| Field | Value |
|---|---|
| Hardware | 1 × NVIDIA DGX Spark (GB10), 128 GB unified LPDDR5X, ATS addressing mode |
| Engine | native ExLlamaV3 — **not** vLLM, **not** `vllm-exl3`, **not** TabbyAPI |
| ExLlamaV3 repo | `https://github.com/vcruz305/exllamav3.git` |
| ExLlamaV3 branch | `feat/gb10-ats-load` |
| ExLlamaV3 commit | `954a8ca6e59d` |
| CUDA | 13.0, `TORCH_CUDA_ARCH_LIST=12.1a` |
| Architecture class | `DeepseekV41ForCausalLM` |
| Drafter | `deepseek_v41_mtp.py` (DSpark / MTP block drafting, block size 5) |
| Model | local 64-byte re-laid build of `vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw`; that exact local build is not published |
| Topology | TP1, one local CUDA device |
| Context | `CTX=6144` |
| Batch | `max_batch_size=1`, single sequence |
| Speculative policy | `EXL3_DSPARK_CONF=0.7` where a drafter is used; otherwise none |
| EXL3 backend | ExLlamaV3 EXL3 kernels, per-expert mixed K (K1–K6 present) |
| MoE dispatch | per-expert path; grouped CUDA-graph modes measured slower and are off |

## Decode, no drafter

| Loading mode | tok/s | Load time | Resident |
|---|---:|---:|---:|
| Fully aliased (`EXL3_ATS_MMAP=1`, no copy) | 13.96 – 14.19 | — | weights stay reclaimable |
| Whole model copied into CUDA | **15.13 – 15.22** | 37.5 s | ~107 GiB |

Copying the whole model is +7.3% over full aliasing. This is the `nodraft`
profile.

## Decode, DSpark drafter, confidence 0.7

Warm and cold are reported separately and are never averaged together.

| Loading mode | Fresh prompt | Repeat prompt | Acceptance |
|---|---:|---:|---:|
| Aliased + drafter | 11.46 median | 16 – 17 | — |
| **Main copied, drafter aliased** (`EXL3_ATS_COPY='^(?!mtp\.)'`) | **17.53 median**, 19.82 mean | **20.11 – 24.67** | 0.889 |
| Interactive chat session | 11.8 cold | 17.4 warm | 0.74 |

The middle row is the `measured` profile and this recipe's default. 107 GiB of
main model plus a drafter cannot both be resident, so the drafter stays in page
cache. (Those are upstream's figures for its own build. This published pack
measures 111.16 GiB of text and a 7.39 GiB drafter against a 121.69 GiB pool --
see README.md -> Profiles.)

The first row is the `aliased` profile.

## Prefill

| Loading mode | Chunk | tok/s | Context |
|---|---:|---:|---|
| Warm | 4096 | 254 – 261 | 4k – 6k |
| Model resident in CUDA | 2048 | 154 – 229 | 2k – 6k |

Chunk 4096 does not fit once the model is resident, which is why `profiles.sh`
renders `chunk_size: 4096` only for the `aliased` profile and 2048 elsewhere.

## Zero-copy aliasing coverage

| Pack layout | Aliased | Copied |
|---|---:|---:|
| As published (tensors on arbitrary offsets) | 48.6 GiB | 67.4 GiB |
| Re-laid at 64-byte alignment | all text-model tensors | none |

This table is the entire justification for `relay.sh`. Skipping it does not
fail loudly; it just quietly copies 67.4 GiB.

## Measured negative results

Recorded so they are not re-tried. Same runtime identity as above.

| Change | Result | Disposition |
|---|---|---|
| `EXL3_MOE_GROUP_GRAPH=1` (per quantization-key groups, 11–22 graphs/layer) | 9.69 tok/s vs 10.97 baseline | off by default |
| `EXL3_MOE_GROUP_GRAPH=2` (per projection, 11–16 groups) | 10.43 tok/s vs 10.97 baseline | off by default |
| `EXL3_ATS_HUGEPAGE=1` | ~1–2%, within run-to-run noise | not recommended |
| Forcing a minimum draft length | slower | rejected |
| Draft early-exit | neutral | not enabled |
| `EXL3_MOE_MIXED_BSZ1=1` | ~5% warm decode, **greedy output not reproducible run to run** | **do not use** |

The grouped-MoE result is the important one: an exact per-slot mgemm loses to
the int8 GEMV path on this hardware, so reducing kernel launch count did not
help.

## Not measured, here or upstream

- **This port's actual serving path.** TabbyAPI has not been run end-to-end
  against this pack on any Spark. No TabbyAPI throughput number exists.
- **Native ExLlamaV3 TP2 / TP4.** ExLlamaV3 tensor parallelism is single-host
  only: one `multiprocessing.Process` per *local* CUDA index, payloads through
  `multiprocessing.shared_memory`, `EXLLAMA_MASTER_ADDR` defaulting to
  `127.0.0.1`, and no multi-host worker. It does not span two Sparks.
- **Long context beyond 6144**, quantized KV, and CUDA-graph capture for the
  heterogeneous mixed-K path.
- **Engram throughput.** The 189.13 GiB of Engram tables are read from disk by
  row during generation (the model card: "the Engram tables are read from disk"),
  competing for the page cache the aliased drafter lives in. No one has measured
  what that costs.
