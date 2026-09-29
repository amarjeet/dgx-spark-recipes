# DGX Spark recipes

Reproducible recipes for serving large models on an **NVIDIA DGX Spark** (GB10,
aarch64, 121.7 GiB unified memory). Each recipe pins a model revision, verifies
every weight shard by SHA-256, asserts its runtime version before a long load,
and exposes an OpenAI-compatible endpoint.

These are working notes that happen to be runnable. Numbers in a recipe README
were measured on the machine described there.

## Recipes

| Recipe | Model | Runtime | Weights | Notes |
|---|---|---|---|---|
| [`Ling-3.0-flash-Fin-GGUF-llamacpp`](recipes/Ling-3.0-flash-Fin-GGUF-llamacpp/) | [inclusionAI/Ling-3.0-flash-Fin](https://huggingface.co/inclusionAI/Ling-3.0-flash-Fin) — `bailingmoe3`, 124B total / 5.1B active | llama.cpp `server-cuda` | GGUF, 57–72 GiB | Hybrid KDA + MLA attention; full 262,144 context at ~46 tok/s; MTP self-speculation, no draft model |
| [`Qwen3.8-Flash-Next-NVFP4-vLLM`](recipes/Qwen3.8-Flash-Next-NVFP4-vLLM/) | [Mia-AiLab/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4) — multimodal, MXFP8 attention + NVFP4 PLE | vLLM `TP=1` | NVFP4, 98.66 GiB | PLE table memory-mapped off the GPU; 262K native / 524K via YaRN; ~1.11M FP8 KV tokens. **AGPL**, see its README |
| [`DeepSeek-V4.1-Flash-EXL3-ExLlamaV3`](recipes/DeepSeek-V4.1-Flash-EXL3-ExLlamaV3/) | [vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw](https://huggingface.co/vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw) — 552B backbone + ~196B Engram, 384 routed experts | **native ExLlamaV3** + TabbyAPI, `TP=1` | EXL3 1.59 bpw, 307.72 GiB on disk / 111.16 GiB resident | **No Docker.** GPU must be in ATS addressing mode; weights aliased from `mmap` rather than copied; 189 GiB of Engram read from disk. Needs a 64-byte re-lay before it will fit. **Serving path unqualified** and **AGPL-3.0-only** — see its README |
| [`Ternary-Bonsai-2-27B-GGUF-llamacpp-prism`](recipes/Ternary-Bonsai-2-27B-GGUF-llamacpp-prism/) | [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) — `qwen35`, 27.36B, ternary g128 at a true 1.72 bpw | **native PrismML llama.cpp fork**, built for `sm_121` | PQ2_0, 7.21 GB on disk / 6.70 GiB resident | **No Docker, and stock llama.cpp produces fluent nonsense on these files.** Hybrid attention (48 of 64 blocks linear) costs only 64 KiB/token of KV, so four concurrent 262,144-token slots fit in 77.9 GiB. 1040 t/s prefill, 29.8 t/s decode. Vision via mmproj; XML tool calls. Needle recalled at 254,032 tokens, though an open upstream bug (#27756) says it should not have been. The PTQ1_0 pack decodes 16% faster here, contradicting the model card — see its README |

## Quick start

```bash
git clone https://github.com/amarjeet/dgx-spark-recipes.git
cd dgx-spark-recipes/recipes/Ling-3.0-flash-Fin-GGUF-llamacpp

./download.sh iq4xs     # 64 GiB, checksum-verified, resumable
./preflight.sh iq4xs    # asserts build number, memory, shards, disk, port
./start.sh iq4xs        # serves on :8008
./scripts/smoke.py
./status.sh
./stop.sh
```

Or from the repo root, which runs preflight then start:

```bash
./scripts/run-ling-3.0-flash-fin-gguf.sh iq4xs
```

Every recipe has the same shape, so the second one reads the same as the first:

```bash
cd dgx-spark-recipes/recipes/Qwen3.8-Flash-Next-NVFP4-vLLM

./download.sh           # 98.66 GiB, pinned revision, checksum-verified
./preflight.sh          # image contents, memory budget, GPU tenancy, port
./start.sh              # serves on :8888
./scripts/smoke.py
```

The other two are native, not Docker, and each has a step before it will
serve. DeepSeek needs a re-lay:

```bash
cd dgx-spark-recipes/recipes/DeepSeek-V4.1-Flash-EXL3-ExLlamaV3

./download.sh           # 307.72 GiB, pinned revision, checksum-verified
./relay.sh              # 64-byte re-lay, ~118 GiB rewritten
./preflight.sh          # ATS mode, toolchain, symlink chain, memory budget, port
./start.sh              # serves on :8009 via TabbyAPI
```

Ternary Bonsai needs a build, because its runtime has no aarch64 binary to pull:

```bash
cd dgx-spark-recipes/recipes/Ternary-Bonsai-2-27B-GGUF-llamacpp-prism

./build.sh              # pinned llama.cpp fork, compiled for sm_121 (~20 min, once)
./download.sh pq2       # 7.8 GB, pinned revision, checksum-verified
./preflight.sh wide     # fork binary, sm_121, Hadamard metadata, budget, port
./start.sh wide         # serves on :8010
./scripts/smoke.py      # includes a needle test at depth -- read that recipe's README
```

**One GPU, one pool.** These recipes cannot run at the same time — each default
profile wants 60–111 GiB of a 121.7 GiB unified pool. Preflight refuses to start
a second and names what is holding the memory.

Note that the DeepSeek and Ternary Bonsai recipes are **native**, so unlike the
Docker recipes nothing caps them from outside: there is no cgroup limit and no
watchdog between a bad budget and a hung kernel. Each one's `profiles.sh`
derives the budget per profile and `preflight.sh` refuses a profile that cannot
fit, but that check is the only guard.

## Requirements

- NVIDIA DGX Spark (GB10, `aarch64`). Recipes assert the architecture and refuse
  to run elsewhere — the measurements and memory budgets assume this machine.
- Docker with the NVIDIA container runtime (`--gpus all`).
- `curl`, `python3` (stdlib only — no pip installs), `nvidia-smi`.
- A Hugging Face token for gated repos. Export `HF_TOKEN`, or log in with
  `hf auth login`; recipes read the standard token file. The Hub works
  anonymously at a lower rate limit.

## Storage: the DGX Spark layout

Every recipe here follows one storage convention, called the **DGX Spark
layout**. In one sentence: *heavyweight reusable data lives exactly once, at
each tool's own standard default location, and never inside a recipe
directory.*

So a recipe directory holds code, config and manifests — a few hundred
kilobytes. Model weights land in `~/.cache/llama.cpp`, the Hugging Face cache
stays at `~/.cache/huggingface`, and both are shared with every other recipe on
the host rather than downloaded per recipe. For Docker recipes the host cache is
bind-mounted onto *the same path the tool defaults to inside the container*, so
nothing needs a cache environment variable at all.

Every path is an environment-overridable variable whose default is that standard
location: the override makes a recipe portable, the default makes it correct
with no setup. [CONVENTIONS.md](CONVENTIONS.md) has the full rules, the complete
environment variable reference, and the reasoning about sharing one GPU between
servers.

The convention is also packaged as an agent skill, so a coding agent working in
this repo applies it without being told:

**[`amarjeet/agent-skills` → `dgx-spark-layout`](https://github.com/amarjeet/agent-skills/tree/main/skills/dgx-spark-layout)**
— storage rules for serving recipes.
**[`amarjeet/agent-skills` → `dgx-spark-training-layout`](https://github.com/amarjeet/agent-skills/tree/main/skills/dgx-spark-training-layout)**
— the companion for fine-tuning runs.

Install either by copying it into `.agents/skills/`, `.claude/skills/` or
`.cursor/skills/`.

## Repository layout

```
dgx-spark-recipes/
├── CONVENTIONS.md      storage rules, recipe shape, reproducibility
├── recipes/            one self-contained directory per model+runtime
└── scripts/            thin launchers (preflight + start); set no paths
```

## Security note

Recipes bind `0.0.0.0` and publish their port on all interfaces, with **no API
key and CORS open** — llama.cpp warns about this at startup. That is fine behind
a trusted network and not fine anywhere else. Set `HOST=127.0.0.1` to keep a
server on loopback.

## License

[MIT](LICENSE) for this repository, **except** where a recipe says otherwise.

Two recipes are licensed differently and cannot be otherwise. Both are ports of
AGPL work, so AGPL §5(c) requires the derived works to stay AGPL. Each ships its
own `LICENSE` and documents the reasoning in its README.

| Recipe | Licence | Copyright |
|---|---|---|
| [`Qwen3.8-Flash-Next-NVFP4-vLLM`](recipes/Qwen3.8-Flash-Next-NVFP4-vLLM/) | **AGPL-3.0-or-later** | © 2026 [MiaAI Lab](https://x.com/MiaAI_lab) (original), © 2026 amarjeet (port) |
| [`DeepSeek-V4.1-Flash-EXL3-ExLlamaV3`](recipes/DeepSeek-V4.1-Flash-EXL3-ExLlamaV3/) | **AGPL-3.0-only** | © 2026 Victor Cruz (original), © 2026 amarjeet (port) |

**The two tags differ, and the difference is load-bearing.** MiaAI Lab chose
`-or-later`; Victor Cruz chose `-only`. Marking the DeepSeek port `-or-later`
would grant permission under a future licence version that upstream never gave,
so each carries its own upstream's tag exactly. One consequence: code may move
from the `-or-later` recipe into the `-only` one — "or later" permits use under
v3 — but **not** the other way.

- [`Qwen3.8-Flash-Next-NVFP4-vLLM`](recipes/Qwen3.8-Flash-Next-NVFP4-vLLM/) is a
  port of [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
  and carries upstream's code verbatim under `files/`.
- [`DeepSeek-V4.1-Flash-EXL3-ExLlamaV3`](recipes/DeepSeek-V4.1-Flash-EXL3-ExLlamaV3/) is a
  port of [vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe)
  (`one-spark-tp1`). It also serves through TabbyAPI, which is independently
  AGPL-3.0 and installed by you rather than vendored here.

MIT code may be incorporated into those recipes; their AGPL code may **not** be
copied back into the MIT parts of this repository. Per AGPL §5, holding all of
them in one repository is an aggregate and does not make the other recipes AGPL.
