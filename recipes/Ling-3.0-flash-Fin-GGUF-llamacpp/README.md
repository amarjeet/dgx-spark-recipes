# Ling-3.0-flash-Fin on llama.cpp (DGX Spark / GB10)

Serves [`inclusionAI/Ling-3.0-flash-Fin`](https://huggingface.co/inclusionAI/Ling-3.0-flash-Fin)
— a finance-domain continued-training of Ling-3.0-flash — on a single GB10 as an
OpenAI-compatible endpoint, using
[bartowski's imatrix GGUFs](https://huggingface.co/bartowski/Ling-3.0-flash-Fin-GGUF)
and the published `ghcr.io/ggml-org/llama.cpp:server-cuda` image.

```
./download.sh iq4xs     # 64 GiB, checksum-verified, resumable
./preflight.sh iq4xs    # asserts build number, memory, shards, port
./start.sh iq4xs        # serves on :8008 (all interfaces)
./scripts/smoke.py      # finance reasoning + thinking-mode check
./status.sh
./stop.sh
```

## Why GGUF and not SGLang/vLLM

The usual pattern on this hardware is NVFP4 weights under SGLang or vLLM. That
is not available for this checkpoint:

| | |
|---|---|
| Architecture | `bailingmoe3` MoE, 124B total / **5.1B active** |
| Layers | 42, hybrid — KDA linear-attention layers plus a minority of full-attention (MLA) layers |
| Experts | 512 routed, 8 active + 1 shared |
| Context | 262,144 |
| Released precision | **BF16 only** (~248 GB) |
| Extras | 1 MTP (multi-token-prediction) layer |

BF16 is roughly twice the Spark's 121.7 GiB of unified memory, and no FP8, AWQ
or NVFP4 repack has been published. So the choice is a quantised GGUF or
nothing, and llama.cpp is the runtime that reads them.

The upside of the architecture is that this is much less painful than "124B"
suggests: only 5.1B parameters are active per token, so decode touches a small
fraction of the resident weights.

## Profiles

Sizes are the actual shard totals at pinned revision `a792b6bd51d2`.

| Profile | Quant | Weights | Default ctx | Notes |
|---|---|---|---|---|
| `q3kxl` | Q3_K_XL | 57.5 GiB | 262,144 | most headroom |
| **`iq4xs`** | **IQ4_XS** | **64.0 GiB** | **262,144** | **default — best quality per byte at 4 bits** |
| `q4km` | Q4_K_M | 72.5 GiB | 65,536 | quality profile; raise ctx after measuring |

Every path and tunable is env-overridable, with defaults chosen so the recipe is
correct with no configuration. The ones you are most likely to touch:

| Variable | Default | Means |
|---|---|---|
| `CTX_SIZE` | per profile | Context window — the largest single lever on memory use |
| `PARALLEL` | `1` | Server slots; `CTX_SIZE` is divided across them |
| `PORT` | `8008` | Listen port, published on all interfaces |
| `HOST` | `0.0.0.0` | Bind address — set `127.0.0.1` to keep the server off the LAN |
| `SPEC_TYPE` | `draft-mtp` | MTP speculative decoding; `none` disables it |
| `MLOCK` | `1` | Lock weights in RAM, never swap; `0` to oversubscribe |
| `BATCH_SIZE` / `UBATCH_SIZE` | `4096` / `2048` | Prefill batch sizes |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | `1.0` / `0.95` / `20` | Server-side sampling defaults |
| `RESTART_POLICY` | `unless-stopped` | Docker restart policy; `no` for a one-off |
| `LLAMA_CACHE` | `~/.cache/llama.cpp` | Where GGUF shards live and what is bind-mounted |
| `OUT_DIR` | `${XDG_STATE_HOME:-~/.local/state}/dgx-spark-recipes/<recipe>` | Bench results, verify stamps |
| `HF_TOKEN` | *(unset)* | Hub token; falls back to `~/.cache/huggingface/token` |
| `FORCE_VERIFY` | `0` | `1` re-hashes the shards instead of trusting the stamp |
| `IMAGE` | `ghcr.io/ggml-org/llama.cpp:server-cuda` | Runtime image |
| `MIN_LLAMA_BUILD` | `10776` | Build floor `preflight.sh` enforces |

The complete reference — including `MODEL_STORE`, `MODEL_ROOT`,
`SERVED_MODEL_NAME`, `CONTAINER_NAME`, `MEM_HEADROOM_BYTES`,
`DISK_RESERVE_BYTES` and `RETRIES` — is in
[CONVENTIONS.md](../../CONVENTIONS.md#environment-variable-reference).

```bash
CTX_SIZE=262144 ./start.sh iq4xs
PARALLEL=4 CTX_SIZE=32768 ./start.sh q3kxl     # 4 slots x 32K
HOST=127.0.0.1 ./start.sh iq4xs                # loopback only
```

## Measured

Measured on a DGX Spark (GB10, 121.7 GiB unified), `iq4xs`, at both context
sizes:

| | `n_ctx=131072` | `n_ctx=262144` (default) |
|---|---|---|
| Cold load | 3 min 48 s | ~3 min 30 s |
| Resident total | ~73 GiB | **76.4 GiB** |
| — weights | 64.0 GiB | 64.0 GiB |
| — overhead | ~9.1 GiB | **~12.3 GiB** |
| Free afterwards | ~47 GiB | **~45 GiB** |
| Decode | ~52 tok/s | ~46 tok/s |
| MTP | active | active |

Doubling the context cost **+3.2 GiB** — the whole basis for making 262,144
the default. Overhead covers KV, compute buffers and the MTP draft context.

### Throughput vs. context depth

`./bench.sh depths`, `iq4xs` at `n_ctx=131072`, 128 decoded tokens per sample,
median of 2, prompt cache disabled so every sample pays a real prefill:

| Depth | Prefill tok/s | Decode tok/s | TTFT |
|---:|---:|---:|---:|
| 2,048 | 1,314 | 52.0 | 1.6 s |
| 8,192 | 1,390 | 53.0 | 5.9 s |
| 16,384 | 1,342 | 46.5 | 12.2 s |
| 32,768 | 1,198 | 48.9 | 27.2 s |
| 65,536 | 997 | 51.9 | 65.3 s |

Two things worth noticing, both consequences of the hybrid architecture:

- **Decode is flat.** ~47–53 tok/s from 2K to 64K, with the variation inside
  run-to-run noise rather than trending down. A dense model's decode decays
  as its KV cache grows; here most layers carry a fixed-size recurrent state,
  so there is nothing to decay.
- **Prefill degrades gently** — 1,390 → 997 tok/s, about −28% across a 32x
  increase in depth, where quadratic attention would cost far more.

The practical constraint for long-document work is therefore **TTFT, not
throughput**: 64K of filings costs about a minute before the first token.
Prompt caching (on by default, `--cache-ram 8192`) is what makes repeated
queries over the same documents usable — this benchmark disables it precisely
to avoid flattering the numbers.

The 9 GiB of overhead settles the question the profile table could not: the
hybrid architecture really does keep KV cheap. Most of the 42 layers are KDA
linear-attention layers holding a *fixed-size* recurrent state regardless of
context length, and the full-attention minority uses MLA-style compression
(`kv_lora_rank: 512`, `qk_rope_head_dim: 64`). A dense 124B model at 131K
context would need an order of magnitude more.

## Context sizing

`iq4xs` defaults to the model's **full 262,144** context, which is a measured
choice rather than an aspirational one: it leaves ~45 GiB free on a 121.7 GiB
machine. The other two
profiles are still set conservatively — raise them the same way once you have
a load to measure.

```bash
CTX_SIZE=262144 ./start.sh q3kxl    # 57.5 GiB of weights, even more headroom
```

To measure it yourself: llama.cpp does **not** print a `KV self size` line at
the image's default verbosity, so resident memory minus the weights is the
honest read. `start.sh` prints exactly that on success:

```
  weights        64.0GiB
  host in use    73.1GiB of 121.7GiB
  overhead       ~9.1GiB (KV at n_ctx=131072, compute buffers, MTP draft ctx)
```

## MTP speculative decoding

This checkpoint ships a multi-token-prediction layer and bartowski's GGUFs keep
it, so llama.cpp can self-speculate with **no draft model** — `start.sh`
defaults to `--spec-type draft-mtp`. On a 5.1B-active MoE, where decode is
memory-bandwidth-bound rather than compute-bound, this is where the throughput
is.

It is a default, not a measured claim. A/B it:

```bash
./bench.sh mtp    # restarts twice, writes both arms to $OUT_DIR
```

If the server refuses to start with it, `start.sh` detects the early exit and
prints the retry command (`SPEC_TYPE=none ./start.sh iq4xs`).

## Thinking mode and sampling

Thinking mode is **on by default** for this checkpoint. The server runs with
`--jinja --reasoning-format deepseek`, so the trace arrives in
`message.reasoning_content` and never inline in `message.content` —
`scripts/smoke.py` asserts both.

Sampling defaults are the model card's recommendation for general inference:
`temperature=1.0`, `top_p=0.95`, `top_k=20`. They are set server-side, so
clients that send nothing get the right values (verify with `/props`).

**Budget tokens generously.** This checkpoint reasons at length — the smoke
test's finance prompt routinely spends >2,000 tokens in the trace before
emitting a single token of content. A `max_tokens` of 2048 truncates
mid-thought and looks exactly like an empty-response bug. `scripts/smoke.py`
defaults to 8192 for this reason.

## Runtime version

`bailingmoe3` landed in llama.cpp **b10776**. The `:server-cuda` tag is
mutable, and an image pulled before that fails the load with an unhelpful
"unknown model architecture" *after* reading the whole first shard — so
`preflight.sh` reads the build number out of the image and refuses to proceed
below `MIN_LLAMA_BUILD`.

```bash
docker pull ghcr.io/ggml-org/llama.cpp:server-cuda
```

No source build is needed: support was already upstream when the current
aarch64 image was cut.

Two harmless things the runtime says that are worth recognising:

- `load: special_eos_id is not in special_eog_ids - the tokenizer config may be
  incorrect` — emitted at load. Generation terminates correctly in practice.
- The image's built-in `HEALTHCHECK` hardcodes port 8080, so a server on any
  other port shows as **`(unhealthy)`** in `docker ps` while serving fine.
  `start.sh` overrides it with the real port and a 30-minute start period, so
  this only affects containers started by hand.

## Storage

Nothing heavy lives in this directory. This recipe follows the **DGX Spark
layout** — heavyweight reusable data lives once, at each tool's own standard
default location. See [CONVENTIONS.md](../../CONVENTIONS.md) at the repo root,
or the packaged skill
[`dgx-spark-layout`](https://github.com/amarjeet/agent-skills/tree/main/skills/dgx-spark-layout).

| What | Where | Override |
|---|---|---|
| GGUF shards | `~/.cache/llama.cpp/Ling-3.0-flash-Fin-GGUF/<QUANT>-<rev12>/` | `LLAMA_CACHE`, `MODEL_STORE` |
| Bench results, verify stamps | `${XDG_STATE_HOME:-~/.local/state}/dgx-spark-recipes/Ling-3.0-flash-Fin-GGUF-llamacpp/` | `OUT_DIR` |
| Container view of the store | `/root/.cache/llama.cpp` (bind mount) | — |

The host store is mounted onto llama.cpp's *own* in-container default, so no
cache env var is set inside the container and the shards are shared with every
other llama.cpp recipe on the host.

`download.sh` pins the Hub revision, resumes partial downloads, and verifies
every shard's size and SHA-256 (`manifests/*.json`). Re-verification is stamped
by `(path, size, mtime)` so a restart does not re-hash 64 GiB; `FORCE_VERIFY=1`
forces it.

## Sharing the box

If the host is already running another model server there will not be room for
both — a single SGLang container can hold ~110 GiB. `preflight.sh` checks free
memory against weights + headroom and, when it fails, lists the other running
containers rather than making you go looking.

Weights are `mlock`ed (`--load-mode mmap+mlock`) so they never reach swap; the
cost is that a load which does not fit fails outright instead of degrading. Set
`MLOCK=0` to opt out.

## Files

| | |
|---|---|
| `profiles.sh` | the only place paths, the profile table and helpers are defined |
| `download.sh` | checksum-verified, resumable shard download |
| `preflight.sh` | build number, GPU, shards, disk, memory, port |
| `start.sh` / `stop.sh` / `status.sh` | container lifecycle |
| `bench.sh` | depth ladder, and the MTP A/B |
| `scripts/download_model.py` | pinned-revision downloader (stdlib only) |
| `scripts/smoke.py` | finance reasoning + thinking-mode assertions |
| `scripts/bench_depths.py` | prefill/decode vs. context depth |
| `manifests/*.json` | pinned revision, per-shard sizes and SHA-256 |

## Launcher

[`scripts/run-ling-3.0-flash-fin-gguf.sh [profile]`](../../scripts/run-ling-3.0-flash-fin-gguf.sh)
at the repo root runs `preflight.sh` and then `start.sh` from anywhere. It sets
no cache paths — the defaults already point at shared storage.

## Endpoint and lifetime

```
http://<host>:8008/v1
model id: ling-3.0-flash-fin-iq4-xs      # or -q3-k-xl / -q4-k-m
```

The server binds `0.0.0.0` and the port is published on all interfaces, so it
is reachable from the LAN, not just loopback. There is **no API key and CORS is
open** — llama.cpp warns about this at startup. That is fine behind a trusted
network and not fine anywhere else.

The container runs with `--restart unless-stopped`, so it comes back after a
docker daemon restart or a reboot. `./stop.sh` removes it outright, which the
policy honours. Set `RESTART_POLICY=no` for a one-off run.
