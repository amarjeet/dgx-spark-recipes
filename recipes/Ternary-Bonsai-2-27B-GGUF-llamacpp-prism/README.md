# Ternary-Bonsai-2-27B on llama.cpp (DGX Spark / GB10)

[`prism-ml/Ternary-Bonsai-2-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
is a 27B-class hybrid-attention VLM derived from
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), quantized to
**ternary g128** — weights from `{-1, 0, +1}` with an FP16 scale per group of
128, at a true 1.72 bits/weight. The PQ2_0 pack is **7.21 GB on disk and
6.70 GiB resident**, and the model card reports 98.2% of the FP16 benchmark
average retained.

Served with the **[PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp)**,
built from source for `sm_121`. Stock llama.cpp cannot run these files.

This recipe is the odd one out in this repo twice over. It is **native**, not
Docker, because the fork publishes no aarch64 CUDA binary and no image. And it
is the only recipe here whose memory goes to *context* rather than weights: at
6.70 GiB the model is 5.5% of the pool, and the hybrid backbone costs only
64 KiB/token of KV, so the default profile holds **four concurrent
262,144-token slots in 77.9 GiB**.

That context is not just allocated: a needle buried mid-prompt was recalled at
every depth from 8k to **254,032 tokens** on this build. There is an open
upstream bug that should have broken it, and
[The 262K caveat](#the-262k-caveat) explains why the test stays in the smoke
suite anyway.

## Quick start

```bash
./build.sh                  # clone the pinned fork and compile for sm_121 (~20 min, once)
./download.sh pq2           # 7.8 GB, checksum-verified, resumable
./preflight.sh wide         # fork binary, sm_121, Hadamard metadata, memory, port
./start.sh wide             # serves on :8010
./scripts/smoke.py          # arithmetic, thinking, tool calls, vision, needle at depth
./status.sh
./stop.sh
```

Or from the repo root, which runs preflight then start:

```bash
./scripts/run-ternary-bonsai-2-27b-gguf.sh wide
```

## Why the fork, and not stock llama.cpp

Two independent reasons, and the second is the dangerous one.

**The types do not exist upstream.** `PQ2_0` (ggml type id 142) and `PTQ1_0`
sit past upstream's `GGML_TYPE_COUNT`, so stock llama.cpp refuses them
outright. That is a clean failure.

**The weights are in a rotated basis.** Every matrix is transformed by a
blockwise Hadamard rotation (block 1024, `normalized-sylvester-walsh-hadamard`)
before the ternary assignment, folded into the stored weights; the runtime has
to apply the matching transform to activations. Only the fork does. The same
model family also ships a plain `Q2_0` band whose type id upstream *does* know
— and upstream loads it without a warning and emits **fluent nonsense**.

So "is this the right binary" and "is this the right file" are both checks that
cannot be skipped, and neither announces itself at run time. `preflight.sh`
asserts both: `pq2_0`/`ptq1_0` in `libggml-base`, and `prism.hadamard.*` in the
GGUF header. `scripts/smoke.py` then asks the model an arithmetic question the
card scores 95–98% on, because a wrong basis passes every structural check and
fails that one.

**Do not build with the fork's own `scripts/build_cuda_linux.sh`.** It hardcodes
`CMAKE_CUDA_ARCHITECTURES="80;86;89;90;100;120a"`, and ggml only selects
architectures for you when that variable is *unset*. `sm_120a` is not
forwards-compatible to GB10's `sm_121` and carries no PTX to JIT from, so that
script produces a binary which cannot run here at all. `./build.sh` passes
`121a-real` and then reads the architectures back out of `libggml-cuda.so` to
prove it.

## Why GGUF and not vLLM or SGLang

There is no FP8/AWQ/NVFP4 repack of this checkpoint, and there is no point
making one: the whole artifact is a ternary quantization. The F16 reference
pack in the same repo is 53.8 GB and would fit this box, but running it defeats
the purpose. The ternary packs only have kernels in this fork, which is a
llama.cpp fork.

## Profiles

| Profile | Pack | Weights | `CTX_SIZE` | Slots | Per slot | Budget | Measured | Notes |
|---|---|---|---|---|---|---|---|---|
| `wide` *(default)* | PQ2_0 | 6.70 GiB | 1048576 | 4 | 262144 | 84.9 GiB | **77.9 GiB** | Four concurrent full-context slots. Needle recalled at 254,032 tokens; read [the caveat](#the-262k-caveat). |
| `deep` | PQ2_0 | 6.70 GiB | 262144 | 1 | 262144 | 36.4 GiB | — | One full-context slot, when four slots' worth of KV is not wanted. |
| `long` | PQ2_0 | 6.70 GiB | 131072 | 1 | 131072 | 28.4 GiB | — | What upstream's own launcher auto-picks for a 27B on a box this size. |
| `safe` | PQ2_0 | 6.70 GiB | 262144 | 4 | 65536 | 36.9 GiB | **28.7 GiB** | Four slots, each below every depth #27756 reports failing, and under 90 s to fill one. |
| `ptq1` | PTQ1_0 | 5.53 GiB | 262144 | 1 | 262144 | 35.3 GiB | — | 16% faster decode, 56% slower prefill. Right for short prompts, wrong for deep ones — see [Packings](#packings). |

`CTX_SIZE` is the **total** context, divided across slots — that is llama.cpp's
own semantics (`n_ctx_seq = n_ctx / n_seq_max`), not a convention of this
recipe. Per-slot context is hard-capped at the model's 262144 training context.

Passing `--parallel` explicitly is load-bearing. Left at its default of `-1`
the server picks 4 slots *and* enables `kv_unified`, which makes `CTX_SIZE` a
single shared pool that is **not** divided. Every profile here sets both.

Knobs worth touching:

| Variable | Default | Means |
|---|---|---|
| `CTX_SIZE` / `PARALLEL` | per profile | Total context and slot count. The largest lever on memory. |
| `REASONING_EFFORT` | *(unset)* | `xhigh` \| `medium` \| `low`, pinned server-wide. Unset leaves the template's own default, which is `xhigh` — not the cheap one. The card notes `low` is not really honoured and behaves close to `xhigh`. |
| `UBATCH_SIZE` | `512` | Drop to `256` if prefill dies in cuBLAS; this family is batch-size fragile on GB10 (upstream #28377). |
| `CUDA_ARCHS` | `121a-real` | Only change this to build for a different GPU. |
| `MMPROJ_CPU` | `0` | `1` keeps the vision projector in system RAM. |
| `IMAGE_MAX_TOKENS` | *(unset)* | Vision-token cap per image. Unset is uncapped, which is what upstream does on CUDA. |
| `PORT` / `HOST` | `8010` / `0.0.0.0` | **No API key, CORS open.** Set `HOST=127.0.0.1` to keep it off the LAN. |

Everything else is in
[`CONVENTIONS.md`](../../CONVENTIONS.md#environment-variable-reference).

## The 262K caveat

**[llama.cpp issue #27756](https://github.com/ggml-org/llama.cpp/issues/27756)
is open against exactly this architecture** — Qwen3.8-27B, 48 gated-DeltaNet
layers plus 16 full-attention layers — and reports silent failure at long
context. The prefill completes cleanly, then the model emits EOS as its very
first token: `tokens_predicted: 1`, empty content, `stop_type: "eos"`, no error
anywhere. A client sees a successful HTTP 200 with nothing in it.

What the issue establishes: 132,375 tokens passes, **~129,864 fails**, and
everything from 174,495 up fails including 243,077. It is **non-monotonic** — a
depth that works can sit directly above one that does not — it reproduces on
the CUDA and CPU backends alike, and a 30-GDN-layer control model passes
243,077 on the same build. That points at per-layer recurrent-state
accumulation scaling with depth rather than at a kernel.

**It did not reproduce here.** `./bench.sh needle` on this build, at seven
depths from 8k to the full context:

| Depth (tokens) | Prefill | Wall | Needle |
|---:|---:|---:|---|
| 8,202 | 986 t/s | 12.2 s | recalled |
| 32,358 | 904 t/s | 40.1 s | recalled |
| 64,629 | 798 t/s | 86.2 s | recalled |
| 96,837 | 720 t/s | 141.7 s | recalled |
| 129,104 | 654 t/s | 207.8 s | recalled |
| 193,583 | 555 t/s | 358.5 s | recalled |
| **254,032** | 487 t/s | 532.7 s | **recalled** |

Note 129,104 — within a few hundred tokens of the depth the issue reports
failing — and 193,583, inside the band where it reports everything failing.
Both answered correctly. The most likely explanation is that this fork (build
10687, commit `5d80cff0`) postdates the build the issue was filed against;
a second possibility is that the Bonsai-2 checkpoint differs from the base
Qwen3.8-27B quants the reporter used.

**This is evidence, not proof.** The issue describes a prompt-dependent,
non-monotonic onset, and the sweep above is one prompt shape per depth — it can
confirm that a depth works, never that every prompt at that depth works. So the
recipe keeps testing rather than assuming:

- `scripts/smoke.py` runs the needle test at the server's own per-slot context
  by default, and distinguishes *answered correctly*, *answered but lost the
  needle*, and *silent EOS*.
- `preflight.sh`, `start.sh` and `status.sh` all name the issue whenever
  per-slot context exceeds 98,304, and print the command to check.
- `safe` (4 × 65,536) sits under every reported failure depth, one word away.

```bash
./scripts/smoke.py --needle-depth 262144          # does this depth answer?
./bench.sh needle                                 # walk depths, find any holes
./start.sh safe                                    # 4 x 65536, conservative
```

If you hit a silent EOS, that is worth adding to #27756 with the exact token
count — the non-monotonic holes are the part upstream has the least data on.

## Measured

DGX Spark, GB10, aarch64, 121.7 GiB unified memory, driver 580.173.02,
CUDA 13.0.88. PrismML fork commit `5d80cff0`, build 10687, compiled for
`sm_121a`. Nothing else running.

`llama-bench -ngl 99 -fa on -r 3`, PQ2_0:

| test | t/s | upstream's GB10 figure |
|---|---:|---:|
| pp512 | **1039.99 ± 13.39** | 1005.19 ± 9.76 |
| tg128 | **29.83 ± 0.05** | 29.16 ± 0.04 |

Run as two invocations: the combined form emits only `pp512` for this model.
Matching upstream within a couple of percent is the evidence that the
`121a-real` build took — a binary falling back to PTX, or missing `sm_121`
entirely, would not be close.

29.83 t/s against a bandwidth roofline of ~273 GB/s ÷ 7.15 GB ≈ 41 t/s is 76%
of peak, so decode is memory-bound and will not improve with a faster context
or a bigger batch. It is also the whole server's rate, shared across slots:
four slots buy concurrency and context, not per-request speed.

`./start.sh wide`, measured at ready:

```
context   : 1048576 total over 4 slot(s) = 262144 per slot
budget    : 84.9GiB expected resident
host in use    77.9GiB of 121.7GiB
context        262144 per slot, as requested
srv load_model: initializing, n_slots = 4, n_ctx_slot = 262144, kv_unified = 'false'
```

At ready the budget over-predicts by ~7 GiB, because the 8 GiB checkpoint cache
and the 5 GiB compute reserve are both floors with slack. Under load it closes:
with one slot mid-way through a 130k-token prefill, `./status.sh` reported
**86.6 GiB in use** against the 84.9 GiB budget — so the budget is a fair
working figure rather than a generous one, and there is no room to raise
`CTX_SIZE` past what `preflight.sh` approves. `budget_bytes` in `profiles.sh`
is the arithmetic; `preflight.sh` prints it itemised.

`./start.sh safe` measured 28.7 GiB resident against a 36.9 GiB budget, and a
full `./scripts/smoke.py` passed on it including the needle at 61,467 tokens.

Needle retrieval passed at all seven depths from 8k to 254,032 tokens; the
table is in [The 262K caveat](#the-262k-caveat), and the throughput curve
behind it is in [Context sizing](#context-sizing).

`./scripts/smoke.py` on the `wide` profile: arithmetic correct (operating
income $582M, net debt resolved as a net cash position), `reasoning_content`
populated with no `<think>` leakage, `reasoning_effort=medium` honoured, tool
call parsed to `get_weather {"city":"Reykjavik"}`, vision projector named both
colours in a generated test image. Serving-path decode 29.3 tok/s.

## Context sizing

KV cost is exact arithmetic off the GGUF header, not an estimate:

```
block_count 64, full_attention_interval 4     -> 16 full-attention layers
head_count_kv 4, key_length 256, value_length 256
16 x 4 x (256 + 256) x 2 B                    =  65536 B = 64 KiB per token
```

`./scripts/gguf_probe.py <model.gguf>` restates this against the file, so a
different checkpoint cannot quietly invalidate it.

The other 48 blocks are linear attention: a fixed recurrent state per sequence,
independent of how full the context is. From `ssm.inner_size 6144`,
`ssm.state_size 128`, `ssm.conv_kernel 4` in fp32, that is ~148 MiB per slot,
and `--parallel N` multiplies it.

| Per-slot context | KV (F16) | Time to fill one slot | Needle test |
|---:|---:|---:|---|
| 32,768 | 2.0 GiB | 40 s | passed |
| 65,536 | 4.0 GiB | 86 s | passed — `safe` |
| 98,304 | 6.0 GiB | 2.4 min | passed |
| 131,072 | 8.0 GiB | 3.5 min | passed — `long` |
| 262,144 | 16.0 GiB | 8.9 min | passed at 254,032 — `wide`, `deep` |

Two things to expect at depth, neither of them memory, and both measured
rather than modelled:

**Throughput falls with depth — decode more than prefill.**

| Depth | Prefill | Decode |
|---:|---:|---:|
| 8,232 | 986 t/s | 26.58 t/s |
| 32,388 | 904 t/s | 21.83 t/s |
| 64,659 | 798 t/s | 17.81 t/s |
| 96,867 | 720 t/s | 14.86 t/s |
| 129,134 | 654 t/s | 12.85 t/s |
| 193,613 | 555 t/s | 10.07 t/s |
| 254,062 | 487 t/s | 8.36 t/s |

Prefill ends at **49%** of its shallow rate, decode at **31%**. The prefill
slope is expected: the 16 full-attention layers cost work proportional to
depth, and setting their FLOPs against the backbone's weight matmuls
(48.7 GFLOP/token vs 393,216 × depth) puts the crossover near 124k tokens.
The decode slope is the one that catches people out — only a quarter of the
blocks attend over the context, yet decode still loses two thirds of its rate,
and upstream issue #28734 reports the same shape for a sibling architecture.

Two consequences. A full-context prefill is **8.7 minutes**, not the
262144 ÷ 1040 ≈ 4.2 minutes a flat rate would predict. And the 29.8 t/s from
`llama-bench` is an upper bound at depth 0, not a serving rate: at 254k it is
8.4 t/s. `./bench.sh depths` re-measures the curve.

**An edited conversation re-prefills.** A hybrid model cannot partially evict
its recurrent state, so llama.cpp can only drop a sequence whole: prompt-cache
reuse works for pure prefix extension, and any client-side history edit costs a
full re-prefill. Context checkpoints soften it — `CTX_CHECKPOINTS=8`,
`CACHE_RAM_MIB=8192`, set explicitly here because each checkpoint is about a
slot's worth of recurrent state and the default per-slot count is large enough
to matter to the budget.

## Packings

PQ2_0 (2.13 bpw) and PTQ1_0 (1.75 bpw) are a genuine trade, not an ordering.
PTQ1_0 moves 17% less weight data per decode step but pays arithmetic to unpack
dense trits, so it wins where bandwidth binds and loses where instruction
throughput does.

The model card says PQ2_0 is "the faster decode on ... the Blackwell cards".
**That is wrong for GB10**, and `./bench.sh packs` says so on this box:

| Pack | Resident | pp512 | tg128 |
|---|---:|---:|---:|
| PQ2_0 | 6.70 GiB | **1033.67 ± 14.25** | 29.57 ± 0.04 |
| PTQ1_0 | 5.53 GiB | 457.19 ± 2.55 | **34.27 ± 0.07** |

PTQ1_0 decodes **16% faster** and prefills **56% slower**, and saves 1.17 GiB.

The card's guidance was measured on RTX 5090 and RTX PRO 6000 — parts with
roughly 6× GB10's memory bandwidth, where batch-1 decode is limited by
instruction throughput rather than memory, so unpacking dense trits costs more
than the traffic it saves. GB10's achieved 208 GB/s puts it with upstream's
**L4** row instead (~300 GB/s, 29.8 t/s PQ2_0 — nearly identical to this box),
where PTQ1_0 also won decode. Calling GB10 "Blackwell" and inheriting the
5090's answer is a category error; the architecture name is not the thing that
decides this, the bandwidth is.

**Which to use.** For a request with `P` prompt tokens and `G` generated
tokens, PTQ1_0 is faster when

```
G / P  >  0.26          (from 1/29.57 - 1/34.27  vs  1/457.19 - 1/1033.67)
```

That threshold is lower than it looks, because this is a thinking model: at the
default `xhigh` effort it routinely spends one to two thousand tokens reasoning
before answering. So **PTQ1_0 is the better pick for short-prompt interactive
chat** — a 1,000-token prompt needs only ~260 generated tokens to favour it.

At depth the conclusion flips hard. A 100k-token prompt would need 26k
generated tokens to pay back PTQ1_0's prefill, which will not happen. Since
this recipe exists for long context, **PQ2_0 stays the default** — but `ptq1`
is a real option, not a footnote, and it is the faster one for the workload
most people try first.

## Thinking, tool calling, vision

The GGUF **carries its own chat template** (8,952 characters), so unlike some
recipes here there is no template file to ship. `--jinja` picks it up.

- **Thinking is on by default** and the template opens `<think>` in the
  assistant prefix, so generation begins *inside* the block and only `</think>`
  is ever emitted. `--reasoning-format auto` (the default, passed explicitly)
  puts the trace in `message.reasoning_content`. Do not copy the
  `--reasoning-format deepseek` some other recipes use — upstream's own enum
  comment says to prefer `auto`.
- **Reasoning effort** is `xhigh` unless told otherwise. Set `REASONING_EFFORT`
  server-wide, or send `chat_template_kwargs: {"reasoning_effort": "medium"}`
  per request.
- **Tool calls work**, and are XML rather than JSON:
  `<tool_call><function=NAME><parameter=P>…`. llama.cpp auto-detects any
  template containing all three markers and routes it to its Qwen3-Coder
  parser, whose own comment names Qwen3.5 — so `--jinja` alone yields parsed
  `tool_calls`. One hazard: on output it cannot parse, that PEG parser
  **throws**, so a malformed call arrives as an HTTP error rather than as
  degraded content.
- **Vision** works through the Q8_0 projector (~600 MiB), loaded by default.
  Images are priced as prefill, roughly one token per 32×32 patch up to ~4096
  tokens. Uncapped on CUDA, per upstream.

## No speculative decoding

Upstream's GB10 benchmark shows a 2.35–2.45× decode speedup from a DSpark
drafter — but that measurement is of the **previous-generation**
`Ternary-Bonsai-27B`. Drafters are target-specific, the Bonsai-2 repo ships no
drafter file, and no `Ternary-Bonsai-2-27B-dspark*` repo exists. So ~29.8 t/s
is the decode ceiling here, and this recipe has no `SPEC_TYPE`.

## Storage

| What | Where | Override |
|---|---|---|
| GGUF weights + projector | `~/.cache/llama.cpp/Ternary-Bonsai-2-27B-gguf/<PACK>-<rev12>/` | `LLAMA_CACHE`, `MODEL_STORE` |
| Fork source checkout | `~/src/llama.cpp-prism` | `SRC_ROOT`, `FORK_DIR` |
| Built binaries | `~/src/llama.cpp-prism/build-cuda/bin` | `BUILD_DIR`, `BIN_DIR` |
| Bench results, verify stamps | `~/.local/state/dgx-spark-recipes/Ternary-Bonsai-2-27B-GGUF-llamacpp-prism/` | `OUT_DIR` |
| Server log, pidfile, active profile | in this directory, gitignored | — |

Weights land in llama.cpp's own standard cache, so they are downloaded once and
shared with every other llama.cpp recipe on the host. This is a native recipe:
there is no mount to redirect, and `LLAMA_CACHE` is the variable llama.cpp
itself reads, so nothing needs exporting. Binaries are run out of the build
tree with `LD_LIBRARY_PATH` pointed at it — no `patchelf`, nothing copied into
the recipe directory.

## Sharing the box

`wide` holds ~78 GiB of one 121.7 GiB pool, so nothing else substantial fits
beside it. `preflight.sh` checks the budget against `MemAvailable`, lists other
containers and other model-server processes, and claims port 8010.

Being native, this recipe has **no cgroup cap and no watchdog** — the budget in
`preflight.sh` is the only guard. On unified memory that matters: exhausting
the pool hangs the kernel with no OOM and no logs. Do not raise `CTX_SIZE`
past what preflight approves.

## Files

```
profiles.sh          paths, profile table, memory model, helpers
build.sh             pinned fork checkout + CUDA build for sm_121, then verify
download.sh          retry wrapper over scripts/download_model.py
preflight.sh         binary, weights, disk, budget, tenancy, port, context
start.sh             launch llama-server natively
stop.sh              SIGTERM the pidfile, then SIGKILL; log preserved
status.sh            up? serving what geometry? host healthy?
bench.sh             depths | packs | needle
manifests/pq2.json   PQ2_0 + projector, pinned revision, sizes, SHA-256
manifests/ptq1.json  PTQ1_0 + projector
scripts/download_model.py  stdlib, resumable, checksum-verified
scripts/gguf_probe.py      read GGUF metadata; assert the Hadamard rotation
scripts/smoke.py           arithmetic, thinking, tools, vision, needle at depth
scripts/bench_depths.py    prefill and decode against context depth
```

## Endpoint and lifetime

OpenAI-compatible at `http://<host>:8010/v1`, model id
`ternary-bonsai-2-27b-pq2-0`. Native, so there is no restart policy and the
server does not survive a reboot — start it again with `./start.sh`.

## Known unknowns

- **Whether #27756's non-monotonic holes exist here at all.** Seven depths
  passed with one prompt shape each; that cannot rule out a prompt-dependent
  failure at some other depth. This is the open question that matters most, and
  `./bench.sh needle` with your own `DEPTHS` is how to narrow it.
- **Why the decode slope is so steep** — 31% of the shallow rate at 254k, when
  only 16 of 64 blocks attend over the context. Not investigated.
- Whether all four slots hold full context *simultaneously* under real
  concurrent load. The geometry is served and the arithmetic fits, but the
  sweep above drove one slot at a time.
- Whether `--kv-unified` would suit four slots that rarely all run deep at
  once. Untested.

## Deviations from `CONVENTIONS.md`

- **Native, not Docker.** The fork ships no aarch64 CUDA binary and no image,
  so there is nothing to pull. This brings an extra `build.sh`, a pidfile
  instead of a container, and an explicit memory budget in place of a cgroup
  cap. `CONVENTIONS.md` has been extended with the native rules rather than
  leaving them implicit here.
- **`stop.sh` signals a process** rather than removing a container, and there
  is no `RESTART_POLICY`, no healthcheck and no `IMAGE`.
- **The default profile runs against an open upstream bug.** Every other recipe
  here defaults to a configuration with no known defect filed against it. This
  one defaults to the full context, because it measured clean at 254,032 tokens
  — but #27756 describes a prompt-dependent, non-monotonic failure that a
  seven-point sweep cannot exclude. The choice is deliberate, the evidence is
  in the README, and `safe` is one word away.
- **An extra `scripts/gguf_probe.py`.** No other recipe needs to prove a file
  is in the basis its runtime expects.

## License

MIT, like the rest of this repo — no AGPL machinery needed here.

- The weights are **Apache-2.0** ([model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)).
- The [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) is
  **MIT**, as upstream llama.cpp is. `build.sh` compiles it from source into
  `~/src`; nothing from it is redistributed here.
- This recipe carries no upstream code, only the commands to fetch and build it.
