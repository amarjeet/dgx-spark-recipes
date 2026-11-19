# Qwen3.8-27B + pi LoRA on vLLM

[harness-hub/Qwen3.8-27B-pi-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted):
a PEFT LoRA (rank 32, alpha 64) on
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), trained with
reinforcement learning (AsyncGRPO) to drive the
[pi](https://github.com/earendil-works/pi) coding agent. The base is a dense
27B hybrid: 48 linear-attention and 16 full-attention layers out of 64. This
recipe runs the adapter card's own **vLLM evaluation command** on the BF16 base:
LoRA served beside the base model, thinking off, `qwen3_xml` tool calls. The one
change is the window: the base's native 262,144 tokens, not the card's 131,072.

**Status: `pi`, `pi-mtp` and `nvfp4` run on this host, and the smoke test
passes on each.** Numbers are in Measurements. `nvfp4` is about 5.6x faster
than `pi` per stream. On `nvfp4`, an agent eval in a pi-shaped harness
(`scripts/pi_eval/`, 180 runs) shows the adapter using about 30% fewer tool
calls than the base, with a slightly higher solve rate (85/90 against 81/90).
The `base` profile has not been started.

## Run

```bash
./download.sh        # 0.6 GiB adapter + 51.8 GiB BF16 base, sha256-verified, resumable
./preflight.sh       # image contents, checkpoints, adapter shape, memory, port, disk
./start.sh           # serves on :8014
./scripts/smoke.py   # thinking off, tool call parsed, adapter changes output
./stop.sh
```

Or `scripts/run-qwen3.8-27b-pi-lora.sh [pi|pi-mtp|base]` from the repo root,
which runs preflight and then start. `DRY_RUN=1 ./start.sh` prints the
`docker run` command.

| Profile | Meaning |
|---|---|
| `pi` (default) | The card's command. Serves `qwen3.8-27b` (base) and `qwen3.8-pi` (base + LoRA) from one server, so A/B comparison needs no restart |
| `pi-mtp` | `pi` plus MTP speculative decoding (2 tokens) from the base's own MTP head. Lossless: same output distribution, about 2x the decode speed. Not in the card |
| `nvfp4` | `pi-mtp` on `nvidia/Qwen3.8-27B-NVFP4` instead of the BF16 base, with the LoRA unmerged on top. About 2.6x `pi-mtp`'s decode speed and 20.4 GiB of weights instead of 50.7. **The adapter was trained against BF16, so outputs drift; task quality is unchecked.** `./download.sh nvfp4` fetches the 20.4 GiB base |
| `base` | The base model alone, no LoRA: a baseline for LoRA overhead, and a way to isolate a failure to the adapter. Does not need the adapter downloaded |

## What is pinned

| | |
|---|---|
| Base | `Qwen/Qwen3.8-27B` @ `37561816e66b…` (32 files, 51.8 GiB, BF16, includes a vision tower and an MTP head) |
| NVFP4 base (`nvfp4` only) | `nvidia/Qwen3.8-27B-NVFP4` @ `482ca0f38322…` (19 files, 20.4 GiB, Apache-2.0). ModelOpt `MIXED_PRECISION`: NVFP4 MLPs and `lm_head`, FP8 attention and linear-attention projections, MTP head left in BF16 |
| Adapter | `harness-hub/Qwen3.8-27B-pi-adapted` @ `fedf3083b1bb…` (12 files, 0.6 GiB; the weights are F32) |
| Image | `vllm/vllm-openai@sha256:3016f449…`, the `v0.26.0-aarch64-ubuntu2404` tag |

Regenerate the manifests with `scripts/make_manifest.py`.

## Departures from the model card

- **The adapter id.** The card's vLLM and merge commands name
  `harness-hub/Qwen3.8-27B-pi`. On 2026-11-18 that repo returned HTTP 401,
  meaning it does not exist or is private. The adapter is at
  `-pi-adapted`, the name the card's own PEFT snippet uses.
- **The adapter is passed by snapshot path**
  (`/root/.cache/huggingface/hub/models--harness-hub--Qwen3.8-27B-pi-adapted/snapshots/<rev>`),
  not by repo id. The served adapter is then the pinned revision even if
  `refs/main` moves. `start.sh` derives that path from the `HF_HOME` mount, so
  the two cannot disagree.
- **The window.** `--max-model-len 262144`, the base model's native window.
  The card evaluated at 131,072, so **the adapter is untested past 131,072
  tokens**. `MAX_MODEL_LEN=131072 ./start.sh` restores the card's setting.
- **Concurrency.** `--max-num-seqs 4`. The card leaves this at vLLM's default.
- **The chat template** is the base's. The adapter repo ships a
  `chat_template.jinja`, but it is byte-identical (sha256 `af16f654…`), so no
  `--chat-template` override is needed.

## Why it should load (checked without loading)

These are read-only checks from 2026-11-18. `preflight.sh` repeats each one
against the image and the files on disk:

- vLLM 0.26.0 registers `Qwen3_5ForConditionalGeneration` (with `IsHybrid`)
  and inherits `SupportsLoRA` from its Qwen3-VL base.
- The adapter's 512 tensors are all named
  `base_model.model.model.language_model.layers.*`. vLLM's Qwen3-VL weight
  mapper rewrites exactly `model.language_model.` → `language_model.model.`.
- The MLP LoRA covers all 64 layers. The q/k/v/o LoRA covers only the 16
  full-attention layers (3, 7, … 63). The linear-attention layers have no
  adapter.
- The `q_proj` LoRA output is 12,288 wide, twice 24 heads × 256, because the
  attention is gated. Only a real load could prove this fits vLLM's merged
  `qkv_proj`, and it does: the adapter loads and is applied (see
  Measurements). If a re-published adapter breaks that, run
  `./start.sh base` to confirm the base on its own works.

## Memory

vLLM budgets `GPU_MEMORY_UTILIZATION` (0.75, about 91 GiB) of total memory and
refuses to start unless that much is free. The expected split:

- About 51 GiB of BF16 weights; the vision tower is skipped with
  `--language-model-only`.
- About 1.2 GiB for the LoRA weights, upcast from F32.
- The rest goes to KV cache, Mamba state and activations. Only 16 layers keep
  KV (4 KV heads × 256, about 64 KiB per token in BF16), so a full
  262,144-token sequence costs about 16 GiB. `align`-mode prefix caching keeps
  additional Mamba state per cached block.

**This server cannot run beside the Qwen3.8-Flash-Next, Nemotron or DeepSeek
recipes.** `preflight.sh` demands the budget plus 6 GiB and names the containers
in the way. If MemAvailable sits well above the budget plus headroom, drop the
page cache (`sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`) before a cold
load. A 52 GiB read on top of a full page cache is where unified memory
gets tight.

## Speed

Decode is bound by memory bandwidth. Each token reads all ~51 GiB of BF16
weights, and the GB10 has about 273 GB/s, so one stream tops out around
5 tok/s. Measured: 4.3 tok/s for the adapter, and about 17 tok/s across four
streams (see Measurements). Prefill is compute-bound and much faster, and
prefix caching keeps agent loops from re-prefilling their shared history.

Two levers, both measured:

- **MTP (`pi-mtp`)** turns one weight read into about 2.3 to 2.8 tokens and
  roughly doubles decode, to 8.4 to 10.3 tok/s. The output distribution is
  unchanged.
- **NVFP4 (`nvfp4`)** cuts the read to about 20 GiB. With MTP that gives
  24 to 26 tok/s per stream and 56 tok/s across four. The cost is drift from
  the BF16 weights the adapter was trained on.

For the card's numbers, use `pi` or `pi-mtp`. For daily use, `nvfp4`, once a
few real pi tasks show it still behaves.

## Storage

Nothing heavy is in this directory. Both checkpoints live in the shared Hugging
Face cache. The vLLM, flashinfer and triton caches are the shared defaults,
mounted onto the container's own default paths (`/root/.cache/huggingface`,
`/root/.cache/vllm`, `/root/.cache/flashinfer`, `/root/.triton`), so no cache
env var is set inside. The container runs with `HF_HUB_OFFLINE=1`. Logs go to
`OUT_DIR` (`~/.local/state/dgx-spark-recipes/Qwen3.8-27B-pi-LoRA-vLLM/`).

A merged checkpoint (the card's alternative to serving the LoRA) would be
about 54 GB of run output. It belongs under `${DGX_SPARK_ROOT}/outputs/`, never
in this directory. This recipe does not merge.

## Knobs

`PORT` (8014), `HOST` (also the address the port is published on), `IMAGE`,
`CONTAINER_NAME`, `SERVED_MODEL_NAME` (`qwen3.8-27b`), `LORA_NAME`
(`qwen3.8-pi`), `GPU_MEMORY_UTILIZATION`, `MAX_MODEL_LEN` (262144, the model's own
window; 131072 is the card's), `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, `SPEC_TOKENS`,
`MEM_HEADROOM_BYTES`, and the storage variables in
[CONVENTIONS.md](../../CONVENTIONS.md). The server has no API key. Set
`HOST=127.0.0.1` to keep it off the LAN.

## Using it with pi

OpenAI-compatible at `http://<host>:8014/v1`. Request model `qwen3.8-pi` for
the adapter and `qwen3.8-27b` for the base. Thinking is off by default,
**as the adapter was trained**. Do not turn it on for the adapter. Tool calls
come back parsed in `tool_calls` (parser `qwen3_xml`).

pi must use the chat-completions API. Without it, pi uses the Responses API,
which the adapter was neither trained nor evaluated with. In
`~/.pi/agent/models.json`, add a provider with `"baseUrl":
"http://localhost:8014/v1"`, `"api": "openai-completions"`, `"apiKey":
"dummy"` and model id `qwen3.8-pi`. pi needs Node.js 22.19 or newer. The card
evaluated with temperature 0.8.

## Claims to keep in perspective

The card reports Terminal-Bench 2.1 rising from 39.5% to 55.8% and SWE-bench
Pro (hard 51) from 29.4% to 62.7%, both under pi against the base. These
are the publisher's own numbers. The checkpoint (step 160 of a run stopped at
step 191) was chosen using scores on SWE-bench Pro and a Terminal-Bench subset,
so expect some selection bias.

## Licenses

Base and adapter are Apache-2.0. The recipe's own scripts are MIT, except
`scripts/download_snapshot.py`, copied unchanged from
`Nemotron-3.5-Super-VL-NVFP4-vLLM` with its AGPL-3.0-or-later header.

## Measurements

DGX Spark (GB10, 121.7 GiB), `pi` profile, image `v0.26.0-aarch64-ubuntu2404`
(by digest), 2026-11-18, nothing else running.

- **Load:** about 9 min cold on the first start. Weights took 356 s
  (50.73 GiB), torch.compile 32 s, and engine init 160 s in total.
- **Memory:** vLLM used 50.73 GiB for weights, 2.55 GiB peak activation,
  2.99 GiB non-torch and 0.35 GiB CUDA graphs, against a 91.3 GiB budget.
  MemAvailable was about 28 GiB with the server up.
- **KV pool:** 34.43 GiB, which is 541,613 tokens at a 131,072-token window
  (4.13x concurrency). With the 262,144 default it is 552,731 tokens, 2.11x
  concurrency. That second start took 379 s for weights and 142 s for engine
  init. The compile ran again, because the window is part of vLLM's cache key. vLLM set the attention block size
  to 784 tokens so its pages match the Mamba state pages (`align` mode).
- **Decode, 512 tokens, temperature 0.8, wall time including prefill:**
  4.58 tok/s for the base alone and **4.41 tok/s for the adapter**, so LoRA
  costs about 4%. Four adapter streams together reached 17.25 tok/s.
- **Adapter applied:** greedy output on an easy coding prompt was identical to
  the base's. On an agent-style prompt the adapter switched the top first
  token (`` ` `` instead of `` ``` ``), and logprobs over 24 tokens differed by
  up to 1.02. The smoke test checks the logprob shift for this reason, not
  the text.
- **Smoke test:** passes. Thinking is off by default with no trace, the tool
  call is parsed into `tool_calls`, and the adapter shifts the distribution.
- vLLM warns that `align`-mode prefix caching for Mamba layers is
  experimental, and that it found no LoRA kernel tuning configs for this GPU
  ("Using default LoRA kernel configs").

### `pi-mtp` and `nvfp4`

Same host and image, 2026-11-18/19, 262,144-token window, nothing else running.
256 generated tokens per run. Decode rate is timed from the first token, so it
excludes the wait before it. "Len" is tokens produced per verification step,
out of a maximum of 3, from vLLM's own spec-decode counters.

| Run | `pi` | `pi-mtp` | `nvfp4` |
|---|---|---|---|
| Adapter, prose, temp 0.8, 1 stream | 4.29 tok/s | 8.44 (66%, len 2.32) | **24.20** (81%, len 2.62) |
| Base, prose, temp 0.8, 1 stream | 4.42 | 9.28 (75%, len 2.51) | 24.60 (74%, len 2.48) |
| Adapter, code, temp 0.8, 1 stream | — | 9.59 (82%, len 2.64) | 24.20 (81%, len 2.62) |
| Adapter, code, greedy, 1 stream | — | 10.29 (89%, len 2.78) | **25.75** (86%, len 2.72) |
| Adapter, prose, 4 streams, total | 16.63 | 26.99 | **55.72** (19.88 per stream) |

- **`pi-mtp`:** weights 51.52 GiB including the MTP layer. KV pool 488,821
  tokens, 1.86x at 262,144.
- **`nvfp4`:** weights 20.4 GiB. KV pool 964,535 tokens, 3.68x at 262,144.
  Load took 124 s and engine init 248 s, including a 52 s compile.
  MemAvailable was about 25 GiB with the server up. vLLM used
  `FlashInferCutlassNvFp4LinearKernel` and loaded the LoRA onto the NVFP4 and
  FP8 layers with no error. FlashInfer had no tuned config for the NVFP4
  `lm_head` GEMM and fell back to defaults, so there may be a little speed
  left on the table.
- **`nvfp4` drift, on the same greedy probe as above.** BF16 base answers
  `` ```bash python -c "import sys; print(sys.path)" `` and BF16 adapter
  `` `python -c "import <module_name>"` ``. On NVFP4 the base answers
  `` `python -m pytest` `` and the adapter answers
  `` `python -c "import sys; print(sys.path)"` ``, the BF16 base's answer. The
  adapter still pushes the first token the same way (`` ` `` at -0.72 on NVFP4,
  -0.94 on BF16), but quantization alone changed the base's top first token.
  The agent eval below runs on this profile.
- The prose and code runs at temperature 0.8 on `nvfp4` gave the same
  numbers to two decimals. Each is a single 256-token run.

### Agent eval: does the adapter do its job under pi?

`scripts/pi_eval/` replays pi's coding-agent system prompt and its four default
tools (read, bash, edit, write), with the descriptions and JSON schemas from
pi's source (`earendil-works/pi` `main` @ `5a14e69`), over chat completions as
pi's `openai-completions` mode does. bash runs in a throwaway container with no
network, as the invoking user. read, edit and write are confined to the task
directory. Each task has hidden checks that are copied in only after the agent
stops.

There are 18 synthetic tasks, all written for this eval:

- **10 easy:** small fixes, a rename across files, and terminal tasks.
- **8 hard:** a traceback that points at the wrong file, three bugs where
  fixing the obvious one is not enough, a circular import, a performance fix
  with an ordering trap, two separate shared-state leaks, mixed encodings, a
  deduplicated report with a tie-break, and an 8-case `.env` parser.

`validate.py` proves each hard task is solvable: doing nothing fails and a
reference fix passes. The tempting wrong fixes were also checked and are
rejected: working around the bug in `report.py`, a `Counter` rewrite, forcing
Latin-1, fixing one cart leak, fixing only the median.

```bash
cd scripts/pi_eval
./validate.py                          # each hard task: no-op fails, reference fix passes
./harness.py                           # both suites, 5 reps, qwen3.8-27b vs qwen3.8-pi
./analyze.py ~/.local/state/dgx-spark-recipes/Qwen3.8-27B-pi-LoRA-vLLM/pi-eval/<run>/results.jsonl
```

Results and transcripts go to `OUT_DIR/pi-eval/<timestamp>/`, not here.

Measured on `nvfp4`, 2026-11-19: 5 reps per task and model, 180 runs,
temperature 0.8, thinking off. Base and adapter are served by the same server,
so the adapter is the only difference.

| | Base `qwen3.8-27b` | Adapter `qwen3.8-pi` |
|---|---|---|
| Solved, easy | 48/50 | 50/50 |
| Solved, hard | 33/40 | 35/40 |
| Tool calls per task, easy / hard | 3.88 / 5.88 | **2.80 / 4.03** (−28% / −31%) |
| Generated tokens per task, all | 437 | 333 (−24%) |
| Seconds per task, all | 49.0 | 38.0 (−22%) |
| Tool errors / malformed calls | 18 / 0 | 5 / 0 |
| Ran code or tests after editing | 69/90 | **17/90** |

- **The adapter is reliably more direct.** It used fewer tool calls on 17
  of 18 tasks. An exact sign-flip test across tasks gives p < 0.001. This is
  the direction the card reports ("55–62% fewer tool calls"), smaller here
  because these tasks leave less to trim.
- **Correctness: a lean, not a result.** The adapter solved 85/90 and the
  base 81/90 (one-sided Fisher p = 0.20). The card's +16 to +33 point
  gains come from far longer tasks than these.
- **It rarely checks its work, and mostly gets away with it.** The adapter
  ran code after editing in 17 of 90 runs. On hard tasks it solved 24 of
  the 28 runs where it did not verify. The base never solved a hard task
  without running something.
- **Failures:**
  - `dedupe_report` (base 0/5, adapter 1/5): almost every run got the totals
    right, then sorted with `reverse=True` on (total, id), which reverses the
    tie-break.
  - The base twice overwrote the `sample.txt` input of `cli_top_flag`, and once
    moved a class out of its module in `circular_import`.

Not measured: the card's benchmarks, the agent eval on BF16 (`pi` or
`pi-mtp`), long-context prefill, and the `base` profile.
