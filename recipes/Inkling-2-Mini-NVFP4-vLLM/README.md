# Inkling 2 Mini · NVFP4 · vLLM + DSpark

[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini): a looped MoE, 26.4B
total / 3.4B active (the middle 12 of 24 layers run three times, ~5.4B effective), 8 of 128 experts per layer,
sliding-window plus 4 global attention layers, text and image in. The official checkpoint is NVFP4 experts (MXFP8
in the last four layers) with BF16 attention, 18.8 GiB, and carries its own **DSpark** speculative drafter
(`mtp.safetensors`, 3 layers, block size 8). Native window 1,048,576 tokens.

Port of [tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark)
by Tech2wild, at `300942d` (2026-11-21), which starts from the [vLLM recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini).
Its three launchers are the three profiles here, flag for flag.

**Status: runs on this host.** `standard` and `max-fp8` started and passed their checks here on 2026-11-22;
`max` (BF16 at 1M) is upstream's and was not started here. Numbers are in [Measurements](#measurements).

## Run

```bash
./download.sh                 # 18.8 GiB into the HF cache, sha256-verified, resumable
docker pull vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e
./preflight.sh [profile]      # image flags, checkpoint, memory, port, disk
./start.sh [profile]          # serves on :8015, waits for /health, watches memory
./scripts/sanity.py chat      # two short chats: correctness, TTFT, decode tok/s
./scripts/sanity.py needle --tokens 200000
./bench.sh [1,2,4,8] [1024]   # decode tok/s per stream and aggregate, results in OUT_DIR
./status.sh
./stop.sh
```

Or `scripts/run-inkling-2-mini-nvfp4.sh [profile]` from the repo root, which runs preflight and then start.
`DRY_RUN=1 ./start.sh max-fp8` prints the `docker run` command.

| Profile | Window | KV | MemAvailable left | Use |
|---|---|---|---|---|
| `standard` (default) | 262,144 | BF16, `--gpu-memory-utilization 0.80`, sized by vLLM after warmup | ~20 GiB | chat, agents, tools; room for the OS |
| `max` | 1,048,576 | BF16, `KV_GIB` = 80 (cap 80) | ~7 GiB (upstream) | past 262K with unquantized KV: one full 1M request plus shorter ones |
| `max-fp8` | 1,048,576 | FP8 E4M3 unit-scale, `KV_GIB` = 74 (cap 74) | ~9 GiB | the most context: 3.3M tokens, three full 1M requests |

The KV in `max-fp8` has no calibrated scales. Upstream measured it within 1.4 points of BF16 on their 69-scenario
tool eval and it passed a ~1M needle there, but check it on your own long-context work. A near-1M prompt takes about
26 minutes to its first token (upstream), so the 1M tiers suit long documents and sessions, not latency.

## What is pinned

| | |
|---|---|
| Model | `1fb4134b9a74…` ("upload model"), 14 files, 18.8 GiB, every one checksummed in `manifests/model.json`. Upstream's tested revision; later commits only touch `config.json` and the card |
| Image | `vllm/vllm-openai:inkling-2-mini` by digest `c2dc759f…` (27 GB, vLLM `0.1.dev25855+g7cb5dae22`, torch 2.14 cu130) |

`preflight.sh` reads the image's vLLM source for `kv_cache_memory_bytes`, the `dspark` method and the
`inkling` reasoning and tool parsers, so a different image fails there rather than minutes into a load.
Regenerate the manifest with `scripts/make_manifest.py thinkingmachines/Inkling-2-Mini <rev> manifests/model.json`.

## Memory

On GB10 the KV pool comes out of the same 121.6 GiB the OS lives on, and an exhausted pool hangs the box rather
than raising an OOM. So:

- `preflight.sh` and `start.sh` refuse to launch unless MemAvailable covers the profile: `standard` needs
  0.80 x MemTotal + 6 GiB (vLLM refuses below the first term anyway); the max tiers need `KV_GIB` + an
  overhead estimate (31 GiB BF16, 36 GiB FP8; measured here as ~35 GiB for FP8) + `HOST_FLOOR_GIB` (6).
- `KV_GIB` above the measured ceiling (80 BF16, 74 FP8) is refused. FP8 at 80 GiB reported a bigger pool but
  fell to 1.9 GiB MemAvailable during warmup upstream: FP8 splits the cache into 99 KV groups (13 with BF16)
  and their per-group buffers cost ~5 GiB outside the pool.
- `start.sh` folds in upstream's `watch_start.sh`: while it waits for `/health` it kills the container if
  MemAvailable drops under `KILL_FLOOR_GIB` (3).
- Before launching, `start.sh` drops clean page cache (the "GB10 page-cache trap") when passwordless sudo is
  available, as upstream does.
- `RESTART_POLICY` defaults to `no`, as upstream: nothing relaunches a load after a memory kill.

**None of the tiers fits beside another model server**, Qwen3.8-Flash-Next included. Stop it first.

## Storage

Nothing heavy is in this directory. Upstream downloads with `--local-dir ~/models/Inkling-2-Mini` and mounts
that; here the checkpoint lives in the shared Hugging Face cache and is loaded by repo id and `--revision` with
`HF_HUB_OFFLINE=1`. The vLLM, flashinfer and triton caches (upstream leaves them inside the container, so every
start re-JITs) are the shared defaults, mounted onto the container's own default paths, so no cache env var is set
inside. Logs, the active profile and verification stamps go to `OUT_DIR`
(`~/.local/state/dgx-spark-recipes/Inkling-2-Mini-NVFP4-vLLM/`).

## Using it

OpenAI-compatible at `http://<host>:8015/v1`; model `inkling-2-mini` or `thinkingmachines/Inkling-2-Mini`.
Thinking comes back in `reasoning_content` and shares `max_tokens`. Effort goes in
`chat_template_kwargs.reasoning_effort`: `none` 0.0, `minimal` 0.1, `low` 0.2, `medium` 0.7, `high` 0.9
(the default), `xhigh`/`max` 0.99, or a number; `none` still thinks briefly. Tool calls use the `inkling`
parser with auto tool choice. Images go in as OpenAI `image_url` parts, 40 to 4096 px a side; the drafter sees
text only, so DSpark helps less on image-heavy turns. Sampling defaults come from `generation_config.json`
(temperature 1.0). Set client timeouts of 30+ minutes for near-1M prompts.

```bash
curl -s http://localhost:8015/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini",
  "messages": [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
  "max_tokens": 1024,
  "chat_template_kwargs": {"reasoning_effort": "low"}
}'
```

## Knobs

`PORT` (8015), `HOST` (`0.0.0.0`; there is no API key, so `127.0.0.1` keeps it off the LAN), `IMAGE`,
`CONTAINER_NAME` (`inkling-2-mini`), `SERVED_MODEL_NAME`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`
(standard), `KV_GIB` (max tiers; step down by 4 for more OS headroom), `SPEC_TOKENS` (7, the only value the
checkpoint accepts; 0 disables DSpark), `MEM_HEADROOM_BYTES`, `HOST_FLOOR_GIB`, `KILL_FLOOR_GIB`,
`RESTART_POLICY`, `EFFORT` (for `bench.sh`, default `low`), and the storage variables in [CONVENTIONS.md](../../CONVENTIONS.md).

## Measurements

DGX Spark (GB10, 121.6 GiB), image digest `c2dc759f`, 2026-11-22, nothing else running, caches cold for the
first start. Upstream's figures for the same tiers are in brackets.

| | `standard` | `max-fp8` (74 GiB) |
|---|---|---|
| Ready after | 336 s, cold JIT caches (5 to 6 min) | 246 s, warm caches |
| Weights | 19.74 GiB, loaded in 112 s | |
| KV pool | 1,354,918 tokens, 5.17x at 262,144 (1,346,671) | 3,308,911 tokens, 3.16x at 1,048,576 (same) |
| MemAvailable after warmup | 19.7 GiB (20.2) | 8.8 GiB, lowest 7.7 during start (8.2 to 8.4; 7.3); 3.7 GiB swap left from startup |
| `sanity.py chat` | pass both: "391" at 72.7 tok/s, planets at 38.0 tok/s, TTFT ~1.5 s | pass both: 74.0 and 37.9 tok/s, TTFT ~1.6 s |
| Tool call | `get_weather({"city":"Atlanta"})` parsed | |
| Needle, 199,985 tokens at 50% | | pass, TTFT 129.9 s (1,540 tok/s prefill; 1,693), decode 41.6 tok/s |

### Decode throughput

`./bench.sh` on `max-fp8`, 2026-11-22 (`scripts/bench_tps.py`): natural-language and code prompts, each held to
exactly 1,024 tokens (`min_tokens`) so the runs compare, `reasoning_effort: low`, after a warm-up request.
Per-stream decode is completion tokens over the time after the first token; aggregate is all tokens over the whole
span; acceptance length is the DSpark mean from vLLM's `/metrics`.

| Concurrency | Per stream tok/s (min / mean / max) | Aggregate tok/s | TTFT | DSpark acceptance |
|---|---|---|---|---|
| 1 | 37.4 to 40.8 (two runs) | 37.1 to 40.4 | 0.2 s | 2.3 to 2.5 |
| 2 | 33.2 / 42.6 / 51.9 | 62.5 | 1.9 s (first time at this width; 0.26 s on a re-run) | 3.1 |
| 4 | 20.2 / 25.6 / 36.4 | 77.8 | 1.9 s (same) | 2.6 |
| 8 | 15.8 / 19.5 / 32.3 | 125.3 | 0.4 s | 2.6 |

Long free-form output accepts fewer drafts than upstream's short chats (2.3 to 2.5 against their 3.1 mean), so about
40 tok/s is the realistic single-stream figure; the 72 to 74 tok/s in the sanity check is a 51-token arithmetic answer.
The spread across streams is the prompt: code and structured text accept more drafts than prose. The `standard` tier
was not benchmarked; decode should be close, since it is the same weights with a BF16 KV.

Not measured here: the `max` tier, the ~1M needle, images, prefill apart from the 200K needle, and upstream's
69-scenario eval (88.4 BF16 / 87.0 FP8; scorecards in the upstream repo).

## Credits and licenses

- **Recipe:** [tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark)
  by Tech2wild, MIT. Its three launch tiers and their flags, the KV sizing (80 GiB BF16, 74 GiB FP8) and the
  measurements behind it, the 99-KV-group finding for FP8, the page-cache drop and the startup memory watch are
  the basis of `profiles.sh`, `start.sh` and `preflight.sh`. `scripts/sanity.py` is its `sanity.py` with
  the default port changed. Upstream's MIT notice is in [`LICENSE`](LICENSE). The 69-scenario eval figures quoted
  above are upstream's, from [2Wild-Model-Eval](https://github.com/tonyd2wild/2Wild-Model-Eval), which follows the Pass/Partial/Fail
  method of wolttam's `tool-eval-bench`.
- **vLLM:** the [Inkling 2 Mini recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini) upstream starts from,
  Inkling 2 support, and the `vllm/vllm-openai:inkling-2-mini` image. Apache 2.0.
- **Model:** [Thinking Machines Lab](https://huggingface.co/thinkingmachines), Inkling 2 Mini, its NVFP4/MXFP8
  checkpoint and the DSpark drafter. Apache 2.0.
- **This port's scripts** are MIT (see [`LICENSE`](LICENSE)), except `scripts/download_snapshot.py`, copied from
  `Qwen3.8-Flash-Next-NVFP4-vLLM` with its AGPL-3.0-or-later header, as in the Nemotron recipe.
