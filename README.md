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

## Requirements

- NVIDIA DGX Spark (GB10, `aarch64`). Recipes assert the architecture and refuse
  to run elsewhere — the measurements and memory budgets assume this machine.
- Docker with the NVIDIA container runtime (`--gpus all`).
- `curl`, `python3` (stdlib only — no pip installs), `nvidia-smi`.
- A Hugging Face token for gated repos. Export `HF_TOKEN`, or log in with
  `hf auth login`; recipes read the standard token file. The Hub works
  anonymously at a lower rate limit.

## Where things are written

Nothing heavy is written inside a recipe directory. Weights land in the tool's
own standard cache and are shared across recipes; bench results and verification
stamps go to a state directory. Every path is an environment-overridable
variable whose default is the standard location.

See [CONVENTIONS.md](CONVENTIONS.md) for the full storage rules, the recipe
layout, and the reasoning about sharing one GPU between servers.

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

[MIT](LICENSE).
