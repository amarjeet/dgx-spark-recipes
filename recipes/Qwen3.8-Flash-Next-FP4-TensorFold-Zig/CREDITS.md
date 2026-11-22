# Credits

This repository is a thin layer of scripts. Almost everything that makes it work was built by others. Its own work is
licensed under the MIT License ([`LICENSE`](LICENSE)). The image `build.sh` builds also carries TensorFold's
license files from the pinned checkout, and the license files of the libraries listed under [Runtime stack](#runtime-stack).

## Model

- **[Qwen3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** by the
  [Qwen team](https://huggingface.co/Qwen) (Alibaba): the model's design, training and evaluations, its chat template
  and its MTP head. Its license, the
  [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE), comes with the
  weights.
- **NVIDIA** ([nvidia](https://huggingface.co/nvidia)): the checkpoint this recipe serves by default,
  [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local): Qwen3.8 Flash Next quantized with
  quantization-aware distillation (QAD) against the bf16 model as teacher, using NVIDIA Model Optimizer: NVFP4 routed experts with CSF-compressed
  scale factors, MXFP8 attention, DeltaNet and shared-expert linears, an NVFP4 n-gram table and quantized MTP and vision towers. On NVIDIA's own
  numbers it scores 91.5% on GPQA Diamond and 88.7% on ToolEvalBench against 92.2% and 89.4% for bf16 (its model card, not measured here). Used under the
  [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/); the base model's
  Qwen Community License 1.0 applies as well.
- **drowzeys** ([drowzeys](https://huggingface.co/drowzeys)): the Keys house-qsa abliteration overlay for the NVIDIA checkpoint,
  [`drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-FP4-QAD-house-qsa-L3-47`](https://huggingface.co/drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-FP4-QAD-house-qsa-L3-47): the requantized
  `self_attn.o_proj` weights of 12 layers that `ABLIT=1` grafts onto the checkpoint. Gated; its license is NVIDIA's and the Qwen license. Not part of this
  repository: `download.sh --ablit` downloads the sidecar. `scripts/ablit_view.py` is MiaAI-Lab's own and uses none of the overlay's scripts.
- **azampatti** ([azampatti](https://huggingface.co/azampatti)): upstream's optional INT4 checkpoint (not carried by this recipe),
  [`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound):
  azampatti authored its top-5 expert cut, the shared-expert healing, and the checkpoint. Under the license on
  its model card (the Qwen license).
- **Intel** ([Intel](https://huggingface.co/Intel)): the AutoRound int4 quantization the INT4-AR checkpoint is built on,
  [`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound).
- **Saren-Arterius** ([Saren-Arterius](https://github.com/Saren-Arterius)): the hybrid checkpoint the INT4-AR
  checkpoint is built from and its FP8 n-gram table,
  [`Saren/Qwen3.8-Flash-Next-ple-table-fp8`](https://huggingface.co/Saren/Qwen3.8-Flash-Next-ple-table-fp8) (in its
  `ple-table/`). The recipe uses their weights only, no code from their repositories.
- **[Vontra](https://huggingface.co/Vontra)**: the checkpoint this recipe served before the Zig engine,
  [`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP): the MLX
  4-bit conversion, the preserved native MTP draft head, validation and packaging.

## Inference engine

- **[TensorFold](https://github.com/ashhart/TensorFold)** and its Zig engine (`tensorfold-native`, release
  [v1.2.7](https://github.com/ashhart/TensorFold/releases/tag/v1.2.7)) by Ash Hart
  ([ashhart](https://github.com/ashhart)) and the TensorFold contributors (Apache 2.0 from v0.6.0; code written before
  v0.6.0 keeps its MIT notice): the engine that serves the model, its CUDA runtime and kernels, MTP drafting with
  exact verification and the OpenAI-compatible server. `build.sh` builds it from TensorFold's source at the
  commit pinned in `profiles.sh`.
  The Flash Next CUDA engine in this recipe is ported from TensorFold's Python Flash Next CUDA engine, written by
  Ash Hart and the TensorFold contributors ([contributors](https://github.com/ashhart/TensorFold/graphs/contributors));
  their authorship is recorded in TensorFold's history.
- **The patches in `patches/` on TensorFold v1.2.7**, under upstream's MIT License. They modify TensorFold's own Flash Next CUDA
  engine, by Ash Hart and the TensorFold contributors, whose code keeps TensorFold's licenses (see `NOTICE`):
  - `0001-vision-nvidia-fp4-tower`: the CUDA vision tower loads NVIDIA's MXFP8 and NVFP4 vision linears, decoded to bf16 at load. MiaAI-Lab.
  - `0002-vision-image-tokens-16384`: image-token budget 16,384. MiaAI-Lab.
  - `0003-kv-cache-fp8`: `--kv-cache fp8` for Flash Next, so five full 262,144-token windows fit. MiaAI-Lab; the format is adapted from
    MiaAI-Lab's GLM recipe patch `0038-glm-kv-fp8` in
    [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold).
  - `0004-prompt-mxfp8-tile`: prompt passes run the wide MXFP8 projections on a 128 x 128 tile (64 x 64 for short calls) with the block
    scales folded into the weights. MiaAI-Lab, built on TensorFold's qmmf kernels and fragment helpers by Ash Hart and the TensorFold contributors.
  - `0005-flashnext-cuda-video`: video parts on the CUDA engine, by **[chai-md](https://github.com/chai-md)**
    ([upstream #90](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/pull/90)), re-based onto v1.2.7 by MiaAI-Lab.
- **Earlier patch series** this recipe applied before the move to v1.2.7: patches 0001 to 0015 on TensorFold `db281878`, by MiaAI-Lab,
  including NVIDIA FP4 support in the Zig engine (`0011` to `0015`) as a port of TensorFold's Python implementation of the same formats,
  written by the TensorFold Contributors. Upstream's [CREDITS](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/b956dba/CREDITS.md)
  lists these and the intermediate v1.1.4 series one by one, with their other authors (tempeste, BadAd84).
- **The Zig CUDA serving path and the CUDA family registry** were authored by Jürgen Schmied
  ([jschmied](https://github.com/jschmied)) in [TensorFold PR #443](https://github.com/ashhart/TensorFold/pull/443)
  (commit [`59e77e8`](https://github.com/ashhart/TensorFold/commit/59e77e8f4b875ce0e863a8c896fc8e424bc539ac)):
  `tensorfold-native` serving CUDA families through one registry, which this recipe's engine runs on.
- TensorFold itself builds on, and credits in its
  [third-party notices](https://github.com/ashhart/TensorFold/blob/v1.2.7/THIRD_PARTY_NOTICES.md), the
  projects whose code it adapts. Those notices name
  [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) (Apple, MIT),
  [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) (Prince Canuma, MIT),
  [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (turboderp, MIT), and the Qwen Flash Next modeling code in
  Hugging Face [transformers](https://github.com/huggingface/transformers) (the Qwen Team and the Hugging Face team,
  Apache 2.0).
- **[b12x](https://github.com/local-inference-lab/b12x)** by local-inference-lab (Apache-2.0): the one-shot RoCE
  all-gather in the engine implements the RoCEnante protocol of b12x by local-inference-lab; that implementation is
  new code written for this engine. This recipe serves one Spark, so it does not turn that path on.
- **[Zig](https://ziglang.org)** by the Zig Software Foundation and the Zig contributors (MIT): the language and
  compiler TensorFold's Zig engine is written in and built with (0.17.0).

## Earlier patches on the Python engine

This recipe no longer applies those patches (the serving path is the Zig engine). The people who authored them:

- `0001-cuda-live-token-counters`: by MiaAI-Lab, submitted upstream as
  [TensorFold #79](https://github.com/ashhart/TensorFold/pull/79). The recipe's earlier typed-tool-parameters patch
  ([#75](https://github.com/ashhart/TensorFold/pull/75)) is part of TensorFold v0.3.6.3.
- `0006-flash-next-prefill-rows`: a port of [TensorFold #40](https://github.com/ashhart/TensorFold/pull/40) by
  **[MovieMaker93](https://github.com/MovieMaker93)**, rebased onto v0.3.6.3.
- `0007-flash-next-copy-drafts`: uses TensorFold's own `CopyIndex` prompt-lookup index from its Qwen3.5 27B engine.
- `0008-flash-next-vision`: builds on TensorFold's Qwen3.5/3.8 dense vision support, runs the vision tower from
  Hugging Face transformers, and follows transformers' Qwen3.5 rotary index and Qwen3-VL's video processing.
  Image and video input on the Zig engine is adapted from this patch and from patch 0009 (many images), both by
  MiaAI-Lab, and from TensorFold's vision code by Ash Hart ([ashhart](https://github.com/ashhart)) and the
  TensorFold contributors.
- `0010-flash-next-draft-languages` (the optional language image, not built anymore): the language token lists come from
  **Javier ([jvr0x](https://github.com/jvr0x))**'s language draft vocabularies for this model's vLLM recipe
  ([MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark#84](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/pull/84)),
  built from each language's Wikipedia by token frequency.
- `0002`-`0005`, `0007`-`0009`: by MiaAI-Lab.
- FP8 KV (`KV_DTYPE=fp8`) is adapted from MiaAI-Lab's GLM recipe patch `0038-glm-kv-fp8` in
  [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold).

## Recipe

- **[321sssrt-bit](https://github.com/321sssrt-bit)** authored the full-window admission fix and
  `tools/context_boundary.py` in
  [Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold)
  ([issue #1](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/issues/1),
  [PR #2](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/pull/2)): admission counted the draft
  window against `--context`, so a prompt plus `max_tokens` equal to the window was refused. The engine already keeps
  those rows beyond the window. The admission change in the engine patches and `tools/context_boundary.py` are theirs.
- **[sparkDash](https://github.com/MiaAI-Lab/sparkDash)** by MiaAI-Lab: the Decode and Prefill benchmarks upstream's performance numbers were measured with.
- **[joernchen](https://github.com/joernchen)** asked for PDF document input in
  [upstream #88](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/pull/88). TensorFold v1.2.7 took the feature into
  its own server; `PDF=1`, the pinned `libpdfium.so` in the image and the `--vision-pdf` wiring are MiaAI-Lab's.
- The scripts (`start.sh`, `stop.sh`, `scripts/`) and the checks in `tools/` are MiaAI-Lab's. `client.py`,
  `needle.py`, `toolcheck.py` and `prompt_reuse.py` are adapted from MiaAI-Lab's two-Spark recipe
  [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold);
  `exact.py` is from MiaAI-Lab's Flash Next two-Spark TensorFold recipe. `scripts/ablit_view.py` and
  `tools/test_ablit_view.py` are MiaAI-Lab's, unchanged.

## Runtime stack

- **[NVIDIA PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)**
  (`nvcr.io/nvidia/pytorch:26.07-py3`), the base of the image, with NVIDIA's CUDA (nvcc builds the engine's kernels),
  NCCL and related libraries. Governed by the NVIDIA Software License Agreement and the Product-Specific Terms for
  NVIDIA AI Products.
- **[libjpeg-turbo](https://libjpeg-turbo.org/)** (IJG/BSD-3-Clause-like) and **[libwebp](https://github.com/webmproject/libwebp)**
  (BSD-3-Clause): the decoders the engine loads for JPEG and WebP input, built from pinned sources into the image by `build.sh`.
- **[FFmpeg](https://ffmpeg.org/)** (LGPL-2.1) and **libtfvideo** (TensorFold's `zig/src/video/tf_video.c`, Apache-2.0): the
  decoder behind `VIDEO=1`, at the release TensorFold pins, built by TensorFold's own `tools/zig/build_ffmpeg.sh`.
- **[PDFium](https://pdfium.googlesource.com/pdfium)** (BSD-3-Clause) as shipped by
  **[pypdfium2](https://pypdfium2.readthedocs.io/)** (pypdfium2 contributors): the `libpdfium.so` behind `PDF=1`, taken from
  pypdfium2 5.14.0's aarch64 wheel, pinned by SHA-256, with the wheel's license files.
- **[xgrammar](https://github.com/mlc-ai/xgrammar)** (MLC-AI team) and **[dlpack](https://github.com/dmlc/dlpack)** (Apache 2.0),
  inside `libtfgrammar.so` from TensorFold's v1.2.7 release archive: structured output and required tool calls.
- **[Hugging Face Hub](https://huggingface.co/)**: model hosting, the `hf` CLI and `huggingface_hub` (Apache 2.0), and
  the [safetensors](https://github.com/huggingface/safetensors) format (Apache 2.0) the checkpoint ships in.
- **[Docker](https://www.docker.com/)** and the
  **[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)** (Apache 2.0): running the server
  on the GPU in a container.

## Hardware

- **[NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)** (GB10 Grace Blackwell,
  128 GB unified memory): every number in the README was measured on one, here or upstream.

## README

- Badges by [Shields.io](https://shields.io/).

If you believe something here is missing or credited wrongly, please open an issue.
