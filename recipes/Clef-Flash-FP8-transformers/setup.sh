#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
#
# Create the serving environment from pyproject.toml + uv.lock, then prove it
# can actually drive this GPU.
#
#   ./setup.sh            uv sync --locked into $VENV (default ~/venvs/clef-flash)
#
# The venv lives outside the recipe (CONVENTIONS.md, VENV); uv is pointed at it
# with UV_PROJECT_ENVIRONMENT. uv's package cache stays at its own default,
# ~/.cache/uv, shared with every other uv project on the host. --locked means
# a pyproject.toml edited without re-locking fails here instead of silently
# resolving something new.
set -euo pipefail

case "${1:-}" in
  -h|--help) sed -n '4,14p' "$0" | sed 's/^# \?//'; exit 0 ;;
esac

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
require_aarch64

command -v uv >/dev/null || { printf 'error: uv is not on PATH (https://docs.astral.sh/uv/)\n' >&2; exit 1; }

printf 'project : %s\n' "${EXPERIMENT_DIR}/pyproject.toml"
printf 'venv    : %s\n\n' "${VENV}"
UV_PROJECT_ENVIRONMENT="${VENV}" uv sync --locked --project "${EXPERIMENT_DIR}"

printf '\nchecking the environment against this GPU\n'
"${PYTHON}" - <<'PYEOF'
import sys
import torch, transformers
from transformers.utils.import_utils import is_flash_linear_attention_available

print(f"  python        {sys.version.split()[0]}")
print(f"  torch         {torch.__version__} (CUDA {torch.version.cuda})")
print(f"  transformers  {transformers.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("error: torch cannot see a CUDA device -- a CPU-only wheel was installed?")
cap = torch.cuda.get_device_capability()
archs = torch.cuda.get_arch_list()
print(f"  device        {torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}; wheel carries {' '.join(archs)}")
# sm_120 SASS runs on sm_121; an "a"-suffixed arch would not, and neither would
# a wheel with nothing for major 12. Prove it with a real kernel, not the list.
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
assert torch.isfinite(x @ x).all()
print("  kernel        bf16 matmul ran on the GPU")
if not is_flash_linear_attention_available():
    raise SystemExit("error: flash-linear-attention is missing; transformers would fall back to a slow torch loop")
print("  fla           flash-linear-attention present (Triton GDN kernels)")
PYEOF
printf '\nok. next: ./download.sh, then ./preflight.sh\n'
