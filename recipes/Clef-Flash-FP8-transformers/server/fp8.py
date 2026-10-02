# SPDX-License-Identifier: MIT
# Copyright (c) 2026 amarjeet
"""FP8 (e4m3) linear layers for the Clef backbone, quantized at load.

Why: in BF16, Clef-Flash costs ~25 GiB of the pool, which beside the
Qwen3.8-Flash-Next server at 1 x 262,144 leaves ~8 GiB -- below the 10 GiB at
which both recipes' watchdogs start treating a low MemFree as danger. FP8
weights for the decoder's linear layers give back ~6.5 GiB.

What is quantized: every nn.Linear inside the language model's decoder layers.
What is not: the token embeddings and lm_head (the joint schema head scores
options by reading lm_head's weight directly, so it must stay exact), the
vision tower, and the joint schema head itself.

Scheme: per-output-row weight scales, computed once; per-token activation
scales, computed every call; torch._scaled_mm with BF16 output, at most 4096
rows per call. Measured on GB10 for a 4096 -> 12288 projection:

    tokens    BF16      FP8 (this)
       512    0.86 ms   0.43 ms
      4096    4.92 ms   3.44 ms
      8192    8.98 ms   8.72 ms   (one unchunked call: 17.9 ms)
     16384   17.36 ms  17.03 ms   (one unchunked call: 35.9 ms)

Per-tensor scaling is faster still (8.8 ms at 16k) but lets one outlier token
set the scale for all of them; it was not adopted.

This changes the model's numbers. The recipe keeps it only because a
comparison against BF16 on a test set (scripts/fp8_drift.py) showed the same
winning answers -- see the README, which records the figures.
"""

from __future__ import annotations

import torch

F8 = torch.float8_e4m3fn
F8_MAX = torch.finfo(F8).max
# Rows per _scaled_mm call; see FP8Linear.forward.
ROWS_PER_CALL = 4096


class FP8Linear(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear) -> None:
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        weight = linear.weight.detach()
        scale = weight.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12) / F8_MAX
        self.register_buffer("weight_fp8", (weight.float() / scale).clamp(-F8_MAX, F8_MAX).to(F8))
        # _scaled_mm wants the second operand column-major and its scale as (1, N).
        self.register_buffer("weight_scale", scale.t().contiguous())
        self.bias = None if linear.bias is None else torch.nn.Parameter(linear.bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, self.in_features)
        rows = x2.shape[0]
        if rows <= ROWS_PER_CALL:
            out = self._mm(x2)
        else:
            # Row-wise _scaled_mm on GB10 falls off a cliff past ~4k rows
            # (measured at 4096 x 12288: 3.5 ms; at 8192: 17.9 ms, against BF16's
            # 9.0). Every row carries its own scale, so splitting by rows is
            # exact -- the same numbers as one call -- and keeps it on the fast
            # side at any length. It also bounds the quantization transient.
            out = torch.empty(rows, self.out_features, dtype=x.dtype, device=x.device)
            for start in range(0, rows, ROWS_PER_CALL):
                out[start:start + ROWS_PER_CALL] = self._mm(x2[start:start + ROWS_PER_CALL])
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1], self.out_features)

    def _mm(self, x2: torch.Tensor) -> torch.Tensor:
        scale = x2.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12) / F8_MAX
        # Scale in the activation's own dtype, not fp32: at 16k tokens an fp32
        # copy of a 12288-wide activation is 800 MB of transient per call, and
        # measured, it raised the warmup's high-water by 1.4 GiB. BF16 rounding
        # (2^-8) is 16x finer than e4m3's (2^-4), so nothing is lost.
        xq = (x2 * scale.reciprocal().to(x2.dtype)).clamp(-F8_MAX, F8_MAX).to(F8)
        return torch._scaled_mm(
            xq, self.weight_fp8.t(), scale_a=scale, scale_b=self.weight_scale, out_dtype=x2.dtype)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, fp8_e4m3 rowwise"


def quantize_decoder(backbone: torch.nn.Module) -> tuple[int, int]:
    """Swap the decoder's nn.Linear layers for FP8Linear, freeing each BF16 weight as it goes.

    Returns (layers converted, bytes of weight given back). Layers whose
    dimensions _scaled_mm cannot take (not multiples of 16) stay BF16.
    """
    converted, saved = 0, 0
    for layer in backbone.model.language_model.layers:
        for parent in list(layer.modules()):
            for name, child in list(parent.named_children()):
                if type(child) is not torch.nn.Linear:
                    continue
                if child.in_features % 16 or child.out_features % 16:
                    continue
                setattr(parent, name, FP8Linear(child))
                saved += child.weight.numel() * (child.weight.element_size() - 1)
                converted += 1
        # Hand each layer's freed BF16 blocks back as we go, so the peak is the
        # BF16 model, not the BF16 model plus a growing FP8 copy.
        torch.cuda.empty_cache()
    return converted, saved
