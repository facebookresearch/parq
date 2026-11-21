# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE.torchao file in this directory.

import torch

from .quantizer import Quantizer


class FastAffineQ2Quantizer(Quantizer):
    """Affine Q2 quantizer selecting three grids followed by one least-squares fit"""

    def __init__(self, compile_search=False, chunk_groups=65536):
        super().__init__(center=False)
        if chunk_groups < 1:
            raise ValueError("chunk_groups must be positive")
        self.compile_search = compile_search
        self.chunk_groups = chunk_groups
        self._compiled = (
            torch.compile(self._calibrate, fullgraph=True, dynamic=True)
            if compile_search
            else None
        )

    def get_quant_size(self, b):
        if b != 2:
            raise ValueError("FastAffineQ2Quantizer requires two bits")
        return 4

    @staticmethod
    def _evaluate(x, scale, bias, dtype):
        scale, bias = scale.to(dtype).float(), bias.to(dtype).float()
        codes = torch.round((x - bias) / scale).clamp(0, 3)
        hard = codes * scale + bias
        loss = (hard - x).square().mean(-1, keepdim=True)
        return hard, codes, loss, scale, bias

    def _calibrate(self, p):
        x = p.float()
        low, high = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
        width = ((high - low) / 3).clamp_min(1e-7)
        radius = torch.minimum(
            2 * x.abs().mean(-1, keepdim=True), x.abs().amax(-1, keepdim=True)
        )
        radius = radius.clamp_min(torch.finfo(p.dtype).eps)
        # MLX's zero-aligned seed, normalized to increasing unsigned codes.
        negative = low.abs() > high.abs()
        native_scale = torch.where(negative, width, -width)
        edge = torch.where(negative, low, high)
        zero = torch.round(edge / native_scale)
        native_scale = torch.where(
            zero != 0, edge / torch.where(zero != 0, zero, 1), native_scale
        )
        native_bias = torch.where(zero != 0, edge, 0)
        native_bias = torch.minimum(native_bias, native_bias + 3 * native_scale)
        seeds = (
            (width, low),
            (2 * radius / 3, -radius),
            (native_scale.abs(), native_bias),
        )
        hard, codes, loss, scale, bias = self._evaluate(x, *seeds[0], p.dtype)
        for candidate_scale, candidate_bias in seeds[1:]:
            new_hard, new_codes, new_loss, new_scale, new_bias = self._evaluate(
                x, candidate_scale, candidate_bias, p.dtype
            )
            better = new_loss < loss
            hard, codes = (
                torch.where(better, new_hard, hard),
                torch.where(better, new_codes, codes),
            )
            scale, bias = (
                torch.where(better, new_scale, scale),
                torch.where(better, new_bias, bias),
            )
            loss = torch.minimum(loss, new_loss)
        mean_x, mean_q = x.mean(-1, keepdim=True), codes.mean(-1, keepdim=True)
        variance = (codes - mean_q).square().mean(-1, keepdim=True)
        covariance = ((codes - mean_q) * (x - mean_x)).mean(-1, keepdim=True)
        refined_scale = torch.where(
            variance > 0, covariance / variance.clamp_min(1e-20), scale
        ).clamp_min(1e-7)
        refined_bias = mean_x - refined_scale * mean_q
        new_hard, _, new_loss, new_scale, new_bias = self._evaluate(
            x, refined_scale, refined_bias, p.dtype
        )
        # Keep the seed if rounding stored metadata makes refinement worse.
        better = new_loss < loss
        hard = torch.where(better, new_hard, hard)
        scale, bias = (
            torch.where(better, new_scale, scale),
            torch.where(better, new_bias, bias),
        )
        levels = bias + scale * torch.arange(4, device=p.device)
        return hard, levels

    @torch.no_grad()
    def quantize(self, p, b, dim=None):
        self.get_quant_size(b)
        if p.ndim != 2 or dim != -1 or p.shape[-1] not in (32, 64, 128):
            raise ValueError("Expected rows of logical input-column groups32/64/128")
        calibrate = self._compiled or self._calibrate
        if p.shape[0] <= self.chunk_groups:
            return calibrate(p)
        hard = torch.empty_like(p, dtype=torch.float32)
        levels = torch.empty((p.shape[0], 4), device=p.device, dtype=torch.float32)
        for start in range(0, p.shape[0], self.chunk_groups):
            stop = min(start + self.chunk_groups, p.shape[0])
            hard[start:stop], levels[start:stop] = calibrate(p[start:stop])
        return hard, levels
