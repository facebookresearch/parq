# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE.torchao file in this directory.

import torch

from .quantizer import Quantizer


def _divide(numerator, denominator):
    """Use IEEE float32 division for agreement with the MLX CPU reference."""
    return (numerator.double() / denominator).float()


class AffineQ2Quantizer(Quantizer):
    """Affine Q2 quantizer with MLX calibration or independent scale/bias search"""

    def __init__(self, calibration="mlx", iterations=8, chunk_groups=65536):
        super().__init__(center=False)
        if calibration not in ("mlx", "minmax", "mse"):
            raise ValueError("calibration must be mlx, minmax or mse")
        if iterations < 1 or chunk_groups < 1:
            raise ValueError("iterations and chunk_groups must be positive")
        self.calibration = calibration
        self.iterations = iterations
        self.chunk_groups = chunk_groups

    def get_quant_size(self, b):
        if b != 2:
            raise ValueError("AffineQ2Quantizer requires two bits")
        return 4

    @staticmethod
    def _mlx(x):
        low, high = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
        scale = _divide(high - low, 3).clamp_min(1e-7)
        negative = low.abs() > high.abs()
        scale = torch.where(negative, scale, -scale)
        edge = torch.where(negative, low, high)
        zero = torch.round(_divide(edge, scale))
        nonzero = zero != 0
        scale = torch.where(
            nonzero, _divide(edge, torch.where(nonzero, zero, 1)), scale
        )
        bias = torch.where(nonzero, edge, 0)
        return scale, bias

    @staticmethod
    def _minmax(x):
        low, high = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
        return _divide(high - low, 3).clamp_min(1e-7), low

    @staticmethod
    def _stored(x, scale, bias, dtype, codes=None):
        scale = scale.to(dtype).float()
        bias = bias.to(dtype).float()
        # Parameters are stored in the input dtype; code selection in native
        # MLX uses its unrounded float32 parameters before that storage cast.
        if codes is None:
            codes = torch.round(_divide(x - bias, scale)).clamp(0, 3)
        hard = codes * scale + bias
        return hard, scale, bias, codes

    def _mse(self, x, dtype):
        scale, bias = self._minmax(x)
        hard, best_scale, best_bias, _ = self._stored(x, scale, bias, dtype)
        best_loss = (hard - x).square().mean(-1, keepdim=True)
        radius = torch.minimum(
            2 * x.abs().mean(-1, keepdim=True), x.abs().amax(-1, keepdim=True)
        )
        radius = radius.clamp_min(torch.finfo(dtype).eps)
        native_scale, native_bias = self._mlx(x)
        # Normalize native descending codes to an increasing affine grid.
        initial = (
            (scale, bias),
            (_divide(2 * radius, 3), -radius),
            (
                native_scale.abs(),
                torch.minimum(native_bias, native_bias + 3 * native_scale),
            ),
        )
        mean_x = x.mean(-1, keepdim=True)
        for scale, bias in initial:
            for _ in range(self.iterations + 1):
                hard, scale, bias, codes = self._stored(x, scale, bias, dtype)
                loss = (hard - x).square().mean(-1, keepdim=True)
                better = loss < best_loss
                best_scale = torch.where(better, scale, best_scale)
                best_bias = torch.where(better, bias, best_bias)
                best_loss = torch.minimum(loss, best_loss)
                mean_q = codes.mean(-1, keepdim=True)
                variance = (codes - mean_q).square().mean(-1, keepdim=True)
                covariance = ((codes - mean_q) * (x - mean_x)).mean(-1, keepdim=True)
                # With assignments fixed, fit scale and bias by least squares.
                scale = torch.where(
                    variance > 0, covariance / variance.clamp_min(1e-20), scale
                ).clamp_min(1e-7)
                bias = mean_x - scale * mean_q
        hard, _, _, _ = self._stored(x, best_scale, best_bias, dtype)
        return hard, best_scale, best_bias

    @torch.no_grad()
    def quantize(self, p, b, dim=None):
        self.get_quant_size(b)
        if p.ndim != 2 or dim != -1 or p.shape[-1] not in (32, 64, 128):
            raise ValueError("Expected rows of logical input-column groups32/64/128")
        hard = torch.empty_like(p, dtype=torch.float32)
        levels = torch.empty((p.shape[0], 4), device=p.device, dtype=torch.float32)
        for start in range(0, p.shape[0], self.chunk_groups):
            stop = min(start + self.chunk_groups, p.shape[0])
            x = p[start:stop].float()
            if self.calibration == "mse":
                q, scale, bias = self._mse(x, p.dtype)
            else:
                scale, bias = (
                    self._mlx(x) if self.calibration == "mlx" else self._minmax(x)
                )
                codes = torch.round(_divide(x - bias, scale)).clamp(0, 3)
                q, scale, bias, _ = self._stored(x, scale, bias, p.dtype, codes)
            hard[start:stop] = q
            levels[start:stop] = (
                (bias + scale * torch.arange(4, device=p.device)).sort(-1).values
            )
        return hard, levels
