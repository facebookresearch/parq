# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from .quantizer import Quantizer


def get_q_max(
    q: Tensor, b: int, dim: int | None = None, scale_method: str = "mean"
) -> Tensor:
    if scale_method == "mean":
        # set range of quantization: min(b * |q|.mean(), |q|.max())
        q_abs = q.abs()
        if dim is not None:
            q_max = torch.minimum(
                b * q_abs.mean(dim=dim, keepdim=True),  # pyre-ignore[6,9]
                torch.max(q_abs, dim=dim, keepdim=True).values,  # pyre-ignore[6]
            )
        else:
            q_max = torch.minimum(b * q_abs.mean(), torch.max(q_abs))  # pyre-ignore[6]
    elif scale_method == "max":
        q_max = (
            q.abs().max(dim=dim, keepdim=True).values
            if dim is not None
            else q.abs().max()
        )
    else:
        raise NotImplementedError(f"Invalid {scale_method=}, choices=('mean','max')")
    return q_max


class UnifQuantizer(Quantizer):
    """Uniform and symmetric quantizer"""

    def __init__(
        self,
        center: bool = False,
        scale_method: str = "mean",
        int_shift: float = 0.5,
        zero_point: float = 0.5,
    ):
        """Set quantization function parameters.

        Args:
            center: whether to subtract p.mean() prior to quantization
            scale_method: compute scale based 'mean', multiples of |p|.mean(),
                or 'max', |p|.max() (default: 'mean')
            int_shift: float value to shift the lower bound of integer range by:
                -2^{b - 1} + int_shift (default: 0.5). Using 0.5 results in 2^b
                values. E.g., [-1.5, -0.5, 0.5, 1.5] for b=2.
            zero_point: float value to shift p by after scale and round.
        """
        assert scale_method in ("max", "mean"), f"Invalid {scale_method=}"
        super().__init__(center=center)

        self.scale_method = scale_method
        self.int_shift = int_shift
        self.zero_point = zero_point

    def get_quant_size(self, b: int) -> int:
        """Levels in [-2^{b-1} + self.int_shift, 2^{b-1} - self.int_shift].

        Note that range_absmax = 2^{b-1} - self.int_shift on both ends of the
        boundary and the interval is closed."""
        return math.floor(2**b - 2 * self.int_shift) + 1

    def quantize(
        self, p: Tensor, b: int, dim: int | None = None
    ) -> tuple[Tensor, Tensor]:
        """Instantiation of Quantizer.quantize() method"""
        assert b != 0, "Please use TernaryUnifQuantizer instead"

        if self.center:
            q, mean = super().remove_mean(p.detach(), dim=dim)
        else:
            q = p.detach().clone()
            mean = torch.zeros(1, dtype=p.dtype, device=p.device)
        q_max = get_q_max(q, b, dim=dim, scale_method=self.scale_method)
        q_max.clamp_(min=torch.finfo(q.dtype).tiny)

        # clamp to quantization range
        q.copy_(torch.minimum(torch.maximum(q, -q_max), q_max))

        # scale from [-2^{b-1}+int_shift, 2^{b-1}-int_shift] to [-q_max, q_max]
        range_absmax = 2 ** (b - 1) - self.int_shift
        s = q_max / range_absmax

        # scale by 1/s -> shift -zero_point -> round -> shift +zero_point ->
        # scale by s, where shift ensures rounding to integers
        q.div_(s).sub_(self.zero_point).round_().add_(self.zero_point).mul_(s)

        # set of all target quantization values
        Q = torch.arange(
            -range_absmax, range_absmax + 1e-5, dtype=p.dtype, device=p.device
        )
        if dim is not None:
            Q = Q.unsqueeze(0).mul(s)  # broadcasted multiply requires copy
        else:
            Q.mul_(s)

        # return quantized tensor and set of possible quantization values
        if self.center:
            q += mean
            Q += mean
        return q, Q


class MaxUnifQuantizer(UnifQuantizer):
    def __init__(
        self,
        center: bool = False,
        scale_method: str = "max",
        int_shift: float = 1.0,
        zero_point: float = 0.0,
    ):
        """Set quantization function with int_shift=1.0.

        The final quantization range includes 2^b - 1 quantized values. E.g.,
        [-1, 0, 1] for b=2. The quantization scale is determined by |p|.max()
        by default and zero point is 0.0.
        """
        super().__init__(
            center=center,
            scale_method=scale_method,
            int_shift=int_shift,
            zero_point=zero_point,
        )


class AsymUnifQuantizer(Quantizer):
    """Uniform asymmetric quantizer: 2^b levels at integer multiples of a scale,
    so zero is exact.

    Min-max scaling lets outliers dominate at low bit-widths. ``clip_ratios``
    tries each range shrink in (0, 1] and keeps the least-squared-error one
    (per channel when ``dim`` is given); values outside saturate. Include 1.0
    to never do worse than min-max.
    """

    def __init__(
        self, center: bool = False, clip_ratios: Sequence[float] | None = None
    ) -> None:
        super().__init__(center)
        if clip_ratios is not None:
            clip_ratios = tuple(float(r) for r in clip_ratios)
            assert clip_ratios and all(0 < r <= 1 for r in clip_ratios), (
                f"clip_ratios must be non-empty and in (0, 1], got {clip_ratios}"
            )
        self.clip_ratios = clip_ratios

    def get_quant_size(self, b: int) -> int:
        """Equivalent to int_max - int_min + 1, where int_min = -2^{b-1} and
        int_max = 2^{b-1} - 1."""
        return 2**b

    @staticmethod
    def _scale_and_zero_point(
        q_min: Tensor, q_max: Tensor, n_levels: int, ratio: float = 1.0
    ) -> tuple[Tensor, Tensor]:
        span = q_max - q_min
        # Constant nonzero groups need a representable scale as well.
        span = torch.where(span > 0, span, q_max.abs())
        s = (span * ratio / (n_levels - 1)).clamp_(min=torch.finfo(q_min.dtype).tiny)
        # Levels start at zero_point, not int_min, unless the range is symmetric.
        # Q must use the same offset or it disagrees with q.
        zero_point = (q_min * ratio).div_(s).round_()
        return s, zero_point

    def quantize(
        self, p: Tensor, b: int, dim: int | None = None
    ) -> tuple[Tensor, Tensor]:
        assert b != 0, "Please use TernaryUnifQuantizer instead"

        if self.center:
            q, mean = super().remove_mean(p.detach(), dim=dim)
        else:
            q = p.detach().clone()
            mean = torch.zeros(1, dtype=p.dtype, device=p.device)

        if dim is not None:
            q_min = q.min(dim=dim, keepdim=True).values
            q_max = q.max(dim=dim, keepdim=True).values
        else:
            q_min = q.min()
            q_max = q.max()

        n_levels = 2**b
        s, zero_point = self._scale_and_zero_point(q_min, q_max, n_levels)
        if self.clip_ratios:
            best_err = None
            for ratio in self.clip_ratios:
                s_r, zp_r = self._scale_and_zero_point(q_min, q_max, n_levels, ratio)
                q_r = q.div(s_r).round_().clamp_(zp_r, zp_r + n_levels - 1).mul_(s_r)
                err = (q_r - q).square_()
                err = err.sum(dim=dim, keepdim=True) if dim is not None else err.sum()
                if best_err is None:
                    best_err, s, zero_point = err, s_r, zp_r
                else:
                    better = err < best_err
                    best_err = torch.where(better, err, best_err)
                    s = torch.where(better, s_r, s)
                    zero_point = torch.where(better, zp_r, zero_point)

        q.div_(s).round_().clamp_(zero_point, zero_point + n_levels - 1).mul_(s)

        Q = torch.arange(n_levels, dtype=p.dtype, device=p.device)
        if dim is not None:
            Q = Q.unsqueeze(0).add(zero_point).mul(s)  # broadcast requires copy
        else:
            Q = Q.add(zero_point).mul(s)

        # return quantized tensor and set of possible quantization values
        if self.center:
            q += mean
            Q += mean
        return q, Q


class TernaryUnifQuantizer(Quantizer):
    """Uniform quantizer for ternary bit case. Quantization range is [-1, 1]."""

    def get_quant_size(self, b: int) -> int:
        return 3

    def quantize(
        self, p: Tensor, b: int, dim: int | None = None
    ) -> tuple[Tensor, Tensor]:
        assert b == 0, f"Unexpected {b=} for ternary case"

        if self.center:
            q, mean = super().remove_mean(p.detach(), dim=dim)
        else:
            q = p.detach().clone()
            mean = torch.zeros(1, dtype=p.dtype, device=p.device)

        q_max = get_q_max(q, b, dim=dim, scale_method="max")
        q_max.clamp_(min=torch.finfo(q.dtype).tiny)
        s = q_max / 1.5
        q.div_(s).round_().clamp_(min=-1, max=1).mul_(s)

        Q = torch.tensor([-1, 0, 1], dtype=p.dtype, device=p.device)
        if dim is not None:
            Q = Q.unsqueeze(0).mul(s)
        else:
            Q.mul_(s)

        if self.center:
            q += mean
            Q += mean
        return q, Q
