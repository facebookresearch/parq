# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE.torchao file in this directory.

import math
from typing import Optional

import torch
from torch import Tensor
from torchao.quantization.quant_primitives import _get_reduction_params

from .uniform_torchao import StretchedUnifTorchaoQuantizer, UnifTorchaoQuantizer


def _choose_stretched_minmax(
    input_float,
    mapping_type,
    block_size,
    target_dtype,
    quant_min=None,
    quant_max=None,
    eps=None,
):
    shape, dims = _get_reduction_params(block_size, input_float.size())
    eps = torch.finfo(input_float.dtype).eps if eps is None else eps
    bound = input_float.view(shape).abs().amax(dim=dims, keepdim=True).clamp_min(eps)
    scale = bound / quant_max
    return scale, torch.full_like(scale, -0.5)


class MinMaxStretchedUnifTorchaoQuantizer(StretchedUnifTorchaoQuantizer):
    """Stretched uniform quantizer that uses each group's maximum absolute weight"""

    def __init__(self, b: int, **kwargs) -> None:
        super().__init__(b=b, **kwargs)
        self._choose_qparams = _choose_stretched_minmax


class ClippedUnifTorchaoQuantizer(UnifTorchaoQuantizer):
    """Uniform quantizer that clips weights using a multiple of their mean magnitude"""

    def __init__(self, clip_factor: float = 3.0, **kwargs) -> None:
        if not math.isfinite(clip_factor) or clip_factor <= 0:
            raise ValueError("clip_factor must be finite and positive")
        super().__init__(**kwargs)
        self.clip_factor = clip_factor

    def quantize(
        self, p: Tensor, b: int, dim: Optional[int] = None
    ) -> tuple[Tensor, Tensor]:
        dims = (-1,) if dim is not None else tuple(range(p.ndim))
        magnitude = p.abs()
        bound = torch.minimum(
            self.clip_factor * magnitude.mean(dim=dims, keepdim=True),
            magnitude.amax(dim=dims, keepdim=True),
        ).clamp_min(
            self.eps if self.eps is not None else torch.finfo(p.dtype).smallest_normal
        )
        return super().quantize(p.clamp(min=-bound, max=bound), b, dim=dim)
