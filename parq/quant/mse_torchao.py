# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE.torchao file in this directory.

from collections.abc import Sequence

import torch
from torchao.quantization.granularity import PerGroup, PerTensor
from torchao.quantization.observer import AffineQuantizedMSEObserver
from torchao.quantization.quant_primitives import (
    ZeroPointDomain,
    _get_reduction_params,
)
from torchao.quantization.utils import get_block_size

from .uniform_torchao import UnifTorchaoQuantizer


class _ClippingMSEObserver(AffineQuantizedMSEObserver):
    """Use a small explicit search with the native observer's affine MSE loss."""

    def line_search(self, input):
        block_size = get_block_size(input.shape, self.granularity)
        shape, dims = _get_reduction_params(block_size, input.shape)
        grouped = input.detach().reshape(shape)
        minimum = grouped.amin(dim=dims)
        maximum = grouped.amax(dim=dims)
        best_min, best_max = minimum, maximum
        best_loss = self.loss_fn(input, minimum, maximum)
        for ratio in self.clip_ratios:
            if ratio == 1.0:
                continue
            new_min, new_max = minimum * ratio, maximum * ratio
            loss = self.loss_fn(input, new_min, new_max)
            better = loss < best_loss
            best_min = torch.where(better, new_min, best_min)
            best_max = torch.where(better, new_max, best_max)
            best_loss = torch.minimum(loss, best_loss)
        return best_min, best_max


class MSEUnifTorchaoQuantizer(UnifTorchaoQuantizer):
    """Uniform quantizer that uses torchao's MSE observer to select clipping ranges"""

    def __init__(
        self,
        clip_ratios: Sequence[float] = (0.8, 0.9, 1.0),
        compile_search: bool = False,
        **kwargs,
    ):
        ratios = tuple(float(r) for r in clip_ratios)
        if not ratios or 1.0 not in ratios or any(not 0 < r <= 1 for r in ratios):
            raise ValueError("clip_ratios must be in (0, 1] and include 1.0")
        if (
            not kwargs.get("preserve_zero", True)
            or kwargs.get("zero_point_domain", ZeroPointDomain.INT)
            != ZeroPointDomain.INT
        ):
            raise ValueError(
                "MSE search requires preserve_zero=True and INT zero points"
            )
        super().__init__(**kwargs)
        self.clip_ratios = tuple(dict.fromkeys(ratios))
        self.compile_search = compile_search
        self._mse_observers = {}
        self._choose_qparams = self._choose_mse_qparams
        self._compiled_inputs = set()
        self._compiled_quantize = (
            torch.compile(super().quantize, fullgraph=True) if compile_search else None
        )

    def quantize(self, p, b, dim=None):
        if self._compiled_quantize is None:
            return super().quantize(p, b, dim)
        key = (p.dtype, b, dim, p.size(-1), p.size(0) == 1)
        if key not in self._compiled_inputs:
            # Initialize the native qparams and observer cache outside tracing.
            super().quantize(p, b, dim)
            self._compiled_inputs.add(key)
        return self._compiled_quantize(p, b, dim)

    def _choose_mse_qparams(
        self, input, mapping_type, block_size, target_dtype, **kwargs
    ):
        per_tensor = tuple(block_size) == tuple(input.shape)
        key = (
            per_tensor,
            block_size[-1],
            input.dtype,
            kwargs["quant_min"],
            kwargs["quant_max"],
        )
        if key not in self._mse_observers:
            granularity = PerTensor() if per_tensor else PerGroup(block_size[-1])
            observer = _ClippingMSEObserver(
                mapping_type, target_dtype, granularity, **kwargs
            )
            observer.clip_ratios = self.clip_ratios
            self._mse_observers[key] = observer, observer.line_search
        observer, search = self._mse_observers[key]
        observer.min_val, observer.max_val = search(input.detach())
        return observer.calculate_qparams()
