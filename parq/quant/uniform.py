# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor

from .quantizer import Quantizer


class UnifQuantizer(Quantizer):
    """Uniform quantizer, range determined by multiples of |p|.mean()"""

    def __init__(self, center: bool = False) -> None:
        super().__init__(center)

    def quantize(
        self, p: Tensor, b: int, dim: int | None = None, int_shift: float = 0.5
    ) -> tuple[Tensor, Tensor]:
        """Instantiation of Quantizer.quantize() method.

        int_shift: float value to shift integer range by (default: 0.5). In
            standard uniform quantization, this value is 0.0. For 2 bits, the
            quantized values would be [-2, -1, 0, 1]). We use 0.5 for a
            symmetric range of quantized values: [-1.5, -0.5, 0.5, 1.5].
        """
        assert b >= 1
        if self.center:
            q, mean = super().remove_mean(p.detach(), dim=dim)
        else:
            q = p.detach().clone()
            mean = torch.zeros(1, dtype=p.dtype, device=p.device)

        # set range of quantization: min( b * |q|.mean(), |q|.max())
        q_abs = q.abs()
        if dim is not None:
            q_max = torch.minimum(
                b * q_abs.mean(dim=dim, keepdim=True),  # pyre-ignore[6,9]
                torch.max(q_abs, dim=dim, keepdim=True)[0],  # pyre-ignore[6]
            )
        else:
            q_max = torch.minimum(b * q_abs.mean(), torch.max(q_abs))  # pyre-ignore[6]

        # clamp to quantization range
        q.copy_(torch.minimum(torch.maximum(q, -q_max), q_max))

        # compute scale from [-2^{b-1}+0.5, 2^{b-1}-0.5] to [-q_max, q_max]
        n_levels = 2 ** (b - 1)
        s = q_max / (n_levels - int_shift)

        # scale by 1/s -> shift -0.5 -> round -> shift +0.5 -> scale by s
        # where shift ensures rounding to integers 2^{b-1}, ..., 2^{b-1}-1
        q.div_(s).sub_(int_shift).round_().add_(int_shift).mul_(s)

        # set of all target quantization values
        Q = torch.arange(-n_levels + int_shift, n_levels, device=p.device)
        if dim is not None:
            Q = Q.unsqueeze(0).mul(s)  # broadcasted multiply requires copy
        else:
            Q.mul_(s)

        # return quantized tensor and set of possible quantization values
        if self.center:
            q += mean
            Q += mean
        return q, Q
