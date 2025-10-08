# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch import Tensor

from ..utils import channel_bucketize
from .proxmap import ProxMap


def normalized_sigmoid(x: Tensor, x1: Tensor, x2: Tensor, t: float) -> Tensor:
    """Sigmoid-like function increasing from 0 to 1 over interval [x1, x2).
    'normalized' means value 0 at starting point x1 and 1 at end point x2."""
    eps = 1e-12  # avoid numerical issue when x=x1=x2
    fx = (x - x1) / (x2 - x1 + eps)  # fraction of progress from x1 to x2
    sx = torch.ones_like(fx) / (
        1 + torch.exp(-t * (fx - 0.5))
    )  # scaled and shifted mirror sigmoid
    # s1 = 1/(1+math.exp(0.5*t))          # sx value when x = x1 -> fx = 0
    s2 = 1 / (1 + math.exp(-0.5 * t))  # sx value when x = x2 -> fx = 1
    return (sx - 0.5) / (s2 - 0.5)  # shift and scale to range (0, 1]


class ProxSoftQuant(ProxMap):
    def __init__(
        self,
        anneal_start: int,
        anneal_end: int,
        t_start: float = 1.0,
        t_end: float = 1e3,
        t_power: int = 1,
    ) -> None:
        """Initialize annealing parameters for soft quantization.
        Annealing with sigmoid shape with increasing temperature
            t = t_start + (frac ** t_power) * (t_end - t_start)
        where
            frac = (step_count - anneal_start) / (anneal_end - anneal_start)
        """
        assert anneal_start < anneal_end, "SoftQuant: anneal start before end"
        assert t_start < t_end, "SoftQuant: anneal temperature only increases"
        assert t_power > 0, "SoftQuant: fraction power for temperature anneal"
        self.anneal_start = anneal_start
        self.anneal_end = anneal_end
        self.t_start = t_start
        self.t_end = t_end
        self.t_power = t_power

    @torch.no_grad()
    def apply_(
        self,
        p: Tensor,
        q: Tensor,
        Q: Tensor,
        step_count: int,
        dim: int | None = None,
    ) -> None:
        """Soft quantization map gradually annealing to hard quantization."""

        if step_count < self.anneal_start:
            return
        elif step_count >= self.anneal_end:
            if q is None:
                # hard quantization to the nearest point in Q
                Q_mid = (Q[..., :-1] + Q[..., 1:]) / 2
                if dim is None:
                    q = Q[torch.bucketize(p, Q_mid)]
                else:
                    q = Q.gather(1, channel_bucketize(p, Q_mid))
            p.copy_(q)
            return
        else:
            frac = (step_count - self.anneal_start) / (
                self.anneal_end - self.anneal_start
            )
            temp = self.t_start + (frac**self.t_power) * (self.t_end - self.t_start)
            if dim is None:
                # it is important to clamp idx-1 and then clamping idx itself
                # idx_lower[k] == idx[k] iff p[k] > Q.max() or p[k] <= Q.min()
                idx = torch.bucketize(p, Q)  # locate quant interval
                idx_lower = (idx - 1).clamp_(min=0)  # index of lower bound
                idx_upper = idx.clamp(max=Q.numel() - 1)  # clamp upper index
                q_lower = Q[idx_lower]  # lower boundary of interval
                q_upper = Q[idx_upper]  # upper boundary of interval
                # concise implementation of annealed soft quantization
                p_clamped = torch.clamp(p, min=q_lower, max=q_upper)
            else:
                idx = channel_bucketize(p, Q)
                idx_lower = (idx - 1).clamp_(min=0)
                idx_upper = idx.clamp(max=Q.size(1) - 1)
                q_lower = Q.gather(1, idx_lower)
                q_upper = Q.gather(1, idx_upper)
                p_clamped = torch.minimum(torch.maximum(p, q_lower), q_upper)
            c = (q_upper + q_lower) / 2  # center of interval
            d = (q_upper - q_lower) / 2  # half length of interval
            q = c + d * normalized_sigmoid(p_clamped, q_lower, q_upper, temp)
            # in-place update of model parameters
            p.copy_(q)
