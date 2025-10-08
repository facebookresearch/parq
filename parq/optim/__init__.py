# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial

from torch.optim import Optimizer

from ..quant import Quantizer
from .binarelax import ProxBinaryRelax  # noqa: F401
from .nm_sgd import NMSGDOptimizer  # noqa: F401
from .parq import ProxPARQ  # noqa: F401
from .proxmap import ProxHardQuant, ProxMap  # noqa: F401
from .quantopt import QuantOptimizer  # noqa: F401
from .softquant import ProxSoftQuant  # noqa: F401


def build_quant_optimizer(
    base_optimizer: Optimizer,
    quantizer: Quantizer,
    prox_map: ProxMap,
    warmup_steps: int = 0,
    quant_period: int = 10,
    quant_per_channel: bool = False,
    quant_shrink: bool = False,
    anneal_wd_frac: float = 0.0,
    nm_gamma: float = 0.0,
) -> QuantOptimizer:
    if nm_gamma > 0:
        prune_opt_cls = partial(NMSGDOptimizer, nm_gamma=nm_gamma)
    else:
        prune_opt_cls = QuantOptimizer

    return prune_opt_cls(
        base_optimizer=base_optimizer,
        quantizer=quantizer,
        prox_map=prox_map,
        warmup_steps=warmup_steps,
        quant_period=quant_period,
        quant_per_channel=quant_per_channel,
        quant_shrink=quant_shrink,
        anneal_wd_frac=anneal_wd_frac,
    )
