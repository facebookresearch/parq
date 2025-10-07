import torch
from torch import Tensor
from torch.optim import Optimizer

from .quantopt import QuantOptimizer
from .proxmap import ProxMap
from ..quant import Quantizer

class NMSGDOptimizer(QuantOptimizer):
    """From "A Normal Map-Based Proximal Stochastic Gradient Method": https://arxiv.org/pdf/2305.05828v2
    Other parameters:
        norm_gamma: float, default -1.0
            If > 0, then normalize gamma by parameter group dimension.
            This is the same as the "normalized" option in N:M sparsity.
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        quantizer: Quantizer,
        prox_map: ProxMap,
        warmup_steps: int = 0,
        quant_period: int = 10,
        quant_per_channel: bool = False,
        quant_shrink: bool = False,
        anneal_wd_frac: float = 0.0,
        nm_gamma: float = 0.0,
    ) -> None:
        super().__init__(
            base_optimizer=base_optimizer,
            quantizer=quantizer,
            prox_map=prox_map,
            warmup_steps=warmup_steps,
            quant_period=quant_period,
            quant_per_channel=quant_per_channel,
            quant_shrink=quant_shrink,
            anneal_wd_frac=anneal_wd_frac,
        )
        self.nm_gamma = nm_gamma
        for group in self.regularized_param_groups():
            group["gamma"] = self.nm_gamma

    def _get_gamma(self, group):
        return self.nm_gamma

    @torch._disable_dynamo
    def restore_latent_params(self) -> None:
        """Restore latent parameters as optimizer parameters"""
        for group in self.regularized_param_groups():
            for p in group["params"]:
                if p.requires_grad:
                    self.state[p]["corrections"] = self.state[p]["latent"] - p
                    p.copy_(self.state[p]["latent"])

    def _correct_param(self, p: Tensor, group) -> None:
        if "corrections" in self.state[p]:
            stepsize = -1.0 * group["lr"] / self.nm_gamma
            p.add_(self.state[p]["corrections"], alpha=stepsize)
