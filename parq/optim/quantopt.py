# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from ..quant import Quantizer
from ..utils import HAS_DTENSOR, is_dtensor
from .proxmap import ProxMap

if HAS_DTENSOR:
    from torch.distributed.tensor import empty as dtensor_empty
    from torch.distributed.tensor.experimental import local_map


class QuantOptimizer(Optimizer):
    """QuantOptimizer assembles functionalities of the following objects:
    a base optimizer (e.g., SGD or AdamW)
        - update the latent variables for QAT
    a quantizer (e.g., UnifQuantizer, LSBQuantizer, LearnedScale)
        - update target quantization values for model parameters
    a proximal mapping (e.g, HardQuant/STE, PARQ, BinaryRelax)
        - update model parameters based on the above two updates
    Other parameters:
        - warmup_steps: int >= 0
        - quant_period: int > 0
        - quant_per_channel: True or False
        - quant_shrink: True or False
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
    ) -> None:
        if not 0 <= anneal_wd_frac <= 1:
            raise ValueError(f"Invalid {anneal_wd_frac=} outside range [0.0, 1.0]")

        # need to reconstruct these objects if loading checkpoint
        self.base_optimizer = base_optimizer
        self.quantizer = quantizer
        self.prox_map = prox_map

        # need to store these attributes in state_dict for checkpoint
        self.warmup_steps = warmup_steps
        self.quant_period = quant_period
        self.quant_per_channel = quant_per_channel
        self.quant_shrink = quant_shrink
        self.anneal_wd_frac = anneal_wd_frac

        # Initialize "cumu_lr" and latent params in optimizer states
        for group in self.regularized_param_groups():
            group["cumu_lr"] = 0.0
            if self.anneal_wd_frac > 0:
                group["initial_wd"] = group["weight_decay"]
        # NOTE: Filling state dict here cause Adam(W) error, which assumes
        # empty state[p] at first step() where optimizer states are initialized

    def __getattribute__(self, name: str):
        try:
            attr = super(Optimizer, self).__getattribute__(name)
        except AttributeError:
            attr = self.base_optimizer.__getattribute__(name)
        return attr

    def __repr__(self) -> str:
        base_optimizer = "\n    ".join(self.base_optimizer.__repr__().split("\n"))
        quantizer = self.quantizer.__class__.__name__
        prox_map = self.prox_map.__class__.__name__
        extra_repr = "\n  ".join(("(", base_optimizer, f"{quantizer=}", f"{prox_map=}"))
        return f"{self.__class__.__name__} {extra_repr}\n)"

    @property
    def state(self) -> defaultdict[Tensor, Any]:  # pyre-ignore[3]
        return self._state if hasattr(self, "_state") else self.base_optimizer.state

    @staticmethod
    def quantize_(
        p: Tensor,
        quants: Tensor,
        quantizer: Quantizer,
        b: int,
        dim: int | None = None,
    ) -> Tensor | None:
        """Optionally update the quantization targets `quants` in place.
        Return the quantized `p` as a by-product if `quant_update=True`.
        """
        q, Q = quantizer.quantize(p, b, dim=dim)  # pyre-ignore[28]
        quants.copy_(Q)
        return q

    @property
    def num_steps(self) -> int:
        for group in self.regularized_param_groups():
            group.setdefault("num_steps", 0)
            return group["num_steps"]

    @num_steps.setter
    def num_steps(self, value: int) -> None:
        for group in self.regularized_param_groups():
            group["num_steps"] = value
            return

    def regularized_param_groups(self):  # pyre-ignore[3]
        """Yield parameter groups that need to be quantized."""
        for group in self.param_groups:
            if group.get("quant_bits", 16) < 16:
                yield group

    @torch._disable_dynamo
    def state_dict(self) -> dict[str, Any]:
        state_dict = self.base_optimizer.state_dict()
        if self.quant_period == 1:
            # Targets are recomputed next update; keep the live cache out of
            # checkpoints while retaining latents and base optimizer state.
            state_dict["state"] = {
                key: (
                    {name: value for name, value in state.items() if name != "quants"}
                    if isinstance(state, dict)
                    else state
                )
                for key, state in state_dict["state"].items()
            }
        return state_dict

    @torch._disable_dynamo
    def load_state_dict(
        self, state_dict: dict[str, Any], start_step: int | None = None
    ) -> None:
        saved_quants: dict[Tensor, Tensor] = {}

        def preserve_quants(optimizer: Optimizer, checkpoint: dict[str, Any]) -> None:
            # Capture after user pre-hooks remap IDs, but before the base optimizer
            # casts floating state to the parameter dtype and rounds FP32 caches.
            saved_ids = [
                key for group in checkpoint["param_groups"] for key in group["params"]
            ]
            parameters = [
                p for group in optimizer.param_groups for p in group["params"]
            ]
            for key, p in zip(saved_ids, parameters):
                state = checkpoint["state"].get(key)
                if isinstance(state, dict) and "quants" in state:
                    saved_quants[p] = state["quants"]

        def restore_quants(optimizer: Optimizer) -> None:
            # User post-hooks must see exact targets and retain their state edits.
            for p, quants in saved_quants.items():
                optimizer.state[p]["quants"] = quants.to(device=p.device, copy=True)

        with (
            self.base_optimizer.register_load_state_dict_pre_hook(preserve_quants),
            self.base_optimizer.register_load_state_dict_post_hook(
                restore_quants, prepend=True
            ),
        ):
            self.base_optimizer.load_state_dict(state_dict)

        # resume from check points usually not corresponds to saved num_steps
        # so allow explicit start_step computed from epochs * steps_per_epoc
        if start_step is not None:
            self.num_steps = start_step

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if self.num_steps < self.warmup_steps:
            # warmup stage: running the base optimizer only
            loss = self.base_optimizer.step(closure=closure)  # pyre-ignore[6]
            self.num_steps += 1
            return loss

        if self.num_steps == self.warmup_steps:
            # first step of qat, save latent params, instead of restore
            self.save_latent_params()
        else:
            # qat: restore latent params for update by the base optimizer
            self.restore_latent_params()

        # call base optimizer step() method to update latent parameters
        loss = self.base_optimizer.step(closure=closure)  # pyre-ignore[6]

        if hasattr(self, "_state"):
            assert self.warmup_steps == 0
            # restore the temporary state to the base optimizer's state
            for p in self._state.keys():
                self.base_optimizer.state[p]["latent"] = self._state[p]["latent"]
            del self._state

        # check if it is time to update set of quantization values Q
        if (self.num_steps - self.warmup_steps) % self.quant_period == 0:
            quant_update = True
        else:
            quant_update = False

        for group in self.regularized_param_groups():
            quantizer = self.quantizer

            # AProx in practice: ensure shrinkage coefficient >= 1
            group["cumu_lr"] += group["lr"]
            gamma = self._get_gamma(group)
            b = group["quant_bits"]
            block_size = group.get("quant_block_size")
            inv_slope = 0.0

            for p in group["params"]:
                if not p.requires_grad:
                    continue

                if is_dtensor(p) and any(
                    placement.is_partial() for placement in p.placements
                ):
                    raise ValueError(
                        "Quantization requires reduced parameters; Partial placements are unsupported"
                    )
                self._correct_param(p, group)

                state = self.state[p]
                # save latent parameters, need detach()? or copy p)
                state["latent"].copy_(p)

                # Complete per-row/input-column groups are independent across
                # shards. Mutate the original local storage directly, following
                # PAT's rule that in-place prox updates must reach their owner.
                local_dtensor = None
                local_parameter = None
                if is_dtensor(p) and self.quant_per_channel and p.dim() == 2:
                    if p.device_mesh.get_coordinate() is None:
                        continue
                    # Column shards are safe for complete blocks; the local modulo
                    # assertion below validates their alignment.
                    if block_size is None and any(
                        placement.is_shard() and placement.dim == 1
                        for placement in p.placements
                    ):
                        raise ValueError(
                            "A per-row quantization group crosses an input-column shard"
                        )
                    local_dtensor = p
                    local_parameter = p.to_local()
                    p = local_parameter.contiguous()

                # in-place scaling of parameters by 1/gamma if specified
                if self.quant_shrink:
                    p.div_(gamma)

                # reshape p according to block size if specified
                if block_size is not None:
                    assert p.size(-1) % block_size == 0, (
                        f"{p.size(-1)=} is not divisible by {block_size=}"
                    )
                    assert p.dim() <= 2, f"Invalid {p.dim()=} for {block_size=}"
                    if p.dim() == 1:
                        p = p.unsqueeze(0)

                    # row-major ordering ensures this is correct
                    p = p.view(-1, block_size)

                # quantization by channel or by layer
                # update quantization targets periodically
                per_channel = self.quant_per_channel and p.dim() > 1
                quant_levels = quantizer.get_quant_size(b)
                quant_size = (
                    (p.size(0), quant_levels) if per_channel else (quant_levels,)
                )
                quant_owner = local_dtensor if local_dtensor is not None else p
                if local_dtensor is not None:
                    # Keep the original row/column-group axes in distributed
                    # state. Flattening global uneven row shards would assign
                    # different numbers of groups than the owning parameters.
                    quant_size = (
                        (
                            local_dtensor.size(0),
                            local_dtensor.size(1) // block_size,
                            quant_levels,
                        )
                        if block_size is not None
                        else (local_dtensor.size(0), quant_levels)
                    )
                quants = state.get("quants")
                rebuild_targets = (
                    quants is None
                    or tuple(quants.shape) != quant_size
                    or quants.device != p.device
                    or quants.dtype != torch.get_default_dtype()
                    or is_dtensor(quants) != is_dtensor(quant_owner)
                    or (
                        is_dtensor(quant_owner)
                        and (
                            quants.placements != quant_owner.placements
                            or quants.device_mesh != quant_owner.device_mesh
                        )
                    )
                )
                parameter_quant_update = quant_update or rebuild_targets
                if rebuild_targets:
                    # Allocate only owned state; there is no data to scatter.
                    # The next quantize_ call overwrites every local target.
                    state["quants"] = (
                        dtensor_empty(
                            quant_size,
                            device_mesh=quant_owner.device_mesh,
                            placements=quant_owner.placements,
                        )
                        if is_dtensor(quant_owner)
                        else torch.empty(quant_size, device=p.device)
                    )

                quants = state["quants"]
                if local_dtensor is not None:
                    quants = quants.to_local().reshape(-1, quants.size(-1))

                dim = -1 if per_channel else None
                if per_channel and p.dim() > 2:
                    p = p.flatten(start_dim=1)

                q = None
                if parameter_quant_update and p.numel() == 0:
                    # Empty owning shards still carry canonical distributed
                    # state, but affine primitives cannot infer their qparams.
                    q = p
                elif parameter_quant_update:
                    qfunc = partial(self.quantize_, quantizer=quantizer, b=b, dim=dim)
                    if is_dtensor(p):
                        qfunc = local_map(
                            qfunc,
                            out_placements=[*p.placements],
                            in_placements=(p.placements, quants.placements),
                            redistribute_inputs=False,
                        )
                    q = qfunc(p, quants)

                # apply (step-dependent) proximal mapping in place
                pfunc = partial(
                    self.prox_map.apply_, step_count=self.num_steps, dim=dim
                )
                if is_dtensor(p):
                    pfunc = local_map(
                        pfunc,
                        out_placements=None,
                        in_placements=(
                            p.placements,
                            None if q is None else q.placements,
                            quants.placements,
                        ),
                        redistribute_inputs=False,
                    )
                inv_slope = pfunc(p, q, quants)
                if local_parameter is not None and not local_parameter.is_contiguous():
                    # Column-shard views may have gaps between rows. Group on
                    # contiguous workspace, then copy the projection to its owner.
                    local_parameter.copy_(p.reshape_as(local_parameter))

            # quantized parameters share the same PARQ inverse slope
            if inv_slope:
                if self.anneal_wd_frac > 0:
                    group["weight_decay"] = (
                        inv_slope * self.anneal_wd_frac * group["initial_wd"]
                        + (1 - self.anneal_wd_frac) * group["initial_wd"]
                    )
                group["inv_slope"] = inv_slope  # save for tensorboard

        self.num_steps += 1
        return loss

    @torch._disable_dynamo
    def restore_latent_params(self) -> None:
        """Restore latent parameters as optimizer parameters"""
        for group in self.regularized_param_groups():
            for p in group["params"]:
                if p.requires_grad:
                    p.copy_(self.state[p]["latent"])

    @torch._disable_dynamo
    def save_latent_params(self) -> None:
        """Save updated latent parameters before applying prox-map"""
        if self.warmup_steps == 0:
            assert len(self.state) == 0, "Expected empty state at first step()"
            # Maintain the invariant that `len(self.state) == 0` before first
            # self.base_optimizer.step() call by using a temporary state buffer
            self._state = defaultdict(dict)

        for group in self.regularized_param_groups():
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    if "latent" not in state:
                        state["latent"] = p.detach().clone()
                    else:
                        state["latent"].copy_(p)

    def _get_gamma(self, group):
        return max(1.0, group["cumu_lr"])

    def _correct_param(self, p: Tensor, group) -> None:
        pass
