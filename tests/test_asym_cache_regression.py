# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license in the repository root.

import copy
import importlib.util
import unittest

import torch

from parq.optim import ProxHardQuant, ProxPARQ, QuantOptimizer
from parq.quant import AffineQ2Quantizer, AsymUnifQuantizer, FastAffineQ2Quantizer
from parq.quant.uniform import UnifQuantizer


class TestAsymmetricTargets(unittest.TestCase):
    def test_zero_lr_cached_targets_retain_all_four_levels(self):
        original = torch.linspace(0.0, 1.0, 12)
        for prox in (ProxHardQuant(), ProxPARQ(0, 2)):
            with self.subTest(prox=type(prox).__name__):
                weight = torch.nn.Parameter(original.clone())
                optimizer = QuantOptimizer(
                    torch.optim.SGD([{"params": [weight], "quant_bits": 2}], lr=0.0),
                    AsymUnifQuantizer(),
                    prox,
                )
                expected = torch.tensor([0.0, 1 / 3, 2 / 3, 1.0])
                for step in range(12):
                    weight.grad = torch.zeros_like(weight)
                    optimizer.step()
                    self.assertEqual(weight.min().item(), 0.0)
                    self.assertEqual(weight.max().item(), 1.0)
                    if step >= 2 or isinstance(prox, ProxHardQuant):
                        torch.testing.assert_close(weight.unique(), expected)

    def test_target_membership_across_seeds_and_centering(self):
        for seed in range(200):
            weights = torch.randn(8, 64, generator=torch.Generator().manual_seed(seed))
            for center in (False, True):
                for dim in (None, -1):
                    with self.subTest(seed=seed, center=center, dim=dim):
                        q, targets = AsymUnifQuantizer(center=center).quantize(
                            weights, 2, dim=dim
                        )
                        grid = targets if dim is None else targets.unsqueeze(1)
                        self.assertTrue((q.unsqueeze(-1) == grid).any(-1).all())

    def test_constant_groups_are_finite(self):
        for value in (0.0, -7.0, 7.0):
            for center in (False, True):
                for dim in (None, -1):
                    weights = torch.full((2, 32), value)
                    q, targets = AsymUnifQuantizer(center=center).quantize(
                        weights, 2, dim=dim
                    )
                    self.assertTrue(torch.isfinite(q).all())
                    self.assertTrue(torch.isfinite(targets).all())
                    torch.testing.assert_close(q, weights)


class TestAdditionalQuantizers(unittest.TestCase):
    def test_affine_quantizers_cache_the_returned_targets(self):
        weights = torch.randn(4, 64, generator=torch.Generator().manual_seed(7))
        quantizers = [
            AffineQ2Quantizer(calibration=kind) for kind in ("mlx", "minmax", "mse")
        ]
        quantizers.append(FastAffineQ2Quantizer())
        for quantizer in quantizers:
            with self.subTest(quantizer=type(quantizer).__name__):
                q, targets = quantizer.quantize(weights, 2, dim=-1)
                self.assertTrue((q.unsqueeze(-1) == targets.unsqueeze(1)).any(-1).all())
                self.assertTrue((targets.diff(dim=-1) >= 0).all())
                parameter = torch.nn.Parameter(weights.clone())
                optimizer = QuantOptimizer(
                    torch.optim.SGD([{"params": [parameter], "quant_bits": 2}], lr=0.0),
                    quantizer,
                    ProxHardQuant(),
                    quant_per_channel=True,
                )
                for _ in range(3):
                    parameter.grad = torch.zeros_like(parameter)
                    optimizer.step()
                    torch.testing.assert_close(parameter, q)

    def test_per_group_quantizer_selection(self):
        # Avoid midpoint ties, where either adjacent symmetric level is valid.
        original = torch.linspace(0.07, 1.07, 12)
        parameters = [torch.nn.Parameter(original.clone()) for _ in range(2)]
        base, override = UnifQuantizer(), AsymUnifQuantizer()
        optimizer = QuantOptimizer(
            torch.optim.SGD(
                [{"params": [p], "quant_bits": 2} for p in parameters], lr=0.0
            ),
            base,
            ProxHardQuant(),
            group_quantizer_map={1: override},
        )
        for _ in range(3):
            for p in parameters:
                p.grad = torch.zeros_like(p)
            optimizer.step()
            for p, quantizer in zip(parameters, (base, override)):
                torch.testing.assert_close(p, quantizer.quantize(original, 2)[0])

    def test_legacy_checkpoint_preserves_qat_clock_and_input(self):
        parameter = torch.nn.Parameter(torch.linspace(0.0, 1.0, 12))
        optimizer = QuantOptimizer(
            torch.optim.SGD([{"params": [parameter], "quant_bits": 2}], lr=0.0),
            AsymUnifQuantizer(),
            ProxHardQuant(),
        )
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        checkpoint = copy.deepcopy(optimizer.state_dict())
        for group in checkpoint["param_groups"]:
            group.pop("num_steps", None)
        checkpoint["qat_state"] = {"num_steps": 7}
        optimizer.load_state_dict(checkpoint)
        self.assertEqual(optimizer.num_steps, 7)
        self.assertEqual(checkpoint["qat_state"]["num_steps"], 7)
        optimizer.load_state_dict(checkpoint, start_step=9)
        self.assertEqual(optimizer.num_steps, 9)

    @unittest.skipUnless(
        importlib.util.find_spec("torchao"), "optional torchao dependency"
    )
    def test_torchao_variants_return_consistent_targets(self):
        from parq.quant import (
            ClippedUnifTorchaoQuantizer,
            MinMaxStretchedUnifTorchaoQuantizer,
            MSEUnifTorchaoQuantizer,
        )

        weights = torch.randn(4, 64, generator=torch.Generator().manual_seed(13))
        quantizers = [
            MSEUnifTorchaoQuantizer(),
            ClippedUnifTorchaoQuantizer(),
            MinMaxStretchedUnifTorchaoQuantizer(b=2),
        ]
        for quantizer in quantizers:
            q, targets = quantizer.quantize(weights, 2, dim=-1)
            self.assertTrue(torch.isfinite(q).all())
            self.assertTrue(torch.isfinite(targets).all())
            self.assertTrue((q.unsqueeze(-1) == targets.unsqueeze(1)).any(-1).all())
