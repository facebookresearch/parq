# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license in the repository root.

import unittest

import torch

from parq.optim import ProxHardQuant, ProxPARQ, QuantOptimizer
from parq.quant.uniform import AsymUnifQuantizer


class TestAsymmetricGridRegression(unittest.TestCase):
    def test_shifted_grid_contains_quantized_weights(self):
        weights = torch.linspace(0.0, 1.0, 12)
        q, targets = AsymUnifQuantizer().quantize(weights, 2)
        expected = torch.tensor([0.0, 1 / 3, 2 / 3, 1.0])
        torch.testing.assert_close(q.unique(), expected)
        torch.testing.assert_close(targets, expected)

    def test_cached_targets_do_not_collapse_at_zero_lr(self):
        for per_channel in (False, True):
            for prox in (ProxHardQuant(), ProxPARQ(0, 2)):
                with self.subTest(per_channel=per_channel, prox=type(prox).__name__):
                    linear = torch.nn.Linear(12, 1, bias=False)
                    with torch.no_grad():
                        linear.weight.copy_(torch.linspace(0.0, 1.0, 12))
                    optimizer = QuantOptimizer(
                        torch.optim.SGD(
                            [{"params": [linear.weight], "quant_bits": 2}], lr=0.0
                        ),
                        AsymUnifQuantizer(),
                        prox,
                        quant_per_channel=per_channel,
                    )
                    for step in range(12):
                        optimizer.zero_grad()
                        linear(torch.zeros(1, 12)).sum().backward()
                        optimizer.step()
                        self.assertEqual(linear.weight.min().item(), 0.0)
                        self.assertEqual(linear.weight.max().item(), 1.0)
                        if step >= 2 or isinstance(prox, ProxHardQuant):
                            torch.testing.assert_close(
                                linear.weight.unique(),
                                torch.tensor([0.0, 1 / 3, 2 / 3, 1.0]),
                            )

    def test_membership_across_seeds_centering_and_channels(self):
        for seed in range(200):
            weights = torch.randn(8, 64, generator=torch.Generator().manual_seed(seed))
            for bits in (2, 3, 4):
                for center in (False, True):
                    for dim in (None, -1):
                        with self.subTest(seed=seed, bits=bits, center=center, dim=dim):
                            q, targets = AsymUnifQuantizer(center=center).quantize(
                                weights, bits, dim=dim
                            )
                            grid = targets if dim is None else targets.unsqueeze(1)
                            self.assertTrue((q.unsqueeze(-1) == grid).any(-1).all())
