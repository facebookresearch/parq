# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
import unittest

from parq.quant.lsbq import LSBQuantizer
from parq.optim import QuantOptimizer, ProxHardQuant


class TestQuantOptimizer(unittest.TestCase):
    def __init__(self, methodName, per_channel=False, block_size=None):
        super(TestQuantOptimizer, self).__init__(methodName)
        self.per_channel = per_channel
        self.block_size = block_size
        self.model = torch.nn.Sequential(*[torch.nn.Linear(128, 64)])
        self.nbit = 2

    def reset_model_and_get_optimizer(self, block_size=None):
        weights = [torch.nn.init.xavier_uniform_(self.model[0].weight)]
        biases = [torch.nn.init.zeros_(self.model[0].bias)]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": weights,
                    "quant_bits": self.nbit,
                    "quant_block_size": block_size,
                },
                {"params": biases, "weight_decay": 0.0},
            ]
        )
        return optimizer

    def test_optimizer_loop(self):
        optimizer = self.reset_model_and_get_optimizer(block_size=self.block_size)
        quantizer = LSBQuantizer()
        prox_map = ProxHardQuant()
        optimizer = QuantOptimizer(
            optimizer, quantizer, prox_map, quant_per_channel=self.per_channel
        )

        N = 5
        dummy_data = torch.randn(N, 128)
        labels = torch.nn.functional.one_hot(torch.arange(0, N), num_classes=64)
        labels = labels.to(dummy_data.dtype)
        loss_fn = torch.nn.MSELoss()
        for i in range(N):
            y_pred = self.model(dummy_data[i])
            loss = loss_fn(y_pred, labels[i])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if self.per_channel and self.block_size is not None:
            q = self.model[0].weight
            q = q.view(-1, q.size(-1) // self.block_size, self.block_size)
            uniq = q[0, 0].unique()
            self.assertFalse(uniq.equal(q[0, 1].unique()))
            self.assertTrue(uniq.numel() == 2**self.nbit)


def load_tests(loader, tests, pattern):
    test_cases = unittest.TestSuite()
    for per_channel, block_size in zip(
        (False, True, True, True, True), (None, None, 8, 16, 64)
    ):
        test_cases.addTest(
            TestQuantOptimizer("test_optimizer_loop", per_channel, block_size)
        )
    return test_cases


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    unittest.main()
