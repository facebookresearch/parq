# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
import unittest

from torch import Tensor

from parq.quant.lsbq import LSBQuantizer


class TestLSBQuantizer(unittest.TestCase):
    def __init__(
        self,
        methodName,
        nbit: int = None,
        dim: int | None = None,
        optimal: bool = False,
    ):
        super(TestLSBQuantizer, self).__init__(methodName)
        self.nbit = nbit
        self.dim = dim
        self.optimal = optimal
        self.model = torch.nn.Sequential(*[torch.nn.Linear(256, 32, bias=False)])

    def get_test_sample(self, p, q) -> tuple[Tensor, Tensor]:
        return p, q if self.dim is None else p[0], q[0]

    @torch.no_grad
    def test_unique_vals(self):
        quantizer = LSBQuantizer(optimal=self.optimal)
        for p in self.model.parameters():
            torch.nn.init.xavier_uniform_(p)
            n_uniq = quantizer.get_quant_size(self.nbit)
            q, Q = quantizer.quantize(p, self.nbit, dim=self.dim)

            q_test, Q_test = (q, Q) if self.dim is None else (q[0], Q[0])
            if not self.optimal or self.nbit > 0:  # handle minor discrepancy
                q_test, Q_test = q_test.round(decimals=6), Q_test.round(decimals=6)
            cur_uniq = q_test.unique()
            self.assertTrue(cur_uniq.numel() == Q_test.numel() == n_uniq)
            self.assertTrue(cur_uniq.equal(Q_test))


def load_tests(loader, tests, pattern):
    test_cases = unittest.TestSuite()
    for nbit in range(5):
        for dim in (None, -1):
            if nbit < 3:
                for optimal in (True, False):
                    test_cases.addTest(
                        TestLSBQuantizer("test_unique_vals", nbit, dim, optimal)
                    )
            else:
                test_cases.addTest(TestLSBQuantizer("test_unique_vals", nbit, dim))
    return test_cases


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    unittest.main()
