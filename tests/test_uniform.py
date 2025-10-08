# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import unittest

import torch
from torch import Tensor

from parq.quant.uniform import (
    AsymUnifQuantizer,
    MaxUnifQuantizer,
    TernaryUnifQuantizer,
    UnifQuantizer,
)


class TestUnifQuantizer(unittest.TestCase):
    def __init__(
        self, methodName, nbit: int = None, dim: int | None = None, sym: bool = True
    ):
        super(TestUnifQuantizer, self).__init__(methodName)
        self.nbit = nbit
        self.dim = dim
        self.sym = sym
        self.model = torch.nn.Sequential(*[torch.nn.Linear(256, 32, bias=False)])

    def get_test_sample(self, p, q) -> tuple[Tensor, Tensor]:
        return p, q if self.dim is None else p[0], q[0]

    @torch.no_grad
    def test_unique_vals(self):
        if self.nbit == 0:
            quantizer = TernaryUnifQuantizer()
        elif self.nbit < 3:
            quantizer = UnifQuantizer()
        else:
            quantizer = MaxUnifQuantizer() if self.sym else AsymUnifQuantizer()

        for p in self.model.parameters():
            torch.nn.init.xavier_uniform_(p)
            n_uniq = quantizer.get_quant_size(self.nbit)
            q, Q = quantizer.quantize(p, self.nbit, dim=self.dim)

            q_test, Q_test = (q, Q) if self.dim is None else (q[0], Q[0])
            cur_uniq = q_test.unique()
            self.assertTrue(cur_uniq.numel() == Q_test.numel() == n_uniq)
            self.assertTrue(cur_uniq.equal(Q_test))


def load_tests(loader, tests, pattern):
    test_cases = unittest.TestSuite()
    for nbit in range(3):
        for dim in (None, -1):
            test_cases.addTest(TestUnifQuantizer("test_unique_vals", nbit, dim))
    for nbit in range(3, 5):
        for dim in (None, -1):
            for sym in (True, False):
                test_cases.addTest(
                    TestUnifQuantizer("test_unique_vals", nbit, dim, sym)
                )
    return test_cases


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    unittest.main()
