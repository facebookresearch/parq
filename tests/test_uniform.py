# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import unittest

from torch import Tensor

from parq.quant.uniform import MaxUnifQuantizer, UnifQuantizer, TernaryUnifQuantizer


class TestUnifQuantizer(unittest.TestCase):
    def __init__(self, methodName, nbit: int = None, dim: int | None = None):
        super(TestUnifQuantizer, self).__init__(methodName)
        self.nbit = nbit
        self.dim = dim
        self.model = torch.nn.Sequential(*[torch.nn.Linear(512, 64, bias=False)])

    def get_test_sample(self, p, q) -> tuple[Tensor, Tensor]:
        return p, q if self.dim is None else p[0], q[0]

    @torch.no_grad
    def test_unique_vals(self):
        if self.nbit == 0:
            quantizer = TernaryUnifQuantizer()
        elif self.nbit < 3:
            quantizer = UnifQuantizer()
        else:
            quantizer = MaxUnifQuantizer()

        for p in self.model.parameters():
            torch.nn.init.xavier_uniform_(p)
            min_val = p.min() if self.dim is None else torch.min(p, dim=self.dim)
            p[p == min_val].copy_(0)
            if self.nbit == 0:
                n_uniq = 3
            elif self.nbit < 3:
                n_uniq = 2**self.nbit
            else:
                n_uniq = 2**self.nbit - 1

            q, Q = quantizer.quantize(p, self.nbit, dim=self.dim)

            q_test, Q_test = (q, Q) if self.dim is None else (q[0], Q[0])
            cur_uniq = q_test.unique()
            self.assertTrue(cur_uniq.numel() == Q_test.numel() == n_uniq)
            self.assertTrue(cur_uniq.equal(Q_test))


def load_tests(loader, tests, pattern):
    test_cases = unittest.TestSuite()
    for nbit in range(5):
        for dim in (None, -1):
            test_cases.addTest(TestUnifQuantizer("test_unique_vals", nbit, dim))
    return test_cases


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
