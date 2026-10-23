# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

import torch

from parq.utils import channel_bucketize


class TestChannelBucketize(unittest.TestCase):
    def test_rows_match_torch_bucketize(self):
        dtypes = [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
        ]
        for dtype, bounds_dtype in [(dtype, dtype) for dtype in dtypes] + [
            (torch.bfloat16, torch.float32)
        ]:
            bounds = torch.tensor([[-2, 0, 0, 3], [-4, -1, 2, 4]], dtype=bounds_dtype)
            if dtype != bounds_dtype:
                # Expose boundary precision lost by a bfloat16 cast.
                bounds.add_(0.001)
            values = torch.tensor(
                [[-5, -2, -1, 0, 1, 3, 5], [-5, -4, -1, 0, 2, 4, 5]], dtype=dtype
            )
            if dtype.is_floating_point:
                values = torch.cat(
                    (
                        values,
                        torch.tensor(
                            [[float("-inf"), float("inf"), float("nan")]] * 2,
                            dtype=dtype,
                        ),
                    ),
                    dim=1,
                )
            for right in [False, True]:
                with self.subTest(dtype=dtype, bounds_dtype=bounds_dtype, right=right):
                    expected = torch.stack(
                        [
                            torch.bucketize(v, b, right=right)
                            for v, b in zip(values, bounds)
                        ]
                    )
                    torch.testing.assert_close(
                        channel_bucketize(values, bounds, right),
                        expected,
                        rtol=0,
                        atol=0,
                    )

    def test_strided_and_empty_inputs(self):
        torch.manual_seed(0)
        for levels in [0, 1, 4, 16, 256]:
            storage = torch.randn(3, 2 * levels).sort(dim=-1).values
            bounds = storage[:, ::2]
            inputs = torch.randn(3, 34)[:, ::2]
            for right in [False, True]:
                expected = torch.stack(
                    [
                        torch.bucketize(v.contiguous(), b.contiguous(), right=right)
                        for v, b in zip(inputs, bounds)
                    ]
                )
                torch.testing.assert_close(
                    channel_bucketize(inputs, bounds, right), expected
                )
        result = channel_bucketize(torch.empty(0, 3), torch.empty(0, 4))
        self.assertEqual(result.shape, (0, 3))
        self.assertEqual(result.dtype, torch.int64)

    def test_no_comparison_mask(self):
        values = torch.randn(3, 17)
        bounds = torch.randn(3, 4).sort(dim=-1).values
        with (
            mock.patch.object(torch.Tensor, "ge", side_effect=AssertionError("mask")),
            mock.patch.object(torch.Tensor, "le", side_effect=AssertionError("mask")),
        ):
            for right in [False, True]:
                self.assertEqual(
                    channel_bucketize(values, bounds, right).shape, (3, 17)
                )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
