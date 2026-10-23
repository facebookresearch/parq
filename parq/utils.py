# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor

try:
    from torch.distributed.tensor import DTensor

    HAS_DTENSOR = True
except ImportError:
    HAS_DTENSOR = False


def is_dtensor(x):
    return HAS_DTENSOR and isinstance(x, DTensor)


def channel_bucketize(input: Tensor, boundaries: Tensor, right: bool = False) -> Tensor:
    """Bucketize each input row against its sorted boundary row.

    This matches torch.bucketize without constructing an input-by-boundary
    comparison mask. Keep boundary precision when input and boundaries differ.
    """
    return torch.searchsorted(boundaries.contiguous(), input.contiguous(), right=right)
