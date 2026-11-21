# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from importlib import import_module

from .affine_q2 import AffineQ2Quantizer  # noqa: F401
from .affine_q2_fast import FastAffineQ2Quantizer  # noqa: F401
from .lsbq import LSBQuantizer  # noqa: F401
from .quantizer import Quantizer  # noqa: F401
from .uniform import (  # noqa: F401
    AsymUnifQuantizer,
    MaxUnifQuantizer,
    TernaryUnifQuantizer,
    UnifQuantizer,
)

_TORCHAO_QUANTIZERS = {
    "UnifTorchaoQuantizer": "uniform_torchao",
    "Int4UnifTorchaoQuantizer": "uniform_torchao",
    "StretchedUnifTorchaoQuantizer": "uniform_torchao",
    "MSEUnifTorchaoQuantizer": "mse_torchao",
    "MinMaxStretchedUnifTorchaoQuantizer": "range_torchao",
    "ClippedUnifTorchaoQuantizer": "range_torchao",
}


def __getattr__(name):
    if name not in _TORCHAO_QUANTIZERS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_TORCHAO_QUANTIZERS[name]}", __name__), name)
    globals()[name] = value
    return value
