# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import h5py
import io

from typing import Any, Callable, Optional, Tuple
from PIL import Image
from torchvision.datasets import VisionDataset


class H5VisionDataset(VisionDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        super().__init__(root, transform)
        self.root = root
        self.transform = transform

        self.image_bytes = None
        self.targets = None

        with h5py.File(self.root, "r") as hf:
            self.n_samples = len(hf["targets"])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.image_bytes is None:
            hf = h5py.File(self.root, "r")
            self.image_bytes = hf["image_bytes"]
            self.targets = hf["targets"]

        sample = Image.open(io.BytesIO(self.image_bytes[index])).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)

        target = self.targets[index]
        return sample, target

    def __len__(self):
        return self.n_samples
