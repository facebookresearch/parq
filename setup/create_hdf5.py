# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Convert ImageNet data directory to HDF5 format for faster data loading."""

import argparse
import h5py
import numpy as np
import os

from typing import List, Tuple
from multiprocessing import cpu_count, Pool
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def get_image_bytes(sample: Tuple[str, int]) -> Tuple[np.ndarray, int]:
    path, target = sample
    with open(path, "rb") as f:
        image_bytes = f.read()
    return image_bytes, target


def read_sample_tups(src_dir: str) -> List[Tuple[np.ndarray, int]]:
    """Read in image bytes and targets from dataset initiated with src_dir."""
    dataset = ImageFolder(src_dir)
    total_count = len(dataset)
    nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
    with Pool(nproc) as p:
        sample_tups = list(
            tqdm(p.imap(get_image_bytes, dataset.samples), total=total_count)
        )
    return sample_tups


def write_hdf5(hdf5_path: str, sample_tups: List[Tuple[np.ndarray, int]]):
    hf = h5py.File(hdf5_path, "w", libver="latest")

    # add all targets to single dataset
    image_bytes, targets = list(zip(*sample_tups))
    hf.create_dataset("targets", data=np.array(targets), chunks=True)

    # serially add image bytes, which are variable length uint8 arrays
    dtype = h5py.special_dtype(vlen=np.dtype("uint8"))
    total_count = len(image_bytes)
    img_dset = hf.create_dataset(
        "image_bytes", (total_count,), dtype=dtype, chunks=True
    )
    for i, ib in tqdm(enumerate(image_bytes), total=total_count):
        img_dset[i] = np.frombuffer(ib, dtype="uint8")

    hf.close()
    print(f"Created {hdf5_path} with {total_count} images")


def main(args):
    # write each data split to its own HDF5 file
    for data_split in ("train", "val"):
        src_dir = os.path.expanduser(os.path.join(args.src_dir, data_split))
        sample_tups = read_sample_tups(src_dir)

        hdf5_path = os.path.join(args.dest_dir, f"{data_split}.hdf5")
        write_hdf5(hdf5_path, sample_tups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=str, help="source data directory")
    parser.add_argument("--dest-dir", type=str, help="where to write HDF5 files")
    args = parser.parse_args()
    main(args)
