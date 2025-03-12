#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=create_hdf5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00

# NOTE: HDF5 data is most helpful on AWS clusters. Do not use on H2/CoreWeave.

# After running this script, each training job should use rsync to move the HDF5
# files from DEST_DIR to its own /scratch disk to maximize training speed. (Many
# jobs reading the same file(s) creates a bottleneck.)

SRC_DIR=/datasets01/imagenet_full_size/061417
DEST_DIR=/fsx-physics-of-language-models/${USER}/datasets/ILSVRC
set -e
if [ ! -d $SRC_DIR ] || [ ! -d $DEST_DIR ]; then
    echo "At least one of ${SRC_DIR} or ${DEST_DIR} does not exist"
    exit 1
fi
set +e

start_time=$(date +%s)
python $HOME/qpat/setup/create_hdf5.py --src-dir $SRC_DIR --dest-dir $DEST_DIR
end_time=$(date +%s)

elapsed_time=$((end_time - start_time))
echo "$elapsed_time seconds elapsed"
