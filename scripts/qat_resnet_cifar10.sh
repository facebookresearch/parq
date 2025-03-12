#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=qat_resnet20_cifar10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --output /checkpoint/%u/qat_resnet20_cifar10_%j/train.out
#SBATCH --error /checkpoint/%u/qat_resnet20_cifar10_%j/train.err
#SBATCH --time=0-2:00:00

SAVE_DIR=/checkpoint/$USER/qat_resnet20_cifar10_${SLURM_JOB_ID}
mkdir -p $SAVE_DIR

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate qpat

N_GPU=2
BATCH_SIZE=$((128 / $N_GPU))
SEED=$RANDOM
srun torchrun \
  --nnodes 1 \
  --nproc-per-node $N_GPU \
  --rdzv-id $SEED \
  --rdzv-endpoint localhost:29500 \
  -m examples.qat_resnet_cifar10 \
    --arch resnet20 \
    --data-dir $HOME/local/datasets/CIFAR10 \
    --save-dir $SAVE_DIR \
    --resume $SAVE_DIR/checkpoint.pth \
    --seed $SEED \
    --weight-decay 2e-4 \
    --quant-bits 2 \
    --quant-method lsbq \
    --quant-proxmap parq \
    --anneal-end 150 \
    --torch-compile
