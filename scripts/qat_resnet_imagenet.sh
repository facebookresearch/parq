#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=qat_resnet50_imagenet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=352G
#SBATCH --output=/checkpoint/%u/qat_resnet50_imagenet_%j/train.out
#SBATCH --error=/checkpoint/%u/qat_resnet50_imagenet_%j/train.err
#SBATCH --time=1-12:00:00

head_node=`scontrol show hostnames $SLURM_JOB_NODELIST | sed 1q`
head_node_ip=`srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address`
echo Node IP: $head_node_ip

SAVE_DIR=/checkpoint/$USER/qat_resnet50_imagenet_${SLURM_JOB_ID}
mkdir -p $SAVE_DIR

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate qpat

N_GPU=8
BATCH_SIZE=$((256 / $N_GPU))
SEED=$RANDOM
ret=0
srun torchrun \
  --nnodes 1 \
  --nproc-per-node $N_GPU \
  --rdzv-id $SEED \
  --rdzv-backend c10d \
  --rdzv-endpoint $head_node_ip:29500 \
  -m examples.qat_resnet_imagenet \
    --arch resnet50 \
    --batch-size $BATCH_SIZE \
    --save-dir $SAVE_DIR \
    --resume $SAVE_DIR/checkpoint.pth \
    --seed $SEED \
    --quant-bits 2 \
    --quant-method lsbq \
    --quant-proxmap parq \
    --quant-per-channel \
    --anneal-steepness 75 \
    --torch-compile \
    || ret=$?

# Resubmit the job up to 3 times if it failed
restart_count=$(scontrol show job $SLURM_JOB_ID | grep -oP 'Restarts=\d+' | cut -d '=' -f 2)
if [ $ret -ne 0 ] && [ $restart_count -le 3 ]; then
  scontrol requeue $SLURM_JOB_ID
fi
