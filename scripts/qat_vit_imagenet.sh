#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=parq_deit_tiny_imagenet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=352G
#SBATCH --output=/checkpoint/%u/parq_deit_tiny_imagenet_%j/train.out
#SBATCH --error=/checkpoint/%u/parq_deit_tiny_imagenet_%j/train.err
#SBATCH --time=1-12:00:00

head_node=`scontrol show hostnames $SLURM_JOB_NODELIST | sed 1q`
head_node_ip=`srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address`
echo Node IP: $head_node_ip

SAVE_DIR=/checkpoint/$USER/parq_deit_tiny_imagenet_${SLURM_JOB_ID}
mkdir -p $SAVE_DIR

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate qpat

N_NODE=1
N_GPU_PER_NODE=8
BATCH_SIZE=$((1024 / ($N_GPU_PER_NODE * $N_NODE)))
SEED=$RANDOM
ret=0
srun torchrun \
  --nnodes $N_NODE \
  --nproc-per-node $N_GPU_PER_NODE \
  --rdzv-id $SEED \
  --rdzv-backend c10d \
  --rdzv-endpoint $head_node_ip:29500 \
  -m examples.qat_vit_imagenet \
    --arch deit_tiny_patch16_224 \
    --batch-size $BATCH_SIZE \
    --save-dir $SAVE_DIR \
    --resume $SAVE_DIR/checkpoint.pth \
    --seed $SEED \
    --lr 2e-3 \
    --lr-min 1e-8 \
    --quant-bits 2 \
    --quant-method lsbq \
    --quant-proxmap parq \
    --quant-per-channel \
    --anneal-steepness 50 \
    --custom-train-transform \
    --cutmix-mixup \
    --torch-compile \
    --amp \
    || ret=$?

# Resubmit the job up to 3 times if it failed
restart_count=$(scontrol show job $SLURM_JOB_ID | grep -oP 'Restarts=\d+' | cut -d '=' -f 2)
if [ $ret -ne 0 ] && [ $restart_count -le 3 ]; then
  scontrol requeue $SLURM_JOB_ID
fi
