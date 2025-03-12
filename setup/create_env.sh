# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
#SBATCH --job-name=create_env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

start_time=$(date +%s)
env_name=parq

source $CONDA_PREFIX/etc/profile.d/conda.sh
if [ -d "${CONDA_PREFIX}/envs/${env_name}" ]; then
    echo "Removing existing $env_name env"
    conda env remove -n $env_name -y
fi

# Downgrade grpcio to avoid bug in https://github.com/grpc/grpc/issues/32758
conda create -n $env_name python=3.11 \
    grpcio=1.51.1 \
    pytorch">=2.0.0" torchvision pytorch-cuda=12.1 \
    jupyterlab matplotlib flake8 black \
    timm=0.9.2 tensorboard h5py \
    -c pytorch -c nvidia -c conda-forge \
    -y

# The below two lines are only necessary for AWS cluster
conda activate $env_name
conda env config vars set MODULE_PATH=/opt/slurm/etc/files/modulesfiles:/usr/share/modules/modulefiles LD_PRELOAD=/usr/local/cuda-12.2/lib/libnccl.so

conda clean --all -y

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

echo "Created $env_name env and installed packages in $elapsed_time seconds"
