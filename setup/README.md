# Setup

## Conda environment

### Option 1: create from YAML file

If not on an AWS cluster, delete the `variables` section of the file first.
```bash
conda env create -f parq.yml
```

### Option 2: build from scratch

If not on an AWS cluster, comment out the line starting with `conda env config vars set`. Replace `bash` with `sbatch` if Slurm is available.
```bash
bash create_env.sh
```

## AWS-specific modifications

### Set CUDA and NCCL environment variables

The AWS cluster requires specific environment variables to be set. For all scripts under examples/scripts, add the below lines to the top of the file.
```bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate parq

source /etc/profile.d/modules.sh
module load cuda/12.1
module load nccl/2.18.3-cuda.12.1
```

### Convert ImageNet image files to HDF5 format

Parallel read speeds on AWS are unfortunately very slow. To combat this, it is recommended to convert ImageNet files to a single HDF5 file per data split.
```bash
# SRC_DIR and DEST_DIR are the respective read and write paths for ImageNet
bash create_hdf5.sh
```

### Copy ImageNet HDF5 data to local disk prior to training

Next, add the below lines to examples/scripts/\*imagenet.sh. Each node should have access to its own copy of the HDF5 files to maximize read speed.
```bash
data_dir=$DEST_DIR   # location of created HDF5 files
for split in train val; do
  if [ ! -f /scratch/${split}.hdf5 ]; then
    rsync -av ${data_dir}/${split}.hdf5 /scratch/
  fi
done
data_dir=/scratch
```

Add the following flags to the training script under the `torchrun` command so that it consumes the HDF5 data using a special `H5VisionDataset` class.
```bash
--data-dir $data_dir \
--hdf5-data \
```
