## DeiT + ImageNet

For distributed training on Slurm, run the following.
```bash
sbatch scripts/parq_vit_imagenet.sh
```
Replace `sbatch` with `bash` to instead run on a local GPU node.

### Multi-node training on Slurm

Adjust the number of nodes after `#SBATCH --nodes` and `--nnodes` based on the model size (e.g., `deit_base_patch16_224` may require 2 nodes). Change `N_GPU` to the total number of GPUs across nodes.

## ResNet + ImageNet

The same options for distributed training as above apply here.
```bash
sbatch scripts/parq_resnet_imagenet.sh
```

## ResNet + CIFAR-10

To run on a single GPU:
```bash
sbatch scripts/parq_resnet_cifar10.sh
```
