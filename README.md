# PARQ: A PyTorch Library for QAT

PARQ is a PyTorch library for quantization-aware training (QAT). It is based on a convex regularization approach called Piecewise-Affine Regularized Quantization (PARQ). The library also implements several other QAT methods and can serve as a toolbox for general QAT research and benchmarking.

PARQ enables QAT without modifying the code/module specifying the neural network model. It instead interfaces with a QAT optimizer only, allowing the users to specify the parameter groups to be quantized and the bit-widths for different groups. The QAT optimizer in PARQ can be configured with three components: a base optimizer, a quantization method, and a proximal map (see [Optimizer-only interface](#optimizer-only-interface) for details).

Reference: [PARQ paper on arXiv](https://arxiv.org/abs/2503.15748)

> [!NOTE]
> PARQ is currently only compatible with [FSDP2](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) data parallelism. Support for its predecessor, [FSDP](https://pytorch.org/docs/stable/fsdp.html), will be added soon.

## Installation

```bash
git clone https://github.com/facebookresearch/parq.git
```

or

```bash
pip install -e .

# for a developer installation
pip install -e '.[dev]'
```

Alternatively, PARQ is included as a prototype in `torchao` and can be imported via [`torchao.prototype.parq`](https://github.com/pytorch/ao/tree/main/torchao/prototype/parq).

To run the QAT examples, follow [these instructions](setup/README.md).

## Optimizer-only interface

This package provides a `QuantOptimizer` that can be constructed with three components:

* the base optimizer: a `torch.optim.Optimizer` object (SGD, Adam or AdamW)
* a `Quantizer` object specifying the quantization method (uniform or LSBQ)
* a `ProxMap` object specifying the proximal map (hard, soft, PARQ, BinaryRelax)

The following code snippet illustrate how to set up QAT with PARQ:

```python
from parq.optim import QuantOptimizer, ProxPARQ
from parq.quant import UnifQuantizer

# create model and loss function
model = torchvision.models.resnet18().cuda()

# split params into quantizable and non-quantizable groups and set bit-widths
weights = [p for name, p in model.named_parameters() if name.endswith("weight")]
others  = [p for name, p in model.named_parameters() if not name.endswith("weight")]

param_groups = [
    {"params": weights, "quant_bits": 2},
    {"params": others, "weight_decay": 0},
]

# create base optimizer (SGD, Adam or AdamW)
base_optimizer = torch.optim.SGD(
    param_groups, lr=0.1, momentum=0.9, weight_decay=1e-4
)

# create quantizer and proximal map objects
quantizer = UnifQuantizer()
prox_map = ProxPARQ(anneal_start=100, anneal_end=20000, steepness=20)

# create QuantOptimizer
optimizer = QuantOptimizer(base_optimizer, quantizer, prox_map)
```

After creating `QuantOptimizer`, QAT follows the common training pipeline:

```python
dataset = torch.utils.data.DataLoader(...)
loss_fn = torch.nn.CrossEntropyLoss().cuda()

for epoch in range(200):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
```

See [`examples/qat_simple.py`](examples/qat_simple.py) for the full code context.

## QAT arguments

| | description | choices |
| --- | --- | --- |
| `quant-bits` | bit-width for quantized weights | 0 (ternary), 1-4 |
| `quant-method` | method for determining quantized values | `lsbq`, `uniform` |
| `quant-proxmap` | proximal mapping to project weights onto quantized values | `hard`, `soft`, `parq`, `binaryrelax` |
| `quant-lsbq-optimal` | use optimal LSBQ algorithm instead of greedy | `store_true` flag |
| `anneal-start` | start epoch for QAT annealing period | (0, `total_steps` - 1) |
| `anneal-end` | end epoch for QAT annealing period | (`anneal_end`, `total_steps`) |
| `anneal-steepness` | sigmoid steepness for PARQ inverse slope schedule | 1-20 |

## Running the examples

### DeiT + ImageNet

To train 2-bit DeiT-T model with [LSBQ](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Pouransari_Least_Squares_Binary_Quantization_of_Neural_Networks_CVPRW_2020_paper.html) and PARQ on 8 GPUs:
```bash
SEED=$RANDOM
torchrun \
  --nnodes 1 --nproc-per-node 8 \
  --rdzv-id $SEED --rdzv-backend c10d --rdzv-endpoint localhost:29500 \
  -m examples.qat_vit_imagenet \
    --arch deit_tiny_patch16_224 \
    --save-dir $SAVE_DIR --data-dir $DATA_DIR \
    --seed $SEED \
    --lr 2e-3 --lr-min 1e-8 \
    --quant-bits 2 --quant-method lsbq --quant-per-channel \
    --quant-proxmap parq --anneal-steepness 50 \
    --custom-train-transform --cutmix-mixup \
    --torch-compile --amp
```
Besides the [QAT arguments](#qat-arguments), details on other arguments can be found via `python -m examples.qat_vit_imagenet --help`.

### ResNet + ImageNet

To train 2-bit ResNet50 with LSBQ and PARQ on 8 GPUs:
```bash
SEED=$RANDOM
torchrun \
  --nnodes 1 --nproc-per-node 8 \
  --rdzv-id $SEED --rdzv-backend c10d --rdzv-endpoint localhost:29500 \
  -m examples.qat_resnet_imagenet \
  --arch resnet50 \
  --save-dir $SAVE_DIR --data-dir $DATA_DIR \
  --seed $SEED \
  --quant-bits 2 --quant-method lsbq --quant-per-channel \
  --quant-proxmap parq --anneal-steepness 75 \
  --torch-compile
```

### ResNet + CIFAR-10

To train 2-bit ResNet-20 with LSBQ and PARQ on a single GPU:
```bash
SEED=$RANDOM
python -m examples.qat_resnet_cifar10 \
  --arch resnet20 \
  --save-dir $SAVE_DIR --data-dir $DATA_DIR \
  --seed $SEED \
  --weight-decay 2e-4 \
  --quant-bits 2 --quant-method lsbq \
  --quant-proxmap parq --anneal-end 150 \
  --torch-compile
```

## License

PARQ is MIT licensed, as found in the LICENSE file.
