# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Simple example demonstrating QAT using PARQ (ResNet on CIFAR-10)
Adapted from https://github.com/akamaster/pytorch_resnet_cifar10
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T

from model import resnet
from parq.optim import (  # noqa: F401
    ProxBinaryRelax,
    ProxHardQuant,
    ProxPARQ,
    ProxSoftQuant,
    QuantOptimizer,
)
from parq.quant import LSBQuantizer, UnifQuantizer  # noqa: F401
from utils.train import (
    load_checkpoint,
    log_stats,
    save_checkpoint,
    split_param_groups,
    train_one_epoch,
    validate,
)


def main(args):
    # Check the save_dir exists or not, and makedir if not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # resnet models: resnet20, resnet32, resnet44, resnet56, resnet101
    model = resnet.resnet20().cuda()

    # create train and validation data loaders
    batch_size = 128
    num_workers = 4
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = DataLoader(
        datasets.CIFAR10(
            args.data_dir,
            train=True,
            transform=T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.RandomCrop(32, 4),
                    T.ToTensor(),
                    normalize,
                ]
            ),
            download=True,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        datasets.CIFAR10(
            args.data_dir,
            train=False,
            transform=T.Compose([T.ToTensor(), normalize]),
        ),
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    ##################################################################
    # specify number of quantization bits for different param_groups
    # NOTE: quantizing biases is not recommended due to performance loss
    b_weights = 2
    params_quant, params_no_wd, params_wd = split_param_groups(model)
    param_groups = [
        {"params": params_quant, "quant_bits": b_weights},
        {"params": params_no_wd, "weight_decay": 0},
        {"params": params_wd},
    ]

    # quantizer: Uniform or LSBQ (Least-Squares Binary Quantization)
    quantizer = UnifQuantizer()
    # quantizer = LSBQuantizer()

    epochs = 200
    steps_per_epoch = len(train_loader)

    # change annealing start and end epochs into steps for prox_map
    anneal_start_epoch = 0
    anneal_end_epoch = epochs - 10  # last 10 epochs hard quantization
    anneal_start_step = anneal_start_epoch * steps_per_epoch
    anneal_end_step = anneal_end_epoch * steps_per_epoch

    # construct the proximal map
    prox_map = ProxPARQ(anneal_start_step, anneal_end_step, steepness=100)
    # Other choices of prox_map (uncomment to try):
    # prox_map = ProxHardQuant()
    # prox_map = ProxSoftQuant(anneal_start_step, anneal_end_step)
    # prox_map = ProxBinaryRelax(anneal_start_step, anneal_end_step)

    # construct the base optimizer
    base_optimizer = torch.optim.SGD(
        param_groups, lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    # Other choice of base optimizer: Adam, AdamW, ...

    # construct the quantization (QAT) optimizer
    optimizer = QuantOptimizer(base_optimizer, quantizer, prox_map)

    # define LR scheduler using the base_optimizer
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        base_optimizer, milestones=[100, 150]
    )

    start_epoch = 0
    best_prec1 = 0
    if args.resume:
        start_epoch, best_prec1 = load_checkpoint(
            model, optimizer, lr_scheduler, args.resume, steps_per_epoch, args.evaluate
        )

    ##########################################################################
    # Quantization-Aware Training, loop over epochs

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    log_path = os.path.join(args.save_dir, "log.txt")
    for epoch in range(start_epoch, epochs):
        # train for one epoch
        print("current lr {:.5e}".format(optimizer.param_groups[0]["lr"]))
        train_stats = train_one_epoch(
            train_loader, model, criterion, optimizer, epoch, args.print_freq
        )
        lr_scheduler.step()

        # evaluate on validation set
        test_stats = validate(val_loader, model, criterion, args.print_freq)
        prec1 = test_stats["test_prec1"]
        log_stats(train_stats, test_stats, log_path, epoch)

        # save best quantized model (after annealing ends)
        if epoch >= anneal_end_epoch and prec1 > best_prec1:
            best_prec1 = prec1
            save_checkpoint(
                epoch,
                model,
                optimizer,
                lr_scheduler,
                best_prec1,
                os.path.join(args.save_dir, "best_model.pth"),
            )

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                epoch,
                model,
                optimizer,
                lr_scheduler,
                prec1,
                os.path.join(args.save_dir, "checkpoint.pth"),
            )


def get_arg_parser():
    parser = argparse.ArgumentParser(description="QAT of ResNets on CIFAR10")
    parser.add_argument(
        "--arch",
        "-a",
        default="resnet20",
        choices=(
            "resnet20",
            "resnet32",
            "resnet44",
            "resnet56",
            "resnet110",
            "resnet1202",
        ),
    )
    parser.add_argument("--data-dir", default="~/data", type=str)
    parser.add_argument("--save-dir", default="checkpoints", type=str)
    parser.add_argument("--print-freq", default=50, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoints every specified number of epochs",
    )
    return parser


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
