# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Simple example of QAT using DDP (ViT on ImageNet)"""

import argparse
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms as T
from torchvision.models import get_model

from parq.optim import (
    ProxBinaryRelax,
    ProxHardQuant,
    ProxPARQ,
    ProxSoftQuant,
    build_quant_optimizer,
)
from parq.quant import LSBQuantizer, UnifQuantizer
from utils.h5_vision_dataset import H5VisionDataset
from utils.train import (
    is_main_process,
    load_checkpoint,
    log_stats,
    save_checkpoint,
    split_param_groups,
    train_one_epoch,
    validate,
)


def main(args):
    # torchrun will set the environment variables for GPU rank
    dist.init_process_group(backend="nccl")
    if is_main_process():
        print(args)

    torch.set_float32_matmul_precision("high")  # improves matmul speed

    # each GPU begins with a different random seed
    global_rank = int(os.environ["RANK"])
    torch.manual_seed(args.seed + global_rank)

    # check whether the save_dir exists or not, and makedir if not
    if is_main_process() and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # set local CUDA device and create model
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    model = get_model(
        args.arch,
        num_classes=1000,
        weights="IMAGENET1K_V1" if args.pretrained else None,
    )
    model = model.to(device)
    model = DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    if args.torch_compile:
        assert hasattr(torch, "compile"), (
            "{torch.__version__=} is missing torch.compile()"
        )
        model = torch.compile(model, backend="inductor")

    # remove torch.compile and DDP wrappers, if they exist
    model_without_ddp = model
    for attr in ("_orig_mod", "module"):
        if hasattr(model, attr):
            model_without_ddp = getattr(model_without_ddp, attr)

    train_loader, val_loader = create_data_loaders(
        args.data_dir,
        args.batch_size,
        args.workers,
        args.evaluate,
        args.seed,
        args.input_size,
        args.hdf5_data,
    )
    steps_per_epoch = len(train_loader)

    # define loss function (criterion)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # specify number of quantization bits for different parameter groups
    params_quant, params_no_wd, params_wd = split_param_groups(model)
    param_groups = [
        {"params": params_quant, "quant_bits": args.quant_bits},
        {"params": params_no_wd, "weight_decay": 0},
        {"params": params_wd},
    ]

    # construct the quantizer
    if args.quant_method.lower() == "unif":
        quantizer = UnifQuantizer()
    elif args.quant_method.lower() == "lsbq":
        quantizer = LSBQuantizer(optimal=args.quant_lsbq_optimal)
    else:
        raise ValueError("Invalid quantizer option")

    # change annealing start and end epochs into steps for QuantOptimizer
    if args.anneal_end < 0:
        args.anneal_end = args.epochs
    anneal_start_step = args.anneal_start * steps_per_epoch
    anneal_end_step = args.anneal_end * steps_per_epoch

    # construct the proximal map
    if args.quant_proxmap.lower() == "hard":
        prox_map = ProxHardQuant()
    elif args.quant_proxmap.lower() == "soft":
        prox_map = ProxSoftQuant(anneal_start_step, anneal_end_step)
    elif args.quant_proxmap.lower() == "parq":
        prox_map = ProxPARQ(
            anneal_start_step, anneal_end_step, steepness=args.anneal_steepness
        )
    elif args.quant_proxmap.lower() == "binaryrelax":
        prox_map = ProxBinaryRelax(anneal_start_step, anneal_end_step)
    else:
        raise ValueError("Invalid prox-map option")

    # construct the base optimizer
    base_optimizer = torch.optim.SGD(
        param_groups,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if args.full_prec:
        optimizer = base_optimizer
    else:
        # construct the quantization (QAT) optimizer
        optimizer = build_quant_optimizer(
            base_optimizer=base_optimizer,
            quantizer=quantizer,
            prox_map=prox_map,
            warmup_steps=args.quant_warmup_steps,
            quant_period=args.quant_period,
            quant_per_channel=args.quant_per_channel,
            quant_shrink=args.quant_shrink,
            anneal_wd_frac=args.anneal_wd_frac,
            nm_gamma=args.nm_gamma,
        )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # optionally resume from a checkpoint -> need to load optimizer state
    best_prec1 = 0
    if args.resume:
        args.start_epoch, best_prec1 = load_checkpoint(
            model_without_ddp,
            optimizer,
            lr_scheduler,
            args.resume,
            steps_per_epoch,
            args.evaluate,
        )

    if args.evaluate:
        validate(val_loader, model, criterion, args.print_freq)
        return

    elapsed_steps = args.start_epoch * steps_per_epoch
    tb_writer = (
        SummaryWriter(os.path.join(args.save_dir, "runs"))
        if is_main_process()
        else None
    )
    log_path = os.path.join(args.save_dir, "log.txt")
    for epoch in range(args.start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        train_stats = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            args.print_freq,
            tb_writer,
        )
        lr_scheduler.step()

        # evaluate on validation set
        test_stats = validate(
            val_loader,
            model,
            criterion,
            args.print_freq,
            elapsed_steps,
            tb_writer,
        )
        prec1 = test_stats["test_prec1"]
        log_stats(train_stats, test_stats, log_path, epoch)

        # remember best prec@1 and model, and save checkpoint regularly
        if prec1 > best_prec1:
            best_prec1 = prec1
            save_checkpoint(
                epoch,
                model_without_ddp,
                optimizer,
                lr_scheduler,
                best_prec1,
                os.path.join(args.save_dir, "best_model.pth"),
            )

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                epoch,
                model_without_ddp,
                optimizer,
                lr_scheduler,
                prec1,
                os.path.join(args.save_dir, "checkpoint.pth"),
            )

        elapsed_steps += steps_per_epoch

    dist.destroy_process_group()


def create_data_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    evaluate: bool,
    seed: int,
    input_size: int,
    hdf5_data: bool,
) -> Tuple[Optional[DataLoader], DataLoader]:
    normalize_transform = T.Normalize(
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    )
    if not evaluate:
        train_transform = T.Compose(
            [
                T.RandomResizedCrop(input_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize_transform,
            ]
        )
        if hdf5_data:
            train_set = H5VisionDataset(
                os.path.join(data_dir, "train.hdf5"), transform=train_transform
            )
        else:
            train_set = datasets.ImageFolder(
                os.path.join(data_dir, "train"), transform=train_transform
            )
        train_sampler = DistributedSampler(train_set, seed=seed)
        train_loader = DataLoader(
            train_set,
            sampler=train_sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:  # no need to load train data for eval-only
        train_loader = None

    val_transform = T.Compose(
        [
            T.Resize(int(input_size / 0.875), interpolation=3),
            T.CenterCrop(input_size),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    if hdf5_data:
        val_set = H5VisionDataset(
            os.path.join(data_dir, "val.hdf5"), transform=val_transform
        )
    else:
        val_set = datasets.ImageFolder(
            os.path.join(data_dir, "val"), transform=val_transform
        )
    val_sampler = DistributedSampler(val_set, shuffle=False)
    val_loader = DataLoader(
        val_set,
        sampler=val_sampler,
        batch_size=int(1.5 * batch_size),
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--arch",
        default="resnet34",
        choices=("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"),
        help="model architecture (default: resnet34)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="image input size",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=8,
        help="number of data loading workers (default: 8)",
    )
    parser.add_argument(
        "--epochs", type=int, default=90, help="number of total epochs to run"
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="mini-batch size per GPU (default: 32)",
    )
    parser.add_argument(
        "--lr", default=0.1, type=float, help="initial learning rate (default: 0.1)"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument(
        "-wd",
        "--weight-decay",
        type=float,
        default=1e-4,
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--anneal-wd-frac",
        type=float,
        default=0.0,
        help="fraction of weight decay to anneal during QAT (default: 0.0)",
    )
    parser.add_argument(
        "--full-prec",
        action="store_true",
        help="use full-precision training",
    )
    parser.add_argument(
        "--quant-bits",
        default=2,
        type=int,
        help="number of bits for QAT (default: 2)",
    )
    parser.add_argument(
        "--nm-gamma",
        type=float,
        default=0.0,
        help="gamma parameter for NM-SGD (default: 0.0, i.e., disabled)",
    )
    parser.add_argument(
        "--quant-warmup-steps",
        default=2500,
        type=int,
        help="number of warmup steps before applying QAT (default: 2500)",
    )
    parser.add_argument(
        "--quant-period",
        default=50,
        type=int,
        help="compute quant set every specified number of steps",
    )
    parser.add_argument(
        "--quant-method",
        default="unif",
        type=str,
        choices=["unif", "lsbq"],
        help="quantization method",
    )
    parser.add_argument(
        "--quant-per-channel",
        action="store_true",
        help="apply per-channel quantization",
    )
    parser.add_argument(
        "--quant-lsbq-optimal",
        action="store_true",
        help="Use optimal LSBQ algorithm instead of greedy",
    )
    parser.add_argument(
        "--quant-shrink",
        action="store_true",
        help="scale quantized weights by gamma inverse",
    )
    parser.add_argument(
        "--quant-proxmap",
        default="hard",
        type=str,
        choices=["hard", "soft", "parq", "binaryrelax"],
        help="proximal mapping for QAT",
    )
    parser.add_argument(
        "--anneal-start",
        default=0,
        type=int,
        help="starting epoch for QAT annealing period",
    )
    parser.add_argument(
        "--anneal-end",
        default=-1,
        type=int,
        help="ending epoch for QAT annealing period",
    )
    parser.add_argument(
        "--anneal-steepness",
        default=100,
        type=float,
        help="Sigmoid steepness for QAT annealing",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=500,
        type=int,
        help="print frequency (default: 500)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="use pre-trained model",
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        default="/datasets01/imagenet_full_size/061417",
        type=str,
    )
    parser.add_argument(
        "--hdf5-data",
        action="store_true",
        help="load HDF5 datasets from --data-dir",
    )
    parser.add_argument("--save-dir", dest="save_dir", default="checkpoints", type=str)
    parser.add_argument(
        "--save-every",
        dest="save_every",
        type=int,
        default=10,
        help="Save checkpoints every specified number of epochs",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="random seed set for reproducibility",
    )
    parser.add_argument(
        "--torch-compile", action="store_true", help="use torch.compile"
    )
    return parser


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
