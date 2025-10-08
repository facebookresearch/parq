# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import json
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
from timm.layers import PatchEmbed
from torch import Tensor

from parq.optim import QuantOptimizer

from .visual import plot_quantized_mapping

NORM_LAYERS = (torch.nn.modules.batchnorm._BatchNorm, torch.nn.LayerNorm)


def is_main_process():
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    return rank == 0


def get_param_groups(
    model: torch.nn.Module,
    params_quant: Dict[str, Tensor],
    params_no_wd: Dict[str, Tensor],
    params_wd: Dict[str, Tensor],
    skip_wd_names: Optional[Set[str]] = None,
    prefix: str = "",
    force_full_prec: bool = False,
) -> None:
    """Recurse over children of model to extract quantizable params_quant, as well as
    non-quantizable params (params_no_wd, params_wd).
    """
    # drop torch.compile and DDP wrapper prefixes, if they exist
    for mn, module in model.named_children():
        cur_prefix = f"{prefix}.{mn}" if prefix else mn

        # leave ViT embedding and final classification layer at full precision
        # TODO: generalize to other architectures
        use_full_prec = (
            force_full_prec
            or isinstance(module, PatchEmbed)
            or isinstance(module, NORM_LAYERS)
        )
        for pn, param in module.named_parameters(recurse=False):
            param_name = f"{cur_prefix}.{pn}"
            for attr in ("_orig_mod", "module"):
                param_name = param_name.rsplit(f"{attr}.", 1)[-1]

            use_full_prec |= param_name.startswith("head.")
            if not use_full_prec and pn == "weight":
                params_quant[param_name] = param
            elif pn == "bias" or skip_wd_names and param_name in skip_wd_names:
                params_no_wd[param_name] = param
            else:
                params_wd[param_name] = param
        get_param_groups(
            module,
            params_quant,
            params_no_wd,
            params_wd,
            skip_wd_names=skip_wd_names,
            prefix=cur_prefix,
            force_full_prec=use_full_prec,
        )


def split_param_groups(
    model: torch.nn.Module,
    skip_wd_names: Optional[Set[str]] = None,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Splits model parameters into 3 groups, described below.

    Returns:
        params_quant: quantized, weight decay
        params_no_wd: unquantized, no weight decay
        params_wd: unquantized, weight decay
    """
    params_quant, params_no_wd, params_wd = {}, {}, {}
    get_param_groups(
        model, params_quant, params_no_wd, params_wd, skip_wd_names=skip_wd_names
    )
    n_found_params = len(params_quant) + len(params_no_wd) + len(params_wd)
    assert n_found_params == len(list(model.parameters()))

    if is_main_process():
        for name, dct in zip(
            ("quant", "no_wd", "wd"), (params_quant, params_no_wd, params_wd)
        ):
            print(f"[params_{name}], {len(dct)}: {tuple(dct.keys())}")
    return (
        list(params_quant.values()),
        list(params_no_wd.values()),
        list(params_wd.values()),
    )


class AverageMeter(object):
    """Computes and stores the average and current value

    The (global) average is synced across all worker processes. The main process
    can optionally have a fixed length buffer to average over its most recent
    `window_size` values.
    """

    def __init__(self, window_size=0):
        self.reset(window_size)

    def reset(self, window_size):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.buf = (
            deque(maxlen=window_size) if is_main_process() and window_size > 0 else None
        )

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.buf is not None:
            self.buf.extend([val] * n)

    @property
    def avg(self):
        """Average across all values. Reduces across workers if applicable."""
        if dist.is_available() and dist.is_initialized():
            # Average over worker values if using distributed training
            t = torch.tensor([self.count, self.sum], device="cuda")
            dist.barrier()
            dist.all_reduce(t)

            self.count = int(t[0].item())
            self.sum = t[1].item()
        return self.sum / self.count

    @property
    def window_avg(self):
        """Worker-specific average over the most recent `window_size` elements"""
        return sum(self.buf) / len(self.buf) if self.buf is not None else -1


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(epoch, model, optimizer, lr_scheduler, best_prec1, save_path):
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": lr_scheduler.state_dict(),
            "best_prec1": best_prec1,
        },
        save_path,
    )


def load_checkpoint(
    model, optimizer, lr_scheduler, resume_path, steps_per_epoch, evaluate
) -> Tuple[int, float]:
    start_epoch = 0
    best_prec1 = 0.0
    if os.path.isfile(resume_path):
        if is_main_process():
            print(f"=> loading checkpoint '{resume_path}'")
        checkpoint = torch.load(resume_path, weights_only=True)
        start_epoch = checkpoint["epoch"]
        best_prec1 = checkpoint["best_prec1"]
        model.load_state_dict(checkpoint["model_state"])
        kwargs = (
            {"start_step": start_epoch * steps_per_epoch}
            if isinstance(optimizer, QuantOptimizer)
            else {}
        )
        optimizer.load_state_dict(checkpoint["optim_state"], **kwargs)
        lr_scheduler.load_state_dict(checkpoint["sched_state"])
        if is_main_process():
            print(f"=> loaded checkpoint ({evaluate=}, {start_epoch=})")
    elif is_main_process():
        print(f"=> no checkpoint found at '{resume_path}'")
    return start_epoch, best_prec1


def train_one_epoch(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    print_freq,
    tb_writer=None,
    cutmix_or_mixup=None,
    scaler=None,
) -> Dict[str, Any]:
    """Run one train epoch"""
    batch_time = AverageMeter(print_freq)
    data_time = AverageMeter(print_freq)
    losses = AverageMeter(print_freq)
    top1 = AverageMeter(print_freq)

    # switch to train mode
    model.train()

    end = time.time()
    start_time = end
    elapsed_steps = len(train_loader) * epoch
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target_orig = target

        if cutmix_or_mixup is not None:
            input, target = cutmix_or_mixup(input, target)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        with torch.amp.autocast(device.type, enabled=scaler is not None):
            output = model(input)
            loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_orig)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if is_main_process() and i % print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.window_avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.window_avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.window_avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.window_avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                )
            )
            if tb_writer:
                tb_writer.add_scalar(
                    "train/loss_inner", losses.window_avg, elapsed_steps
                )
        elapsed_steps += 1

    loss_avg = losses.avg
    prec1_avg = top1.avg
    train_stats = {"train_loss": loss_avg, "train_prec1": prec1_avg}
    if is_main_process():
        lr = optimizer.param_groups[0]["lr"]
        lr_sum = None
        parq_inv_slope = None
        if isinstance(optimizer, QuantOptimizer):
            group = next(optimizer.regularized_param_groups())
            lr_sum = group["cumu_lr"]
            if "inv_slope" in group:
                parq_inv_slope = group["inv_slope"]

        if tb_writer:
            tb_writer.add_scalar("train/lr", lr, elapsed_steps)
            tb_writer.add_scalar("train/loss", loss_avg, elapsed_steps)
            tb_writer.add_scalar("train/prec1", prec1_avg, elapsed_steps)

            if lr_sum is not None:
                tb_writer.add_scalar("train/lr_sum", lr_sum, elapsed_steps)
            if parq_inv_slope is not None:
                tb_writer.add_scalar(
                    "train/parq_inv_slope", parq_inv_slope, elapsed_steps
                )

            plot_quantized_mapping(model, optimizer, tb_writer, elapsed_steps)

        epoch_time = datetime.timedelta(seconds=int(time.time() - start_time))
        train_stats["lr"] = lr
        train_stats["train_minutes"] = epoch_time.total_seconds() / 60
        print(
            "Epoch (train):\tTime {0}\tLoss {1:.4f}\tPrec@1 {2:.3f}\tLR {3:.2e}".format(
                epoch_time, loss_avg, prec1_avg, lr
            )
        )
    return train_stats


@torch.inference_mode()
def validate(
    val_loader, model, criterion, print_freq, elapsed_steps=0, tb_writer=None, amp=False
) -> Dict[str, Any]:
    """Run evaluation"""
    is_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.backends.cudnn.deterministic = True
    batch_time = AverageMeter(print_freq)
    losses = AverageMeter(print_freq)
    top1 = AverageMeter(print_freq)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    start_time = end
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    for i, (input, target) in enumerate(val_loader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast(device.type, enabled=amp):
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if is_main_process() and i % print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.window_avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.window_avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.window_avg:.3f})".format(
                    i,
                    len(val_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                )
            )
    loss_avg = losses.avg
    prec1_avg = top1.avg
    test_stats = {"test_loss": loss_avg, "test_prec1": prec1_avg}
    if is_main_process():
        if tb_writer:
            tb_writer.add_scalar("test/loss", loss_avg, elapsed_steps)
            tb_writer.add_scalar("test/prec1", prec1_avg, elapsed_steps)

        epoch_time = datetime.timedelta(seconds=int(time.time() - start_time))
        test_stats["test_minutes"] = epoch_time.total_seconds() / 60
        print(
            "Epoch (test):\tTime {0}\tLoss {1:.4f}\tPrec@1 {2:.3f}".format(
                epoch_time, loss_avg, prec1_avg
            )
        )
    torch.backends.cudnn.deterministic = is_deterministic
    return test_stats


def log_stats(
    train_stats: Dict[str, Any], test_stats: Dict[str, Any], log_path: str, epoch: int
):
    if is_main_process():
        with open(log_path, "a") as f:
            train_stats["epoch"] = epoch
            for k, v in test_stats.items():
                train_stats[k] = v
            f.write(f"{json.dumps(train_stats)}\n")
