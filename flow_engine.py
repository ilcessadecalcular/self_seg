# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from util.metric import dice_func
from util.datafunction import z_norm,normalize_hu
import util.misc as misc
import util.lr_sched as lr_sched
from util.crop import rand_crop_flow

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    crop_size: int = 50,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets,flows_forward, flows_backward) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples, targets, flows_forward,flows_backward = rand_crop_flow(samples, targets, flows_forward,flows_backward,crop_size)


        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        flows_forward = flows_forward.to(device, non_blocking=True)
        flows_backward = flows_backward.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples,flows_forward,flows_backward )
            loss = criterion(outputs, targets)
            # print(loss)
        loss_value = loss.item()
        # print(samples.size)
        # print(outputs.size())
        # print(targets.size())
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    from util.loss_function import SoftDiceLoss, BCELoss2d, DiceCeloss
    criterion = DiceCeloss()
    # criterion = SoftDiceLoss()
    # criterion = torch.nn.BCEWithLogitsLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 5, header):
        images = batch[0]
        target = batch[1]
        flows_forward = batch[2]
        flows_backward = batch[3]
        # n, t, c, h, w = images.size()

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        flows_forward = flows_forward.to(device, non_blocking=True)
        flows_backward = flows_backward.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast():
            # images1 = images[:,:300,:,:,:]
            # flows_forward1 = flows_forward[:,:299,:,:,:]
            # flows_backward1 = flows_backward[:, :299, :, :, :]
            # output1 = model(images1,flows_forward1,flows_backward1)
            # images2 = images[:,300:,:,:,:]
            # flows_forward2 = flows_forward[:, 300: , :, :, :]
            # flows_backward2 = flows_backward[:, 300: , :, :, :]
            # output2 = model(images2,flows_forward2,flows_backward2)
            # output = torch.cat((output1,output2),1)
            output = model(images,flows_forward,flows_backward)
            loss = criterion(output, target)

        logits = torch.sigmoid(output)
        # print(output)
        # print(logits)
        labels = logits.clone()
        labels[labels > 0.5] = 1
        labels[labels <= 0.5] = 0
        # print(labels)
        # print(logits.sum())
        # print(labels.sum())
        dice = dice_func(labels, target)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['dice'].update(dice.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* dice {dices.global_avg:.3f}  loss {losses.global_avg:.3f}'
          .format(dices=metric_logger.dice, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}