#!/usr/bin/env python3

# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert

"""
Workflow of spinal semantic segmentation based on multiviews (2D semantic segmentation).

It learns how to differentiate between spine head, spine neck and spine shaft.
Caution! The input dataset was not manually corrected.
"""

import argparse
import os
from elektronn3.models.fcn_2d import *
from elektronn3.models.tiramisu_2d import FCDenseNet
import torch
from torch import nn
from torch import optim
from elektronn3.training.loss import BlurryBoarderLoss


def get_model():
    # vgg_model = VGGNet(model='vgg13', requires_grad=True, in_channels=4)
    # model = FCNs(pretrained_net=vgg_model, n_class=5)
    model = FCDenseNet(in_channels=4, n_classes=5)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a network.')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('-n', '--exp-name', default="tiramisu", help='Manually set experiment name')
    parser.add_argument(
        '-m', '--max-steps', type=int, default=500000,
        help='Maximum number of training steps to perform.'
    )
    args = parser.parse_args()

    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f'Running on device: {device}')

    # Don't move this stuff, it needs to be run this early to work
    import elektronn3
    elektronn3.select_mpl_backend('Agg')

    from elektronn3.training import Trainer, Backup
    from elektronn3.data.cnndata import MultiviewData

    torch.manual_seed(0)


    # USER PATHS
    save_root = os.path.expanduser('~/e3training/')

    max_steps = args.max_steps
    lr = 0.004
    lr_stepsize = 500
    lr_dec = 0.99
    batch_size = 6

    model = get_model()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        batch_size = batch_size * torch.cuda.device_count()
        # dim = 0 [20, xxx] -> [10, ...], [10, ...] on 2 GPUs
        model = nn.DataParallel(model)
    model.to(device)


    # Specify data set
    train_dataset = MultiviewData(train=True)
    valid_dataset = MultiviewData(train=False)

    # Set up optimization
    optimizer = optim.Adam(
        model.parameters(),
        weight_decay=0.5e-4,
        lr=lr,
        amsgrad=True
    )
    lr_sched = optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)

    criterion = BlurryBoarderLoss().to(device)

    # Create and run trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batchsize=batch_size,
        num_workers=2,
        save_root=save_root,
        exp_name=args.exp_name,
        schedulers={"lr": lr_sched},
        ipython_on_error=False
    )

    # Archiving training script, src folder, env info
    bk = Backup(script_path=__file__,save_path=trainer.save_path).archive_backup()

    trainer.train(max_steps)
