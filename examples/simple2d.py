#!/usr/bin/env python3

# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert

"""
Demo of a 2D semantic segmentation workflow.

It doesn't really learn anything useful, since both model and dataset
are far too small. It just serves as a quick demo for how 2D stuff can
be implemented.
"""

import argparse
import os

import torch
from torch import nn
from torch import optim

parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('-n', '--exp-name', default=None, help='Manually set experiment name')
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

from elektronn3.training import Trainer
from elektronn3.data import SimpleNeuroData2d

torch.manual_seed(0)


# USER PATHS
save_root = os.path.expanduser('~/e3training/')

max_steps = args.max_steps
lr = 0.0004
lr_stepsize = 1000
lr_dec = 0.995
batch_size = 1

# Initialize neural network model
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
    nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
    nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
    nn.Conv2d(32, 2, 1)
).to(device)

# Specify data set
train_dataset = SimpleNeuroData2d(train=True)
valid_dataset = SimpleNeuroData2d(train=False)

# Set up optimization
optimizer = optim.Adam(
    model.parameters(),
    weight_decay=0.5e-4,
    lr=lr,
    amsgrad=True
)
lr_sched = optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)

criterion = nn.CrossEntropyLoss().to(device)

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
    schedulers={"lr": lr_sched}
)
trainer.train(max_steps)
