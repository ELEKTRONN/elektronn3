#!/usr/bin/env python3

# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert

import argparse
import os

import torch
from torch import nn
from torch import optim

parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--exp-name', default=None, help='Manually set experiment name')
parser.add_argument(
    '--epoch-size', type=int, default=100,
    help='How many training samples to process between '
         'validation/preview/extended-stat calculation phases.'
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

from elektronn3.data.cnndata import PatchCreator
from elektronn3.training.trainer import StoppableTrainer
from elektronn3.models.unet import UNet

torch.manual_seed(0)


# USER PATHS
save_root = os.path.expanduser('~/e3training/')
os.makedirs(save_root, exist_ok=True)
data_root = os.path.expanduser('~/neuro_data_cdhw/')
input_h5data = [
    (os.path.join(data_root, f'raw_{i}.h5'), 'raw')
    for i in range(3)
]
target_h5data = [
    (os.path.join(data_root, f'barrier_int16_{i}.h5'), 'lab')
    for i in range(3)
]

max_steps = 500000
lr = 0.0004
lr_stepsize = 1000
lr_dec = 0.995
batch_size = 1

# Initialize neural network model
model = UNet(
    n_blocks=3,
    start_filts=32,
    planar_blocks=(1,),
    activation='relu',
    batch_norm=True
).to(device)

# Specify data set
common_data_kwargs = {  # Common options for training and valid sets.
    'mean': 155.291411,
    'std': 41.812504,
    'aniso_factor': 2,
    'patch_shape': (48, 96, 96),
    'squeeze_target': True,  # Workaround for neuro_data_cdhw,
    'device': device,
}
train_dataset = PatchCreator(
    input_h5data=input_h5data[:2],
    target_h5data=target_h5data[:2],
    train=True,
    epoch_size=args.epoch_size,
    class_weights=True,
    warp=0.5,
    warp_kwargs={
        'sample_aniso': True,
        'perspective': True,
    },
    **common_data_kwargs
)
valid_dataset = PatchCreator(
    input_h5data=[input_h5data[2]],
    target_h5data=[target_h5data[2]],
    train=False,
    epoch_size=10,  # How many samples to use for each validation run
    preview_shape=(64, 144, 144),
    warp=0,
    warp_kwargs={
        'sample_aniso': True,
    },
    **common_data_kwargs
)

# Set up optimization
optimizer = optim.Adam(
    model.parameters(),
    weight_decay=0.5e-4,
    lr=lr,
    amsgrad=True
)
lr_sched = optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)
# lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

criterion = nn.CrossEntropyLoss(weight=train_dataset.class_weights).to(device)
# TODO: Dice loss? (used in original V-Net) https://github.com/mattmacy/torchbiomed/blob/661b3e4411f7e57f4c5cbb56d02998d2d8bddfdb/torchbiomed/loss.py

# Create and run trainer
trainer = StoppableTrainer(
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
