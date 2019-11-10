# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Jonathan Klimesch

import argparse
import os
import random
import torch
from torch import nn
import numpy as np

# Don't move this stuff, it needs to be run this early to work
import elektronn3
elektronn3.select_mpl_backend('Agg')

from elektronn3.training import Trainer, Backup
from elektronn3.training import metrics
from elektronn3.models.convpoint import ConvPoint


# PARSE ARGUMENTS #


parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('-r', '--resume', metavar='PATH',
                    help='Path to pretrained model state dict from which to resume training.')
parser.add_argument('--seed', type=int, default=0, help='Base seed for all RNGs.')

args = parser.parse_args()


# SET UP ENVIRONMENT #


# Set up all RNG seeds, set level of determinism
random_seed = args.seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
deterministic = args.deterministic

torch.backends.cudnn.benchmark = True  # Improves overall performance in *most* cases

# Set up CUDA if needed
if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Running on device: {device}')


# CREATE NETWORK #


# TODO change channels
input_channels = 1
output_channels = 1
model = ConvPoint(input_channels, output_channels).to(device)

# Load pretrained network if required
if args.resume is not None:
    model.load_state_dict(torch.load(os.path.expanduser(args.resume)))

# define parameters
epochs = 200
milestones = [60, 120]
lr = 1e-3
batch_size = 16

# set paths
save_root = os.path.expanduser('~/e3training/')


# PREPARE DATA SET #


# TODO add transformations

# TODO set up data set
train_dataset = 1
valid_dataset = 1


# PREPARE AND START TRAINING #


# set up optimization
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones)

valid_metrics = {
    'val_accuracy': metrics.bin_accuracy,
    'val_precision': metrics.bin_precision,
    'val_recall': metrics.bin_recall,
    'val_DSC': metrics.bin_dice_coefficient,
    'val_IoU': metrics.bin_iou,
}

criterion = nn.CrossEntropyLoss().to(device)

trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    batchsize=batch_size,
    num_workers=1,
    save_root=save_root,
    exp_name=args.exp_name,
    schedulers={"lr": scheduler},
    valid_metrics=valid_metrics,
)

# Archiving training script, src folder, env info
bk = Backup(script_path=__file__, save_path=trainer.save_path).archive_backup()

# Start training
trainer.run(epochs)
