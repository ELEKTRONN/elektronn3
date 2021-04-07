# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Jonathan Klimesch

import os
import torch
import argparse
import random
import numpy as np
# Don't move this stuff, it needs to be run this early to work
import elektronn3
elektronn3.select_mpl_backend('Agg')
import morphx.processing.clouds as clouds
from torch import nn
from morphx.data.torchset import TorchSet
from elektronn3.models.convpoint import SegSmall, SegNoBatch
from elektronn3.training import Trainer3d, Backup


# PARSE PARAMETERS #

parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--na', type=str, required=True, help='Experiment name')
parser.add_argument('--tp', type=str, required=True, help='Train path')
parser.add_argument('--sr', type=str, required=True, help='Save root')
parser.add_argument('--bs', type=int, default=16, help='Batch size')
parser.add_argument('--sp', type=int, default=1000, help='Number of sample points')
parser.add_argument('--ra', type=int, default=10000, help='Radius')
parser.add_argument('--cl', type=int, default=5, help='Number of classes')
parser.add_argument('--co', action='store_true', help='Disable CUDA')
parser.add_argument('--big', action='store_true', help='Use big SegBig Convpoint network')
parser.add_argument('--seed', default=0, help='Random seed')
parser.add_argument('--ana', default=0, help='Cloudset size of previous analysis')

args = parser.parse_args()

for arg in vars(args):
    print(arg, getattr(args, arg))

# SET UP ENVIRONMENT #

random_seed = args.seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# define parameters
use_cuda = not args.co
name = args.na
batch_size = args.bs
npoints = args.sp
radius = args.ra
num_classes = args.cl
milestones = [10, 20, 40, 80, 160, 320, 640, 1280]
lr = 1e-3
lr_stepsize = 1000
lr_dec = 0.995
max_steps = 500000
size = args.ana

if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Running on device: {device}')

# set paths
save_root = os.path.expanduser(args.sr)
train_path = os.path.expanduser(args.tp)


# CREATE NETWORK AND PREPARE DATA SET #

input_channels = 1
if args.big:
    model = SegNoBatch(input_channels, num_classes)
else:
    model = SegSmall(input_channels, num_classes)

if use_cuda:
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        batch_size = batch_size * torch.cuda.device_count()
        model = nn.DataParallel(model)
    model.to(device)

# Transformations to be applied to samples before feeding them to the network
train_transform = clouds.Compose([clouds.RandomVariation((-10, 10)),
                                  clouds.Normalization(radius),
                                  clouds.RandomRotate(),
                                  clouds.Center()])

train_ds = TorchSet(train_path, radius, npoints, train_transform, class_num=num_classes, size=size)

# PREPARE AND START TRAINING #

# set up optimization
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.5)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000)

criterion = torch.nn.CrossEntropyLoss()
if use_cuda:
    criterion.cuda()

# Create trainer
trainer = Trainer3d(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    train_dataset=train_ds,
    batchsize=batch_size,
    num_workers=0,
    save_root=save_root,
    exp_name=name,
    schedulers={"lr": scheduler},
    num_classes=num_classes
)

# Archiving training script, src folder, env info
bk = Backup(script_path=__file__,
            save_path=trainer.save_path).archive_backup()

# Start training
trainer.run(max_steps)
