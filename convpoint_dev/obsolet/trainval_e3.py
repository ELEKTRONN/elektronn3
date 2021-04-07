# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Jonathan Klimesch

import os
import torch
import argparse
import logging
# Don't move this stuff, it needs to be run this early to work
import elektronn3
elektronn3.select_mpl_backend('Agg')
import morphx.processing.clouds as clouds
from morphx.data.torchset import TorchSet
from elektronn3.models.convpoint import SegSmall, SegNoBatch
from elektronn3.training import Trainer3d, Backup
from elektronn3.training import metrics


# PARSE PARAMETERS #

parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--na', type=str, required=True, help='Experiment name')
parser.add_argument('--tp', type=str, required=True, help='Train path')
parser.add_argument('--vp', type=str, required=True, help='Validation path')
parser.add_argument('--sr', type=str, required=True, help='Save root')
parser.add_argument('--bs', type=int, default=16, help='Batch size')
parser.add_argument('--sp', type=int, default=1000, help='Number of sample points')
parser.add_argument('--ra', type=int, default=10000, help='Radius')
parser.add_argument('--cl', type=int, default=2, help='Number of classes')
parser.add_argument('--co', action='store_true', help='Disable CUDA')
parser.add_argument('--big', action='store_true', help='Use big SegBig Convpoint network')

args = parser.parse_args()


# SET UP ENVIRONMENT #

use_cuda = not args.co
if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

logger = logging.getLogger('elektronn3log')
logger.info(f'Running on device: {device}')

# define parameters
name = args.na
batch_size = args.bs
npoints = args.sp
radius = args.ra
num_classes = args.cl
milestones = [60, 120]
lr = 1e-3
lr_stepsize = 1000
lr_dec = 0.995
max_steps = 500000

# set paths
save_root = os.path.expanduser(args.sr)
train_path = os.path.expanduser(args.tp)
val_path = os.path.expanduser(args.vp)


# CREATE NETWORK AND PREPARE DATA SET#

input_channels = 1
if args.big:
    model = SegNoBatch(input_channels, num_classes).to(device)
else:
    model = SegSmall(input_channels, num_classes).to(device)

# Transformations to be applied to samples before feeding them to the network
train_transform = clouds.Compose([clouds.RandomRotate(), clouds.Center()])
val_transform = clouds.Center()

train_ds = TorchSet(train_path, radius, npoints, train_transform,
                    class_num=num_classes,
                    elektronn3=True)

valid_ds = TorchSet(val_path, radius, npoints, val_transform,
                    class_num=num_classes,
                    elektronn3=True)

# PREPARE AND START TRAINING #

# set up optimization
optimizer = torch.optim.Adam(model.parameters(),
                             weight_decay=0.5e-4,
                             lr=lr,
                             amsgrad=True)
lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)

valid_metrics = {
    'val_accuracy': metrics.accuracy,
    'val_precision': metrics.precision,
}

criterion = torch.nn.CrossEntropyLoss().to(device)

# Create trainer
trainer = Trainer3d(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    train_dataset=train_ds,
    v_path=valid_ds,
    batchsize=batch_size,
    num_workers=1,
    save_root=save_root,
    exp_name=name,
    schedulers={"lr": lr_sched},
    valid_metrics=valid_metrics,
    num_classes=num_classes
)

# Archiving training script, src folder, env info
bk = Backup(script_path=__file__,
            save_path=trainer.save_path).archive_backup()

# Start training
trainer.run(max_steps)
