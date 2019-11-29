# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Jonathan Klimesch

import os
import torch

# Don't move this stuff, it needs to be run this early to work
import elektronn3
elektronn3.select_mpl_backend('Agg')

import morphx.processing.clouds as clouds
from morphx.data.torchset import TorchSet
from elektronn3.models.convpoint import ConvPoint
from elektronn3.training import Trainer3d, Backup
from elektronn3.training import metrics

# DEFINE PARAMETERS #

epoch_size = 2048
lr = 1e-3
lr_stepsize = 1000
lr_dec = 0.995
batch_size = 16
npoints = 800
radius = 10000
use_cuda = True
max_steps = 500000
num_classes = 5

input_channels = 1
# dendrite, axon, soma, bouton, terminal
output_channels = 5

# set paths
save_root = os.path.expanduser('~/gt/e3training/')
train_path = os.path.expanduser('~/gt/training/')
val_path = os.path.expanduser('~/gt/validation/')

# CREATE NETWORK AND PREPARE DATA SET#

model = ConvPoint(input_channels, output_channels)
if use_cuda:
    model.cuda()

# Transformations to be applied to samples before feeding them to the network
train_transform = clouds.Compose([clouds.RandomRotate(),
                                  clouds.RandomVariation(limits=(-1, 1)),
                                  clouds.Center()])

valid_transform = clouds.Compose([clouds.Center()])

train_ds = TorchSet(train_path, radius, npoints, train_transform, epoch_size=epoch_size)
valid_ds = TorchSet(val_path, radius, npoints, valid_transform, epoch_size=epoch_size)

# PREPARE AND START TRAINING #

# set up optimization
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)

criterion = torch.nn.CrossEntropyLoss()
if use_cuda:
    criterion.cuda()

valid_metrics = {
    'val_accuracy': metrics.accuracy,
    'val_precision': metrics.precision,
    'val_IoU': metrics.bin_iou
}

if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Create trainer
trainer = Trainer3d(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    train_dataset=train_ds,
    valid_dataset=valid_ds,
    batchsize=batch_size,
    num_workers=1,
    num_classes=num_classes,
    save_root=save_root,
    exp_name="ConvPoint_0",
    schedulers={"lr": scheduler},
    valid_metrics=valid_metrics,
)

# Archiving training script, src folder, env info
bk = Backup(script_path=__file__,
            save_path=trainer.save_path).archive_backup()

# Start training
trainer.run(max_steps)
