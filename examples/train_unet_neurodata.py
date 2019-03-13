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
parser.add_argument('-n', '--exp-name', default=None, help='Manually set experiment name')
parser.add_argument(
    '-s', '--epoch-size', type=int, default=100,
    help='How many training samples to process between '
         'validation/preview/extended-stat calculation phases.'
)
parser.add_argument(
    '-m', '--max-steps', type=int, default=500000,
    help='Maximum number of training steps to perform.'
)
parser.add_argument(
    '-t', '--max-runtime', type=int, default=3600 * 24 * 4,  # 4 days
    help='Maximum training time (in seconds).'
)
parser.add_argument(
    '-r', '--resume', metavar='PATH',
    help='Path to pretrained model state dict from which to resume training.'
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

from elektronn3.data import PatchCreator, transforms, utils
from elektronn3.training import Trainer, Backup, DiceLoss
from elektronn3.training import metrics
from elektronn3.models.unet import UNet


torch.manual_seed(0)


# USER PATHS
save_root = os.path.expanduser('~/e3training/mahsaBranchWithElasticTransformAndRGB')
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

max_steps = args.max_steps
max_runtime = args.max_runtime
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
if args.resume is not None:  # Load pretrained network params
    model.load_state_dict(torch.load(os.path.expanduser(args.resume)))

# These statistics are computed from the training dataset.
# Remember to re-compute and change them when switching the dataset.
dataset_mean = (155.291411,)
dataset_std = (42.599973,)

# Transformations to be applied to samples before feeding them to the network
common_transforms = [
    transforms.SqueezeTarget(dim=0),  # Workaround for neuro_data_cdhw
    transforms.Normalize(mean=dataset_mean, std=dataset_std)
]
train_transform = transforms.Compose(common_transforms + [
    #transforms.RandomGrayAugment(channels=[0], prob=0.3),
    # transforms.AdditiveGaussianNoise(sigma=0.1,prob=0.3),
    # transforms.RandomBlurring({'probability': 0.5}),
    #transforms.RandomGaussianBlur(channels=[0], prob=0.3),
    #transforms.ElasticTransform()
])
valid_transform = transforms.Compose(common_transforms + [])

# Specify data set
common_data_kwargs = {  # Common options for training and valid sets.
    'aniso_factor': 2,
    'patch_shape': (48, 96, 96),
    'classes': [0, 1],
}
train_dataset = PatchCreator(
    input_h5data=input_h5data[:2],
    target_h5data=target_h5data[:2],
    train=True,
    epoch_size=args.epoch_size,
    warp=0.5,
    warp_kwargs={
        'sample_aniso': True,
        'perspective': True,
    },
    transform=train_transform,
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
        'warp_amount': 0.8,  # Strength
    },
    transform=valid_transform,
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

valid_metrics = {
    'val_accuracy': metrics.bin_accuracy,
    'val_precision': metrics.bin_precision,
    'val_recall': metrics.bin_recall,
    'val_DSC': metrics.bin_dice_coefficient,
    'val_IoU': metrics.bin_iou,
    'val_AP': metrics.bin_average_precision,  # expensive
    'val_AUROC': metrics.bin_auroc,  # expensive
}

# Class weights for imbalanced dataset
class_weights = torch.tensor([0.2653,  0.7347])

# criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = DiceLoss()

# Create trainer
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
    valid_metrics=valid_metrics,
)

# Archiving training script, src folder, env info
Backup(script_path=__file__,save_path=trainer.save_path).archive_backup()

# Start training
trainer.train(max_steps=max_steps, max_runtime=max_runtime)


# How to re-calculate mean, std and class_weights for other datasets:
#  dataset_mean = utils.calculate_means(train_dataset.inputs)
#  dataset_std = utils.calculate_stds(train_dataset.inputs)
#  class_weights = torch.tensor(utils.calculate_class_weights(train_dataset.targets))
