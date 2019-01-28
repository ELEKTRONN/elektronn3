#!/usr/bin/env python3

# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert

import argparse
import os
import _pickle

import torch
from torch import nn
from torch import optim

# TODO: Make torch and numpy RNG seed configurable

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
    help='Path to pretrained model state dict or a compiled and saved '
         'ScriptModule from which to resume training.'
)
parser.add_argument(
    '-d', '--disable-trace', action='store_true',
    help='Disable tracing JIT compilation of the model.'
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

from elektronn3.data import PatchCreator, transforms, utils, get_preview_batch
from elektronn3.training import Trainer, Backup, DiceLoss, metrics, Padam
from elektronn3.models.unet import UNet


model = UNet(
    n_blocks=3,
    start_filts=32,
    planar_blocks=(0,),
    activation='relu',
    batch_norm=True,
    # conv_mode='valid',
    adaptive=True  # Experimental. Disable if results look weird.
).to(device)
if not args.disable_trace:
    x = torch.randn(1, 1, 32, 64, 64, device=device)
    model = torch.jit.trace(model, x)


# USER PATHS
save_root = os.path.expanduser('~/e3training/')
os.makedirs(save_root, exist_ok=True)
if os.getenv('CLUSTER') == 'WHOLEBRAIN':  # Use bigger, but private data set
    data_root = '/wholebrain/scratch/j0126/barrier_gt_phil/'
    fnames = [f for f in os.listdir(data_root) if f.endswith('.h5')]
    input_h5data = [(os.path.join(data_root, f), 'raW') for f in fnames]
    target_h5data = [(os.path.join(data_root, f), 'labels') for f in fnames]
    valid_indices = [1, 3, 5, 7]

    # These statistics are computed from the training dataset.
    # Remember to re-compute and change them when switching the dataset.
    dataset_mean = (0.6170815,)
    dataset_std = (0.15687169,)
    # Class weights for imbalanced dataset
    class_weights = torch.tensor([0.2808, 0.7192])
else:  # Use publicly available neuro_data_cdhw dataset
    data_root = os.path.expanduser('~/neuro_data_cdhw/')
    input_h5data = [
        (os.path.join(data_root, f'raw_{i}.h5'), 'raw')
        for i in range(3)
    ]
    target_h5data = [
        (os.path.join(data_root, f'barrier_int16_{i}.h5'), 'lab')
        for i in range(3)
    ]
    valid_indices = [2]

    dataset_mean = (155.291411,)
    dataset_std = (42.599973,)
    class_weights = torch.tensor([0.2653, 0.7347])

max_steps = args.max_steps
max_runtime = args.max_runtime

if args.resume is not None:  # Load pretrained network
    try:  # Assume it's a state_dict for the model
        model.load_state_dict(torch.load(os.path.expanduser(args.resume)))
    except _pickle.UnpicklingError as exc:
        # Assume it's a complete saved ScriptModule
        model = torch.jit.load(os.path.expanduser(args.resume), map_location=device)

# Transformations to be applied to samples before feeding them to the network
common_transforms = [
    transforms.SqueezeTarget(dim=0),  # Workaround for neuro_data_cdhw
    transforms.Normalize(mean=dataset_mean, std=dataset_std)
]
train_transform = transforms.Compose(common_transforms + [
    # transforms.RandomGrayAugment(channels=[0], prob=0.3),
    # transforms.RandomGammaCorrection(gamma_std=0.25, gamma_min=0.25, prob=0.3),
    # transforms.AdditiveGaussianNoise(sigma=0.1, channels=[0], prob=0.3),
    # transforms.RandomBlurring({'probability': 0.5})
])
valid_transform = transforms.Compose(common_transforms + [])

# Specify data set
common_data_kwargs = {  # Common options for training and valid sets.
    'aniso_factor': 2,
    'patch_shape': (48, 96, 96),
    # 'offset': (8, 20, 20),
    'num_classes': 2,
}
train_dataset = PatchCreator(
    input_h5data=[input_h5data[i] for i in range(len(input_h5data)) if i not in valid_indices],
    target_h5data=[target_h5data[i] for i in range(len(input_h5data)) if i not in valid_indices],
    train=True,
    epoch_size=args.epoch_size,
    warp_prob=0.2,
    warp_kwargs={
        'sample_aniso': True,
        'perspective': True,
        'warp_amount': 0.1,
    },
    transform=train_transform,**common_data_kwargs
)
valid_dataset = PatchCreator(
    input_h5data=[input_h5data[i] for i in range(len(input_h5data)) if i in valid_indices],
    target_h5data=[target_h5data[i] for i in range(len(input_h5data)) if i in valid_indices],
    train=False,
    epoch_size=10,  # How many samples to use for each validation run
    warp_prob=0,
    transform=valid_transform,
    **common_data_kwargs
)

# Use first validation cube for previews. Can be set to any other data source.
preview_batch = get_preview_batch(
    h5data=input_h5data[valid_indices[0]],
    preview_shape=(32, 320, 320),
    transform=transforms.Normalize(mean=dataset_mean, std=dataset_std)
)

# Set up optimization
# optimizer = optim.Adam(
#     model.parameters(),
#     weight_decay=0.5e-4,
#     lr=0.0004,
#     amsgrad=True
# )
optimizer = Padam(
    model.parameters(),
    lr=0.1,
    weight_decay=0.5e-4,
    partial=1/4,
)

# All these metrics assume a binary classification problem. If you have
#  non-binary targets, remember to adapt the metrics!
valid_metrics = {
    'val_accuracy': metrics.bin_accuracy,
    'val_precision': metrics.bin_precision,
    'val_recall': metrics.bin_recall,
    'val_DSC': metrics.bin_dice_coefficient,
    'val_IoU': metrics.bin_iou,
    # 'val_AP': metrics.bin_average_precision,  # expensive
    # 'val_AUROC': metrics.bin_auroc,  # expensive
}


# criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = DiceLoss(apply_softmax=True, weight=class_weights)

# Create trainer
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    batchsize=1,
    num_workers=1,
    save_root=save_root,
    exp_name=args.exp_name,
    # schedulers={"lr": optim.lr_scheduler.StepLR(optimizer, 1000, 0.995)},
    valid_metrics=valid_metrics,
    preview_batch=preview_batch,
    preview_interval=5,
    offset=train_dataset.offset,
    apply_softmax_for_prediction=True,
    num_classes=train_dataset.num_classes,
    # TODO: Tune these:
    preview_tile_shape=(32, 64, 64),
    preview_overlap_shape=(32, 64, 64)
)

# Archiving training script, src folder, env info
Backup(script_path=__file__,save_path=trainer.save_path).archive_backup()

# Start training
trainer.run(max_steps=max_steps, max_runtime=max_runtime)


# How to re-calculate mean, std and class_weights for other datasets:
#  dataset_mean = utils.calculate_means(train_dataset.inputs)
#  dataset_std = utils.calculate_stds(train_dataset.inputs)
#  class_weights = torch.tensor(utils.calculate_class_weights(train_dataset.targets))
