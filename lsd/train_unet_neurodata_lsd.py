#!/usr/bin/env python3

# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert

import argparse
import logging
import os
import random

import torch
from torch import nn
from torch import optim
import numpy as np

parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('-n', '--exp-name', default=None, help='Manually set experiment name')
parser.add_argument(
    '-s', '--epoch-size', type=int, default=800,
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
    '-j', '--jit', metavar='MODE', default='onsave',
    choices=['disabled', 'train', 'onsave'],
    help="""Options:
"disabled": Completely disable JIT (TorchScript) compilation;
"onsave": Use regular Python model for training, but JIT-compile it on-demand for saving training state;
"train": Use JIT-compiled model for training and serialize it on disk."""
)
parser.add_argument('--seed', type=int, default=0, help='Base seed for all RNGs.')
parser.add_argument(
    '--deterministic', action='store_true',
    help='Run in fully deterministic mode (at the cost of execution speed).'
)
parser.add_argument('-i', '--ipython', action='store_true',
    help='Drop into IPython shell on errors or keyboard interrupts.'
)
parser.add_argument('-c', '--criterion', default="L1",type=str,
    help='Loss function'
)
args = parser.parse_args()

# Set up all RNG seeds, set level of determinism
random_seed = args.seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
deterministic = args.deterministic
criterion_string=args.criterion
if deterministic:
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.benchmark = True  # Improves overall performance in *most* cases

# Don't move this stuff, it needs to be run this early to work
import elektronn3
elektronn3.select_mpl_backend('Agg')
logger = logging.getLogger('elektronn3log')
# Write the flags passed to python via argument passer to logfile
# They will appear as "Namespace(arg1=val1, arg2=val2, ...)" at the top of the logfile
logger.debug("Arguments given to python via flags: {}".format(args))

from elektronn3.data import PatchCreator, transforms, utils, get_preview_batch
from elektronn3.training import Trainer, Backup, metrics
from elektronn3.training import SWA
from elektronn3.modules import DiceLoss, CombinedLoss
from elektronn3.models.unet import UNet


if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
logger.info(f'Running on device: {device}')

# You can store selected hyperparams in a dict for logging to tensorboard, e.g.
# hparams = {'n_blocks': 4, 'start_filts': 32, 'planar_blocks': (0,)}
hparams = {}

out_channels = 5 #change this for the lsd_description
model = UNet(
    out_channels=out_channels,
    n_blocks=4,
    start_filts=32,
    planar_blocks=(0,),
    activation='relu',
    normalization='batch',
    # conv_mode='valid',
    # full_norm=False,  # Uncomment to restore old sparse normalization scheme
    # up_mode='resizeconv_nearest',  # Enable to avoid checkerboard artifacts
).to(device)

#model = nn.DataParallel(model)
#model.to(device)

# Example for a model-compatible input.
example_input = torch.ones(1, 1, 32, 64, 64)

save_jit = None if args.jit == 'disabled' else 'script'
if args.jit == 'onsave':
    # Make sure that compilation works at all
    jitmodel = torch.jit.script(model)
elif args.jit == 'train':
    jitmodel = torch.jit.script(model)
    model = jitmodel


# USER PATHS
save_root = os.path.expanduser('/wholebrain/scratch/fkies/e3training/lsd/')
os.makedirs(save_root, exist_ok=True)
if os.getenv('CLUSTER') == 'WHOLEBRAIN':  # Use bigger, but private data set
    data_root = '/wholebrain/scratch/j0126/barrier_gt_phil/'
    # data_root = '/wholebrain/u/mdraw/barrier_gt_phil_r2r_bn/'
    # data_root = '/wholebrain/u/mdraw/barrier_gt_phil_n2v/'
    data_root_lab = '/wholebrain/scratch/j0126/barrier_gt_phil/'
    fnames = sorted([f for f in os.listdir(data_root) if f.endswith('.h5')])
    input_h5data = [(os.path.join(data_root, f), 'raW') for f in fnames]
    target_h5data = [(os.path.join(data_root_lab, f), 'labels') for f in fnames]
    valid_indices = [1, 3, 5, 7]

    # These statistics are computed from the training dataset.
    # Remember to re-compute and change them when switching the dataset.
    dataset_mean = (0.6170815,)
    dataset_std = (0.15687169,)
    # Class weights for imbalanced dataset
    class_weights = torch.tensor([0.2808, 0.7192]).to(device)
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
    class_weights = torch.tensor([0.2653, 0.7347]).to(device)


max_steps = args.max_steps
max_runtime = args.max_runtime

optimizer_state_dict = None
lr_sched_state_dict = None
if args.resume is not None:  # Load pretrained network
    pretrained = os.path.expanduser(args.resume)
    logger.info(f'Loading model from {pretrained}')
    if pretrained.endswith('.pt'):  # nn.Module
        model = torch.load(pretrained, map_location=device)
    elif pretrained.endswith('.pts'):  # ScriptModule
        model = torch.jit.load(pretrained, map_location=device)
    elif pretrained.endswith('.pth'):
        state = torch.load(pretrained)
        model.load_state_dict(state['model_state_dict'], strict=False)
        optimizer_state_dict = state.get('optimizer_state_dict')
        lr_sched_state_dict = state.get('lr_sched_state_dict')
        if optimizer_state_dict is None:
            logger.warning('optimizer_state_dict not found.')
        if lr_sched_state_dict is None:
            logger.warning('lr_sched_state_dict not found.')
    else:
        raise ValueError(f'{pretrained} has an unkown file extension. Supported are: .pt, .pts and .pth')

# Transformations to be applied to samples before feeding them to the network

from lsd import LSDGaussVdtCom
dataset_mean = (0.,)
dataset_std = (255.,)
local_shape_descriptor = LSDGaussVdtCom()
common_transforms = [
    transforms.Normalize(mean=dataset_mean, std=dataset_std),
#    transforms.AdditiveGaussianNoise(),
#    transforms.RandomBrightnessContrast(),
#    transforms.RandomGammaCorrection(),
#    transforms.RandomFlip(ndim_spatial=3),
    local_shape_descriptor

]
train_transform = transforms.Compose(common_transforms + [
])
valid_transform = transforms.Compose(common_transforms + [])

# Specify data set
aniso_factor = 2  # Anisotropy in z dimension. E.g. 2 means half resolution in z dimension.
common_data_kwargs = {  # Common options for training and valid sets.
    'aniso_factor': aniso_factor,
    'patch_shape': (44, 88, 88),
    # 'offset': (8, 20, 20),
    # 'in_memory': True  # Uncomment to avoid disk I/O (if you have enough host memory for the data)
}
from new_knossos import KnossosLabelsNozip
train_dataset = KnossosLabelsNozip(
    conf_path_raw_data='/wholebrain/songbird/j0251/j0251_72_clahe2/mag1/knossos.conf',#philipp said to use this dataset
    conf_path_label='/ssdscratch/songbird/j0251/segmentation/j0251_72_seg_20210127_agglo2/knossos.pyk.conf',
    patch_shape=common_data_kwargs['patch_shape'],
    transform=train_transform,
    epoch_size=args.epoch_size,
    raw_mode='caching',
    raw_cache_size = 64,
    raw_cache_reuses = 8)


valid_dataset = KnossosLabelsNozip(
    conf_path_raw_data='/wholebrain/songbird/j0251/j0251_72_clahe2/mag1/knossos.conf',#philipp said to use this dataset
    conf_path_label='/ssdscratch/songbird/j0251/segmentation/j0251_72_seg_20210127_agglo2/knossos.pyk.conf',
    patch_shape=common_data_kwargs['patch_shape'],
    transform=train_transform,
    epoch_size=args.epoch_size,
    raw_mode='caching',
    raw_cache_size = 64,
    raw_cache_reuses = 8,
    threshold_background_fraction = 1.0)

# Use first validation cube for previews. Can be set to any other data source.
preview_batch = get_preview_batch(
    h5data=input_h5data[valid_indices[0]],
    preview_shape=(32, 320, 320),
)

# knossos_preview_config = {
#     'dataset': '/wholebrain/songbird/j0126/areaxfs_v5/knossosdatasets/mag1/knossos.conf',
#     'offset': [1000, 1000, 1000],  # Offset (min) coordinates
#     'size': [256, 256, 64],  # Size (shape) of the region
#     'mag': 1,  # source mag
#     'target_mags': [1, 2, 3],  # List of target mags to which the inference results should be written
#     'scale_brightness': 255 if os.getenv('CLUSTER') == 'WHOLEBRAIN' else 1.
# }

# Options for the preview inference (see elektronn3.inference.Predictor).
# Attention: These values are highly dependent on model and data shapes!
inference_kwargs = {
    'tile_shape': (32, 64, 64),
    'overlap_shape': (32, 64, 64),
    'offset': None,
    'apply_softmax': False,
    'transform': valid_transform,
}

optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,  # Learning rate is set by the lr_sched below
    momentum=0.9,
    weight_decay=0.5e-4,
)
#optimizer = SWA(optimizer)  # Enable support for Stochastic Weight Averaging

# Set to True to perform Cyclical LR range test instead of normal training
#  (see https://arxiv.org/abs/1506.01186, sec. 3.3).
#do_lr_range_test = False
#if do_lr_range_test:
#    # Begin with a very small lr and double it every 1000 steps.
#    for grp in optimizer.param_groups:
#        grp['lr'] = 1e-7  # Note: lr will be > 1.0 after 24k steps.
#    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, 1000, 2)
#else:
#    lr_sched = torch.optim.lr_scheduler.CyclicLR(
#        optimizer,
#        base_lr=1e-6,
#        max_lr=1e-3,
#        step_size_up=2000,
#        step_size_down=6000,
#        cycle_momentum=True if 'momentum' in optimizer.defaults else False
#    )
#    if optimizer_state_dict is not None:
#        optimizer.load_state_dict(optimizer_state_dict)
#    if lr_sched_state_dict is not None:
#        lr_sched.load_state_dict(lr_sched_state_dict)
lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, 1000, 0.9)

from newmetric import LossFunctorMetric

lf_vdt = LossFunctorMetric('L1', [0,3])
lf_vdt_norm = LossFunctorMetric('L1', [3,3])
lf_gauss_div = LossFunctorMetric('L1', [4,4])

# Validation metrics
valid_metrics = {}
for evaluator in []:
    valid_metrics[f'val_{evaluator.name}_mean'] = evaluator()  # Mean metrics
    for c in range(out_channels):
        valid_metrics[f'val_{evaluator.name}_c{c}'] = evaluator(c)

#class CustomLSDLoss(nn.Module):
#    def __init__(self, lossstr):
#        super().__init__()
#        if lossstr == 'L2':
#            self.criterion = torch.nn.MSELoss()
#        elif lossstr == 'L1':
#            self.criterion = torch.nn.L1Loss()
#        else:
#            raise NotImplementedError
#    
#    def forward(self, out, targ):
#        return self.criterion(out[:,:3], targ[:,:3]) + self.criterion(out[:,3], targ[:,3]) + self.criterion(out[:,4], targ[:,4]) + self.criterion(torch.norm(out, dim=1, keepdim=True), targ[:,3])


if criterion_string == "L1":
    criterion = torch.nn.L1Loss()
elif criterion_string == "L2":
    criterion = torch.nn.MSELoss()
else:
    raise NotImplementedError


#criterion = CustomLSDLoss(criterion_string)

batch_size = 8
num_workers = 2
# Create trainer
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    save_root=save_root,
    exp_name=args.exp_name,
    example_input=example_input,
    save_jit=save_jit,
    schedulers={'lr': lr_sched},
    valid_metrics=valid_metrics,
    preview_batch=preview_batch,
    # knossos_preview_config=knossos_preview_config,
    preview_interval=5,
    inference_kwargs=inference_kwargs,
    hparams=hparams,
    # enable_videos=True,  # Uncomment to enable videos in tensorboard
    out_channels=out_channels,
    ipython_shell=args.ipython,
    # extra_save_steps=range(0, max_steps, 10_000),
    # mixed_precision=True,  # Enable to use Apex for mixed precision training
)

from elektronn3.training.trainer import _worker_init_fn

from torch.utils.data import DataLoader

trainer.train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True,
    timeout=60 if num_workers > 0 else 0,
    worker_init_fn=_worker_init_fn, persistent_workers=True
)

if args.deterministic:
    assert trainer.num_workers <= 1, 'num_workers > 1 introduces indeterministic behavior'

# Archiving training script, src folder, env info
Backup(script_path=__file__,save_path=trainer.save_path).archive_backup()

# Start training
trainer.run(max_steps=max_steps, max_runtime=max_runtime)


# How to re-calculate mean, std and class_weights for other datasets:
#  dataset_mean = utils.calculate_means(train_dataset.inputs)
#  dataset_std = utils.calculate_stds(train_dataset.inputs)
#  class_weights = torch.tensor(utils.calculate_class_weights(train_dataset.targets))
