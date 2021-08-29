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
from typing import Callable, Tuple, Union, Sequence, Optional

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

parser.add_argument('--blackening-size', default=None, nargs='+',type=int,required=True,
    help='Size of block of input data that is set to zero before being fed into the UNet'
)
parser.add_argument('-c', '--criterion', default="L2",type=str,
    help='Loss function'
)
parser.add_argument('--beta', default=1.,type=float,
    help='weighting of input mask in loss'
)
args = parser.parse_args()

# Set up all RNG seeds, set level of determinism
random_seed = args.seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
blackening_size = np.array(args.blackening_size)
criterion_string=args.criterion
deterministic = args.deterministic
beta=args.beta
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

from elektronn3.data import PatchCreator,transforms, utils, get_preview_batch
from elektronn3.data.knossos import KnossosRawData
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

out_channels = 1
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
save_root = os.path.expanduser('/wholebrain/scratch/fkies/e3training/')
os.makedirs(save_root, exist_ok=True)
if os.getenv('CLUSTER') == 'WHOLEBRAIN':  # Use bigger, but private data set
    data_root ='/wholebrain/scratch/j0126/barrier_gt_phil/' 
    # data_root = '/wholebrain/u/mdraw/barrier_gt_phil_r2r_bn/'
    #data_root = '/wholebrain/u/mdraw/barrier_gt_phil_n2v/'
    #data_root_lab = '/wholebrain/songbird/j0251/j0251_72_clahe2/'
    data_root_lab ='wholebrain/scratch/j0126/barrier_gt_phil'
    fnames = sorted([f for f in os.listdir(data_root) if f.endswith('.h5')])
    input_h5data = [(os.path.join(data_root, f), 'raW') for f in fnames]
    target_h5data = [(os.path.join(data_root_lab, f), 'labels') for f in fnames]
    valid_indices = [1, 3, 5, 7]

    # These statistics are computed from the training dataset.
    # Remember to re-compute and change them when switching the dataset.
    dataset_mean = (0.0,)
    dataset_std = (255.,)
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


class NormalizeBoth:
    """Normalizes inputs and targets with supplied per-channel means and stds.

    Args:
        mean: Global mean value(s) of the inputs. Can either be a sequence
            of float values where each value corresponds to a channel
            or a single float value (only for single-channel data).
        std: Global standard deviation value(s) of the inputs. Can either
            be a sequence of float values where each value corresponds to a
            channel or a single float value (only for single-channel data).
        inplace: Apply in-place (works faster, needs less memory but overwrites
            inputs).
        channels: If ``channels`` is ``None``, the change is applied to
            all channels of the input tensor.
            If ``channels`` is a ``Sequence[int]``, change is only applied
            to the specified channels.
            E.g. with mean [a, b], std [x, y] and channels [0, 2],
            following normalizations will be allied:
            - channel 0 with mean a and std x
            - channel 2 with mean b and std y
    """
    def __init__(
            self,
            mean: Union[Sequence[float], float],
            std: Union[Sequence[float], float],
            inplace: bool = False,
            channels: Optional[Sequence[int]] = None
    ):
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.inplace = inplace
        self.channels = channels
        # Unsqueeze first dimensions if mean and scalar are passed as scalars
        if self.mean.ndim == 0:
            self.mean = self.mean[None]
        if self.std.ndim == 0:
            self.std = self.std[None]

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None  # returned without modifications
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.inplace:
            normalized_inp = inp  # Refer to the same memory space
            normalized_target = target
        else:
            normalized_inp = inp.copy()
            normalized_target = target.copy()
        channels = range(inp.shape[0]) if self.channels is None else self.channels
        if not len(channels) == self.mean.shape[0] == self.std.shape[0]:
            raise ValueError(f'mean ({self.mean.shape[0]}) and std ({self.std.shape[0]}) must have the same length as the C '
                             f'axis (number of channels) of the input ({inp.shape[0]}).')
        for c in channels:
            normalized_inp[c] = (inp[c] - self.mean[c]) / self.std[c]
            normalized_target[c] = (target[c] - self.mean[c])/self.std[c]
        return normalized_inp, normalized_target

    def __repr__(self):
        return f'Normalize(mean={self.mean}, std={self.std}, inplace={self.inplace})'




class Lambda:
    """Wraps a function of the form f(x, y) = (x', y') into a transform.

    Args:
        func: A function that takes two arrays and returns a
            tuple of two arrays."""

    def __init__(
            self,
            func: Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
            zerosize: np.ndarray
    ):
        self.func = func
        self.zerosize = zerosize

    def __call__(
            self,
            inp: np.ndarray,
            target: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.func(inp, target, self.zerosize)

def zero_cube(inp,targ,kwargs):
    blackening_size=kwargs
    #print("inp shape: {}".format(inp.shape))
    nc, zs, xs, ys = inp.shape    
    midpos = np.array([zs, xs, ys])//2
    
    if (np.array([zs,xs,ys])<blackening_size).any():
        raise ValueError("""blackening size of center cube is too large for input s     ize:
                        blackening size: {},
                        input size: {},
                        midpos (where is the mid of input tensor?): {} """.format(
                        blackening_size, inp.size(),midpos))
    else:
        #inp[:,:,(midpos[0]-blackening_size[0]//2):(midpos[0]+blackening_size[0]//2),(midpos[1]-blackening_size[1]//2):(midpos[1]+blackening_size[1]//2),(midpos[2]-blackening_size[2]//2):(midpos[2]+blackening_size[2]//2)]=0.

        b0 = blackening_size[0]
        b1 = blackening_size[1]
        b2 = blackening_size[2]
        inp[:,int(midpos[0]-b0//2):int(midpos[0]+b0//2),int(midpos[1]-b1//2):int(midpos[1]+b1//2),int(midpos[2]-b2//2):int(midpos[2]+b2//2)]=0.
        #debug: print(((inp-target)==0).all()) sformations to be applied to samples before feeding them to the network
    return inp, targ

zero_transform = Lambda(zero_cube, blackening_size)

common_transforms = [
    NormalizeBoth(mean=dataset_mean, std=dataset_std),
    transforms.RandomFlip(3),
    zero_transform
]

train_transform = transforms.Compose(common_transforms + [
    # transforms.RandomRotate2d(prob=0.9),
    # transforms.RandomGrayAugment(channels=[0], prob=0.3),
    # transforms.RandomGammaCorrection(gamma_std=0.25, gamma_min=0.25, prob=0.3),
    # transforms.AdditiveGaussianNoise(sigma=0.1, channels=[0], prob=0.3),
])
valid_transform = transforms.Compose(common_transforms + [])

# Specify data set
aniso_factor = 2  # Anisotropy in z dimension. E.g. 2 means half resolution in z dimension.
common_data_kwargs = {  # Common options for training and valid sets.
    'aniso_factor': aniso_factor,
    'patch_shape': (44, 88, 88)
    # 'offset': (8, 20, 20),
    # 'in_memory': True  # Uncomment to avoid disk I/O (if you have enough host memory for the data)
}

train_dataset = KnossosRawData(
    #conf_path='/wholebrain/songbird/j0126/areaxfs_v5/knossosdatasets/mag1/knossos.conf',
    conf_path='/wholebrain/songbird/j0251/j0251_72_clahe2/mag1/knossos.conf',#philipp said to use this dataset
    patch_shape=common_data_kwargs['patch_shape'],
    transform=train_transform,
    epoch_size=args.epoch_size,
    mode='caching',
    cache_size=64,
    cache_reuses=8)

#valid_dataset = KnossosRawData(
#    conf_path='/wholebrain/songbird/j0126/areaxfs_v5/knossosdatasets/mag2/knossos.conf',
#    patch_shape=common_data_kwargs['patch_shape'],
#    train=False,
#    transform=valid_transform,
#    epoch_size=40, #how many samples to use for each validation run
#    mode='caching',
#    cache_size=64,
#    cache_reuses=8)
#valid_dataset = None if not valid_indices else PatchCreator(
#    input_sources=[input_h5data[i] for i in range(len(input_h5data)) if i in valid_indices],
#    target_sources=[target_h5data[i] for i in range(len(input_h5data)) if i in valid_indices],
#    train=False,
#    epoch_size=40,  # How many samples to use for each validation run
#    warp_prob=0,
#    warp_kwargs={'sample_aniso': aniso_factor != 1},
#    transform=valid_transform,
#    **common_data_kwargs
#)

# Use first validation cube for previews. Can be set to any other data source.
#preview_batch = get_preview_batch(
#    h5data=input_h5data[valid_indices[0]],
#    preview_shape=(32, 320, 320),
#)
#preview_batch = KnossosRawData(
#    #conf_path='/wholebrain/songbird/j0126/areaxfs_v5/knossosdatasets/mag1/knossos.conf',
#    conf_path='/wholebrain/songbird/j0251/j0251_72_clahe2/mag1/knossos.conf',#philipp said to use this dataset
#    patch_shape=common_data_kwargs['patch_shape'],
#    transform=train_transform,
#    epoch_size=args.epoch_size,
#    mode='caching',
#    cache_size=64,
#    cache_reuses=8)
knossos_preview_config = {
     'dataset': '/wholebrain/songbird/j0126/areaxfs_v5/knossosdatasets/mag1/knossos.conf',
     'offset': [1000, 1000, 1000],  # Offset (min) coordinates
     'size': [256, 256, 64],  # Size (shape) of the region
     'mag': 1,  # source mag
     'target_mags': [1, 2, 3],  # List of target mags to which the inference results should be written
     'scale_brightness': 255 if os.getenv('CLUSTER') == 'WHOLEBRAIN' else 1.
} 

# Options for the preview inference (see elektronn3.inference.Predictor).
# Attention: These values are highly dependent on model and data shapes!
inference_kwargs = {
    'tile_shape': (32, 64, 64),
    'overlap_shape': (32, 64, 64),
    'offset': None,
    'apply_softmax': False,
    'transform': transforms.Normalize(mean=dataset_mean, std=dataset_std),
}

optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,  # Learning rate is set by the lr_sched below
    momentum=0.9,
    weight_decay=0.5e-4,
)
#optimizer = optim.AdamW(
#    model.parameters(),
#    lr=1e-3,  # Learning rate is set by the lr_sched below
#    weight_decay=0.5e-4,
#)
optimizer = SWA(optimizer)  # Enable support for Stochastic Weight Averaging

# Set to True to perform Cyclical LR range test instead of normal training
#  (see https://arxiv.org/abs/1506.01186, sec. 3.3).
do_lr_range_test = False
if do_lr_range_test:
    # Begin with a very small lr and double it every 1000 steps.
    for grp in optimizer.param_groups:
        grp['lr'] = 1e-7  # Note: lr will be > 1.0 after 24k steps.
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, 1000, 2)
else:
    lr_sched = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-6,
        max_lr=1e-3,
        step_size_up=2000,
        step_size_down=6000,
        cycle_momentum=True if 'momentum' in optimizer.defaults else False
    )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    if lr_sched_state_dict is not None:
        lr_sched.load_state_dict(lr_sched_state_dict)
# lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, 1000, 0.9)

# Validation metrics
valid_metrics = {}
for evaluator in [metrics.Accuracy, metrics.Precision, metrics.Recall, metrics.DSC, metrics.IoU]:
    valid_metrics[f'val_{evaluator.name}_mean'] = evaluator()  # Mean metrics
    for c in range(out_channels):
        valid_metrics[f'val_{evaluator.name}_c{c}'] = evaluator(c)




from scipy.ndimage import distance_transform_edt as dt


class StaticWeightedL1L2(nn.Module):
    """Weighted L1/L2 loss where elements in the blackening_size region are weighted more than
    outside."""

    def __init__(self, loss_string, blackening_size, beta = 1.)->None:
        super().__init__()
        self.loss_string= loss_string
        if self.loss_string == "L1":
            self.criterion = torch.nn.L1Loss(reduction="none")
        elif self.loss_string == "L2":
            self.criterion = torch.nn.MSELoss(reduction="none")
        else:
            raise ValueError("Unknown loss {}".format(self.loss_string))

        self.mask_generator = dt
        self.mask_set = False
        self.register_buffer('blackening_size', torch.as_tensor(blackening_size, dtype = torch.float32))
        self.vol_in = np.prod(blackening_size)
        self.beta = beta
        #to account for the difference in volume, the loss inside the volume will be recaled according to the ratio
        #vol_out/vol_in*beta where factor beta determines how much more the loss inside is supposed to be weighted
        #compared to teh loss outside the blacked region

    def forward(self, out, target):
        if self.mask_set is False:
            test = torch.zeros_like(out)
            #print("input shape: {}".format(test.shape))
            _, nc ,zs, xs, ys = test.shape
            self.vol_out = test[0,0,:,:,:].numel()-self.vol_in
            #print("blackened volume: {}".format(self.vol_in))
            #print("volume outside the blacked cube: {}".format(self.vol_out))
            #print("scale: {}".format(self.vol_out/self.vol_in**2))
            #print("loss marker blackening_size: {}".format(self.blackening_size))
            mid_position_array = np.array([zs, xs, ys])//2
            if (torch.tensor([zs,xs,ys]).cuda()<self.blackening_size).any():
                raise ValueError("""blackening size of center cube is too large for input size:
                                blackening size: {},
                                input size: {},
                                mid_position_array (where is the mid of input tensor?): {} """.format(
                                self.blackening_size, inp.size(),mid_position_array))
            else:
                b0 = self.blackening_size.cpu().numpy()[0]
                b1 = self.blackening_size.cpu().numpy()[1]
                b2 = self.blackening_size.cpu().numpy()[2]
                test[:,:,int(mid_position_array[0]-b0//2):int(mid_position_array[0]+b0//2),int(mid_position_array[1]-b1//2):int(mid_position_array[1]+b1//2),int(mid_position_array[2]-b2//2):int(mid_position_array[2]+b2//2)]=1.
                mask = np.power(self.mask_generator(test.cpu().numpy()),2)*self.beta#*self.vol_out/(self.vol_in**2)
                mask[np.where(mask==0)]=1/2#/self.vol_out
                #print("mask minimum: {}".format(mask.min()))
                #print("mask maximum: {}".format(mask.max()))
                mask = torch.tensor(mask, dtype=torch.float32).to(device=out.device)
                self.register_buffer('mask', torch.as_tensor(mask, dtype=mask.dtype, device=out.device))
                self.mask_set ==True

        err = self.criterion(out, target)
        err *= self.mask
        loss = err.mean()
        return loss


#crossentropy = nn.CrossEntropyLoss(weight=class_weights)
#dice = DiceLoss(apply_softmax=True, weight=class_weights)
#criterion = CombinedLoss([crossentropy, dice], weight=[0.5, 0.5], device=device)


criterion = StaticWeightedL1L2(criterion_string, blackening_size, beta)

# Create trainer
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    train_dataset=train_dataset,
    valid_dataset=None,
    batch_size=8,
    num_workers=2,
    save_root=save_root,
    exp_name=args.exp_name,
    example_input=example_input,
    save_jit=save_jit,
    schedulers={'lr': lr_sched},
    valid_metrics=valid_metrics,
    #preview_batch=preview_batch,
    knossos_preview_config=knossos_preview_config,
    preview_interval=5,
    inference_kwargs=inference_kwargs,
    hparams=hparams,
    # enable_videos=True,  # Uncomment to enable videos in tensorboard
    out_channels=out_channels,
    ipython_shell=args.ipython,
    # extra_save_steps=range(0, max_steps, 10_000),
    # mixed_precision=True,  # Enable to use Apex for mixed precision training
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
