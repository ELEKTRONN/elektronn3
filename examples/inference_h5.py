#!/usr/bin/env python3

# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch

"""
Example script that shows how ``elektronn3.inference.Predictor`` can be used
with a trained PyTorch model of a 3D CNN to perform inference on a 3D HDF5 file
and write the results to another HDF5 file.

This example script does not just work as is, but is meant as a **template**
and needs to be adapted to the paths and configurations specific to the data and
training setup.

IMPORTANT: Especially remember to adapt ``norm_mean`` and ``norm_std``!
Forgetting these will not result in a crash or warning, but your predictions
will be damaged.
"""

import argparse
import logging
import os
from typing import Optional, Tuple

import h5py
import numpy as np
import torch

import elektronn3

elektronn3.select_mpl_backend('Agg')
logger = logging.getLogger('elektronn3log')

from elektronn3.data import transforms
from elektronn3.inference import Predictor

parser = argparse.ArgumentParser(description='Inference from and to HDF5 files.')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
args = parser.parse_args()


if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
logger.info(f'Running on device: {device}')


# Important: Change these variables according to your needs. There are no reasonable default values.

# Change this to the normalization params from the training
norm_mean: Tuple[float, ...] = (0.6170815,)
norm_std: Tuple[float, ...] = (0.15687169,)

# These should be chosen for maximum GPU memory usage but without running out of memory.
tile_shape: Tuple[int, ...] = (64, 128, 128)
overlap_shape: Tuple[int, ...] = (32, 64, 64)

apply_softmax: bool = True  # Disable this if you don't want to apply softmax on the model outputs.
float16: bool = False  # Save memory by setting this to True. Can also improve speed on some setups.
out_channels: int = 2  # Adjust this if your model outputs more channels/classes
export_channel: int = 1  # Specify a channel whose contents will be written to the output file
model_path: str = '~/e3training/base/model_best.pts'  # Change this to the model you want to use.
inpath: str = '~/neuro_data_cdhw/raw_0.h5'  # Path to the file on which to run inference.
inkey: str = 'raw'  # Change this to the name of the HDF5 dataset from which inputs should be read
outpath: Optional[str] = None  # 'path/to/outputfile.h5'  # Default: Write to a file next to infile
outkey = 'out'
out_dtype = np.uint8

model_path = os.path.expanduser(model_path)
inpath = os.path.expanduser(inpath)
if outpath is None:
    r, e = os.path.splitext(inpath)
    outpath = f'{r}_out{e}'

logger.info(f'\nLoading input from {inpath}[{inkey}]...')
with h5py.File(inpath, 'r') as infile:
    inp = infile[inkey][()]
logger.info(f'Input: shape={inp.shape}, dtype={inp.dtype}')

inp = inp.astype(np.float32)[None]  # Prepend singleton batch dim
if inp.ndim == 4:
    inp = inp[None]  # Assume implicit channel dim, make it explicit

# Normalize input because the network was trained on normalized data
logger.info(f'Using mean={norm_mean}, std={norm_std} for input normalization...')
logger.info(' (↑ Make sure these are the same as the values for training the model!)')
transform = transforms.Normalize(mean=norm_mean, std=norm_std, inplace=True)

out_shape = (out_channels, *inp.shape[2:])  # Assuming in_shape = out_shape
logger.info(f'Loading model file {model_path} and setting up Predictor...')
predictor = Predictor(
    model=model_path,
    device=device,
    batch_size=1,
    tile_shape=tile_shape,
    overlap_shape=overlap_shape,
    out_shape=out_shape,
    apply_softmax=apply_softmax,
    float16=float16,
    transform=transform,
    verbose=True
)
logger.info('\nPredicting...')
out: torch.Tensor = predictor.predict(inp)
logger.info('\nConverting outputs to numpy representation and slicing relevant data...')
out_np: np.ndarray = out.to(torch.float32).numpy()

if export_channel is not None:  # Select only the export_channel channel (D, H, W)
    logger.info(f'Only using output channel (class) {export_channel}, discarding other channels.')
    out_export = out_np[0][export_channel]
else:  # Export all channels (C, D, H, W)
    out_export = out_np[0]

if out_dtype == np.uint8:
    out_min, out_max = np.min(out_export), np.max(out_export)
    if out_min < -0.01 or out_max > 1.01:
        logger.error(f'min={out_min}, max={out_max} before rescaling. Please check your setup!')
        logger.error('Conversion to uint8 would lead to data corruption.')
    logger.info('Scaling outputs from [0, 1] to [0, 255] range...')
    out_export = (out_export * 255).astype(out_dtype)

logger.info(f'Output: shape={out_export.shape}, dtype={out_export.dtype}')
logger.info(f'Writing output to {outpath}[{outkey}]...')
with h5py.File(outpath, 'w') as outfile:
    outfile.create_dataset(outkey, data=out_export, dtype=out_dtype)
logger.info('Done.')
