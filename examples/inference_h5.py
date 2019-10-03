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
norm_mean: Tuple[float, ...] = (0., )
norm_std: Tuple[float, ...] = (255., )

# These should be chosen for maximum GPU memory usage but without running out of memory.
tile_shape: Tuple[int, ...] = (64, 64, 64)
overlap_shape: Tuple[int, ...] = (32, 32, 32)

apply_softmax: bool = True  # Disable this if you don't want to apply softmax on the model outputs.
float16: bool = False  # Save memory by setting this to True. Can also improve speed on some setups.
num_out_channels: int = 2  # Adjust this if your model outputs more channels/classes
export_channel: int = 1  # Specify a channel whose contents will be written to the output file
model_path: str = 'path/to/model.pt'
inpath: str = 'path/to/inputfile.h5'
inkey: str = 'raw'  # Change this to the name of the HDF5 dataset from which inputs should be read
outpath: Optional[str] = None  #  'path/to/outputfile.h5'
outkey = 'out'

inpath = os.path.expanduser(inpath)
if outpath is None:
    r, e = os.path.splitext(inpath)
    outpath = f'{r}_out{e}'


with h5py.File(inpath, 'r') as infile:
    inp = infile[inkey]

inp = inp.astype(np.float32)[None]  # Prepend singleton batch dim
if inp.ndim == 4:
    inp = inp[None]  # Assume implicit channel dim, make it explicit

# Normalize input because the network was trained on normalized data
transform = transforms.Normalize(mean=norm_mean, std=norm_std, inplace=True)

out_shape = (num_out_channels, *inp.shape[2:])  # Assuming in_shape = out_shape
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
out: torch.Tensor = predictor.predict(inp)
out_np: np.ndarray = out.to(torch.float32).numpy()
out_export = out_np[0][export_channel]  # Select only one channel./

with h5py.File(outpath, 'w') as outfile:
    outfile.create_dataset(outkey, data=out_np)