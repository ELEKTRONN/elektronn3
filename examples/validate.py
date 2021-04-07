"""Example of an offline model validation script.

This example script does not just work as is, but is meant as a **template**
and needs to be adapted to the paths and configurations specific to the data and
training setup.

IMPORTANT: This example assumes that the model to be validated was trained on
the neuro_data_cdhw example dataset. For other trainings, remember to change
the data paths and the normalization values accordingly
"""

import argparse
import os
import pprint
import zipfile
from typing import Callable, Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from elektronn3.data import PatchCreator, transforms
from elektronn3.training import metrics


def validate(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        metrics: Dict[str, Callable] = {}
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    model.eval()
    stats = {name: [] for name in metrics.keys()}
    for batch in tqdm(loader):
        # Everything with a "d" prefix refers to tensors on self.device (i.e. probably on GPU)
        dinp = batch['inp'].to(device, non_blocking=True)
        target = batch['target']
        with torch.no_grad():
            dout = model(dinp)
            out = dout.detach().cpu()
            for name, evaluator in metrics.items():
                stats[name].append(evaluator(target, out))
    for name in metrics.keys():
        stats[name] = np.nanmean(stats[name])
    return stats


def load_model(model_path: str, device: torch.device):
    if not os.path.isfile(model_path):
        raise ValueError(f'Model path {model_path} not found.')
    # TorchScript serialization can be identified by checking if
    #  it's a zip file. Pickled Python models are not zip files.
    #  See https://github.com/pytorch/pytorch/pull/15578/files
    if model_path.endswith('.pts'):
        return torch.jit.load(model_path, map_location=device)
    elif model_path.endswith('.pt'):
        return torch.load(model_path, map_location=device)
    else:
        raise ValueError(f'{model_path} has an unkown file extension. Supported are .pt and .pts')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate.')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--model', help='Path to the network model')
    args = parser.parse_args()

    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    data_root = os.path.expanduser('~/neuro_data_cdhw/')
    input_h5data = [(os.path.join(data_root, f'raw_{i}.h5'), 'raw') for i in range(3)]
    target_h5data = [(os.path.join(data_root, f'barrier_int16_{i}.h5'), 'lab') for i in range(3)]

    aniso_factor = 2  # Anisotropy in z dimension. E.g. 2 means half resolution in z dimension.
    common_data_kwargs = {  # Common options for training and valid sets.
        'aniso_factor': aniso_factor,
        'patch_shape': (44, 88, 88),
        # 'offset': (8, 20, 20),
        'out_channels': 2,
        # 'in_memory': True  # Uncomment to avoid disk I/O (if you have enough host memory for the data)
    }
    norm_mean = (155.291411,)
    norm_std = (42.599973,)
    valid_transform = transforms.Normalize(mean=norm_mean, std=norm_std, inplace=True)

    print('Loading dataset...')
    valid_dataset = PatchCreator(
        input_sources=input_h5data,
        target_sources=target_h5data,
        train=False,
        epoch_size=40,  # How many samples to use for each validation run
        warp_prob=0,
        warp_kwargs={'sample_aniso': aniso_factor != 1},
        transform=valid_transform,
        **common_data_kwargs
    )
    valid_loader = DataLoader(valid_dataset, num_workers=4, pin_memory=True)

    # Validation metrics
    valid_metrics = {}
    for evaluator in [metrics.Accuracy, metrics.Precision, metrics.Recall, metrics.DSC, metrics.IoU]:
        valid_metrics[f'val_{evaluator.name}_mean'] = evaluator()  # Mean metrics
        for c in range(valid_dataset.out_channels):
            valid_metrics[f'val_{evaluator.name}_c{c}'] = evaluator(c)

    print('Loading model...')
    model = load_model(args.model, device)

    print('Calculating metrics...')
    stats = validate(model=model, loader=valid_loader, device=device, metrics=valid_metrics)
    print('Done. Results:\n')
    pprint.pprint(stats, indent=4, width=100, compact=False)
