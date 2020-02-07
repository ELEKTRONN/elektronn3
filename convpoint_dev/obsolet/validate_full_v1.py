# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Jonathan Klimesch

""" Inference script to use model in save_root + experiment_name + state_dict and output full cells for comparisons
    with original cell files """

import os
import glob
import torch
import random
import argparse
import numpy as np
import morphx.processing.clouds as clouds
from morphx.classes.pointcloud import PointCloud
from elektronn3.models.convpoint import SegSmall, SegNoBatch
from morphx.data.torchset import TorchSet
from tqdm import tqdm


# PARSE PARAMETERS #

parser = argparse.ArgumentParser(description='Validate a network.')
parser.add_argument('--na', type=str, required=True, help='Experiment name')
parser.add_argument('--vp', type=str, required=True, help='Path to validation data')
parser.add_argument('--sr', type=str, required=True, help='Save root')
parser.add_argument('--sd', type=str, required=True, help='State dict name')
parser.add_argument('--sp', type=int, default=1000, help='Number of sample points')
parser.add_argument('--ra', type=int, default=10000, help='Radius')
parser.add_argument('--cl', type=int, default=5, help='Number of classes')
parser.add_argument('--big', action='store_true', help='Use big SegBig Convpoint network')

args = parser.parse_args()


# SET UP ENVIRONMENT #

random_seed = 4
np.random.seed(random_seed)
random.seed(random_seed)

# define parameters
name = args.na
npoints = args.sp
radius = args.ra
n_classes = args.cl

# set paths (full validations get saved to saved_root + name + full_validation)
val_path = os.path.expanduser(args.vp)
save_root = os.path.expanduser(args.sr)
folder = save_root + name + '/'
val_examples = folder + 'full_validation/'

files = glob.glob(val_path + '*.pkl')


# LOAD TRAINED NETWORK #

input_channels = 1
# dendrite, axon, soma, bouton, terminal
output_channels = args.cl

if args.big:
    model = SegNoBatch(input_channels, output_channels)
else:
    model = SegSmall(input_channels, output_channels)

full = torch.load(os.path.join(folder, args.sd))
model.load_state_dict(full['model_state_dict'])
model.eval()


# PREPARE DATA SET #

# Transformations to be applied to one half of the samples before feeding them to the network
transform = clouds.Compose([clouds.Normalization(radius), clouds.RandomRotate(), clouds.Center()])

# Define two datasets, one for applying the same transformation as during training (most important: center()). The
# transformed data gets fed into the network. The other dataset doesn't apply any transformation in order to stitch
# together the original cell afterwards.
t_ds = TorchSet('', radius, npoints, transform=transform, class_num=n_classes)
ds = TorchSet('', radius, npoints, class_num=n_classes)


# PREPARE AND START TRAINING #

for file in files:
    slashs = [pos for pos, char in enumerate(file) if char == '/']
    name = file[slashs[-1] + 1:-4]

    # switch cloudset to selected file
    hc = clouds.load_cloud(file)

    # Switch both dataset to processing only the given cloud
    t_ds.activate_single(hc)
    ds.activate_single(hc)

    # define pytorch loaders for each TorchSet (cannot be defined before because they need to know the sizes of the
    # datasets which only get set after activate_single)
    t_loader = torch.utils.data.DataLoader(t_ds, batch_size=16, shuffle=False, num_workers=1)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=1)

    # load data from both datasets, feed transformed points to network and transfer output to untransformed points
    chunk_build = None
    it = iter(loader)
    batch_counter = 0
    for t_batch in tqdm(t_loader):
        t_pts = t_batch['pts']
        t_features = t_batch['features']
        t_lbs = t_batch['target']

        if t_pts is None:
            continue
        else:
            batch = next(it)
            pts = batch['pts']
            features = batch['features']
            lbs = batch['target']

            with torch.no_grad():
                outputs = model(t_features, t_pts)
                output_np = outputs.cpu().detach().numpy()
                output_np = np.argmax(output_np, axis=2).copy()

                # add processed chunks incrementally to full cell
                chunks = []
                for i in range(t_pts.size(0)):
                    t_inp = PointCloud(t_pts[i].cpu().detach().numpy(), t_lbs[i].cpu().detach().numpy())
                    t_out = PointCloud(t_pts[i].cpu().detach().numpy(), output_np[i])

                    inp = PointCloud(pts[i].cpu().detach().numpy(), lbs[i].cpu().detach().numpy())

                    chunks.append(inp)
                    chunks.append(t_inp)

                    chunk = PointCloud(pts[i], output_np[i])
                    if chunk_build is None:
                        chunk_build = chunk
                    else:
                        chunk_build = clouds.merge_clouds(chunk_build, chunk)

                clouds.save_cloudlist(chunks, val_examples, name+'_{}'.format(batch_counter))
                batch_counter += 1

    clouds.save_cloud(chunk_build, val_examples, name)
