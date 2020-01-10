# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Jonathan Klimesch

""" Inference script to use model in save_root + experiment_name + state_dict and output full cells for comparisons
    with original cell files """

import os
import math
import glob
import torch
import ipdb
import argparse
import numpy as np
import morphx.processing.clouds as clouds
from morphx.classes.pointcloud import PointCloud
from elektronn3.models.convpoint import SegSmall, SegBig
from morphx.data.cloudset import CloudSet
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

# define parameters
name = args.na
npoints = args.sp
radius = args.ra
n_classes = args.cl
batch_size = 16

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
    model = SegBig(input_channels, output_channels)
else:
    model = SegSmall(input_channels, output_channels)

full = torch.load(os.path.join(folder, args.sd))
model.load_state_dict(full['model_state_dict'])
model.eval()


# PREPARE DATA SET #

# Transformations to be applied to the samples before feeding them to the network
transform = clouds.Compose([clouds.Normalization(radius), clouds.Center()])
ds = CloudSet('', radius, npoints, class_num=n_classes)


# PREPARE AND START TRAINING #

for file in files:
    slashs = [pos for pos, char in enumerate(file) if char == '/']
    name = file[slashs[-1] + 1:-4]

    # switch cloudset to selected file
    hc = clouds.load_cloud(file)
    ds.activate_single(hc)
    chunk_build = None

    for i in tqdm(range(math.ceil(len(ds) / batch_size))):
        t_pts = torch.zeros((batch_size, npoints, 3))
        t_features = torch.ones((batch_size, npoints, 1))
        cloud_arr = []

        # load batch_size samples, save original sample for later evaluation, apply transformation and save samples
        # into torch batch
        for j in range(batch_size):
            cloud = ds[0]
            if cloud is not None:
                cloud_arr.append(cloud)
                t_cloud = PointCloud(cloud.vertices, cloud.labels)
                transform(t_cloud)
                t_pts[j] = torch.from_numpy(t_cloud.vertices)
            else:
                break

        # apply model to batch of samples
        outputs = model(t_features, t_pts)
        output_np = outputs.cpu().detach().numpy()
        output_np = np.argmax(output_np, axis=2).copy()

        # map predictions onto original samples and merge all samples into full object
        for j in range(batch_size):
            if j < len(cloud_arr):
                chunk = PointCloud(cloud_arr[j].vertices, output_np[j])

                # add processed chunks incrementally to full cell
                if chunk_build is None:
                    chunk_build = chunk
                else:
                    chunk_build = clouds.merge_clouds(chunk_build, chunk)

    clouds.save_cloud(chunk_build, val_examples, name)
