# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Jonathan Klimesch

import os
import torch
import argparse
import numpy as np
import convpoint_dev.obsolet.metrics as metrics
import morphx.processing.clouds as clouds
from morphx.classes.pointcloud import PointCloud
from sklearn.metrics import confusion_matrix
from elektronn3.models.convpoint import SegSmall, SegBig
from morphx.data.torchset import TorchSet
from tqdm import tqdm

# PARSE PARAMETERS #

parser = argparse.ArgumentParser(description='Validate a network.')
parser.add_argument('--na', type=str, required=True, help='Experiment name')
parser.add_argument('--vp', type=str, required=True, help='Validation path')
parser.add_argument('--sr', type=str, required=True, help='Save root')
parser.add_argument('--sd', type=str, required=True, help='State dict name')
parser.add_argument('--bs', type=int, default=16, help='Batch size')
parser.add_argument('--sp', type=int, default=1000, help='Number of sample points')
parser.add_argument('--ra', type=int, default=10000, help='Radius')
parser.add_argument('--cl', type=int, default=5, help='Number of classes')
parser.add_argument('--co', action='store_true', help='Disable CUDA')
parser.add_argument('--big', action='store_true', help='Use big SegBig Convpoint network')

args = parser.parse_args()

# SET UP ENVIRONMENT #

use_cuda = not args.co

if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# define parameters
name = args.na
batch_size = args.bs
npoints = args.sp
radius = args.ra
n_classes = args.cl

# set paths
val_path = os.path.expanduser(args.vp)
save_root = os.path.expanduser(args.sr)
folder = save_root + name + '/'
val_examples = folder + 'val_examples/'

logs = open(os.path.join(folder, "validation.txt"), "w")

logs.write("Name: " + name + '\n')
logs.write("Batch size: " + str(batch_size) + '\n')
logs.write("Number of sample points: " + str(npoints) + '\n')
logs.write("Radius: " + str(radius) + '\n')
logs.write("Number of classes: " + str(n_classes) + '\n')
logs.write("Validation data: " + val_path + '\n')
logs.write("\n###--- Validation results: ---###\n\n")
logs.flush()

# CREATE NETWORK #

input_channels = 1
# dendrite, axon, soma, bouton, terminal
output_channels = args.cl

if args.big:
    model = SegBig(input_channels, output_channels).to(device)
else:
    model = SegSmall(input_channels, output_channels).to(device)

full = torch.load(os.path.join(folder, args.sd))
model.load_state_dict(full['model_state_dict'])

if use_cuda:
    model.cuda()

# PREPARE DATA SET #

# Transformations to be applied to samples before feeding them to the network
val_transform = clouds.Compose([clouds.Normalization(radius), clouds.RandomRotate(), clouds.Center()])

ds = TorchSet(val_path, radius, npoints, val_transform, class_num=n_classes)
train_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=1)

# PREPARE AND START TRAINING #

model.eval()
with torch.no_grad():
    cm = np.zeros((n_classes, n_classes))
    t = tqdm(train_loader, ncols=120)
    batch_num = 0
    for batch in t:
        pts = batch['pts']
        features = batch['features']
        lbs = batch['target']

        features.to(device)
        pts.to(device)
        lbs.to(device)

        outputs = model(features, pts)

        outputs_np = outputs.cpu().detach().numpy()
        output_np = np.argmax(outputs_np, axis=2).copy()
        target_np = lbs.cpu().numpy().copy()

        cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(n_classes)))
        cm += cm_

        # save results for later visualization
        results = []
        for i in range(pts.size(0)):
            orig = PointCloud(pts[i].cpu().numpy(), labels=target_np[i])
            pred = PointCloud(pts[i].cpu().numpy(), labels=output_np[i])
            results.append(orig)
            results.append(pred)

        clouds.save_cloudlist(results, val_examples, 'batch_{}'.format(batch_num))

        batch_num += 1

    oa = "{:.3f}".format(metrics.stats_overall_accuracy(cm))
    aa = "{:.3f}".format(metrics.stats_accuracy_per_class(cm)[0])

    # write the logs
    logs.write("{} {}\n".format(oa, aa))
    logs.flush()

logs.close()
