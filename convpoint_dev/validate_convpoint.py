# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Jonathan Klimesch

import os
import torch
import pickle
import numpy as np
import convpoint_dev.metrics as metrics
import morphx.processing.clouds as clouds
from morphx.classes.pointcloud import PointCloud
from sklearn.metrics import confusion_matrix
from elektronn3.models.convpoint import ConvPoint
from morphx.data.torchset import TorchSet
from tqdm import tqdm

# SET UP ENVIRONMENT #

use_cuda = False

if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# define parameters
epoch_size = 2048
milestones = [60, 120]
lr = 1e-3
batch_size = 16
npoints = 800
radius = 10000
n_classes = 5

# set paths
val_path = os.path.expanduser('~/gt/validation/')
save_root = os.path.expanduser('~/gt/simple_training/')
folder = os.path.join(save_root, "SegSmall_b{}_r{}_s{}".format(batch_size, radius, npoints))

logs = open(os.path.join(folder, "validation.txt"), "w")

# CREATE NETWORK #

input_channels = 1
# dendrite, axon, soma, bouton, terminal
output_channels = 5
model = ConvPoint(input_channels, output_channels).to(device)
model.load_state_dict(torch.load(os.path.join(folder, "state_dict.pth")))

if use_cuda:
    model.cuda()

# PREPARE DATA SET #

# Transformations to be applied to samples before feeding them to the network
val_transform = clouds.Compose([clouds.Center()])

ds = TorchSet(val_path, radius, npoints, val_transform, epoch_size=epoch_size)
train_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=1)

# PREPARE AND START TRAINING #

model.eval()
with torch.no_grad():
    cm = np.zeros((n_classes, n_classes))
    t = tqdm(train_loader, ncols=120)
    batch_num = 0
    for pts, features, lbs in t:
        features.to(device)
        pts.to(device)
        lbs.to(device)

        outputs = model(features, pts)

        outputs_np = outputs.cpu().detach().numpy()
        output_np = np.argmax(outputs_np, axis=2).copy()
        target_np = lbs.cpu().numpy().copy()

        cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(n_classes)))
        cm += cm_

        results = []
        # save results for later visualization
        for i in range(pts.size(0)):
            orig = PointCloud(pts[i].cpu().numpy(), labels=target_np[i])
            pred = PointCloud(pts[i].cpu().numpy(), labels=output_np[i])
            results.append(orig)
            results.append(pred)

        with open(folder+'/val_examples/batch_num_{}.pkl'.format(batch_num), 'wb') as f:
            pickle.dump(results, f)
        f.close()

        batch_num += 1

    oa = "{:.3f}".format(metrics.stats_overall_accuracy(cm))
    aa = "{:.3f}".format(metrics.stats_accuracy_per_class(cm)[0])

    # write the logs
    logs.write("{} {}\n".format(oa, aa))
    logs.flush()

logs.close()
