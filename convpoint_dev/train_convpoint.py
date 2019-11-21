# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Jonathan Klimesch

import os
import torch
import torch.nn.functional as func
import numpy as np
import convpoint_dev.metrics as metrics

from sklearn.metrics import confusion_matrix
from elektronn3.models.convpoint import ConvPoint
from elektronn3.data.transforms import transforms3d
from elektronn3.data.cnndata import PointCloudLoader
from tqdm import tqdm
from datetime import datetime

# SET UP ENVIRONMENT #

device = torch.device('cpu')

print(f'Running on device: {device}')

# CREATE NETWORK #

input_channels = 1
# dendrite, axon, soma, bouton, terminal
output_channels = 5
model = ConvPoint(input_channels, output_channels).to(device)

# define parameters
epochs = 200
epoch_size = 2000
milestones = [60, 120]
lr = 1e-3
batch_size = 16
npoints = 5000
radius = 10000
n_classes = 5

# set paths
train_path = os.path.expanduser('~/gt/gt_results/')
time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
folder = os.path.join(train_path, "SegSmall_{}_{}_{}_{}".format(batch_size, npoints, radius, time_string))
os.makedirs(folder, exist_ok=True)
logs = open(os.path.join(folder, "log.txt"), "w")

# PREPARE DATA SET #

# Transformations to be applied to samples before feeding them to the network
train_transform = transforms3d.Compose3d([transforms3d.RandomRotate3d(),
                                          transforms3d.RandomVariation3d(limits=(-10, 10)),
                                          transforms3d.Center3d()])

ds = PointCloudLoader(train_path, radius, npoints, train_transform, epoch_size=epoch_size)
train_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                           num_workers=1)

# PREPARE AND START TRAINING #

# set up optimization
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones)

for epoch in range(epochs):
    scheduler.step()
    cm = np.zeros((n_classes, n_classes))
    t = tqdm(train_loader, ncols=120, desc="Epoch {}".format(epoch))
    for pts, features, lbs in t:
        features.to(device)
        pts.to(device)
        lbs.to(device)

        optimizer.zero_grad()
        outputs = model(features, pts)

        loss = 0
        for i in range(pts.size(0)):
            loss = loss + func.cross_entropy(outputs[i], lbs[i])

        loss.backward()
        optimizer.step()

        outputs_np = outputs.cpu().detach().numpy()
        output_np = np.argmax(outputs_np, axis=2).copy()
        target_np = lbs.cpu().numpy().copy()

        cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(n_classes)))
        cm += cm_

        oa = "{:.3f}".format(metrics.stats_overall_accuracy(cm))
        aa = "{:.3f}".format(metrics.stats_accuracy_per_class(cm)[0])
        t.set_postfix(OA=oa, AA=aa)

    # save the model
    torch.save(model.state_dict(), os.path.join(folder, "state_dict.pth"))

    # write the logs
    logs.write("{} {} {} \n".format(epoch, oa, aa))
    logs.flush()

logs.close()
