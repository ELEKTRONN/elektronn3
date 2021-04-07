# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Jonathan Klimesch

import os
import torch
import random
import argparse
import torch.nn.functional as func
import numpy as np
import convpoint_dev.obsolet.metrics as metrics
from morphx.classes.pointcloud import PointCloud
from morphx.processing import clouds
from sklearn.metrics import confusion_matrix
from elektronn3.models.convpoint import SegSmall, SegNoBatch
from elektronn3.training.trainer import Backup
from morphx.data.torchset import TorchSet
from tqdm import tqdm

# PARSE PARAMETERS #

parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--na', type=str, required=True, help='Experiment name')
parser.add_argument('--tp', type=str, required=True, help='Train path')
parser.add_argument('--sr', type=str, required=True, help='Save root')
parser.add_argument('--ep', type=int, default=200, help='Number of epochs')
parser.add_argument('--bs', type=int, default=16, help='Batch size')
parser.add_argument('--sp', type=int, default=1000, help='Number of sample points')
parser.add_argument('--ra', type=int, default=10000, help='Radius')
parser.add_argument('--cl', type=int, default=2, help='Number of classes')
parser.add_argument('--co', action='store_true', help='Disable CUDA')
parser.add_argument('--big', action='store_true', help='Use big SegBig Convpoint network')

args = parser.parse_args()

# SET UP ENVIRONMENT #

use_cuda = not args.co

if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# CREATE NETWORK #

input_channels = 1
output_channels = args.cl

if args.big:
    model = SegNoBatch(input_channels, output_channels).to(device)
else:
    model = SegSmall(input_channels, output_channels).to(device)

if use_cuda:
    model.cuda()

# define parameters
name = args.na
epochs = args.ep
batch_size = args.bs
npoints = args.sp
radius = args.ra
n_classes = args.cl
milestones = [60, 120]
lr = 1e-3

# set paths
train_path = os.path.expanduser(args.tp)
save_root = os.path.expanduser(args.sr)
folder = save_root + name + '/'
if os.path.exists(folder):
    raise ValueError("Experiment with given name already exists, please choose a different name.")
else:
    os.makedirs(folder)
train_examples = folder + 'train_examples/'
os.makedirs(train_examples)

print(folder)
print(train_path)

logs = open(folder + "log.txt", "a")

logs.write("Name: " + name + '\n')
logs.write("Batch size: " + str(batch_size) + '\n')
logs.write("Number of sample points: " + str(npoints) + '\n')
logs.write("Radius: " + str(radius) + '\n')
logs.write("Number of classes: " + str(n_classes) + '\n')
logs.write("Training data: " + train_path + '\n')
logs.write("\n###--- Epoch evaluation: ---###\n\n")
logs.flush()

backup = Backup(__file__, folder)
backup.archive_backup()

# PREPARE DATA SET #

# Transformations to be applied to samples before feeding them to the network
train_transform = clouds.Compose([clouds.RandomRotate(),
                                  clouds.RandomVariation(limits=(-10, 10)),
                                  clouds.Center()])

# create dataset with filter for axon, bouton and terminal
ds = TorchSet(train_path, radius, npoints, train_transform, class_num=n_classes)
train_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=1)

# get class weights
weights = ds.weights
t_weights = torch.tensor(weights, dtype=torch.float)

# PREPARE AND START TRAINING #

# set up optimization
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones)

model.train()
for epoch in range(epochs):
    scheduler.step()
    cm = np.zeros((n_classes, n_classes))
    t = tqdm(train_loader, ncols=120, desc="Epoch {}".format(epoch))
    oa = 0
    aa = 0
    batch_num = 0
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

        # save random sample results for later visualization
        if random.random() > 0.9:
            results = []
            for i in range(pts.size(0)):
                orig = PointCloud(pts[i].cpu().numpy(), labels=target_np[i])
                var = orig.class_num
                # don't save if sample has only one label
                if var <= 1:
                    continue
                pred = PointCloud(pts[i].cpu().numpy(), labels=output_np[i])
                results.append(orig)
                results.append(pred)

            clouds.save_cloudlist(results, train_examples, 'epoch_{}_batch_{}'.format(epoch, batch_num))
        batch_num += 1

    # save the model
    torch.save(model.state_dict(), folder + "state_dict.pth")

    # write the logs
    logs.write("{} {} {} \n".format(epoch, oa, aa))
    logs.flush()

logs.close()
