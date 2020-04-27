# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Jonathan Klimesch, Yang Liu
import open3d as o3d
import os
import torch
import argparse
import numpy as np
# Don't move this stuff, it needs to be run this early to work
import elektronn3
elektronn3.select_mpl_backend('Agg')
import morphx.processing.clouds as clouds
from torch import nn
from morphx.data.torchset import TorchSet, TorchSetSkeleton
from elektronn3.models.convpoint import SegSmall, SegBig, SegSkeleton, SegSkeleton_v2
from elektronn3.training.trainer3d import Trainer3d, Backup
from elektronn3.training import metrics
from syconn.proc.meshes import write_mesh2kzip, triangulation
from plyfile import PlyData, PlyElement

# PARSE PARAMETERS #

parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--na', type=str, default="merger_Mar18_continue01", help='Experiment name')
# parser.add_argument('--na', type=str, default="test_multi_worker", help='Experiment name')
parser.add_argument('--tp', type=str, default="/wholebrain/scratch/yliu/merger_gt_semseg_pointcloud/gt_convpoint/", help='Train path')
# parser.add_argument('--tp', type=str, default="/wholebrain/scratch/yliu/merger_gt_semseg_pointcloud/gt_results/", help='Train path')
parser.add_argument('--sr', type=str, default="/wholebrain/scratch/yliu/pointcloud_train_result/", help='Save root')
parser.add_argument('--bs', type=int, default=32, help='Batch size')
parser.add_argument('--sp', type=int, default=20000, help='Number of sample points, default is 10e3')
parser.add_argument('--ra', type=int, default=10000, help='Radius')
parser.add_argument('--cl', type=int, default=2, help='Number of classes')
parser.add_argument('--co', action='store_true', help='Disable CUDA')
parser.add_argument('--big', action='store_true', help='Use big SegBig Convpoint network')

NUM_WORKER = 10
# TODO: detele this line by April.14

args = parser.parse_args()

# helper functions
def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)


import matplotlib.pyplot as pyplot
def write_ply_color(points, labels, filename, num_classes=None, colormap=pyplot.cm.jet):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels) + 1
    else:
        assert (num_classes > np.max(labels))

    vertex = []
    # colors = [pyplot.cm.jet(i/float(num_classes)) for i in range(num_classes)]
    colors = [colormap(i / float(num_classes)) for i in range(num_classes)]
    for i in range(N):
        c = colors[labels[i]]
        c = [int(x * 255) for x in c]
        vertex.append((points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    vertex = np.array(vertex,
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(filename)


def test_chunked_pcl(train_ds, num_iter=10):
    for i in range(num_iter):
        test = train_ds.__getitem__(i)

        vertices = test['pts']
        nodes = test['nodes'].numpy()
        features = test['features']
        node_labels = test['target']
        assert len(nodes) == len(node_labels)
        assert len(node_labels) == 500
        # convert 1 in node_labels to 2 for visualization
        node_labels = np.where(node_labels == 0, 2, node_labels)
        node_labels = np.where(node_labels == 1, 3, node_labels)
        vertex_labels = test['vert_labels']

        # write vertices to mesh and test
        pts = vertices.numpy()
        BASE_DIR = os.path.dirname(os.path.realpath(__file__))
        output_path = "/wholebrain/scratch/yliu/ConvPoint_chunk_test/"
        if 1 in vertex_labels:
            fname = "test_{}_merger.ply".format(i)
        else:
            fname = "test_{}.ply".format(i)

        # write_ply(vertices, output_path + "test_1.ply")
        # concat nodes into vertices
        pts_all = np.concatenate((pts, nodes), axis=0)
        labels_all = np.concatenate((vertex_labels.cpu().numpy(), node_labels))

        # write_ply_color(pts, vertex_labels.cpu().numpy(), output_path + fname)
        write_ply_color(pts_all, labels_all, output_path + fname)
        print("{}: Wrote ply file to {}".format(i, output_path))

    import pdb
    pdb.set_trace()


def test_num_nodes(train_ds, num_iter=None):
    max_num = 0
    min_num = 10e4
    if num_iter is None:
        num_iter = len(train_ds)
    for i in range(num_iter):
        test = train_ds.__getitem__(i)

        vertices = test['pts']
        nodes = test['nodes'].numpy()
        features = test['features']
        node_labels = test['target']
        assert len(nodes) == len(node_labels)
        assert len(node_labels) == 500
        # convert 1 in node_labels to 2 for visualization
        node_labels = np.where(node_labels == 0, 2, node_labels)
        node_labels = np.where(node_labels == 1, 3, node_labels)
        vertex_labels = test['vert_labels']

        assert len(nodes) == len(node_labels)
        if len(nodes) > max_num:
            max_num = len(nodes)
        if len(nodes) < min_num:
            min_num = len(nodes)

        # print("{}: node_length: {}".format(i, len(nodes)))
        if i % 1 == 0 and i > 0:
            print(i)
            # print("max: {}".format(max_num))
            # print("min: {}".format(min_num))
            print("num_nodes: {}".format(len(nodes)))
    print(i)
    print("max: {}".format(max_num))
    print("min: {}".format(min_num))



def test_loading_time(train_ds):
    import time
    start_time = time.time()
    for i in range(200):
        iter_start_time = time.time()
        test = train_ds.__getitem__(i)
        iter_end_time = time.time()
        print("i: {}, time consumed: {}".format(i, iter_end_time - iter_start_time))

        vertices = test['pts']
        features = test['features']
        vertex_labels = test['target']

        if i % 100 == 0 and i != 0:
            end_time = time.time()
            print("i: {}, Time elapsed for 100 ds: {}".format(i, end_time - start_time))
            start_time = time.time()



# SET UP ENVIRONMENT #

use_cuda = not args.co

# define parameters
name = args.na
batch_size = args.bs
npoints = args.sp
radius = args.ra
num_classes = args.cl
milestones = [60, 120]
lr = 1e-3
# lr = 1e-5
lr_stepsize = 1000
lr_dec = 0.995
# lr_dec = 0.998
max_steps = 500000

if use_cuda:
    # device = torch.device('cuda:1')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Running on device: {device}')

# set paths
save_root = os.path.expanduser(args.sr)
train_path = os.path.expanduser(args.tp)


# CREATE NETWORK AND PREPARE DATA SET#

input_channels = 1
# if args.big:
#     model = SegBig(input_channels, num_classes, dropout=0.1)
# else:
#     model = SegSmall(input_channels, num_classes)
# model = SegSkeleton(input_channels, num_classes)
model = SegSkeleton_v2(input_channels, num_classes)
print("Using {} model".format(model.model_name))

# Load model weights
folder = "/wholebrain/scratch/yliu/pointcloud_train_result/merger_Mar18_1e-5_r10_sam20k_adamStep/"
checkpoint = torch.load(os.path.join(folder, "state_dict.pth"))
model.load_state_dict(checkpoint['model_state_dict'])

if use_cuda:
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     batch_size = batch_size * torch.cuda.device_count()
    #     model = nn.DataParallel(model)
    model.to(device)

# Transformations to be applied to samples before feeding them to the network
# train_transform = clouds.Compose([clouds.Center(), clouds.RandomRotate(), clouds.Normalize(radius)])
train_transform = clouds.Compose([clouds.RandomVariation((-10, 10)),
                                  clouds.Normalization(radius),
                                  clouds.RandomRotate(),
                                  clouds.Center()])

valid_transform = clouds.Compose([clouds.Normalization(radius), clouds.Center()])

dataset_train = TorchSetSkeleton(train_path, radius, npoints, train_transform, class_num=num_classes, data_type='merger', epoch_size=10000)
dataset_valid = TorchSetSkeleton(train_path, radius, npoints, valid_transform, class_num=num_classes, data_type='merger', epoch_size=1000)
# split train and validation
# indices = torch.randperm(len(dataset)).tolist()
# train_ds = torch.utils.data.Subset(dataset, indices[:20000])
# train_ds = torch.utils.data.Subset(dataset, indices[:10])
# valid_ds = torch.utils.data.Subset(dataset, indices[20000:21000])


#### TEST
# test_chunked_pcl(dataset_train, num_iter=100)
# test_loading_time(train_ds)
# test_num_nodes(dataset_train, num_iter=None)

# import pdb
# pdb.set_trace()
#### END of TEST


# PREPARE AND START TRAINING #

# set up optimization
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print("learning rate at the beginning:")
for param_group in optimizer.param_groups:
    print(param_group['lr'])
# Step
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)
###############################
# Cyclic Learning Rate
###############################
# try:
#     from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# except ModuleNotFoundError as e:
#     print(e)
#     from elektronn3.training.schedulers import CosineAnnealingWarmRestarts
# # base_lr = 1e-8
# # max_lr = lr
# # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=lr)
# optimizer = torch.optim.SGD(
#     model.parameters(),
#     lr=lr,  # Learning rate is set by the lr_sched below
#     momentum=0.9,
#     weight_decay=0.5e-5,
# )
# scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=2)


criterion = torch.nn.CrossEntropyLoss()
if use_cuda:
    criterion.cuda()

valid_metrics = {
    'val_accuracy': metrics.bin_accuracy,
    'val_precision': metrics.bin_precision,
    'val_recall': metrics.bin_recall,
    'val_DSC': metrics.bin_dice_coefficient,
    'val_IoU': metrics.bin_iou,
}

# Create trainer
trainer = Trainer3d(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    train_dataset=dataset_train,
    valid_dataset=dataset_valid,
    valid_metrics=valid_metrics,
    batchsize=batch_size,
    num_workers=NUM_WORKER,
    save_root=save_root,
    num_classes=2,
    exp_name=name,
    schedulers={"lr": scheduler},
)

# Archiving training script, src folder, env info
bk = Backup(script_path=__file__,
            save_path=trainer.save_path).archive_backup()

# Start training
trainer.run(max_steps)
