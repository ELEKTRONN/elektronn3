# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Yang Liu

""" Inference script to use model in save_root + experiment_name + state_dict and output full cells for
    merger segemetation """

import os
import glob
import torch
import argparse
import numpy as np
import morphx.processing.clouds as clouds
from scipy.spatial import cKDTree
from morphx.classes.pointcloud import PointCloud
from elektronn3.models.convpoint import SegSmall, SegBig, SegSkeleton
from morphx.data.torchset import TorchSet, TorchSetSkeleton
from morphx.data.torchset import TorchSet
from morphx.data.merger_cloudset import MergerCloudSet
from tqdm import tqdm


# PARSE PARAMETERS #

parser = argparse.ArgumentParser(description='Validate a network.')
parser.add_argument('--na', type=str,
                    default="merger_Feb25_for_inference", help='Experiment name')
parser.add_argument('--vp', type=str,
                    default="/wholebrain/scratch/yliu/merger_gt_semseg_pointcloud/gt_convpoint/", help='Validation path')
parser.add_argument('--sr', type=str,
                    default="/wholebrain/scratch/yliu/pointcloud_train_result/",  help='Save root')
parser.add_argument('--sd', type=str,
                    default="state_dict.pth", help='State dict name')
parser.add_argument('--sp', type=int, default=10000, help='Number of sample points')
parser.add_argument('--ra', type=int, default=10000, help='Radius')
parser.add_argument('--cl', type=int, default=2, help='Number of classes')
parser.add_argument('--big', action='store_true', help='Use big SegBig Convpoint network')

args = parser.parse_args()


import matplotlib.pyplot as pyplot
from plyfile import PlyData, PlyElement
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


def save_pcl_with_node_labels(points, nodes, node_labels, filename, num_classes=None):
    # convert 0, 1 in node_labels to 2, 3 for visualization
    node_labels = np.where(node_labels == 0, 2, node_labels)
    node_labels = np.where(node_labels == 1, 3, node_labels)
    dummy_vert_labels = np.zeros((points.shape[0],), dtype=int)
    labels_all = np.concatenate((dummy_vert_labels, node_labels))
    pts_all = np.concatenate((points, nodes), axis=0)
    write_ply_color(pts_all, labels_all, filename, num_classes)


def node2verts(nodes, vertices) -> dict:
    """ Creates python dict with indices of skel_nodes as keys and lists of vertex
    indices which have their key node as nearest skeleton node.

    Returns:
        Dict with mapping information
    """

    tree = cKDTree(nodes)
    dist, ind = tree.query(vertices, k=1)

    dict_node2verts = {ix: [] for ix in range(len(nodes))}
    for vertex_idx, skel_idx in enumerate(ind):
        dict_node2verts[skel_idx].append(vertex_idx)
    return dict_node2verts


# SET UP ENVIRONMENT #

# define parameters
name = args.na
npoints = args.sp
radius = args.ra
n_classes = args.cl
batch_size = 5
device = 'cuda'

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

# if args.big:
#     model = SegBig(input_channels, output_channels)
# else:
#     model = SegSmall(input_channels, output_channels)
num_classes = 2
model = SegSkeleton(input_channels, num_classes)

full = torch.load(os.path.join(folder, args.sd))
model.load_state_dict(full['model_state_dict'])
model.to('cuda')
model.eval()

# PREPARE DATA SET #

# Transformations to be applied to one half of the samples before feeding them to the network
# transform = clouds.Center()

# Define two datasets, one for applying the same transformation as during training (most important: center()). The
# transformed data gets fed into the network. The other dataset doesn't apply any transformation in order to stitch
# together the original cell afterwards.
transform = clouds.Compose([clouds.Normalization(radius), clouds.Center()])
# ds = MergerCloudSet('', radius, npoints, class_num=n_classes)
data_path = "/wholebrain/scratch/yliu/merger_gt_semseg_pointcloud/gt_convpoint/"
# ds = TorchSetSkeleton(data_path, radius, npoints, transform, class_num=num_classes, data_type='merger', epoch_size=None)

model_time = 0
model_counter = 0

########################################################
if __name__ == "__main__":
    files = files[101:200]
    count = 0
    for file in files:
        print("Number of prediction so far: {}".format(count))
        count += 1

        split = file.split('/')
        name = split[-1][:-4]

        # switch cloudset to selected file
        hc = clouds.load_cloud(file)

        # keep a set of the nodes that are visited
        # keep another list of nodes that remains to be predicted
        node_idx_all = np.arange(0, len(hc.nodes)).tolist()
        visited_node_idx = set()
        node_idx_left = set(node_idx_all.copy())
        pred_node_labels = np.array([-1] * len(hc.nodes), dtype=int)
        kdtree_node = cKDTree(hc.nodes)
        kdtree_vert = cKDTree(hc.vertices)

        iter = 0
        while len(node_idx_left) > 0:
            # iter += 1
            # print("iter: {}; node_left: {}".format(iter, len(node_idx_left)))
            query_center = hc.nodes[list(node_idx_left)[0]]

            chunk_node_ixs = kdtree_node.query_ball_point(query_center, r=radius)
            chunk_nodes = hc.nodes[chunk_node_ixs]
            chunk_vert_ixs = kdtree_vert.query_ball_point(query_center, r=radius)
            chunk_vertices = hc.vertices[chunk_vert_ixs]

            subset = PointCloud(vertices=chunk_vertices)
            sample_cloud = clouds.sample_cloud(subset, npoints)

            # Temporally add nodes in to _vertices in point cloud
            num_nodes = len(chunk_nodes)
            sample_cloud._vertices = np.concatenate((sample_cloud._vertices, chunk_nodes), axis=0)

            # apply transformations
            if len(sample_cloud.vertices) > 0:
                transform(sample_cloud)

            # Remove nodes coordinates from _vertices
            chunk_nodes = sample_cloud._vertices[npoints:]
            sample_cloud._vertices = sample_cloud._vertices[:-num_nodes]

            # Predict node label on current chunk
            # Prepare data for inference
            pts = torch.from_numpy(sample_cloud.vertices).float()
            features = torch.ones(len(sample_cloud.vertices), 1).float()
            nodes = torch.from_numpy(chunk_nodes).float()

            pts = torch.reshape(pts, (1, pts.shape[0], pts.shape[1])).to(device, non_blocking=True)
            features = torch.reshape(features, (1, features.shape[0], features.shape[1])).to(device, non_blocking=True)
            nodes = torch.reshape(nodes, (1, nodes.shape[0], nodes.shape[1])).to(device, non_blocking=True)

            # Prediction
            with torch.no_grad():
                outputs = model(features, pts, nodes)
                output_np = outputs.cpu().detach().numpy()
                output_np = np.argmax(output_np, axis=2).copy()

                # add processed chunks node labels incrementally to full cell
                pred_node_labels[chunk_node_ixs] = output_np

                # Update visited_node_idx and node_idx_left
                visited_node_idx.update(chunk_node_ixs)
                node_idx_left = set(node_idx_all).difference(visited_node_idx)

        # Finished prediction for one cell
        if -1 in pred_node_labels:
            import warnings
            warnings.warn("predition incomplete")

        # Output the point cloud with labeled vertices
        fname = name + ".ply"
        output_path = "/wholebrain/scratch/yliu/ConvPoint_inference/" + fname

        # Map node_labels to vertices_labels
        import time
        start = time.time()

        vert_labels = np.zeros(hc.vertices.shape[0])
        node2verts = node2verts(hc.nodes, hc.vertices)
        for node_idx, vert_ixs in node2verts.items():
            if pred_node_labels[node_idx] == 1:
                vert_labels[vert_ixs] = 1

        end = time.time()
        print("Time elapsed: {}".format(end - start))
        write_ply_color(hc.vertices, vert_labels, output_path)


