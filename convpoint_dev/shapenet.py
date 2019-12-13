# MODELNET40 Example with ConvPoint

import os
import argparse
import torch
import torch.utils.data

import numpy as np
import torch.nn.functional as F
import convpoint_dev.metrics as metrics
import convpoint_dev.data_utils as data_utils

from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import BallTree
from elektronn3.models.convpoint import SegSmall as Net


class PartNormalDataset(torch.utils.data.Dataset):
    def __init__(self, data, data_num, label, npoints, num_iter_per_shape=1):
        self.data = data
        self.data_num = data_num
        self.label = label
        self.npoints = npoints
        self.num_iter_per_shape = num_iter_per_shape

    def __getitem__(self, index):
        index = index // self.num_iter_per_shape

        npts = self.data_num[index]
        pts = self.data[index, :npts]
        choice = np.random.choice(npts, self.npoints, replace=True)

        pts = pts[choice]
        lbs = self.label[index][choice]
        features = torch.ones(pts.shape[0], 1).float()

        pts = torch.from_numpy(pts).float()
        lbs = torch.from_numpy(lbs).long()

        return pts, features, lbs, index

    def __len__(self):
        return self.data.shape[0] * self.num_iter_per_shape


def nearest_correspondance(pts_src, pts_dest, data_src):
    tree = BallTree(pts_src, leaf_size=2)
    _, indices = tree.query(pts_dest, k=1)
    indices = indices.ravel()
    data_dest = data_src[indices]
    return data_dest


def train(args):
    threads = 4
    use_cuda = True
    n_classes = 50
    epochs = 200
    milestones = [60, 120]
    shapenet_labels = [['Airplane', 4], ['Bag', 2], ['Cap', 2], ['Car', 4], ['Chair', 4], ['Earphone', 3],
                       ['Guitar', 3], ['Knife', 2], ['Lamp', 4], ['Laptop', 2], ['Motorbike', 6], ['Mug', 2],
                       ['Pistol', 3], ['Rocket', 3], ['Skateboard', 3], ['Table', 3]]
    category_range = []
    count = 0
    for element in shapenet_labels:
        part_start = count
        count += element[1]
        part_end = count
        category_range.append([part_start, part_end])

    # Prepare inputs
    print('{}-Preparing datasets...'.format(datetime.now()))
    is_list_of_h5_list = not data_utils.is_h5_list(args.filelist)
    if is_list_of_h5_list:
        seg_list = data_utils.load_seg_list(args.filelist)
        seg_list_idx = 0
        filelist_train = seg_list[seg_list_idx]
    else:
        filelist_train = args.filelist
    data_train, labels, data_num_train, label_train, _ = data_utils.load_seg(filelist_train)
    print("Done", data_train.shape)

    print("Computing class weights (if needed, 1 otherwise)...")
    if args.weighted:
        frequences = []
        for i in range(len(shapenet_labels)):
            frequences.append((labels == i).sum())
        frequences = np.array(frequences)
        frequences = frequences.mean() / frequences
    else:
        frequences = [1 for _ in range(len(shapenet_labels))]
    weights = torch.FloatTensor(frequences)
    if use_cuda:
        weights = weights.cuda()
    print("Done")

    print("Creating network...")

    net = Net(1, n_classes)
    if use_cuda:
        net.cuda()
    print("parameters", sum(p.numel() for p in net.parameters() if p.requires_grad))

    ds = PartNormalDataset(data_train, data_num_train, label_train, npoints=args.npoints)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=args.batchsize, shuffle=True, num_workers=threads)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones)

    # create the model folder
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(args.savedir, "{}_b{}_pts{}_weighted{}_{}"
                               .format(args.model, args.batchsize, args.npoints, args.weighted, time_string))
    os.makedirs(root_folder, exist_ok=True)

    # create the log file
    logs = open(os.path.join(root_folder, "log.txt"), "w")
    for epoch in range(epochs):
        scheduler.step()
        cm = np.zeros((n_classes, n_classes))
        t = tqdm(train_loader, ncols=120, desc="Epoch {}".format(epoch))
        oa = 0
        aa = 0

        # process each batch
        for pts, features, seg, indices in t:

            if use_cuda:
                features = features.cuda()
                pts = pts.cuda()
                seg = seg.cuda()

            optimizer.zero_grad()

            # get output of size (batch_size, sample_number, n_classes)
            outputs = net(features, pts)

            loss = 0
            for i in range(pts.size(0)):
                # extract the labels which correspond to the current object
                object_label = labels[indices[i]]
                part_start, part_end = category_range[object_label]
                part_nbr = part_end - part_start
                loss = loss + weights[object_label] * F.cross_entropy(
                    outputs[i, :, part_start:part_end].view(-1, part_nbr), seg[i].view(-1) - part_start)

                # outputnp = outputs[i, :, part_start:part_end].view(-1, part_nbr).detach().numpy()
                # pco = PointCloud(pts[i].numpy(), np.argmax(outputnp, axis=1))
                # pci = PointCloud(pts[i].numpy(), (seg[i].view(-1) - part_start).numpy())
                # clouds.save_cloud(pco, "/u/jklimesch/", name="pco_{}".format(i))
                # clouds.save_cloud(pci, "/u/jklimesch/", name="pci_{}".format(i))

            loss.backward()
            optimizer.step()

            outputs_np = outputs.cpu().detach().numpy()
            for i in range(pts.size(0)):
                # extract the labels which correspond to the current object
                object_label = labels[indices[i]]
                part_start, part_end = category_range[object_label]

                # set all other labels to -1e7
                outputs_np[i, :, :part_start] = -1e7
                outputs_np[i, :, part_end:] = -1e7

            # extract label with maximum probability
            output_np = np.argmax(outputs_np, axis=2).copy()
            target_np = seg.cpu().numpy().copy()

            cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(n_classes)))
            cm += cm_

            oa = "{:.3f}".format(metrics.stats_overall_accuracy(cm))
            aa = "{:.3f}".format(metrics.stats_accuracy_per_class(cm)[0])

            t.set_postfix(OA=oa, AA=aa)

        # save the model
        torch.save(net.state_dict(), os.path.join(root_folder, "state_dict.pth"))

        # write the logs
        logs.write("{} {} {} \n".format(epoch, oa, aa))
        logs.flush()

    logs.close()


def test(args):
    threads = 4
    use_cuda = True
    n_classes = 50

    args.data_folder = os.path.join(args.rootdir, "test_data")

    # create the output folders
    output_folder = os.path.join(args.savedir, '_predictions2')
    category_list = [(category, int(label_num)) for (category, label_num) in
                     [line.split() for line in open(args.category, 'r')]]
    offset = 0
    category_range = dict()
    for category, category_label_seg_max in category_list:
        category_range[category] = (offset, offset + category_label_seg_max)
        offset = offset + category_label_seg_max
        folder = os.path.join(output_folder, category)
        if not os.path.exists(folder):
            os.makedirs(folder)

    input_filelist = []
    output_filelist = []
    output_ply_filelist = []
    for category in sorted(os.listdir(args.data_folder)):
        data_category_folder = os.path.join(args.data_folder, category)
        for filename in sorted(os.listdir(data_category_folder)):
            input_filelist.append(os.path.join(args.data_folder, category, filename))
            output_filelist.append(os.path.join(output_folder, category, filename[0:-3] + 'seg'))
            output_ply_filelist.append(os.path.join(output_folder + '_ply', category, filename[0:-3] + 'ply'))

    # Prepare inputs
    print('{}-Preparing datasets...'.format(datetime.now()))
    data, label, data_num, label_test, _ = data_utils.load_seg(args.filelist_val)  # no segmentation labels

    net = Net(1, n_classes)
    net.load_state_dict(torch.load(os.path.join(args.savedir, "state_dict.pth")))
    net.cuda()
    net.eval()

    ds = PartNormalDataset(data, data_num, label_test, npoints=args.npoints, num_iter_per_shape=args.ntree)
    test_loader = torch.utils.data.DataLoader(ds, batch_size=args.batchsize, shuffle=False, num_workers=threads)

    t = tqdm(test_loader, ncols=120)

    predictions = [None for _ in range(data.shape[0])]
    predictions_max = [[] for _ in range(data.shape[0])]
    with torch.no_grad():
        for pts, features, seg, indices in t:
            if use_cuda:
                features = features.cuda()
                pts = pts.cuda()

            outputs = net(features, pts)
            indices = np.int32(indices.numpy())
            outputs = np.float32(outputs.cpu().numpy())

            # iterate over examples in batch
            for i in range(pts.size(0)):
                # shape id
                shape_id = indices[i]

                # pts_src
                pts_src = pts[i].cpu().numpy()

                # pts_dest
                point_num = data_num[shape_id]
                pts_dest = data[shape_id]
                pts_dest = pts_dest[:point_num]

                # get corresponding range of labels
                object_label = label[indices[i]]
                category = category_list[object_label][0]
                part_start, part_end = category_range[category]

                # get the output range which corresponds to current object
                seg_ = outputs[i][:, part_start:part_end]

                # pco = PointCloud(pts_src, np.argmax(seg_, axis=1))
                # pci = PointCloud(pts_src, (seg[i] - part_start).numpy())
                # clouds.save_cloud(pco, "/u/jklimesch/", name="pco_{}".format(i))
                # clouds.save_cloud(pci, "/u/jklimesch/", name="pci_{}".format(i))

                # interpolate to original points
                seg_ = nearest_correspondance(pts_src, pts_dest, seg_)

                # save output label in predictions
                if predictions[shape_id] is None:
                    predictions[shape_id] = seg_
                else:
                    predictions[shape_id] += seg_

                predictions_max[shape_id].append(seg_)

    for i in range(len(predictions)):
        a = np.stack(predictions_max[i], axis=1)
        a = np.argmax(a, axis=2)
        a = np.apply_along_axis(np.bincount, 1, a, minlength=6)
        predictions_max[i] = np.argmax(a, axis=1)

    # compute labels
    for i in range(len(predictions)):
        predictions[i] = np.argmax(predictions[i], axis=1)

    metrics.scores_from_predictions(predictions, category_list, label, category_range, data_num, label_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", action="store_true", help="save ply files (test mode)")
    parser.add_argument("--savedir", default="results/", type=str)
    parser.add_argument("--batchsize", "-b", default=16, type=int)
    parser.add_argument("--ntree", default=1, type=int)
    parser.add_argument("--npoints", default=2500, type=int)
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--model", default="SegSmall", type=str)
    args = parser.parse_args()

    args.rootdir = '/u/jklimesch/shapenet/shapenet_partseg'
    args.savedir = '/u/jklimesch/shapenet/results/SegSmall/trained'
    args.test = True

    args.filelist = os.path.join(args.rootdir, "train_files.txt")
    args.filelist_val = os.path.join(args.rootdir, "test_files.txt")
    args.category = os.path.join(args.rootdir, "categories.txt")

    if args.test:
        test(args)
    else:
        train(args)
