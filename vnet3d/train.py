#!/usr/bin/env python3

# Requires at least Python 3.6.

# Copyright (c) 2017 Martin Drawitsch
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import sys
import numpy as np
import h5py
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils import data
from socket import gethostname
import matplotlib
from scipy.misc import imsave
from vnet import VNet
if gethostname().startswith('synapse'):  # No X server there, so they need the Agg backend
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data.cnndata import BatchCreatorImage
from data.utils import get_filepaths_from_dir
from torch.nn.init import xavier_normal


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


### UTILS

cuda_enabled = torch.cuda.is_available()


# cuda_enabled = False  # Uncommenct to only use CPU


def inference(loader, model):
    model.eval()
    # assume single GPU / batch size 1
    for data, target in loader:
        # convert names to batch tensor
        if cuda_enabled:
            data.pin_memory()
            data = data.cuda()
        data = Variable(data, volatile=True)
        output = model(data)
        _, output = output.max(1)
        output = output.cpu()
        # merge subvolumes and save
        raise()


def ndarray(x):
    """ Convert torch Tensor or autograd Variable to numpy ndarray.

    If it already is an ndarray, it is just returned without modification.
    Tensors and autograd Variables on GPU are copied to CPU if necessary.
    """
    if isinstance(x, Variable):
        return x.cpu().data.numpy()
    elif isinstance(x, torch._TensorBase):  # torch.Tensor would be too specific here.
        return x.cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise ValueError(f'Input x has to be of type Variable, Tensor or ndarray. Actual type is {type(x)}.')


def slice2d(x, batch=0, channel=0, z=0):
    """ Slice 5D, 4D or 3D tensors to 2D using batch and channel args.

    In the 5D case, the z-th xy slice is returned (assuming zyx axis order).
    2D tensors are returned without modification.
    """
    try:
        ndim = len(x.shape)
    except AttributeError:  # autograd Variables don't have a shape attribute
        ndim = len(x.size())

    if ndim == 5:
        return x[batch, channel, z]
    elif ndim == 4:
        return x[batch, channel]
    elif ndim == 3:
        return x[channel]
    elif ndim == 2:
        return x
    else:
        raise ValueError(f'Input x is {ndim}D, but it has to be 2D, 3D, 4D or 5D.')


def tplot(x, filename=None, batch=0, channel=0, z=0):
    """ Helper for quickly plotting Tensors, autograd Variables or ndarrays.

    Slices 4D or 3D tensors to 2D using batch and channel args and plots the
    2D slice.
    """
    img_tensor = slice2d(x, batch=batch, channel=channel)
    img = ndarray(img_tensor)
    plt.imshow(img, cmap='gray')
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')


def tshow(x, batch=0, channel=0, z=0):
    tplot(x, filename=None, batch=batch, channel=channel)


def pred_preview_plot(num, img, lab, out, prefix=''):
    tplot(img, f'{prefix}{num:04d}_img.png')
    tplot(lab, f'{prefix}{num:04d}_lab.png')
    tplot(out, f'{prefix}{num:04d}_out.png')


def flush():
    """ Flush stdout and stderr so the next output is guaranteed to be printed first.
    """
    sys.stdout.flush()
    sys.stderr.flush()


def cast_arr_to_shape(arr_a, arr_b):
    sh_a = np.array(arr_a.size())[-3:] # only use last/spatial axes only, e.g. b, ch, [z, x, y]
    sh_b = np.array(arr_b.size())[-3:]
    off = (sh_a - sh_b) // 2
    return arr_a[..., off[0]:-off[0] or None, off[1]:-off[1] or None, off[2]:-off[2] or None].contiguous()


def train_nll(epoch, model, trainLoader, optimizer, weights=None):
    model.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        data, target = Variable(data), Variable(target)
        if cuda_enabled:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        # target = cast_arr_to_shape(target, output)
        target = target.view(target.numel())
        loss = F.nll_loss(output, target, weight=weights)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/target.numel()
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tError: {:.3f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], err))


def test_nll(epoch, model, testLoader, optimizer):
    model.eval()
    test_loss = 0
    dice_loss = 0
    incorrect = 0
    numel = 0
    for data, target in testLoader:
        data, target = Variable(data, volatile=True), Variable(target)
        if cuda_enabled:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        target = target.view(target.numel())
        numel += target.numel()
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss /= len(testLoader)  # loss function already averages over batch size
    dice_loss /= len(testLoader)
    err = 100.*incorrect/numel
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.3f}%) Dice: {:.6f}\n'.format(
        test_loss, incorrect, numel, err, dice_loss))
    # print("std:", data.std(), np.array(pred.tolist()).std(), np.array(target.data.tolist()).std())
    imsave("/u/pschuber/vnet/raw%d.png" % (epoch), np.array(data.data.view(data.size()).tolist())[0, 0, 8])
    imsave("/u/pschuber/vnet/pred%d.png" % (epoch), np.array(pred.view(data.size()).tolist())[0, 0, 8])
    imsave("/u/pschuber/vnet/target%d.png" % (epoch), np.array(target.data.view(data.size()).tolist())[0, 0, 8])
    return err


### MODEL
torch.manual_seed(0)
if cuda_enabled:
    torch.cuda.manual_seed(0)
model = VNet(relu=False, nll=True)
model = nn.parallel.DataParallel(model, device_ids=[0, 1])
if cuda_enabled:
    model = model.cuda()
model.apply(weights_init)


### DATA
wd = '/wholebrain/scratch/j0126/'
h5_fnames = get_filepaths_from_dir('%s/barrier_gt_phil/' % wd, ending="rawbarr-zyx.h5")
data_init_kwargs = {
    'zxy': True,
    'd_path' : '%s/barrier_gt_phil/' % wd,
    'l_path': '%s/barrier_gt_phil/' % wd,
    'd_files': [(os.path.split(fname)[1], 'raW') for fname in h5_fnames],
    'l_files': [(os.path.split(fname)[1], 'labels') for fname in h5_fnames],
    'aniso_factor': 2, "source": "train",
    'valid_cubes': [2], 'patch_size': (16, 128, 128),
    'grey_augment_channels': [0], "epoch_size": 40,
    'warp': 0.5,
    'warp_args': {
        'sample_aniso': True,
        'perspective': True
    }}
train_set = BatchCreatorImage(**data_init_kwargs)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=2, shuffle=True, num_workers=1, pin_memory=cuda_enabled)
_ = train_set.getbatch()
test_set = train_set
test_set.source = "valid"
test_set.epoch_size = 10

test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=1, shuffle=True, num_workers=1, pin_memory=cuda_enabled
)

### TRAINING
nEpochs = int(100)
best_prec1 = 100.
wd = 0.5e-4
lr = 0.0003
opt = "adam"
lr_dec = 0.98
# target_mean = train_set.target_mean
# bg_weight = target_mean / (1. + target_mean)
# fg_weight = 1. - bg_weight
# print("\n", bg_weight)
# class_weights = torch.FloatTensor([bg_weight, fg_weight])
# if cuda_enabled:
#     class_weights = class_weights.cuda()
class_weights = None
if opt == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=wd)
elif opt == 'adam':
    optimizer = optim.Adam(model.parameters(), weight_decay=wd, lr=lr)
elif opt == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), weight_decay=wd, lr=lr)
print("Start with training")
for epoch in range(1, nEpochs + 1):
    train_nll(epoch, model, train_loader, optimizer, weights=class_weights)
    err = test_nll(epoch, model, test_loader, optimizer)
    is_best = False
    if err < best_prec1:
        is_best = True
        best_prec1 = err
    lr *= lr_dec
    print("Learning rate:", lr)
    optimizer = optim.Adam(model.parameters(), weight_decay=wd, lr=lr)

inference(test_loader, model)
