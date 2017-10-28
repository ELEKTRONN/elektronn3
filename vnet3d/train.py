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
from tqdm import tqdm
from time import time
import traceback
from socket import gethostname
import IPython
from IPython import embed as ie
import matplotlib
from vnet import VNet
if gethostname().startswith('synapse'):  # No X server there, so they need the Agg backend
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


### MODEL

n_out_channels = 2  # TODO: Maybe infer from data set?

# Actual neuro3d model
neuro3d_seq = torch.nn.Sequential(
    nn.Conv3d(1, 20, (1, 3, 3),), nn.ReLU(),
    nn.MaxPool3d((1, 1, 1)),
    nn.Conv3d(20, 30, (1, 3, 3),), nn.ReLU(),
    nn.MaxPool3d((1, 1, 1)),
    nn.Conv3d(30, 40, (1, 3, 3),), nn.ReLU(),
    nn.Conv3d(40, 80, (3, 3, 3),), nn.ReLU(),
    nn.MaxPool3d((1, 1, 1)),

    nn.Conv3d(80, n_out_channels, (1, 1, 1)), nn.ReLU()
)


class Neuro3DNetFlatSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.neuro3d_seq = neuro3d_seq

    def forward(self, x):
        # Apply main convnet
        x = self.neuro3d_seq(x)
        # Flip N, C axes for loss compat
        x_dims = x.size()
        x = x.permute(1, 0, 2, 3, 4).contiguous()
        # Flatten to 2D Tensor of dimensions (N, C)
        x = x.view(-1, n_out_channels)
        # Compute softmax (just for compat - replace this later by log_softmax)
        x = F.softmax(x)
        return x.view(x_dims)


class Simple3DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 10, 3, padding=1), nn.ReLU(),
            nn.Conv3d(10, 10, 3, padding=1), nn.ReLU(),
            nn.Conv3d(10, n_out_channels, 1), nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        # x = x.permute(1, 0, 2, 3, 4).contiguous()
        # x = x.view(-1, n_out_channels)
        # x = F.log_softmax(x)
        return x


# model = simplenet
# model = neuro2dnet
# model = neuro3d_seq
# model = Simple3DNet()
model = VNet(relu=False, nll=True)
model = nn.parallel.DataParallel(model, device_ids=[0, 1])
# model = Neuro3DNetFlatSoftmax()  # Needs adjustments in the training loop.
# model = Neuro3DNetFlatSoftmax()
criterion = nn.CrossEntropyLoss()
if cuda_enabled:
    model = model.cuda()
    criterion = criterion.cuda()


### DATA SET
z_thickness = 16
class NeuroData3D(data.Dataset):
    """ 2D Dataset class for neuro_data_zxy, reading from HDF5 files.

    Delivers 2D image slices from the xy plane.
    Not scalable, keeps everything in memory.

    See https://elektronn2.readthedocs.io/en/latest/examples.html#data-set
    Download link: http://elektronn.org/downloads/neuro_data_zxy.zip

    TODO: Images and labels don't seem to overlay correctly (MAJOR PROBLEM).
    TODO: Support multiple hdf5 files as one dataset.
    TODO: (nop): Make a 3D version.
    TODO: (nop) Create new files with the right data types so data can be read
          directly from the file while iterating over it.
    """

    def __init__(self, img_path, lab_path, img_key, lab_key, vol_size=None, pool=(1, 1, 1)):
        super().__init__()
        self.img_file = h5py.File(os.path.expanduser(img_path), 'r')
        self.lab_file = h5py.File(os.path.expanduser(lab_path), 'r')
        self.img = self.img_file[img_key].value.astype(np.float32) / 255
        self.lab = self.lab_file[lab_key].value.astype(np.int64)
        self.vol_size = np.array(vol_size, dtype=np.int)
        self.lab = self.lab[::pool[0], ::pool[1], ::pool[2]]  # Handle pooling (Filty, dirty hack TODO)

        # Cut img and lab to same size
        img_sh = np.array(self.img.shape, dtype=np.int)
        lab_sh = np.array(self.lab.shape, dtype=np.int)
        diff = img_sh - lab_sh
        offset = diff // 2  # offset from image boundaries
        self.img = self.img[offset[0]: img_sh[0] - offset[0], offset[1]: img_sh[1] - offset[1], offset[2]: img_sh[2] - offset[2],]

        self.close()  # Using file contents from memory -> no need to keep the file open.

    def __getitem__(self, index):
        # use index just as counter, subvolumes will be chosen randomly
        sh = np.array(self.img.shape, dtype=np.int)
        z_ix = np.random.randint(0, sh[0] - self.vol_size[0], 1)[0]
        x_ix = np.random.randint(0, sh[1] - self.vol_size[1], 1)[0]
        y_ix = np.random.randint(0, sh[2] - self.vol_size[2], 1)[0]

        x = torch.from_numpy(self.img[None, z_ix:z_ix+self.vol_size[0], x_ix:x_ix+self.vol_size[1], y_ix:y_ix+self.vol_size[2]])  # Prepending C axis
        y = torch.from_numpy(self.lab[z_ix:z_ix+self.vol_size[0], x_ix:x_ix+self.vol_size[1], y_ix:y_ix+self.vol_size[2]])
        return x, y

    def __len__(self):
        return np.prod(np.ceil(np.array(self.lab.shape) / self.vol_size).astype(np.int))

    def close(self):
        self.img_file.close()
        self.lab_file.close()


def cast_arr_to_shape(arr_a, arr_b):
    sh_a = np.array(arr_a.size())[-3:] # only use last/spatial axes only, e.g. b, ch, [z, x, y]
    sh_b = np.array(arr_b.size())[-3:]
    off = (sh_a - sh_b) // 2
    return arr_a[..., off[0]:-off[0] or None, off[1]:-off[1] or None, off[2]:-off[2] or None].contiguous()

from scipy.misc import imsave

def train_nll(epoch, model, trainLoader, optimizer):
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
        loss = F.nll_loss(output, target)
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
    imsave("~/raw%d.png" % (epoch), np.array(data.data.view(data.size()).tolist())[0, 0, 8])
    imsave("~/pred%d.png" % (epoch), np.array(pred.view(data.size()).tolist())[0, 0, 8])
    return err

# train_set = NeuroData2D(
train_set = NeuroData3D(
    img_path='~/neuro_data_zxy/raw_0.h5',
    lab_path='~/neuro_data_zxy/barrier_int16_0.h5',
    img_key='raw',
    lab_key='lab',
    vol_size=(16, 128, 128)
)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=1, shuffle=True, num_workers=1, pin_memory=cuda_enabled
)

# test_set = NeuroData2D(
test_set = NeuroData3D(
    img_path='~/neuro_data_zxy/raw_2.h5',
    lab_path='~/neuro_data_zxy/barrier_int16_2.h5',
    img_key='raw',
    lab_key='lab',
    vol_size=(16, 128, 128)
)

test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=1, shuffle=True, num_workers=1, pin_memory=cuda_enabled
)

### TRAINING
nEpochs = int(1e5)
best_prec1 = 100.
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(1, nEpochs + 1):
    print(epoch)
    train_nll(epoch, model, train_loader, optimizer)
    err = test_nll(epoch, model, test_loader, optimizer)
    is_best = False
    if err < best_prec1:
        is_best = True
        best_prec1 = err

inference(test_loader, model)
