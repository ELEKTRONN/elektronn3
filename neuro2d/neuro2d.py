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
from socket import gethostname
import IPython
from IPython import embed as ie
import matplotlib
if gethostname().startswith('synapse'):  # No X server there, so they need the Agg backend
    matplotlib.use('Agg') 
import matplotlib.pyplot as plt


### UTILS

cuda_enabled = torch.cuda.is_available()
# cuda_enabled = False  # Uncommenct to only use CPU


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


def slice2d(x, batch=0, channel=0):
    """ Slice 4D or 3D tensors to 2D using batch and channel args.

    2D tensors are returned without modification. 
    """
    try:
        ndim = len(x.shape)
    except AttributeError:  # autograd Variables don't have a shape attribute
        ndim =  len(x.size())
    
    if ndim == 4:
        return x[batch, channel]
    elif ndim == 3:
        return x[channel]
    elif ndim == 2:
        return x
    else:
        raise ValueError(f'Input x has to be 2D, 3D or 4D. Actual shape is {xnp.shape}.')


def tplot(x, filename=None, batch=0, channel=0):
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


def tshow(x, batch=0, channel=0):
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

simplenet = torch.nn.Sequential(
    nn.Conv2d(1, 20, 5),
    nn.ReLU(),
    nn.Conv2d(20, 2, 5)
)

# Like neuro3d.py, but completely in 2D. See https://github.com/ELEKTRONN/ELEKTRONN2/blob/master/examples/neuro3d.py
neuro2dnet = torch.nn.Sequential(
    nn.Conv2d(1, 20, 6), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(20, 30, 5), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(30, 40, 5), nn.ReLU(),
    nn.Conv2d(40, 80, 4), nn.ReLU(),

    nn.Conv2d(80, 100, 4), nn.ReLU(),
    nn.Conv2d(100, 100, 4), nn.ReLU(),
    nn.Conv2d(100, 150, 4), nn.ReLU(),
    nn.Conv2d(150, 200, 4), nn.ReLU(),
    nn.Conv2d(200, 200, 4), nn.ReLU(),
    nn.Conv2d(200, 200, 1), nn.ReLU(),
    nn.Conv2d(200, 2, 1)
)

class ExampleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_inp = nn.Conv2d(1, 20, 6)
        self.conv_mid = nn.Conv2d(20, 20, 6)
        #self.conv_out = nn.Conv2d(20, 2, 6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc = nn.Linear(20, 2)

    def forward(self, x):
        x = F.relu(self.conv_inp(x))

        x = F.relu(self.conv_mid(x))
        x = F.relu(self.conv_mid(x))
        x = F.relu(self.conv_mid(x))

        x = self.fc(x)
        x = x.view(x.size(0), -1)
        x = F.softmax(x)
        return x


# neural = simplenet
model = neuro2dnet
# neural = ExampleNet()
criterion = nn.CrossEntropyLoss()
if cuda_enabled:
    model = model.cuda()
    criterion = criterion.cuda()


### DATA SET

class NeuroData2D(data.Dataset):
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
    def __init__(self, img_path, lab_path, img_key, lab_key, img_offset=(0, 50, 50), pool=(1, 1, 1)):
        super().__init__()
        self.img_file = h5py.File(os.path.expanduser(img_path), 'r')
        self.lab_file = h5py.File(os.path.expanduser(lab_path), 'r')
        self.img = self.img_file[img_key].value.astype(np.float32) / 255
        self.lab = self.lab_file[lab_key].value.astype(np.int64)  # int16 is unsupported
        self.img_offset = img_offset

        self.lab = self.lab[::pool[0], ::pool[1], ::pool[2]]  # Handle pooling (Filty, dirty hack TODO)

        # Cut img and lab to same size
        img_sh = np.array(self.img.shape)
        lab_sh = np.array(self.lab.shape)
        diff = img_sh - lab_sh
        # offset = diff // 2  # offset from image boundaries
        offset = self.img_offset
        self.img = self.img[
            offset[0] : img_sh[0] - offset[0],
            offset[1] : img_sh[1] - offset[1],
            offset[2] : img_sh[2] - offset[2],
        ]

        self.close()  # Using file contents from memory -> no need to keep the file open.


    def __getitem__(self, index):
        # Get z slices
        x = torch.from_numpy(self.img[None, index, ...])  # Prepending C axis
        y = torch.from_numpy(self.lab[index, ...])
        # y = F.max_pool2d(y, 2 * 2) 
        return x, y

    def __len__(self):
        return self.lab.shape[0]
    
    def close(self):
        self.img_file.close()
        self.lab_file.close()


# pool = None  # for simplenet
pool = (1, 4, 4)  # for neuro2dnet
# img_offset = (0, 96, 96)  # for simplenet
img_offset = (0, 48, 48)  # for neuro2dnet


train_set = NeuroData2D(
    img_path='~/neuro_data_zxy/raw_0.h5',
    lab_path='~/neuro_data_zxy/barrier_int16_0.h5',
    img_key='raw',
    lab_key='lab',
    img_offset=img_offset,
    pool=pool
)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=1, shuffle=True, num_workers=2, pin_memory=cuda_enabled
)

test_set = NeuroData2D(
    img_path='~/neuro_data_zxy/raw_2.h5',
    lab_path='~/neuro_data_zxy/barrier_int16_2.h5',
    img_key='raw',
    lab_key='lab',
    img_offset=img_offset,
    pool=pool
)

test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=4, shuffle=True, num_workers=2, pin_memory=cuda_enabled
)


### TRAINING

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

stat_interval = len(train_loader)
preview_prediction_times = [  # (epoch, mini-batch) tuples at which preview predictions should be made.
    (50, 0),
    (1000, 0),
    (2000, 0),
    (4000, 0),
    (7999, 0),
]
assert stat_interval <= len(train_loader)  # If stat_interval is larger, stats will never be printed.
n_epochs = 8000

print(f'Starting training for {n_epochs} epochs on {len(train_loader)} mini-batches.')
print(f'Extended stats (including loss) are printed every {stat_interval} iterations.')
device_name = torch.cuda.current_device if cuda_enabled else 'CPU'
if cuda_enabled:
    print(f'Using GPU {torch.cuda.current_device()}.\n')
else:
    print('Using CPU.\n')

for epoch in range(n_epochs):
    try:
        running_loss = 0.0
        # t0 = time()
        # train_loader = tqdm(train_loader)
        for i, data in enumerate(train_loader):
            # Get mini-batch
            print(f'Batch: ({epoch:4d}, {i:4d})', end='\r')
            img, lab = data
            if cuda_enabled:
                img, lab = img.cuda(), lab.cuda()
            img, lab = Variable(img), Variable(lab)

            # lab = lab.view(lab.data.shape[0], -1)  # ???

            # Train
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, lab)
            loss.backward()
            optimizer.step()

            # Preview predictions
            if (epoch, i) in preview_prediction_times:
                pred_preview_plot(i, img, lab, out)  # Visualize prediction on current mini-batch (from train set!)
                # TODO: Use test set
            
            # Loss stats
            running_loss += loss.data[0]
            if i % stat_interval == stat_interval - 1:
                print(f'batch: ({epoch:4d}, {i:4d}),  loss: {running_loss/stat_interval:.4f}')
                # t0 = time() ; print(f'{time() - t0} iterations per second.')
                running_loss = 0.0
    except KeyboardInterrupt:
        flush()
        answer = input(
            '\n\n=== KeyboardInterrupt during training: Enter IPython '
            'from here?\n=== (Everything but "y" ends the training) [y/N] '
        )
        if answer == 'y':
            IPython.embed()
            print('Continuing training...')
        else:
            break

print('Finished Training')


test_iter = iter(test_loader)
timg, tlab = next(test_iter)
if cuda_enabled:
    timg, tlab = timg.cuda(), tlab.cuda()
    timg, tlab = Variable(timg), Variable(tlab)

flush()
answer = input(
    '\n\n=== End of training: Enter IPython '
    'from here?\n=== (Everything but "y" ends the process) [y/N] '
)
if answer == 'y':
    IPython.embed()
