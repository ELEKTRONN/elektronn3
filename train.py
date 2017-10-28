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
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils import data
from elektronn3.data.cnndata import BatchCreatorImage
from elektronn3.data.utils import get_filepaths_from_dir
from elektronn3.training.trainer import StoppableTrainer
from elektronn3.model.vnet import VNet
from elektronn3 import cuda_enabled


### UTILS
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


def inference(dataset, model):
    dataset.validate()
    data_loader = torch.utils.data.DataLoader(
        train_set, batch_size=2, shuffle=True, num_workers=1, pin_memory=cuda_enabled)
    model.eval()
    raw = dataset.valid_d[0]
    # assume single GPU / batch size 1
    for data, target in data_loader:
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


### MODEL
torch.manual_seed(0)
if cuda_enabled:
    torch.cuda.manual_seed(0)
model = VNet(relu=False)
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
    'valid_cubes': [0, 5], 'patch_size': (16, 128, 128),
    'grey_augment_channels': [0], "epoch_size": 100,
    'warp': 0.5, 'class_weights': True,
    'warp_args': {
        'sample_aniso': True,
        'perspective': True
    }}
train_set = BatchCreatorImage(**data_init_kwargs)

### TRAINING
nEpochs = int(250)
best_prec1 = 100.
wd = 0.5e-4
lr = 0.0008
opt = "adam"
lr_dec = 0.98

if opt == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
elif opt == 'adam':
    optimizer = optim.Adam(model.parameters(), weight_decay=wd, lr=lr)
elif opt == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), weight_decay=wd, lr=lr)

# start training
st = StoppableTrainer(model, criterion=F.nll_loss, optimizer=optimizer, dataset=train_set, dest_path=None)
st.run(nEpochs)

# start ifnerence
inference(train_set, model)
