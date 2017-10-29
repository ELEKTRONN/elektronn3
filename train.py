#!/usr/bin/env python3
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
from elektronn3.neural.vnet import VNet
from elektronn3 import cuda_enabled
from torch.optim.lr_scheduler import ExponentialLR


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
        dataset, batch_size=2, shuffle=True, num_workers=1, pin_memory=cuda_enabled)
    model.eval()
    raw = torch.from_numpy(dataset.valid_d[0][None, :, :128, :288, :288])
    if cuda_enabled:
        # raw.pin_memory()
        raw = raw.cuda()
    raw = Variable(raw, volatile=True)
    # assume single GPU / batch size 1
    out = model(raw)
    pred = np.array(out.data.view([2, 128, 288, 288]).tolist(), dtype=np.float32)[1]
    from elektronn3.data.utils import h5save, save_to_h5py

    save_to_h5py([pred[0, 0], dataset.valid_d[0][0, :240, :400, :400].astype(np.float32)], "/u/pschuber/test_longier3.h5",
                 hdf5_names=["pred", "raw"])
    # h5save([pred[0, 0], dataset.valid_d[0][0, :128, :288, :288].astype(np.float32)], "/u/pschuber/test_longi.h5", keys=["pred", "raw"])
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

### TRAINING
nIters = int(19500)
wd = 0.5e-4
lr = 0.0015
opt = "sgd"
lr_dec = 0.98
bs = 2
progress_steps = 100

### DATA
d_path = '/wholebrain/scratch/j0126/'
h5_fnames = get_filepaths_from_dir('%s/barrier_gt_phil/' % d_path, ending="rawbarr-zyx.h5")
data_init_kwargs = {
    'zxy': True,
    'd_path' : '%s/barrier_gt_phil/' % d_path,
    'l_path': '%s/barrier_gt_phil/' % d_path,
    'd_files': [(os.path.split(fname)[1], 'raW') for fname in h5_fnames],
    'l_files': [(os.path.split(fname)[1], 'labels') for fname in h5_fnames],
    'aniso_factor': 2, "source": "train",
    'valid_cubes': [6], 'patch_size': (16, 128, 128),
    'grey_augment_channels': [0], "epoch_size": progress_steps*bs,
    'warp': 0.5, 'class_weights': True,
    'warp_args': {
        'sample_aniso': True,
        'perspective': True
    }}
dataset = BatchCreatorImage(**data_init_kwargs)

if opt == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
elif opt == 'adam':
    optimizer = optim.Adam(model.parameters(), weight_decay=wd, lr=lr)
elif opt == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), weight_decay=wd, lr=lr)

lr_sched = ExponentialLR(optimizer, lr_dec)

# start training
st = StoppableTrainer(model, criterion=F.nll_loss, optimizer=optimizer, dataset=dataset, batchsize=bs,
                      save_path="/u/pschuber/vnet6/", schedulers={"lr": lr_sched})
st.run(nIters)

# start ifnerence
inference(dataset, model)
