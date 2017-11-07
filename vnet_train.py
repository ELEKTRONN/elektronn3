#!/usr/bin/env python3
import os
import datetime
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch import optim
from torch.utils import data

# Don't move this stuff, it needs to be run this early to work
from elektronn3 import select_mpl_backend
mpl_backend = 'agg'  # TODO: Make this a CLI option
select_mpl_backend(mpl_backend)

from elektronn3.data.cnndata import BatchCreatorImage
from elektronn3.data.utils import get_filepaths_from_dir, save_to_h5py
from elektronn3.training.trainer import StoppableTrainer
from elektronn3.neural.vnet import VNet
from elektronn3.neural.fcn import fcn32s
from elektronn3 import cuda_enabled
from torch.optim.lr_scheduler import ExponentialLR

### USER PATHS
path_prefix = os.path.expanduser('~/vnet/')
os.makedirs(path_prefix, exist_ok=True)
state_dict_path = '/u/pschuber/vnet/vnet-99900-model.pkl'  # TODO: Make variable
test_cube_path = '/u/pschuber/test_pred.h5'  # TODO: Make variable

timestamp = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
save_name = 'vnet__' + timestamp
save_path = os.path.join(path_prefix, save_name)


### TRAINING
nIters = int(500000)
wd = 0.5e-4
lr = 0.0004
opt = 'adam'
lr_dec = 0.999
bs = 1
progress_steps = 100

### UTILS
def pred(dataset):
    model = VNet(relu=False)
    state_dict = torch.load(state_dict_path)
    # corr_state_dict = state_dict.copy()
    # for k, v in state_dict.items():
    #     corr_state_dict[k[7:]] = v
    #     del corr_state_dict[k]
    # state_dict = corr_state_dict
    if bs >= 4:
        model = nn.parallel.DataParallel(model, device_ids=[0, 1])
    if cuda_enabled:
        model = model.cuda()
        model.load_state_dict(state_dict)
    inference(dataset, model, test_cube_path)
    raise ()

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


def inference(dataset, model, fname):
    # logger.info("Starting preview prediction")
    model.eval()
    raw = torch.from_numpy(dataset.valid_d[0][None, :, :160, :288, :288])
    if cuda_enabled:
        # raw.pin_memory()
        raw = raw.cuda()
    raw = Variable(raw, volatile=True)
    # assume single GPU / batch size 1
    out = model(raw)
    clf = out.data.max(1)[1].view(raw.size())
    pred = np.array(clf.tolist(), dtype=np.float32)[0, 0]
    save_to_h5py([pred, dataset.valid_d[0][0, :160, :288, :288].astype(np.float32)], fname,
                 hdf5_names=["pred", "raw"])
    save_to_h5py([np.exp(np.array(out.data.view([1, 2, 160, 288, 288]).tolist())[0, 1], dtype=np.float32)], fname+"prob.h5",
                 hdf5_names=["prob"])


### DATA
d_path = '/wholebrain/scratch/j0126/'  # TODO: Make variable
h5_fnames = get_filepaths_from_dir('%s/barrier_gt_phil/' % d_path, ending="rawbarr-zyx.h5")[:2]
data_init_kwargs = {
    'zxy': True,
    'd_path' : '%s/barrier_gt_phil/' % d_path,
    'l_path': '%s/barrier_gt_phil/' % d_path,
    'd_files': [(os.path.split(fname)[1], 'raW') for fname in h5_fnames],
    'l_files': [(os.path.split(fname)[1], 'labels') for fname in h5_fnames],
    'aniso_factor': 2, "source": "train",
    'valid_cubes': [6], 'patch_size': (64, 64, 64),
    'grey_augment_channels': [0], "epoch_size": progress_steps*bs,
    'warp': 0.5, 'class_weights': True,
    'warp_args': {
        'sample_aniso': True,
        'perspective': True
    }}
dataset = BatchCreatorImage(**data_init_kwargs)

### MODEL
torch.manual_seed(0)
if cuda_enabled:
    torch.cuda.manual_seed(0)
# model = VNet(relu=False)
model = fcn32s(learned_billinear=False)
if bs >= 4 and cuda_enabled:
    model = nn.parallel.DataParallel(model, device_ids=[0, 1])
if cuda_enabled:
    model = model.cuda()
model.apply(weights_init)

if opt == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
elif opt == 'adam':
    optimizer = optim.Adam(model.parameters(), weight_decay=wd, lr=lr)
elif opt == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), weight_decay=wd, lr=lr)

lr_sched = ExponentialLR(optimizer, lr_dec)

# loss
criterion = CrossEntropyLoss(weight=dataset.class_weights)

# start training
st = StoppableTrainer(model, criterion=criterion, optimizer=optimizer,
                      dataset=dataset, batchsize=bs, save_path=save_path, schedulers={"lr": lr_sched})
st.run(nIters)

# start ifnerence
# inference(dataset, model)
