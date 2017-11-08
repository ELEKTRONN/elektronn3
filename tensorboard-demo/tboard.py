#!/usr/bin/env python3

# Tensorboard-PyTorch integration test code based on the usage example at
# https://github.com/lanpa/tensorboard-pytorch/blob/162034215abe948d5139c0eafba842f818940c7d/README.md
#
# All files are written to subdirectories of ~/tboard/.
# To test the tensorboard server on this script's output, run
# $ tensorboard --logdir ~/tboard

import os
from socket import gethostname
from datetime import datetime

import numpy as np
import torch
import torchvision
import tensorboardX


comment = ''
root_dir = os.path.expanduser('~/tboard/')
log_dir = os.path.join(
    root_dir,
    datetime.now().strftime('%b%d_%H-%M-%S') + '_' + gethostname()
)

resnet18 = torchvision.models.resnet18(False)
writer = tensorboardX.SummaryWriter(log_dir=log_dir)
sample_rate = 44100
freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]

for n_iter in range(100):
    s1 = torch.rand(1) # value to keep
    s2 = torch.rand(1)
    writer.add_scalar('data/scalar1', s1[0], n_iter)  # data grouping by `slash`
    writer.add_scalar('data/scalar2', s2[0], n_iter)
    writer.add_scalars('data/scalar_group', {"xsinx":n_iter*np.sin(n_iter),
                                             "xcosx":n_iter*np.cos(n_iter),
                                             "arctanx": np.arctan(n_iter)}, n_iter)
    x = torch.rand(32, 3, 64, 64)  # output from network
    if n_iter%10==0:
        x = torchvision.utils.make_grid(x, normalize=True, scale_each=True)
        writer.add_image('Image', x, n_iter)
        x = torch.zeros(sample_rate*2)
        for i in range(x.size(0)):
            x[i] = np.cos(freqs[n_iter//10]*np.pi*float(i)/float(sample_rate))
        writer.add_audio('myAudio', x, n_iter, sample_rate=sample_rate)
        writer.add_text('Text', 'text logged at step:'+str(n_iter), n_iter)
        for name, param in resnet18.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)

dataset = torchvision.datasets.MNIST(
    os.path.join(root_dir, 'mnist'),
    train=False,
    download=True
)
images = dataset.test_data[:100].float()
label = dataset.test_labels[:100]
features = images.view(100, 784)
writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))

# export scalar data to JSON for external processing
writer.export_scalars_to_json(os.path.join(root_dir, 'all_scalars.json'))

writer.close()