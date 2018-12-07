# Model based on https://github.com/mariogeiger/se3cnn/blob/master/examples/movie/modelnet_trained.py
# Requires https://github.com/mariogeiger/se3cnn to be installed:
#    pip install 'git+https://github.com/AMLab-Amsterdam/lie_learn'
#    pip install 'git+https://github.com/mariogeiger/se3cnn'

import torch
import torch.utils.data
import torch.nn.functional as F
from se3cnn.blocks import GatedBlock
import matplotlib
matplotlib.use('Agg')


def low_pass_filter(image, scale):
    """
    :param image: [..., x, y, z]
    :param scale: float
    """
    if scale >= 1:
        return image

    dtype = image.dtype
    device = image.device

    sigma = 0.5 * (1 / scale ** 2 - 1) ** 0.5

    size = int(1 + 2 * 2.5 * sigma)
    if size % 2 == 0:
        size += 1

    rng = torch.arange(size, dtype=dtype, device=device) - size // 2  # [-(size // 2), ..., size // 2]
    x = rng.view(size, 1, 1).expand(size, size, size)
    y = rng.view(1, size, 1).expand(size, size, size)
    z = rng.view(1, 1, size).expand(size, size, size)

    kernel = torch.exp(- (x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    out = F.conv3d(image.view(-1, 1, *image.size()[-3:]), kernel.view(1, 1, size, size, size), padding=size//2)
    out = out.view(*image.size())
    return out


class SE3Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        features = [
            (1, ),
            (4, 2, 1),
            (4, 2, 1),
            (8, 4, 2),
            (8, 4, 2),
            (4, 2, 1),
            (4, 2, 1),
            (2, )
        ]

        block_params = [
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': None},
        ]

        common_block_params = [{
                'size': 3,
                'stride': 1,
                'padding': 1,
                'dilation': 1,
                'normalization': None,
            }, {
                'size': 3,
                'stride': 1,
                'padding': 1,
                'dilation': 1,
                'normalization': 'batch',
            }, {
                'size': 3,
                'stride': 1,
                'padding': 2,
                'dilation': 2,
                'normalization': None,
            }, {
                'size': 3,
                'stride': 1,
                'padding': 2,
                'dilation': 2,
                'normalization': 'batch',
            }, {
                'size': 3,
                'stride': 1,
                'padding': 4,
                'dilation': 4,
                'normalization': None,
            }, {
                'size': 3,
                'stride': 1,
                'padding': 4,
                'dilation': 4,
                'normalization': 'batch',
            }, {
                'size': 3,
                'stride': 1,
                'padding': 8,
                'dilation': 8,
                'normalization': None,
            }, {
                'size': 1,
                'stride': 1,
                'padding': 0,
                'dilation': 1,
                'normalization': None,
            },
        ]

        assert len(block_params) + 1 == len(features)

        blocks = [
            GatedBlock(features[i], features[i + 1], **common_block_params[i], **block_params[i])
            for i in range(len(block_params))
        ]

        self.sequence = torch.nn.Sequential(*blocks)
        self.post_activations = None

    def forward(self, x):  # pylint: disable=W
        '''
        :param x: [batch, features, x, y, z]
        '''
        x = low_pass_filter(x, 1 / 3)  # dilation == 3

        self.post_activations = []
        for op in self.sequence:
            x = op(x)
            self.post_activations.append(x)
        return x
