# Trainer code for 2D and 3D Noise2Void (https://arxiv.org/abs/1811.10980)
# Adapted from https://github.com/juglab/pn2v/blob/master/pn2v/training.py,
# ported from NumPy to PyTorch and generalized to support 3D.

import torch
from torch import nn
import numpy as np
import itertools

from tqdm import tqdm

from elektronn3.training.trainer import Trainer, NaNException

import logging
logger = logging.getLogger('elektronn3log')


@torch.no_grad()
def get_stratified_coords(ratio, shape):
    """
    Produce a list of approx. ``num_pix`` random coordinates, sampled from
    ``shape`` using startified sampling. Supports n-dimensional shapes.
    """
    # total_num = torch.prod(shape).to(torch.float32)
    # sample_num = total_num * ratio
    ratio = torch.as_tensor(ratio)
    ndim = len(shape)
    shape = torch.as_tensor(shape, dtype=torch.int32)
    box_size = int(torch.round(torch.sqrt(1. / ratio)))
    coords = []
    box_counts = torch.ceil(shape.float() / box_size).int()
    for steps in itertools.product(*[range(bc) for bc in box_counts]):
        steps = torch.as_tensor(steps, dtype=torch.int32)
        co = torch.randint(0, box_size, (ndim,)) + box_size * steps
        if torch.all(co < shape):
            coords.append(co)
    if not coords:
        raise ValueError(f'ratio {ratio:.1e} is too close to zero. Choose a higher value.')
    coords = torch.stack(coords)
    return coords


# TODO: Is the hardcoded small ROI size sufficient?
@torch.no_grad()
def prepare_sample(img, ratio=1e-3, channels=None):
    """Prepare binary mask and target image for Noise2Void from a given image"""
    ndim = img.ndim - 2  # Subtract (N, C) dims
    if channels is None:
        channels = range(img.shape[1])
    inp = img.clone()
    target = img
    mask = torch.zeros_like(img)
    for n, c in itertools.product(range(img.shape[0]), channels):
        hotcoords = get_stratified_coords(ratio, img[n, c].shape)
        maxsh = np.array(img[n, c].shape) - 1
        for hc in hotcoords:
            roimin = np.clip(hc - 2, 0, None)
            roimax = np.clip(hc + 3, None, maxsh)
            roi = img[n, c, roimin[0]:roimax[0], roimin[1]:roimax[1]]
            if ndim == 3:
                roi = roi[..., roimin[2]:roimax[2]]  # slice 3rd dim if input is 3D
            rc = np.full((ndim,), 2)
            while np.all(rc == 2):
                rc = np.random.randint(0, roi.shape, (ndim,))
            repl = roi[tuple(rc)]  # Select point at rc in current ROI for replacement
            inp[(n, c, *hc)] = repl
            mask[(n, c, *hc)] = 1.0

    return inp, target, mask


def masked_mse_loss(out, target, mask):
    err = nn.functional.mse_loss(out, target, reduction='none')
    err *= mask
    loss = err.sum() / mask.sum()  # Scale by ratio of masked pixels
    return loss


class Noise2Void(nn.Module):
    def __int__(self, ratio=1e-3):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        inp, target, mask = prepare_sample(x, ratio=self.ratio)
        loss = masked_mse_loss(inp, target, mask)
        return loss


class Noise2VoidTrainer(Trainer):
    """Trainer subclass with custom training and validation code for Noise2Void training."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.criterion = Noise2Void()
        self.ratio = 1e-3

    def _train_step(self, batch):
        img = batch['inp'].to(self.device, non_blocking=True)
        dinp, dtarget, dmask = prepare_sample(img, ratio=self.ratio)
        # forward pass
        dout = self.model(dinp)
        dloss = masked_mse_loss(dout, dtarget, dmask)
        if torch.isnan(dloss):
            logger.error('NaN loss detected! Aborting training.')
            raise NaNException
        # update step
        self.optimizer.zero_grad()
        dloss.backward()
        self.optimizer.step()
        return dloss, dout

    @torch.no_grad()
    def _validate(self):
        self.model.eval()  # Set dropout and batchnorm to eval mode

        val_loss = []
        outs = []
        targets = []
        stats = {name: [] for name in self.valid_metrics.keys()}
        batch_iter = tqdm(
            enumerate(self.valid_loader), 'Validating', total=len(self.valid_loader),
            dynamic_ncols=True
        )
        for i, batch in batch_iter:
            # Everything with a "d" prefix refers to tensors on self.device (i.e. probably on GPU)
            dimg = batch['inp'].to(self.device, non_blocking=True)
            dinp, dtarget, dmask = prepare_sample(dimg, ratio=self.ratio)
            # forward pass
            dout = self.model(dinp)
            dloss = masked_mse_loss(dout, dtarget, dmask)
            val_loss.append(dloss.item())
            out = dout.detach().cpu()
            outs.append(out)
            targets.append(dtarget)
        images = {
            'inp': dinp.cpu().numpy(),
            'out': dout.cpu().numpy(),
            'target': None if dtarget is None else dtarget.cpu().numpy(),
            'fname': batch.get('fname'),
        }
        self._put_current_attention_maps_into(images)

        stats['val_loss'] = np.mean(val_loss)
        stats['val_loss_std'] = np.std(val_loss)

        for name, evaluator in self.valid_metrics.items():
            mvals = [evaluator(target, out) for target, out in zip(targets, outs)]
            if np.all(np.isnan(mvals)):
                stats[name] = np.nan
            else:
                stats[name] = np.nanmean(mvals)

        self.model.train()  # Reset model to training mode

        return stats, images


if __name__ == '__main__':
    # Demo of Noise2Void training sample generation
    import matplotlib.pyplot as plt
    import scipy.misc

    # co = get_stratified_coords(16, (8, 8, 3))
    # print(co)
    im = scipy.misc.ascent()[::2, ::2]
    imt = torch.as_tensor(im)[None, None]
    inp, target, mask = prepare_sample(imt, 1e-3)
    fig, axes = plt.subplots(ncols=3, constrained_layout=True, figsize=(20, 12))
    axes[0].imshow(im, cmap='gray')
    axes[0].set_title('Original image')
    axes[1].imshow(mask[0,0])
    axes[1].set_title('Mask')
    axes[2].imshow(inp[0,0], cmap='gray')
    axes[2].set_title('Manipulated image for Noise2Void training')
    plt.show()
