# These are default plotting handlers that work in some common training
#  scenarios, but won't work in every case:

from typing import Optional, Dict

import numpy as np
import torch
from torch.nn import functional as F
from skimage.color import label2rgb

from elektronn3.data.utils import squash01


# TODO: Support regression scenario
def _tb_log_preview(
        trainer: 'Trainer',  # For some reason Trainer can't be imported
        z_plane: Optional[int] = None,
        group: str = 'preview'
) -> None:
    """Preview from constant region of preview batch data."""
    inp_batch = trainer.preview_batch.numpy()
    # TODO: Replace this with elektronn3.inference.Predictor usage
    out_batch = trainer._preview_inference(
        inp=inp_batch,
        tile_shape=trainer.preview_tile_shape,
        overlap_shape=trainer.preview_overlap_shape
    )
    if trainer.apply_softmax_for_prediction:
        out_batch = F.softmax(out_batch, 1)  # Apply softmax before plotting

    batch2img = trainer._get_batch2img_function(out_batch, z_plane)

    out_slice = batch2img(out_batch)
    pred_slice = out_slice.argmax(0)

    for c in range(out_slice.shape[0]):
        trainer.tb.log_image(f'{group}/c{c}', out_slice[c], trainer.step, cmap='gray')
    trainer.tb.log_image(f'{group}/pred', pred_slice, trainer.step, num_classes=trainer.num_classes)

    # This is only run once per training, because the ground truth for
    # previews is constant (always the same preview inputs/targets)
    if trainer._first_plot:
        inp_slice = batch2img(trainer.preview_batch)[0]
        trainer.tb.log_image(f'{group}/inp', inp_slice, step=0, cmap='gray')
        trainer._first_plot = False


# TODO: There seems to be an issue with inp-target mismatches when batch_size > 1
def _tb_log_sample_images(
        trainer: 'Trainer',
        images: Dict[str, torch.Tensor],
        z_plane: Optional[int] = None,
        group: str = 'sample'
) -> None:
    """Preview from last training/validation sample

    Since the images are chosen randomly from the training/validation set
    they come from random regions in the data set.

    Note: Training images are possibly augmented, so the plots may look
        distorted/weirdly colored.
    """

    out_batch = images['out']
    if trainer.apply_softmax_for_prediction:
        out_batch = F.softmax(out_batch, 1)  # Apply softmax before plotting

    batch2img = trainer._get_batch2img_function(out_batch, z_plane)

    inp_slice = batch2img(images['inp'])[0]
    target_batch = images['target']

    # Check if the network is being trained for classification
    is_classification = target_batch.dim() == out_batch.dim() - 1
    # If it's not classification, we assume a regression scenario
    is_regression = np.all(target_batch.shape == out_batch.shape)
    # If not exactly one of the scenarios is detected, we can't handle it
    assert is_regression != is_classification

    if is_classification:
        # In classification scenarios, targets have one dim less than network
        #  outputs, so if we want to use the same batch2img function for
        #  targets, we have to add an empty channel axis to it after the N dimension
        target_batch = target_batch[:, None]

    target_slice = batch2img(target_batch)
    out_slice = batch2img(out_batch)
    if is_classification:
        target_slice = target_slice.squeeze(0)  # Squeeze empty axis that was added above
    elif target_slice.shape[0] == 3:  # Assume RGB values
        # RGB images need to be transposed to (H, W, C) layout so matplotlib can handle them
        target_slice = np.moveaxis(target_slice, 0, -1)  # (C, H, W) -> (H, W, C)
        out_slice = np.moveaxis(out_slice, 0, -1)
    else:
        raise RuntimeError(
            f'Can\t prepare targets of shape {target_batch.shape} for plotting.'
        )

    trainer.tb.log_image(f'{group}/inp', inp_slice, step=trainer.step, cmap='gray')
    trainer.tb.log_image(
        f'{group}/target', target_slice, step=trainer.step, num_classes=trainer.num_classes
    )

    # Only make pred and overlay plots in classification scenarios
    if is_classification:
        # Plot each class probmap individually
        for c in range(out_slice.shape[0]):
            trainer.tb.log_image(f'{group}/c{c}', out_slice[c], step=trainer.step, cmap='gray')

        pred_slice = out_slice.argmax(0)
        trainer.tb.log_image(
            f'{group}/pred_slice', pred_slice, step=trainer.step, num_classes=trainer.num_classes
        )

        inp01 = squash01(inp_slice)  # Squash to [0, 1] range for label2rgb and plotting
        target_slice_ov = label2rgb(target_slice, inp01, bg_label=0, alpha=trainer.overlay_alpha)
        pred_slice_ov = label2rgb(pred_slice, inp01, bg_label=0, alpha=trainer.overlay_alpha)
        trainer.tb.log_image(
            f'{group}/target_overlay', target_slice_ov, step=trainer.step, colorbar=False
        )
        trainer.tb.log_image(
            f'{group}/pred_overlay', pred_slice_ov, step=trainer.step, colorbar=False
        )
        # TODO: Synchronize overlay colors with pred_slice- and target_slice colors
        # TODO: What's up with the colorbar in overlay plots?
        # TODO: When plotting overlay images, they appear darker than they should.
        #       This normalization issue gets worse with higher alpha values
        #       (i.e. with more contribution of the overlayed label map).
        #       Don't know how to fix this currently.
    elif is_regression:
        trainer.tb.log_image(f'{group}/out', out_slice, step=trainer.step)
