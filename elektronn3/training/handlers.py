# These are default plotting handlers that work in some common training
#  scenarios, but won't work in every case:

from typing import Dict, Optional, Callable

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import label2rgb
from torch.nn import functional as F

from elektronn3.data.utils import squash01


def plot_image(
        image: np.ndarray,
        cmap=None,
        num_classes=None,
        colorbar=True,
) -> matplotlib.figure.Figure:
    """Plots a 2D image to a malplotlib figure.

    For gray-scale images, use ``cmap='gray'``.
    For label matrices (segmentation targets or class predictions),
    specify the global number of possible classes in ``num_classes``."""

    # Determine colormap and set discrete color values if needed.
    vmin, vmax = None, None
    ticks = None
    if cmap is None and num_classes is not None:
        # Assume label matrix with qualitative classes, no meaningful order
        # Using rainbow because IMHO all actually qualitative colormaps
        #  are incredibly ugly.
        cmap = plt.cm.get_cmap('viridis', num_classes)
        ticks = np.arange(num_classes)

    if num_classes is not None:  # For label matrices
        # Prevent colormap normalization. If vmax is not set, the colormap
        #  is dynamically rescaled to fit between the minimum and maximum
        #  values of the image to be plotted. This could lead to misleading
        #  visualizations if the maximum value of the array to be plotted
        #  is less than the global maximum of classes.
        vmin = 0
        vmax = num_classes

    fig, ax = plt.subplots()
    aximg = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        fig.colorbar(aximg, ticks=ticks)  # TODO: Centered tick labels
    return fig


def _get_batch2img_function(
        batch: np.ndarray,
        z_plane: Optional[int] = None
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Defines ``batch2img`` function dynamically, depending on tensor shapes.

    ``batch2img`` slices a 4D or 5D tensor to (C, H, W) shape, moves it to
    host memory and converts it to a numpy array.
    By arbitrary choice, the first element of a batch is always taken here.
    In the 5D case, the D (depth) dimension is sliced at z_plane.

    This function is useful for plotting image samples during training.

    Args:
        batch: 4D or 5D tensor, used for shape analysis.
        z_plane: Index of the spatial plane where a 5D image tensor should
            be sliced. If not specified, this is automatically set to half
            the size of the D dimension.

    Returns:
        Function that slices a plottable 2D image out of a np.ndarray
        with batch and channel dimensions.
    """
    if batch.ndim == 5:  # (N, C, D, H, W)
        if z_plane is None:
            z_plane = batch.shape[2] // 2
        assert z_plane in range(batch.shape[2])
        return lambda x: x[0, :, z_plane]
    elif batch.ndim == 4:  # (N, C, H, W)
        return lambda x: x[0, :]
    else:
        raise ValueError('Only 4D and 5D tensors are supported.')


# TODO: Support regression scenario
def _tb_log_preview(
        trainer: 'Trainer',  # For some reason Trainer can't be imported
        z_plane: Optional[int] = None,
        group: str = 'preview'
) -> None:
    """Preview from constant region of preview batch data."""
    inp_batch = trainer.preview_batch
    out_batch = trainer._preview_inference(
        inp=inp_batch,
        tile_shape=trainer.preview_tile_shape,
        overlap_shape=trainer.preview_overlap_shape
    )
    inp_batch = inp_batch.numpy()
    if trainer.apply_softmax_for_prediction:
        out_batch = F.softmax(out_batch, 1).numpy()

    batch2img = _get_batch2img_function(out_batch, z_plane)

    inp_sh = np.array(inp_batch.shape[2:])
    out_sh = np.array(out_batch.shape[2:])
    # TODO: This does not fire yet, because out_batch is always of the same
    #       spatial shape as the input if it comes out of
    #       elektronn3.inference.Predictor...
    #       We probably need to move the padding code to the Predictor itself.
    if out_batch.shape[2:] != inp_batch.shape[2:]:
        # Zero-pad output and target to match input shape
        # Create a central slice with the size of the output
        lo = (inp_sh - out_sh) // 2
        hi = inp_sh - lo
        slc = tuple([slice(None)] * 2 + [slice(l, h) for l, h in zip(lo, hi)])

        padded_out_batch = np.zeros(
            (inp_batch.shape[0], out_batch.shape[1], *inp_batch.shape[2:]),
            dtype=out_batch.dtype
        )
        padded_out_batch[slc] = out_batch
        out_batch = padded_out_batch

    if inp_batch.ndim == 5:  # 5D tensors -> 3D images -> We can make 2D videos out of them
        # See comments in the 5D section in _tb_log_sample_images
        inp_video = squash01(inp_batch)
        trainer.tb.add_video(
            f'{group}_vid/inp', inp_video, global_step=trainer.step
        )
        for c in range(out_batch.shape[1]):
            trainer.tb.add_video(
                f'{group}_vid/out{c}',
                squash01(out_batch[:, c][None]),  # Slice C, but keep dimensions intact
                global_step=trainer.step
            )

    out_slice = batch2img(out_batch)
    pred_slice = out_slice.argmax(0)

    for c in range(out_slice.shape[0]):
        trainer.tb.add_figure(
            f'{group}/c{c}',
            plot_image(out_slice[c], cmap='gray'),
            trainer.step
        )
    trainer.tb.add_figure(
        f'{group}/pred',
        plot_image(pred_slice, num_classes=trainer.num_classes),
        trainer.step
    )

    # This is only run once per training, because the ground truth for
    # previews is constant (always the same preview inputs/targets)
    if trainer._first_plot:
        inp_slice = batch2img(trainer.preview_batch)[0]
        trainer.tb.add_figure(
            f'{group}/inp',
            plot_image(inp_slice, cmap='gray'),
            global_step=0
        )
        trainer._first_plot = False


# TODO: There seems to be an issue with inp-target mismatches when batch_size > 1
def _tb_log_sample_images(
        trainer: 'Trainer',
        images: Dict[str, np.ndarray],
        z_plane: Optional[int] = None,
        group: str = 'sample'
) -> None:
    """Preview from last training/validation sample

    Since the images are chosen randomly from the training/validation set
    they come from random regions in the data set.

    Note: Training images are possibly augmented, so the plots may look
        distorted/weirdly colored.
    """
    # Always only use the first element of the batch dimension
    inp_batch = images['inp'][:1]
    target_batch = images['target'][:1]
    out_batch = images['out'][:1]

    if trainer.apply_softmax_for_prediction:
        out_batch = F.softmax(torch.as_tensor(out_batch), 1).numpy()

    batch2img = _get_batch2img_function(out_batch, z_plane)

    inp_slice = batch2img(images['inp'])[0]

    # TODO: Support one-hot targets
    # Check if the network is being trained for classification
    is_classification = target_batch.ndim == out_batch.ndim - 1
    # If it's not classification, we assume a regression scenario
    is_regression = np.all(target_batch.shape == out_batch.shape)
    # If not exactly one of the scenarios is detected, we can't handle it
    assert is_regression != is_classification

    if is_classification:
        # In classification scenarios, targets have one dim less than network
        #  outputs, so if we want to use the same batch2img function for
        #  targets, we have to add an empty channel axis to it after the N dimension
        target_batch = target_batch[:, None]

    inp_sh = np.array(inp_batch.shape[2:])
    out_sh = np.array(out_batch.shape[2:])
    if out_batch.shape[2:] != inp_batch.shape[2:]:
        # Zero-pad output and target to match input shape
        # Create a central slice with the size of the output
        lo = (inp_sh - out_sh) // 2
        hi = inp_sh - lo
        slc = tuple([slice(None)] * 2 + [slice(l, h) for l, h in zip(lo, hi)])

        padded_out_batch = np.zeros(
            (inp_batch.shape[0], out_batch.shape[1], *inp_batch.shape[2:]),
            dtype=out_batch.dtype
        )
        padded_out_batch[slc] = out_batch
        out_batch = padded_out_batch

        # Assume that target has the same shape as the output and pad it, too
        padded_target_batch = np.zeros(inp_batch.shape, dtype=target_batch.dtype)
        padded_target_batch[slc] = target_batch
        target_batch = padded_target_batch



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
            f'Can\'t prepare targets of shape {target_batch.shape} for plotting.'
        )

    if inp_batch.ndim == 5:  # 5D tensors -> 3D images -> We can make 2D videos out of them
        # We re-interpret the D dimension as the temporal dimension T of the video
        #  -> (N, C, T, H, W)
        # Inputs and outputs need to be squashed to the (0, 1) intensity range
        #  for video rendering, otherwise they will appear as random noise.
        # Since tensorboardX's add_video only supports (N, C, T, H, W) tensors,
        #  we have to add a fake C dimension to the (N, D, H, W) target tensors
        #  and replace the C dimension of output tensors by empty C dimensions
        #  to visualize each channel separately.
        inp_video = squash01(inp_batch)
        target_video = target_batch
        if target_video.ndim == 4:
            target_video = target_video[:, None]
        trainer.tb.add_video(
            f'{group}_vid/inp', inp_video, global_step=trainer.step
        )
        trainer.tb.add_video(
            f'{group}_vid/target', target_video, global_step=trainer.step
        )
        for c in range(out_batch.shape[1]):
            trainer.tb.add_video(
                f'{group}_vid/out{c}',
                squash01(out_batch[:, c][None]),  # Slice C, but keep dimensions intact
                global_step=trainer.step
            )

    trainer.tb.add_figure(
        f'{group}/inp',
        plot_image(inp_slice, cmap='gray'),
        global_step=trainer.step
    )
    trainer.tb.add_figure(
        f'{group}/target',
        plot_image(target_slice, num_classes=trainer.num_classes),
        global_step=trainer.step
    )

    # Only make pred and overlay plots in classification scenarios
    if is_classification:
        # Plot each class probmap individually
        for c in range(out_slice.shape[0]):
            trainer.tb.add_figure(
                f'{group}/c{c}',
                plot_image(out_slice[c], cmap='gray'),
                global_step=trainer.step
            )

        pred_slice = out_slice.argmax(0)
        trainer.tb.add_figure(
            f'{group}/pred_slice',
            plot_image(pred_slice, num_classes=trainer.num_classes),
            global_step=trainer.step
        )

        inp01 = squash01(inp_slice)  # Squash to [0, 1] range for label2rgb and plotting
        target_slice_ov = label2rgb(target_slice, inp01, bg_label=0, alpha=trainer.overlay_alpha)
        pred_slice_ov = label2rgb(pred_slice, inp01, bg_label=0, alpha=trainer.overlay_alpha)
        trainer.tb.add_figure(
            f'{group}/target_overlay',
            plot_image(target_slice_ov, colorbar=False),
            global_step=trainer.step
        )
        trainer.tb.add_figure(
            f'{group}/pred_overlay',
            plot_image(pred_slice_ov, colorbar=False),
            global_step=trainer.step
        )
        # TODO: Synchronize overlay colors with pred_slice- and target_slice colors
        # TODO: What's up with the colorbar in overlay plots?
        # TODO: When plotting overlay images, they appear darker than they should.
        #       This normalization issue gets worse with higher alpha values
        #       (i.e. with more contribution of the overlayed label map).
        #       Don't know how to fix this currently.
    elif is_regression:
        trainer.tb.add_figure(f'{group}/out', plot_image(out_slice), global_step=trainer.step)
