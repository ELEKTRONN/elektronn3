# These are default plotting handlers that work in some common training
#  scenarios, but won't work in every case:

import logging
import os

from typing import Dict, Optional, Callable, Sequence

import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import torch
from torch.nn import functional as F

from elektronn3.data.utils import squash01
from elektronn3.data.transforms import RemapTargetIDs

E3_CMAP: str = os.getenv('E3_CMAP')

logger = logging.getLogger('elektronn3log')


def get_cmap(out_channels: int):
    if E3_CMAP is not None:
        cmname = E3_CMAP
    # Else, use defaults:
    elif out_channels <= 10:
        cmname = 'tab10'
    elif out_channels <= 20:
        cmname = 'tab20'
    else:
        raise RuntimeError(
            f'Default cmaps only support up to 20 colors, which are not enough to label '
            f'{out_channels} different output channels.\nPlease set a different cmap '
            'with the E3_CMAP envvar.'
        )
    return matplotlib.cm.get_cmap(cmname, out_channels)


def plot_image(
        image: np.ndarray,
        overlay: Optional[np.ndarray] = None,
        overlay_alpha=0.5,
        cmap=None,
        colorbar=True,
        filename=None,
        vmin=None,
        vmax=None
) -> matplotlib.figure.Figure:
    """Plots a 2D image to a malplotlib figure.

    For gray-scale images, use ``cmap='gray'``.
    For label matrices (segmentation targets or class predictions),
    specify the global number of possible classes in ``out_channels``."""

    # Determine colormap and set discrete color values if needed.
    ticks = None
    ticklabels = None
    cmap_name = cmap if isinstance(cmap, str) else cmap.name
    if cmap_name in {E3_CMAP, 'tab10', 'tab20'}: # qualitative cmap
        ticks = np.linspace(0.5, vmax - 0.5, vmax) # 0.5 for centered ticks
        ticklabels = np.arange(vmax)

    fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 10))
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if overlay is None:
        aximg = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
    else:
        ax.imshow(image, cmap='gray')
        masked_overlay = np.ma.masked_where(overlay == 0, overlay)
        aximg = ax.imshow(masked_overlay, cmap=cmap, vmin=vmin, vmax=vmax, alpha=overlay_alpha, interpolation='none')
    if filename is not None:
        max_filename_length = 50  # Truncate long file names from the left
        if len(filename) > max_filename_length:
            filename = f'...{filename[-max_filename_length:]}'
        ax.set_title(filename)
    if colorbar:
        bar = fig.colorbar(aximg, ticks=ticks)
        if ticklabels is not None:
            bar.set_ticklabels(ticklabels)
        bar.solids.set(alpha=1) # otherwise uses image’s opacity
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
    elif batch.ndim == 2:  # (N, C) -> img2scalar
        return lambda x: x[None, ]  # (1, N, C)  -> Image will show N x C probabilities
    else:
        raise ValueError('Only 4D and 5D tensors are supported.')


def write_to_kzip(trainer: 'Trainer', pred_batch: np.ndarray) -> None:
    from knossos_utils import KnossosDataset
    ks = trainer.knossos_preview_config
    if isinstance(ks['dataset'], str):
        dataset_path = ks['dataset']
    else:
        dataset_path = ks['dataset'][0]
    ds = KnossosDataset(dataset_path)
    seg = pred_batch[0].swapaxes(0, 2)  # (N, D, H, W) -> (W, H, D)

    # Set movement are in k.zip
    area_min = ks['offset']
    area_sz = ks['size']
    anno_str = f"""<?xml version="1.0" encoding="UTF-8"?>
<things>
    <parameters>
        <MovementArea min.x="{area_min[0]}" min.y="{area_min[1]}" min.z="{area_min[2]}" size.x="{area_sz[0]}" size.y="{area_sz[1]}" size.z="{area_sz[2]}"/>
    </parameters>
    <comments/>
    <branchpoints/>
</things>"""

    kzip_path = f'{trainer.save_path}/preview_{trainer.step}.k.zip'
    logger.info(f'Writing preview inference to {kzip_path}')
    ds.save_to_kzip(
        data=seg,
        data_mag=ks.get('mag', 1),
        kzip_path=kzip_path,
        offset=ks['offset'],
        mags=ks.get('target_mags', [1, 2]),
        gen_mergelist=False,
        upsample=False,
        fast_resampling=False,
        annotation_str=anno_str
    )


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
        inference_kwargs=trainer.inference_kwargs,
    ).to(torch.float32)
    inp_batch = inp_batch.numpy()
    if trainer.inference_kwargs['apply_softmax']:
        out_batch = F.softmax(out_batch, 1).numpy()
    else:
        out_batch = out_batch.numpy()

    batch2img = _get_batch2img_function(out_batch, z_plane)

    if inp_batch.ndim == 5 and trainer.enable_videos:
        # 5D tensors -> 3D images -> We can make 2D videos out of them
        # See comments in the 5D section in _tb_log_sample_images
        inp_video = squash01(inp_batch)
        # (N, C, T=D, H, W) -> (N, T=D, C, H, W) because of add_video API
        inp_video = np.swapaxes(inp_video, 1, 2)
        trainer.tb.add_video(
            f'{group}_vid/inp', inp_video, global_step=trainer.step
        )
        for c in range(out_batch.shape[1]):
            outc_video = squash01(out_batch[:, c][None])  # Slice C, but keep dimensions intact
            # (N, C=1, T=D, H, W) -> (N, T=D, C=1, H, W)
            outc_video = np.moveaxis(outc_video, 1, 2)
            trainer.tb.add_video(
                f'{group}_vid/out{c}',
                outc_video,
                global_step=trainer.step
            )

    out_slice = batch2img(out_batch)
    pred_slice = out_slice.argmax(0)

    if trainer.knossos_preview_config is not None:
        pred_batch = out_batch.argmax(1)
        remap_ids = trainer.knossos_preview_config.get('remap_ids')
        if remap_ids is not None:
            remap = RemapTargetIDs(remap_ids, reverse=True)
            _, pred_batch = remap(None, pred_batch)

        write_to_kzip(trainer, pred_batch)

    for c in range(out_slice.shape[0]):
        trainer.tb.add_figure(
            f'{group}/out{c}',
            plot_image(out_slice[c], cmap='gray'),
            trainer.step
        )
    class_cmap = get_cmap(trainer.max_plot_id)
    trainer.tb.add_figure(
        f'{group}/pred',
        plot_image(pred_slice, vmin=0, vmax=trainer.max_plot_id, cmap=class_cmap),
        trainer.step
    )
    inp_slice = batch2img(inp_batch)[0]
    trainer.tb.add_figure(
        f'{group}/pred_overlay',
        plot_image(inp_slice, overlay=pred_slice, overlay_alpha=trainer.overlay_alpha, vmin=0, vmax=trainer.max_plot_id, cmap=class_cmap),
        global_step=trainer.step
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
    target_batch = images.get('target')
    if target_batch is not None:
        target_batch = target_batch[:1]
    out_batch = images['out'][:1]

    name = images.get('fname')
    if name is not None:
        name = name[0]

    continuous_cmap = 'viridis'

    if trainer.inference_kwargs['apply_softmax']:
        out_batch = F.softmax(torch.as_tensor(out_batch, dtype=torch.float32), 1).numpy()
    elif trainer.inference_kwargs.get('apply_sigmoid'):
        out_batch = torch.sigmoid(torch.as_tensor(out_batch, dtype=torch.float32)).numpy()
    else:
        out_batch = out_batch.astype(np.float32)

    batch2img_inp = _get_batch2img_function(inp_batch, z_plane)
    inp_slice = batch2img_inp(images['inp'])

    uinp_batch = images.get('unlabeled')
    if uinp_batch is not None:
        trainer.tb.add_figure(
            f'{group}/unlabeled_inp',
            plot_image(batch2img_inp(uinp_batch['inp'].cpu().numpy())[0], cmap='gray'),
            global_step=trainer.step
        )

    # TODO: Support one-hot targets
    # TODO: Support multi-label targets
    # TODO: Output vis missing if target_batch is None
    # Check if the network is being trained for classification with class index target tensors
    if target_batch is not None:
        is_classification = target_batch.ndim == out_batch.ndim - 1
        class_cmap = get_cmap(trainer.max_plot_id)
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
    if out_batch.shape[2:] != inp_batch.shape[2:] and not (out_batch.ndim == 2):
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
        if target_batch is not None:
            padded_target_batch = np.zeros((*target_batch.shape[:2], *inp_batch.shape[2:]), dtype=target_batch.dtype)
            padded_target_batch[slc] = target_batch
            target_batch = padded_target_batch

    target_cmap = E3_CMAP
    batch2img = _get_batch2img_function(out_batch, z_plane)
    if target_batch is not None:
        target_slice = batch2img(target_batch)
    out_slice = batch2img(out_batch)
    if target_batch is not None:
        if is_classification:
            target_slice = target_slice.squeeze(0)  # Squeeze empty axis that was added above
        elif target_slice.shape[0] == 3:  # Assume RGB values
            # RGB images need to be transposed to (H, W, C) layout so matplotlib can handle them
            target_slice = np.moveaxis(target_slice, 0, -1)  # (C, H, W) -> (H, W, C)
            out_slice = np.moveaxis(out_slice, 0, -1)

    if inp_batch.ndim == 5 and trainer.enable_videos:
        # 5D tensors -> 3D images -> We can make 2D videos out of them
        # We re-interpret the D dimension as the temporal dimension T of the video
        #  -> (N, T, C, H, W)
        # Inputs and outputs need to be squashed to the (0, 1) intensity range
        #  for video rendering, otherwise they will appear as random noise.
        # Since tensorboardX's add_video only supports (N, T, C, H, W) tensors,
        #  we have to add a fake C dimension to the (N, D, H, W) target tensors
        #  and replace the C dimension of output tensors by empty C dimensions
        #  to visualize each channel separately.
        inp_video = squash01(inp_batch)
        # (N, C, T=D, H, W) -> (N, T=D, C, H, W) because of add_video API
        inp_video = np.swapaxes(inp_video, 1, 2)
        trainer.tb.add_video(
            f'{group}_vid/inp', inp_video, global_step=trainer.step
        )
        if target_batch is not None:
            target_video = target_batch
            if target_video.ndim == 4:
                # TODO: This fails with 2D multi-channel targets. Handle these reliably
                target_video = target_video[:, None]
            target_video = np.swapaxes(target_video, 1, 2)
            trainer.tb.add_video(
                f'{group}_vid/target', target_video, global_step=trainer.step
            )
        for c in range(out_batch.shape[1]):
            outc_video = squash01(out_batch[:, c][None])  # Slice C, but keep dimensions intact
            # (N, C=1, T=D, H, W) -> (N, T=D, C=1, H, W)
            outc_video = np.moveaxis(outc_video, 1, 2)
            trainer.tb.add_video(
                f'{group}_vid/out{c}',
                outc_video,
                global_step=trainer.step
            )
        # TODO: Add output and target overlay videos (not straightforward
        #       because the 2D overlay code currently uses matplotlib)

    for channel in range(inp_slice.shape[0]):
        trainer.tb.add_figure(
            f'{group}/inp{channel}',
            plot_image(inp_slice[channel], cmap='gray', filename=name),
            global_step=trainer.step
        )

    if target_batch is not None:
        _out_channels = trainer.max_plot_id if is_classification else None
        _cmap = class_cmap if is_classification else continuous_cmap
        if target_slice.ndim == 2:
            trainer.tb.add_figure(
                f'{group}/target',
                plot_image(
                    target_slice, vmin=0, vmax=trainer.max_plot_id, filename=name, cmap=_cmap
                    # vmin=0., vmax=1.
                ),
                global_step=trainer.step
            )
        elif target_slice.ndim == 3:
            for c in range(target_slice.shape[0]):
                trainer.tb.add_figure(
                    f'{group}/target{c}',
                    plot_image(
                        target_slice[c], vmin=0, vmax=trainer.max_plot_id, filename=name, cmap=_cmap
                        # vmin=0., vmax=1.
                    ),
                    global_step=trainer.step
                )

    for key, img in images.items():
        if key.startswith('att'):
            trainer.tb.add_figure(
                f'{group}/{key}',
                plot_image(img, cmap='viridis'),
                global_step=trainer.step
            )

    # Plot each output channel c individually as "out{c}"
    for c in range(out_slice.shape[0]):
        trainer.tb.add_figure(
            f'{group}/out{c}',
            plot_image(
                out_slice[c], cmap=continuous_cmap, filename=name,
                # vmin=0., vmax=1.
            ),
            global_step=trainer.step
        )

    # Only make pred and overlay plots in classification scenarios
    if target_batch is not None:
        if is_classification:
            pred_slice = out_slice.argmax(0)
            trainer.tb.add_figure(
                f'{group}/pred_slice',
                plot_image(pred_slice, vmin=0, vmax=trainer.max_plot_id, cmap=class_cmap, filename=name),
                global_step=trainer.step
            )
            if target_batch is not None and not target_batch.ndim == 2:  # TODO: Make this condition more reliable and document it
                for c in range(inp_slice.shape[0]):
                    trainer.tb.add_figure(
                        f'{group}/target_overlay{c}',
                        plot_image(inp_slice[c], overlay=target_slice, overlay_alpha=trainer.overlay_alpha, vmin=0, vmax=trainer.max_plot_id, cmap=class_cmap, filename=name),
                        global_step=trainer.step
                    )
                    trainer.tb.add_figure(
                        f'{group}/pred_overlay',
                        plot_image(inp_slice[c], overlay=pred_slice, overlay_alpha=trainer.overlay_alpha, vmin=0, vmax=trainer.max_plot_id, cmap=class_cmap, filename=name),
                        global_step=trainer.step
                    )


def _tb_log_sample_images_all_img(
        trainer: 'Trainer',
        images: Dict[str, np.ndarray],
        z_plane: Optional[int] = None,
        group: str = 'sample'
) -> None:
    """Tensorboard plotting handler that plots all arrays in the ``images``
    dict as 2D grayscale images. Multi-channel images are split along the
    C dimension and plotted separately.
    """
    name = images.pop('fname', [None])[0]
    # TODO: Clean up/remove the messy name handling. Figure out how to pass non-image data cleanly.

    for key, img in images.items():
        img = img[:1]  # Always only use the first element of the batch dimension
        batch2img = _get_batch2img_function(img, z_plane)
        img = batch2img(img)
        if img.shape[0] == 1:
            trainer.tb.add_figure(
                f'{group}/{key}',
                plot_image(img[0], cmap='gray', filename=name),
                global_step=trainer.step
            )
        else:
            for c in range(img.shape[0]):
                trainer.tb.add_figure(
                    f'{group}/{key}{c}',
                    plot_image(img[c], cmap='gray', filename=name),
                    global_step=trainer.step
                )
