# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Martin Drawitsch
import itertools
import logging
import os
import time
import zipfile
from collections import OrderedDict
from typing import Optional, Tuple, Union, Callable, Sequence

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


# TODO: It's confusing that tiled_apply expects out_shape to include the N dim, but
#     Predictor has a parameter with the same name but which doesn't include N.

logger = logging.getLogger('elektronn3log')


def _extend_nc(spatial_slice: Sequence[slice]) -> Tuple[slice, ...]:
    """Extend a spatial slice ([D,] H, W) to also include the non-spatial (N, C) dims."""
    # Slice everything in N and C dims (equivalent to [:, :] in direct notation)
    nonspatial_slice = [slice(None)] * 2
    return tuple(nonspatial_slice + list(spatial_slice))


def tiled_apply(
        func: Callable[[torch.Tensor], torch.Tensor],
        inp: torch.Tensor,
        device,
        dtype,
        tile_shape: Sequence[int],
        overlap_shape: Sequence[int],
        offset: Optional[Sequence[int]],
        out_shape: Sequence[int],
        argmax_with_threshold = None,
        verbose: bool = False
) -> torch.Tensor:
    """Splits a tensor into overlapping tiles and applies a function on them independently.

    Each tile of the output results from applying a callable ``func`` on an
    input tile which is sliced from a region that has the same center but a
    larger extent (overlapping with other input regions in the vicinity).
    Input tensors are also padded with zeros to at the boundaries according to
    the `overlap_shape``to enable consistent tile shapes.

    The overlapping behavior prevents imprecisions of CNNs (and image
    processing algorithms in general) that appear near the boundaries of
    inner tiles when applying them on a tiled representation of the input.

    By default this function assumes that ``inp.shape[2:] == func(inp).shape[2:]``,
    i.e. that the function keeps the spatial shape unchanged.
    If ``func`` reduces the spatial shape (e.g. by performing valid convolutions)
    and its output is centered w.r.t. the input, you should specify this shape
    offset in the ``offset`` parameter. This is the same offset that
    :py:class:`elektronn3.data.cnndata.PatchCreator` expects.

    It can run on GPU or CPU transparently, depending on the device that
    ``inp`` is allocated on.

    Although this function is mainly intended for the purpose of neural network
    inference, ``func`` doesn't have to be a neural network but can be
    any ``Callable[[torch.Tensor], torch.Tensor]`` that operates on n-dimensional
    image data of shape (N, C, ...) and preserves spatial shape or has a
    constant ``offset``.
    ("..." is a placeholder for the spatial dimensions, so for example
    H(eight) and W(idth).)


    Args:
        func: Function to be applied on input tiles. Usually this is a neural
            network model.
        inp: Input tensor, usually of shape (N, C, [D,], H, W).
            n-dimensional tensors of shape (N, C, ...) are supported.
        tile_shape: Spatial shape of the output tiles to use for inference.
        overlap_shape: Spatial shape of the overlap by which input tiles are
            extended w.r.t. the output ``tile_shape``.
        offset: Determines the offset by which the output contents are shifted
            w.r.t. the inputs by ``func``.
            This should generally be set to half the spatial shape difference
            between inputs and outputs:

            >>> in_sh = np.array(inp.shape[2:])
            >>> out_sh = np.array(func(inp).shape[2:])
            >>> offset = (in_sh - out_sh) // 2

        out_shape: Expected shape of the output tensor that would result from
            applying ``func`` to ``inp`` (``func(inp).shape``).
            It doesn't just refer to spatial shape, but to the actual tensor
            shape including N and C dimensions.
            Note: ``func(inp)`` is never actually executed – ``out_shape`` is
            merely used to pre-allocate the output tensor so it can be filled
            later.
        verbose: If ``True``, a progress bar will be shown while iterating over
            the tiles.
Tensor
    Returns:
        Output tensor, as a torch tensor of the same shape as the input tensor.
    """
    if not (inp.dim() - 2 == len(tile_shape) == len(overlap_shape)):
        raise ValueError(
            f'tile shape (ndim={len(tile_shape)}) and overlap shape '
            f'(ndim={len(overlap_shape)}) don\'t match input shape '
            f'(ndim={inp.dim()}.'
        )
    if not np.all(np.mod(inp.shape[2:], tile_shape) == 0):
        raise ValueError(
            f'spatial inp shape {tuple(inp.shape[2:])} has to be divisible '
            f'by tile_shape {tile_shape}.'
        )

    if offset is None:
        offset = np.zeros_like(tile_shape)
    else:
        offset = np.array(offset)
    inp_shape = np.array(inp.shape)
    out_dtype = torch.uint8 if argmax_with_threshold is not None else dtype
    out = torch.empty(out_shape, dtype=out_dtype, device='cpu')
    out_shape = np.array(out.shape)
    tile_shape = np.array(tile_shape)
    overlap_shape = np.array(overlap_shape)

    # Create padded input with overlap
    padded_shape = inp_shape + np.array((0, 0, *overlap_shape * 2))
    inp_padded = torch.empty(tuple(padded_shape), dtype=dtype, device='cpu')

    padslice = _extend_nc(
        [slice(l, h) for l, h in zip(overlap_shape, padded_shape[2:] - overlap_shape)]
    )
    inp_padded[padslice] = inp
    del inp

    # Offset is subtracted here because otherwise, the final output tensor's
    #  content will be shifted w.r.t. the input content.
    crop_low_corner = overlap_shape.copy() - offset
    crop_high_corner = tile_shape + overlap_shape - offset
    # Used to crop the output tile to the relevant, unpadded region
    #  that will be written to the final output
    final_crop_slice = _extend_nc([slice(l, h) for l, h in zip(crop_low_corner, crop_high_corner)])

    tiles = np.ceil(out_shape[2:] / tile_shape).astype(int)
    num_tiles = np.prod(tiles)

    tile_ranges = [range(t) for t in tiles]
    # TODO: Handle fractional inputshape-to-tile ratio
    pbar = tqdm(
        itertools.product(*tile_ranges), 'Predicting',
        total=num_tiles, disable=not verbose
    )
    for tile_pos in pbar:
        tile_pos = np.array(tile_pos)
        # Calculate corner coordinates of the current output tile
        out_low_corner = tile_shape * tile_pos
        out_high_corner = tile_shape * (tile_pos + 1)

        # Note: To understand why the input corners are chosen in this
        #  particular way, it helps to draw this on paper in the 1d case,
        #  representing the input tensor as a line and slicing it into
        #  input and output tiles, where input tiles have a certain overlap
        #  (note that input and output coordinates exist in two different
        #  coordinate systems: input corner coordinates are shifted "right" by
        #  the ``overlap_shape`` w.r.t. the output corner coordinate system
        #  due to the initial padding.
        inp_low_corner = out_low_corner.copy()
        inp_high_corner = out_high_corner.copy() + 2 * overlap_shape

        assert np.all(np.less_equal(inp_high_corner, inp_padded.shape[2:])), inp_high_corner
        # Slice only the current tile region in ([D,] H, W) dims
        # Slice input with overlap
        inp_slice = _extend_nc([slice(l, h) for l, h in zip(inp_low_corner, inp_high_corner)])
        # Output slice without overlap (this is the region where the current
        #  inference result will be stored)
        out_slice = _extend_nc([slice(l, h) for l, h in zip(out_low_corner, out_high_corner)])
        inp_tile = inp_padded[inp_slice].clone().detach().to(device, dtype, non_blocking=True)
        out_tile = func(inp_tile)
        # Slice the relevant tile_shape-sized region out of the model output
        #  so it can be written to the final output
        out_tile = out_tile[final_crop_slice]
        if argmax_with_threshold is not None:
            out[out_slice] = (out_tile > argmax_with_threshold).argmax(dim=1).to(torch.uint8)
        else:
            out[out_slice] = out_tile

    return out


class Predictor:
    """Class to perform inference using a ``torch.nn.Module`` object either
    passed directly or loaded from a file.

    If both ``tile_shape`` and ``overlap_shape`` are ``None``, input tensors
    are fed directly into the ``model`` (best for scalar predictions,
    medium-sized 2D images or very small 3D images).
    If you define ``tile_shape`` and ``overlap_shape``, these are used to
    slice a large input into smaller overlapping tiles and perform predictions
    on these tiles independently and later put the output tiles together into
    one dense tensor without overlap again. Use this features if your model
    has spatially interpretable (dense) outputs and if passing one input sample
    to the ``model`` would result in an out-of-memory error. For more details
    on this tiling mode, see
    :py:meth:`elektronn3.inference.inference.tiled_apply()`.

    Args:
        model: Network model to be used for inference.
            The model can be passed as an ``torch.nn.Module``, or as a path
            to either a model file or to an elektronn3 save directory:

            - If ``model`` is a ``torch.nn.Module`` object, it is used
              directly.
            - If ``model`` is a path (string) to a serialized TorchScript
              module (.pts), it is loaded from the file and mapped to the
              specified ``device``.
            - If ``model`` is a path (string) to a pickled PyTorch module (.pt)
              (**not** a pickled ``state_dict``), it is loaded from the file
              and mapped to the specified ``device`` as well.
        state_dict_src: Path to ``state_dict`` file (.pth) or loaded
            ``state_dict`` or ``None``. If not ``None``, the ``state_dict`` of
            the ``model`` is replaced with it.
        device: Device to run the inference on. Can be a ``torch.device`` or
            a string like ``'cpu'``, ``'cuda:0'`` etc.
            If not specified (``None``), available GPUs are automatically used;
            the CPU is used as a fallback if no GPUs can be found.
        batch_size: Maximum batch size with which to perform
            inference. In general, a higher ``batch_size`` will give you
            higher prediction speed, but prediction will consume more
            GPU memory. Reduce the ``batch_size`` if you run out of memory.
            If this is ``None`` (default), the input batch size is used
            as the prediction batch size.
        tile_shape: Spatial shape of the output tiles to use for inference.
            The spatial shape of the input tensors has to be divisible by
            the ``tile_shape``.
        overlap_shape: Spatial shape of the overlap by which input tiles are
            extended w.r.t. the ``tile_shape`` of the resulting output tiles.
            The ``overlap_shape`` should be close to the effective receptive
            field of the network architecture that's used for inference.
            Note that ``tile_shape + 2 * overlap`` needs to be a valid
            input shape for the inference network architecture, so
            depending on your network architecture (especially pooling layers
            and strides), you might need to adjust your ``overlap_shape``.
            If your inference fails due to shape issues, as a rule of thumb,
            try adjusting your ``overlap_shape`` so that
            ``tile_shape + 2 * overlap`` is divisible by 16 or 32.
        out_shape: Expected shape of the output tensor.
            It doesn't just refer to spatial shape, but to the actual tensor
            shape of one sample, including the channel dimension C, but
            **excluding** the batch dimension N.
            Note: ``model(inp)`` is never actually executed if tiling is used
            – ``out_shape`` is merely used to pre-allocate the output tensor so
            it can be filled later.
            If you know how many channels your model output has
            (``num_classes``, ``num_out_channels``) and if your model
            preserves spatial shape, you can easily calculate ``out_shape``
            yourself as follows:

            >>> num_out_channels: int = ?  # E.g. for binary classification it's 2
            >>> out_shape = (num_out_channels, *inp.shape[2:])
        float16: If ``True``, deploy the model in float16 (half) precision.
        apply_softmax: If ``True``
            (default), a softmax operator is automatically appended to the
            model, in order to get probability tensors as inference outputs
            from networks that don't already apply softmax.
        verbose: If ``True``, report inference speed.

    Examples:
        >>> model = nn.Sequential(
        ...     nn.Conv2d(5, 32, 3, padding=1), nn.ReLU(),
        ...     nn.Conv2d(32, 2, 1))
        >>> inp = np.random.randn(2, 5, 10, 10)
        >>> predictor = Predictor(model)
        >>> out = predictor.predict(inp)
        >>> assert np.all(np.array(out.shape) == np.array([2, 2, 10, 10]))
    """
    def __init__(
            self,
            model: Union[nn.Module, str],
            state_dict_src: Optional[Union[str, dict]] = None,
            device: Optional[Union[torch.device, str]] = None,
            batch_size: Optional[int] = None,
            tile_shape: Optional[Tuple[int, ...]] = None,
            overlap_shape: Optional[Tuple[int, ...]] = None,
            out_shape: Optional[Tuple[int, ...]] = None,
            offset: Optional[Tuple[int, ...]] = None,
            float16: bool = False,
            apply_softmax: bool = True,
            argmax_with_threshold = None,
            verbose: bool = False,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.batch_size = batch_size
        if tile_shape is not None:
            tile_shape = np.array(tile_shape)
        self.tile_shape = tile_shape
        if overlap_shape is not None:
            overlap_shape = np.array(overlap_shape)
        if offset is not None:
            offset = np.array(offset)
        self.overlap_shape = overlap_shape
        self.offset = offset
        self.out_shape = out_shape
        self.float16 = float16
        if float16 and not isinstance(model, str):
            raise NotImplementedError(
                'float16 inference is currently only supported for models '
                'that are passed as file paths (strings).'
            )
        self.dtype = torch.float16 if float16 else torch.float32
        self.argmax_with_threshold = argmax_with_threshold
        self.verbose = verbose
        if isinstance(model, str):
            if os.path.isfile(model):
                # TorchScript serialization can be identified by checking if
                #  it's a zip file. Pickled Python models are not zip files.
                #  See https://github.com/pytorch/pytorch/pull/15578/files
                if zipfile.is_zipfile(model):
                    model = torch.jit.load(model, map_location=device)
                else:
                    model = torch.load(model, map_location=device)
            else:
                raise ValueError(f'Model path {model} not found.')
        self.model = model
        if isinstance(state_dict_src, str):
            state_dict = torch.load(state_dict_src)
        elif isinstance(state_dict_src, dict) or state_dict_src is None:
            state_dict = state_dict_src
        else:
            raise ValueError(
                '"state_dict_src" has to be either a path to a .pth file (str),'
                ' a state_dict object (dict) or None.')
        if state_dict is not None:
            set_state_dict(model, state_dict)
        if apply_softmax:
            self.model = nn.Sequential(self.model, nn.Softmax(1))
        if float16:
            self.model.half()  # This is destructive. float32 params are lost!
        if self.tile_shape is None and self.overlap_shape is None:
            #  have no spatial dimensions, so tiling doesn't make sense here.
            self.enable_tiling = False
        else:
            self.enable_tiling = True
        self.model.eval()

    def _predict(self, inp: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(inp)

    def _tiled_predict(self, inp: torch.Tensor) -> torch.Tensor:
        """Tiled inference with overlapping input tiles.

        Tiling is not used if ``tile_shape`` and ``overlap_shape`` are
        undefined."""
        if self.enable_tiling:
            return tiled_apply(
                self._predict,
                inp=inp,
                device=self.device,
                dtype=self.dtype,
                tile_shape=self.tile_shape,
                overlap_shape=self.overlap_shape,
                offset=self.offset,
                out_shape=(inp.shape[0], *self.out_shape),
                argmax_with_threshold=self.argmax_with_threshold,
                verbose=self.verbose
            )
        # Otherwise: No tiling, apply model to the whole input in one step
        inp = inp.to(self.device, dtype=self.dtype, non_blocking=True)
        if self.device.type == 'cuda':
            inp.pin_memory()
        return self._predict(inp)

    def _splitbatch_predict(
            self,
            inp: torch.Tensor,
            num_batches: int,
    ) -> torch.Tensor:
        """Split the input batch into smaller batches of the specified
        ``batch_size`` and perform inference on each of them separately."""
        if self.out_shape is None:
            raise ValueError('If you define a batch_size, you also need to supply out_shape.')
        out = torch.empty((inp.shape[0], *self.out_shape), dtype=self.dtype)
        for k in range(0, num_batches):
            low = self.batch_size * k
            high = self.batch_size * (k + 1)
            out[low:high] = self._tiled_predict(inp[low:high])
        return out

    def predict(
            self,
            inp: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """ Perform prediction on ``inp`` and return prediction.

        Args:
            inp: Input data, e.g. of shape (N, C, H, W).
                Can be an ``np.ndarray`` or a ``torch.Tensor``.
                Note that ``inp`` is automatically converted to
                the specified ``dtype`` (default: ``torch.float32``) before
                inference.

        Returns:
            Model output
        """
        if self.verbose:
            start = time.time()

        from elektronn3.data import PatchCreator, transforms, utils
        try:
            print('input dist', utils.calculate_means(inp.numpy()), utils.calculate_stds(inp.numpy()))
        except:
            print('input dist', utils.calculate_means(inp), utils.calculate_stds(inp))

        orig_shape = inp.shape
        new_shape = np.array(inp.shape)
        new_shape[2:] = np.ceil(inp.shape[2:] / self.tile_shape) * self.tile_shape
        foo = np.zeros(new_shape.tolist())
        foo[0:1,0:1,0:orig_shape[2],0:orig_shape[3],0:orig_shape[4]] = inp
        inp = foo
        self.out_shape = (self.out_shape[0], *new_shape[2:])

        print(self.out_shape, orig_shape)
        inp = torch.as_tensor(inp)

        inp_batch_size = inp.shape[0]
        spatial_shape = np.array(inp.shape[2:])
        # Lazily figure out these Predictor options based on the input it
        #  receives if they are not already set.
        # Not sure if that's a good idea because these are object-changing
        #  side-effects in the otherwise pure predict() function.
        if self.tile_shape is None:
            self.tile_shape = spatial_shape
        if self.overlap_shape is None:
            self.overlap_shape = np.zeros_like(spatial_shape)
        if self.batch_size is None:
            self.batch_size = inp_batch_size
        num_batches = int(np.ceil(inp_batch_size / self.batch_size))
        if num_batches == 1:  # Predict everything in one step
            out = self._tiled_predict(inp=inp)
        else:  # Split input batch into smaller batches and predict separately
            out = self._splitbatch_predict(inp=inp, num_batches=num_batches)

        # Explicit synchronization so the next GPU operation won't be
        #  mysteriously slow. If we don't synchronize manually here, profilers
        #  will misleadingly report a huge amount of time spent in out.cpu()
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        out = out[0:1,0:self.out_shape[0],0:orig_shape[2],0:orig_shape[3],0:orig_shape[4]].cpu()
        if self.verbose:
            dtime = time.time() - start
            speed = inp.numel() / dtime / 1e6
            # TODO: Report speed in terms of output, not input (This is not as easy as replacing
            #       inp by out because out may contain padding that we don't want to count)
            print(f'Inference speed: {speed:.2f} MVox/s, time: {dtime:.2f}.')
        return out

    def predict_proba(self, inp):
        logger.warning('Predictor.predict_proba(inp) is deprecated. Please use Predictor.predict(inp) instead.')
        return self.predict(inp)



# TODO: This can be replaced with a single model.load_state_dict(state_dict) call
#       after a while, because Trainer._save_model() now always saves unwrapped
#       modules if a parallel wrapper is detected. Or should we still keep this
#       for better support of models accidentally saved in wrapped state?
def set_state_dict(model: torch.nn.Module, state_dict: dict):
    """Set state dict of a model.

    Also works with ``torch.nn.DataParallel`` models."""
    try:
        model.load_state_dict(state_dict)
    # If self.model was saved as nn.DataParallel then remove 'module.' prefix
    # in every key
    except RuntimeError:  # TODO: Is it safe to catch all runtime errors here?
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        model.load_state_dict(new_state_dict)
