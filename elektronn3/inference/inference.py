# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Martin Drawitsch
import itertools
import os
import time
from collections import OrderedDict
from typing import Optional, Tuple, Union, Callable, Sequence

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


def _extend_nc(spatial_slice: Sequence[slice]) -> Tuple[slice, ...]:
    """Extend a spatial slice ([D,] H, W) to also include the non-spatial (N, C) dims."""
    # Slice everything in N and C dims (equivalent to [:, :] in direct notation)
    nonspatial_slice = [slice(None)] * 2
    return tuple(nonspatial_slice + list(spatial_slice))


def tiled_apply(
        func: Callable[[torch.Tensor], torch.Tensor],
        inp: Union[np.ndarray, torch.Tensor],
        tile_shape: Sequence[int],
        overlap_shape: Sequence[int],
        out_shape: Sequence[int],
        final_crop_enabled: bool = True,
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

    This function assumes that ``inp.shape[2:] == func(inp).shape[2:]``,
    i.e. that the function keeps the spatial shape unchanged.

    Although this function is mainly intended for the purpose of neural network
    inference, ``func`` doesn't have to be a neural network but can be
    any ``Callable[[torch.Tensor], torch.Tensor]`` that operates on n-dimensional
    image data of shape (N, C, ...) and preserves spatial shape.
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
        out_shape: Expected shape of the output tensor that would result from
            applying ``func`` to ``inp`` (``func(inp).shape``).
            It doesn't just refer to spatial shape, but to the actual tensor
            shape including N and C dimensions.
            Note: ``func(inp)`` is never actually executed – ``out_shape`` is
            merely used to pre-allocate the output tensor so it can be filled
            later.
        final_crop_enabled: If ``True``, crop the output to not include the
            overlap regions. If ``False``, ``func`` has is expected to handle
            cropping itself.
        verbose: If ``True``, a progress bar will be shown while iterating over
            the tiles.
Tensor
    Returns:
        Output tensor, as a torch tensor of the same shape as the input tensor.
    """
    inp = torch.as_tensor(inp)
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

    if verbose:
        _tqdm = tqdm
    else:  # Make _tqdm a no-op (disabling progress indicator)
        def _tqdm(x, *_, **__):
            return x

    inp_shape = np.array(inp.shape)
    out = torch.empty(out_shape, dtype=inp.dtype)
    out_shape = np.array(out.shape)
    tile_shape = np.array(tile_shape)
    overlap = np.array(overlap_shape)

    # TODO: The comment below needs to be re-evaluated because we're now using torch instead of numpy
    # Create padded input with overlap
    # np.pad() was used here previously, but it's been replaced due to
    #  performance issues. We should give it a try again though, because maybe
    #  it was just a temporary bug (TODO). Possibly related:
    #  https://github.com/numpy/numpy/issues/11126
    padded_shape = inp_shape + np.array((0, 0, *overlap * 2))
    inp_padded = torch.zeros(tuple(padded_shape), dtype=inp.dtype)

    padslice = _extend_nc([slice(l, h) for l, h in zip(overlap, padded_shape[2:] - overlap)])
    inp_padded[padslice] = inp
    del inp

    if final_crop_enabled:
        crop_low_corner = overlap.copy()
        crop_high_corner = tile_shape + overlap
        # Used to crop the output tile to the relevant, unpadded region
        #  that will be written to the final output
        final_crop_slice = _extend_nc([slice(l, h) for l, h in zip(crop_low_corner, crop_high_corner)])

    tiles = np.ceil(out_shape[2:] / tile_shape).astype(int)
    num_tiles = np.prod(tiles)

    tile_ranges = [range(t) for t in tiles]
    # TODO: Handle fractional inputshape-to-tile ratio
    for tile_pos in _tqdm(itertools.product(*tile_ranges), total=num_tiles):
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
        inp_high_corner = out_high_corner.copy() + 2 * overlap

        assert np.all(np.less_equal(inp_high_corner, inp_padded.shape[2:])), inp_high_corner
        # Slice only the current tile region in ([D,] H, W) dims
        # Slice input with overlap
        inp_slice = _extend_nc([slice(l, h) for l, h in zip(inp_low_corner, inp_high_corner)])
        # Output slice without overlap (this is the region where the current
        #  inference result will be stored)
        out_slice = _extend_nc([slice(l, h) for l, h in zip(out_low_corner, out_high_corner)])
        inp_tile = inp_padded[inp_slice]
        out_tile = func(inp_tile).to(torch.float32)
        # Slice the relevant tile_shape-sized region out of the model output
        #  so it can be written to the final output
        if final_crop_enabled:
            out_tile = out_tile[final_crop_slice]
        out[out_slice] = out_tile
    return out


class Predictor:
    """Class to perform inference using a ``torch.nn.Module`` object either
    passed directly or loaded from a file.

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
        out_shape: Expected shape of the output tensor
            It doesn't just refer to spatial shape, but to the actual tensor
            shape including N and C dimensions.
            Note: ``model(inp)`` is never actually executed – ``out_shape`` is
            merely used to pre-allocate the output tensor so it can be filled
            later.
            If you know how many channels your model output has
            (``num_classes``, ``num_out_channels``) and if your model
            preserves spatial shape, you can easily calculate ``out_shape``
            yourself as follows:
            >>> num_out_channels: int = ?  # E.g. for binary classification it's 2
            >>> out_shape = (inp.shape[0], num_out_channels, *inp.shape[2:])
        float16: If ``True``, deploy the model in float16 (half) precision.
        apply_softmax: If ``True``
            (default), a softmax operator is automatically appended to the
            model, in order to get probability tensors as inference outputs
            from networks that don't already apply softmax.
        verbose: If ``True``, report inference speed.

    Examples:
        >>> cnn = nn.Sequential(
        ...     nn.Conv2d(5, 32, 3, padding=1), nn.ReLU(),
        ...     nn.Conv2d(32, 2, 1))
        >>> inp = np.random.randn(2, 5, 10, 10)
        >>> model = Predictor(cnn)
        >>> out = model.predict_proba(inp)
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
            float16: bool = False,
            apply_softmax: bool = True,
            verbose: bool = False,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.batch_size = batch_size
        if tile_shape is not None:
            tile_shape = np.array(tile_shape)
        self.tile_shape = tile_shape
        if overlap_shape is not None:
            overlap_shape = np.array(overlap_shape)
        self.overlap_shape = overlap_shape
        self.out_shape = out_shape
        self.float16 = float16
        if float16 and not isinstance(model, str):
            raise NotImplementedError(
                'float16 inference is currently only supported for models '
                'that are passed as file paths (strings).'
            )
        self.verbose = verbose
        if isinstance(model, str):
            if os.path.isfile(model):
                # TODO: Find a better way to find out beforehand if it's a TorchScript module.
                #  We're just pretending to know .pts means TorchScript, although no-one except
                #  us even uses this extension...
                if model.endswith('.pts'):
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
        self.model.eval()

    def _predict(self, inp: torch.Tensor) -> torch.Tensor:
        if self.float16:
            inp = inp.to(torch.float16)
        inp = inp.to(self.device)
        with torch.no_grad():
            out = self.model(inp)
            # Crop away the overlap to reduce expensive device-to-host transfers
            crop_slice = _extend_nc([
                slice(l, h)
                for l, h in zip(self.overlap_shape, self.overlap_shape + self.tile_shape)
            ])
            out = out[crop_slice].cpu()
        return out

    def _tiled_predict(self, inp: torch.Tensor) -> torch.Tensor:
        """ Tiled inference with overlapping input tiles."""
        return tiled_apply(
            self._predict,
            inp=inp,
            tile_shape=self.tile_shape,
            overlap_shape=self.overlap_shape,
            out_shape=self.out_shape,
            final_crop_enabled=False,  # See crop in self._predict()
            verbose=self.verbose
        )

    def _splitbatch_predict(
            self,
            inp: torch.Tensor,
            num_batches: int,
    ) -> torch.Tensor:
        """Split the input batch into smaller batches of the specified
        ``batch_size`` and perform inference on each of them separately."""
        out = torch.empty(tuple(self.out_shape), dtype=np.float32)
        for k in range(0, num_batches):
            low = self.batch_size * k
            high = self.batch_size * (k + 1)
            out[low:high] = self._tiled_predict(inp[low:high])
        return out

    def predict_proba(
            self,
            inp: Union[np.ndarray, torch.Tensor],
    ):
        """ Predict class probabilites of an input tensor.

        Args:
            inp: Input data, e.g. of shape (N, C, H, W).
                Can be an ``np.ndarray`` or a ``torch.Tensor``.
                Note that ``inp`` is automatically converted to
                the specified ``dtype`` (default: ``torch.float32``) before
                inference.

        Returns:
            Model output
        """
        # TODO: Maybe change signature to not require out_shape but num_classes
        #       and then calculate out_shape internally from it?
        if self.verbose:
            start = time.time()
        # Unfortunately we need to force inputs to be float32 because PyTorch
        #  lacks support for even the most basic operations on float16 tensors
        #  on CPU. If float16-mode is enabled, inputs are down-cast to float16
        #  just in time for inference in self._predict()
        inp = torch.as_tensor(inp, dtype=torch.float32)
        inp.requires_grad_(False)
        # inp.pin_memory()
        inp_batch_size = inp.shape[0]
        spatial_shape = np.array(inp.shape[2:])
        if self.tile_shape is None:
            self.tile_shape = spatial_shape
        if self.overlap_shape is None:
            self.overlap_shape = np.zeros(len(spatial_shape), dtype=np.int)
        if self.batch_size is None:
            self.batch_size = inp_batch_size
        num_batches = int(np.ceil(inp_batch_size / self.batch_size))
        if num_batches == 1:  # Predict everything in one step
            out = self._tiled_predict(inp=inp)
        else:  # Split input batch into smaller batches and predict separately
            out = self._splitbatch_predict(inp=inp, num_batches=num_batches)

        if self.verbose:
            dtime = time.time() - start
            speed = inp.numel() / dtime / 1e6
            print(f'Inference speed: {speed:.2f} MPix/s, time: {dtime:.2f}.')
        return out


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
