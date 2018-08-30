# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Martin Drawitsch

import torch
import torch.nn as nn
import glob
from collections import OrderedDict
import numpy as np
import time
from typing import Union, Optional, Callable


# TODO: Optionally apply softmax
class InferenceModel:
    """Class to perform inference using a ``torch.nn.Module`` object either
    passed directly or loaded from an elektronn3 training folder.

    Args:
        model: Path to training folder of elektronn3 model or already
            loaded/initialized ``nn.Module`` defining the model.
        state_dict_src: Path to ``state_dict`` file (.pth) or loaded
            ``state_dict`` or ``None``. If not ``None``, the ``state_dict`` of
            the ``model`` is replaced with it.
        device: Device to run the inference on. Can be a ``torch.device`` or
            a string like ``'cpu'``, ``'cuda:0'`` etc.
            If not specified (``None``), available GPUs are automatically used;
            the CPU is used as a fallback if no GPUs can be found.
        multi_gpu: Enable multi-GPU inference (using
            ``torch.nn.DataParallel``).
        model_has_softmax_outputs: If ``True``, it is assumed that the outputs
            of ``model`` are already softmax probabilities. If ``False``
            (default), a softmax operator is automatically appended to the
            model, in order to get probability tensors as inference outputs.
    Examples:
        >>> cnn = nn.Sequential(
        ...     nn.Conv2d(5, 32, 3, padding=1), nn.ReLU(),
        ...     nn.Conv2d(32, 2, 1))
        >>> inp = np.random.randn(2, 5, 10, 10)
        >>> model = InferenceModel(cnn)
        >>> out = model.predict_proba(inp)
        >>> assert np.all(np.array(out.shape) == np.array([2, 2, 10, 10]))
    """
    def __init__(
            self,
            model: nn.Module,
            state_dict_src: Optional[Union[str, dict]] = None,
            device: Optional[Union[torch.device, str]] = None,
            multi_gpu: bool = True,
            model_has_softmax_outputs: bool = False
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model = model
        if type(state_dict_src) is str:
            state_dict = torch.load(state_dict_src[0])
        elif type(state_dict_src) is dict or state_dict_src is None:
            state_dict = state_dict_src
        else:
            raise ValueError(
                '"state_dict_src" has to be either a path to a .pth file (str),'
                ' a state_dict object (dict) or None.')
        if state_dict is not None:
            try:
                self.model.load_state_dict(state_dict)
            # if self.model was saved as nn.DataParallel then remove 'module.'
            #  prefix in every key
            # TODO: Is it really safe to catch all runtime errors here?
            except RuntimeError:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict[k.replace('module.', '')] = v
                self.model.load_state_dict(new_state_dict)
        if not model_has_softmax_outputs:
            self.model = nn.Sequential(self.model, nn.Softmax(1))
        self.model.eval()
        if multi_gpu:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def predict_proba(
            self,
            inp: Union[np.ndarray, torch.Tensor],
            batch_size: int = 2,
            post_process: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            verbose: Optional[bool] = False,
            out_shape: Optional[tuple] = None,
    ):
        """

        Args:
            inp: Input data, e.g. of shape (N, C, H, W).
                Can be an ``np.ndarray`` or a ``torch.Tensor``.
                Note that ``inp`` is automatically converted to
                the specified ``dtype`` (default: ``torch.float32``) before
                inference.
            batch_size: Maximum batch size with which to perform
                inference. In general, a higher ``batch_size`` will give you
                higher prediction speed, but prediction will consume more
                GPU memory. Reduce the ``batch_size`` if you run out of memory.
            post_process: Post processing function that is executed on the
                network output tensors before returning. The expected function
                signature is ``y = post_process(x)``, where x and y both are
                ``np.ndarray``s.
            verbose: If ``True``, report inference speed.
            out_shape: Output shape, will be inferred if ``None``.

        Returns:
            Model output
        """
        # TODO (high priority!): Spatial chunking ("imposed_patch_size")
        # TODO: Refactor this function to be less complex by default, calling
        #     "special" code for spatial chunking (todo) and batch chunking
        #     (batch_size) only if needed.
        # TODO: Optimize performance, especially: Can we use cuda's async data transfer
        #       in a better way? (begin transfer earlier, defer transfer to CPU)
        if verbose:
            start = time.time()
        inp = torch.as_tensor(inp, dtype=torch.float32)
        inp_batch_size = inp.shape[0]
        with torch.no_grad():
            if out_shape is None:
                # Test inference to figure out shapes
                test_out = self.model(inp[:1].to(self.device))
                out_shape = tuple([inp_batch_size] + list(test_out.shape)[1:])
                del test_out
            out = np.empty(out_shape, dtype=np.float32)
            # Split the input batch into small batches of the specified
            # batch_size and perform inference on each of them separately
            num_batches = int(np.ceil(inp_batch_size / batch_size))
            for k in range(0, num_batches):
                low = batch_size * k
                high = batch_size * (k + 1)
                # Only moving data to GPU in the loop to save memory
                inp_stride = inp[low:high].to(self.device)
                out_stride = self.model(inp_stride)
                out[low:high] = out_stride.cpu().numpy()
                del inp_stride, out_stride  # Free some GPU memory
                # TODO: Make sure the GC does its work or manually gc.collect()
            assert high >= inp_batch_size, "Prediction has less samples than given in input."
        if post_process is not None:
            out = post_process(out)
        if verbose:
            dtime = time.time() - start
            speed = inp.numel() / dtime / 1e6
            print(f'Inference speed: {speed:.2f} MPix /s, time: {dtime:.2f}.')
        return out


# TODO: Handle multiple state dict files being available
def load_trained_model(src: str) -> InferenceModel:
    """
    Load trained elektronn3 model.

    Args:
        src: Source path to model directory. Directory must contain training
            script and model-checkpoint.pth.

    Returns:
        Trained model
    """
    # get architecture definition
    train_script = glob.glob(f'{src}/*.py')
    assert len(train_script) == 1, "Multiple/None trainer file(s). " \
                                   "Ill-defined trainer script."
    exec(open(train_script[0]).read(), globals())
    assert "get_model" in globals(), "'get_model' not defiend in trainer script."
    # get state dict path
    state_dict_p = glob.glob(f'{src}/*.pth')
    if len(state_dict_p) > 1:
        final_p = ["final" in sp for sp in state_dict_p]
        if np.sum(final_p) == 1:
            state_dict_p = [state_dict_p[np.argmax(final_p)]]
    assert len(state_dict_p) == 1, "Multiple/None state dict file(s). " \
                                   "Ill-defined state dict file."
    model = InferenceModel(get_model(), state_dict_p[0])
    return model
